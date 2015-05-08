//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.cu
/// @author Declan Russell
/// @date 08/03/2015
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "CudaSPHKernals.h"
#include "helper_math.h"  //< some math operations with cuda types

#define pi 3.14159265359f

//----------------------------------------------------------------------------------------------------------------------
/// @brief Kernal designed to produce a has key based on the location of a particle
/// @brief Hash function taken from Teschner, M., Heidelberger, B., Mueller, M., Pomeranets, D. and Gross, M.
/// @brief (2003). Optimized spatial hashing for collision detection of deformable objects
/// @param d_hashArray - pointer to a buffer to output our hash keys
/// @param d_posArray - pointer to the buffer that holds our particle positions
/// @param numParticles - the number of particles in our buffer
/// @param resolution - the resolution of our hash table
/// @param _gridScaler - Scales our points to between 0-1.
//----------------------------------------------------------------------------------------------------------------------
__global__ void pointHash(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float resolution, float _gridScaler){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //dont want to start accessing data that doesn't exist! Could be deadly!
    if(idx<numParticles){
        //calculate our hash key and store it in our hash key array
        float3 normalizeCoords = d_posArray[idx]*_gridScaler;
        //if our normalized coords are not between 0-1 then we need to
        if(normalizeCoords.x<0){
            normalizeCoords.x = 0;
        }
        if(normalizeCoords.y<0){
            normalizeCoords.y = 0;
        }
        if(normalizeCoords.z<0){
            normalizeCoords.z = 0;
        }
        if(normalizeCoords.x>1){
            normalizeCoords.x = 1;
        }
        if(normalizeCoords.y>1){
            normalizeCoords.y = 1;
        }
        if(normalizeCoords.z>1){
            normalizeCoords.z = 1;
        }

        float3 gridPos = floorf(normalizeCoords*resolution);

        //give our particles a hash value
        d_hashArray[idx] = gridPos.x * resolution * resolution + gridPos.y * resolution + gridPos.z;
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This kernal is designed to count the cell occpancy of a hash table
/// @param d_hashArray - pointer to hash table buffer
/// @param d_cellOccArray - output array of cell occupancy count
/// @param _hashTableSize - the size of our hash table
/// @param _numPoints - the number of particles in our hashed array
//----------------------------------------------------------------------------------------------------------------------
__global__ void countCellOccKernal(unsigned int *d_hashArray, unsigned int *d_cellOccArray, int _hashTableSize, unsigned int _numPoints){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Make sure our idx is valid and add the occupancy count to the relevant cell
    if ((idx < _numPoints) && (d_hashArray[idx] < _hashTableSize)) {
        atomicAdd(&(d_cellOccArray[d_hashArray[idx]]), 1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our desity weighting kernal used in our navier stokes equations
/// @param _dst - the distance away of the neighbouring
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _densKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
//----------------------------------------------------------------------------------------------------------------------
__device__ float densityWeighting(float _dst,float _smoothingLength, float _densKernConst){
    float weighting = 0;
    if(_dst<=_smoothingLength){
        float temp = (_smoothingLength * _smoothingLength) - (_dst*_dst);
        weighting = _densKernConst * temp * temp * temp;
    }
    return weighting;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our desity weighting kernal used in our navier stokes equations
/// @param _r - vector from our neighbour particle to our current particle between 0<x<=smoothingLength
/// @param _dst - the distance between our particles
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _pressKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 pressureWeighting(float3 _r, float _dst,float _smoothingLength, float _pressKernConst){
    float weighting = 0.f;
    weighting = _pressKernConst * (_smoothingLength-_dst) * (_smoothingLength-_dst);
    _r/=_dst;
    _r *= weighting;
    return _r;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our viscosty weighting kernal used in our navier stokes equations
/// @param _r - vector from our neighbour particle to our current particle
/// @param _dst - the distance between our particles
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _viscKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 viscosityWeighting(float3 _r, float _dst,float _smoothingLength, float _viscKernConst){
    float weighting = 0;
    weighting = _viscKernConst * _smoothingLength - _dst;
    _r *= weighting;
    return _r;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverPerCellKernal(int _maxSamples, float3 *d_posArray, float3 *d_velArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, int _hashResolution, int _hashTableSize, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float _velCorrection, float densKernConst, float pressKernConst, float viscKernConst){


    // In this solver we will be exploiting the shared memory of the this block
    // to store our neighbouring particles properties rather than loading it
    // from a buffer.
    // This gives us great speed advantages! So hold on to your seats!
    __shared__ int cellOcc;
    if(threadIdx.x==0){
        // Read in our how many particles our cell holds
        cellOcc = d_cellOccArray[blockIdx.x];
    }
    __syncthreads();
    //If there is nothing in the cell its faster if we dont declare this shared
    //memory as just allocating it takes time!
    if(cellOcc<1){
        return;
    }
    extern __shared__ particleCellProp nParticleData[];
    __shared__ unsigned short int sampleSum;
    __shared__ unsigned int cellStartIdx;

    //get the index's of our 27 cells in parrallel and store in shared memory
    if(threadIdx.x<27){
        int resX = _hashResolution * _hashResolution;
        int x = floor((float)threadIdx.x/9.f);
        int y = floor((float)(threadIdx.x-9*x)/3.f);
        nParticleData[threadIdx.x].idx = blockIdx.x +  (x-1) * resX + (y-1) * _hashResolution + (threadIdx.x%3)-1;
    }
    __syncthreads();

    //now calculate the index positions of our neighbouring particles and store into shared memory
    //Note: This could possibly be done in parallel with the use of mutex locks to increment the sum.
    //      I however have run into many problems with these and decided to play it safe.
    int i;
    //we want to make sure that the first particles to sample are our center cell particles
    if(threadIdx.x==0){
        sampleSum = min(_maxSamples,cellOcc);
        // Calculate our index for these particles in our buffer
        cellStartIdx = d_cellIndxArray[blockIdx.x];
        int j;
        for(i=0;i<27;i++){
            if(sampleSum>=_maxSamples) break;
            if(nParticleData[i].idx>0&&nParticleData[i].idx<_hashTableSize){
                if(nParticleData[i].idx==blockIdx.x) continue;
                int nCellStart = d_cellIndxArray[nParticleData[i].idx];
                //see how much space we have left in our shared memory
                int nStart = sampleSum;
                int dif = min(_maxSamples-nStart,d_cellOccArray[nParticleData[i].idx]);
                sampleSum+=dif;
                //load in our neightbour data idecies
                for(j=0; j<dif;j++){
                    nParticleData[nStart+j].idx = nCellStart+j;
                }
            }
        }
    }
    __syncthreads();
    //now actually load in our particle data to shared memory
    if(threadIdx.x<cellOcc){
        nParticleData[threadIdx.x].idx = cellStartIdx+threadIdx.x;
    }
    if(threadIdx.x<sampleSum){
            nParticleData[threadIdx.x].pos = d_posArray[nParticleData[threadIdx.x].idx];
            nParticleData[threadIdx.x].vel = d_velArray[nParticleData[threadIdx.x].idx];
    }
    __syncthreads();
    // Calculate the density of our particle
    nParticleData[threadIdx.x].density = 0;
    for(i=0;i<sampleSum; i++){
        nParticleData[threadIdx.x].density += _particleMass * densityWeighting(length(nParticleData[threadIdx.x].pos-nParticleData[i].pos),_smoothingLength,densKernConst);
    }
    __syncthreads();
    //make sure that what were going to access is in our range
    if(threadIdx.x<cellOcc){
        //Once this is done we can finally do some navier-stokes!!
        float3 viscosityForce = make_float3(0.0f,0.0f,0.0f);
        float3 acc = make_float3(0.0f,0.0f,0.0f);
        {
            float massDivDen;
            float3 pressWeightTemp, viscWeightTemp, r;
            float currPressTemp,nPressTemp,dst;
            //calculate the pressure force of our current particle
//            currPressTemp = (_gasConstant * (nParticleData[threadIdx.x].density - _restDensity));
            currPressTemp = (_gasConstant * (nParticleData[threadIdx.x].density - _restDensity))/(nParticleData[threadIdx.x].density*nParticleData[threadIdx.x].density);
            //if(currPressTemp<_restDensity)printf("p %f\n",currPressTemp);
            for(i=0;i<sampleSum;i++){
                r = nParticleData[threadIdx.x].pos - nParticleData[i].pos;
                dst = length(r);
                //branching conditions are bad but you can't devide by zero
                if(nParticleData[threadIdx.x].density>0&&nParticleData[i].density>0){
                    //less calculations if we have a smoothing range tested here rather than in smoothing device functions
                    if(dst>0 && dst<=_smoothingLength){
                        //calculate the pressure force of our neighbour particle
                        nPressTemp = (_gasConstant * (nParticleData[i].density - _restDensity));
                        pressWeightTemp = pressureWeighting(r,dst,_smoothingLength,pressKernConst);
                        acc-=  (currPressTemp + (nPressTemp/(nParticleData[i].density*nParticleData[i].density))) * pressWeightTemp;
                        //acc-=( (currPressTemp + nPressTemp)/(2.f*nParticleData[i].density)) * pressWeightTemp;
                        //calculate our viscosity force
                        viscWeightTemp = viscosityWeighting(r,dst,_smoothingLength,viscKernConst);
                        massDivDen = _particleMass/nParticleData[i].density;
                        viscosityForce += (nParticleData[threadIdx.x].vel - nParticleData[i].vel) * massDivDen * viscWeightTemp;
                    }
                }
            }
        }

        //Finish our calculations
        acc*=_particleMass;
        acc+=make_float3(0.f,-9.8f,0.f) + viscosityForce*_visCoef;

        if(acc.x!=acc.x)printf("d %d\n",nParticleData[threadIdx.x].density);

        //calculate our new velocity
        //euler intergration (Rubbish over large time steps)
        //if your looking for instability uncomment here
        //float3 newVel = nParticleData[threadIdx.x].vel + (acc * _timestep);
        //float3 newPos = nParticleData[threadIdx.x].pos + (newVel * _timestep);

        //leap frog integration
        //more stable if we move by half steps than full
        float3 velHalfBack = nParticleData[threadIdx.x].vel - 0.5f * _timestep * acc;
        float3 velHalfFor = velHalfBack + _timestep * acc;

        //XSPH velocity correction
        //can be found in Paiva, A., Petronetto, F., Lewiner, T. and Tavares, G. (2009).
        //Particle-based viscoplastic fluid/solid simulation,
        //To achieve this lets take advantage of our shared memory again
        nParticleData[threadIdx.x].vel = velHalfFor;
        __syncthreads();
        float3 newVel = make_float3(.0f);
        for(i=0;i<sampleSum;i++){
            if((nParticleData[threadIdx.x].density>0)&&(nParticleData[i].density>0))
            newVel += (2.0f/(nParticleData[threadIdx.x].density+nParticleData[i].density)) * (nParticleData[i].vel - velHalfBack) * densityWeighting(length(nParticleData[threadIdx.x].pos-nParticleData[i].pos),_smoothingLength,densKernConst);
        }
        newVel = velHalfFor + _velCorrection * newVel;

        //finally we calculate our position from our velocity
        float3 newPos = nParticleData[threadIdx.x].pos + (newVel * _timestep);


        //update our particle position and velocity
        d_velArray[nParticleData[threadIdx.x].idx] = newVel;
        d_posArray[nParticleData[threadIdx.x].idx] = newPos;

    }
}

//----------------------------------------------------------------------------------------------------------------------
/// @brief implimentation of our container collide function. Very simple AABB but it works!
/// @param _pos - particle position to collide with
/// @param _vel - velocity of the particle
//----------------------------------------------------------------------------------------------------------------------
__device__ void SimpleCuboidCollisionObject::collide(float3 &_pos, float3 &_vel){
    if(_pos.x<p1.x){
        _pos.x = p1.x;
        _vel.x = -_vel.x * restitution.x;
    }
    if(_pos.x>p2.x){
        _pos.x = p2.x;
        _vel.x = -_vel.x * restitution.x;
    }
    if(_pos.y<p1.y){
        _pos.y = p1.y;
        _vel.y = -_vel.y * restitution.y;
    }
    if(_pos.y>p2.y){
        _pos.y = p2.y;
        _vel.y = -_vel.y * restitution.y;
    }
    if(_pos.z<p1.z){
        _pos.z = p1.z;
        _vel.z = -_vel.z * restitution.z;
    }
    if(_pos.z>p2.z){
        _pos.z = p2.z;
        _vel.z = -_vel.z * restitution.z;
    }

}

//----------------------------------------------------------------------------------------------------------------------
__global__ void collisionDetKernal(SimpleCuboidCollisionObject *d_CollisionObjectArray, unsigned int _numObjects, float3 *d_posArray, float3 *d_velArray, unsigned int _numParticles, float _timeStep){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //Make sure we're in our range
    if(idx<_numParticles){
        // Exploit some shared memory for fast access of our plane information
        extern __shared__ SimpleCuboidCollisionObject objects[];
        if(threadIdx.x<_numObjects){
            objects[threadIdx.x] = d_CollisionObjectArray[threadIdx.x];
        }
        __syncthreads();

        float3 pos = d_posArray[idx];
        float3 vel = d_velArray[idx];
        for(int i=0; i<_numObjects;i++){
            objects[i].collide(pos,vel);
        }
        d_posArray[idx] = pos;
        d_velArray[idx] = vel;
    }

}

//----------------------------------------------------------------------------------------------------------------------
void createHashTable(cudaStream_t _stream, unsigned int* d_hashArray, float3* d_posArray, unsigned int _numParticles, float _smoothingLength, float _gridSize, int _maxNumThreads){
    //std::cout<<"createHashTable"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(_numParticles/_maxNumThreads)+1;
    pointHash<<<blocks,_maxNumThreads,0,_stream>>>(d_hashArray,d_posArray,_numParticles,_gridSize/_smoothingLength,1.0f/_gridSize);


    //DEBUG: uncomment to print out counted cell occupancy, WARNING SUPER SLOW!
    //thrust::device_ptr<unsigned int> t_hashPtr = thrust::device_pointer_cast(d_hashArray);
    //thrust::copy(t_hashPtr, t_hashPtr+_numParticles, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("createHashTable CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void sortByKey(unsigned int *d_hashArray, float3 *d_posArray, float3 *d_velArray, unsigned int _numParticles){
    //std::cout<<"sortByKey"<<std::endl;
    //Turn our raw pointers into thrust pointers so we can use
    //thrusts sort algorithm
    thrust::device_ptr<unsigned int> t_hashPtr = thrust::device_pointer_cast(d_hashArray);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(d_posArray);
    thrust::device_ptr<float3> t_velPtr = thrust::device_pointer_cast(d_velArray);


    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_velPtr)));


    //DEBUG: uncomment to print out sorted hash keys
    //thrust::copy(t_hashPtr, t_hashPtr+_numParticles, std::ostream_iterator<unsigned int>(std::cout, " "));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("sortByKey CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
void countCellOccupancy(cudaStream_t _stream,unsigned int *d_hashArray, unsigned int *d_cellOccArray,unsigned int _hashTableSize, unsigned int _numPoints, unsigned int _maxNumThreads){
    //std::cout<<"countCellOccupancy"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(_numPoints/_maxNumThreads)+1;
    countCellOccKernal<<<blocks,_maxNumThreads,0,_stream>>>(d_hashArray,d_cellOccArray,_hashTableSize,_numPoints);


    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::device_ptr<unsigned int> t_occPtr = thrust::device_pointer_cast(d_cellOccArray);
    //thrust::copy(t_occPtr, t_occPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("countCellOccupancy CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void fillUint(unsigned int *_pointer, unsigned int _arraySize, unsigned int _fill){
    //std::cout<<"fillUint"<<std::endl;
    //Turn our raw pointers into thrust pointers so we can use
    //them in thrust fill
    thrust::device_ptr<unsigned int> t_Ptr = thrust::device_pointer_cast(_pointer);
    //fill our buffer
    thrust::fill(t_Ptr, t_Ptr+_arraySize, _fill);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("FillUint CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
void createCellIdx(unsigned int* d_cellOccArray, unsigned int _size,unsigned int* d_cellIdxArray){
    //std::cout<<"createCellIdx"<<std::endl;
    //Turn our raw pointers into thrust pointers so we can use
    //them in thrust
    thrust::device_ptr<unsigned int> t_cellOccPtr = thrust::device_pointer_cast(d_cellOccArray);
    thrust::device_ptr<unsigned int> t_cellIdxPtr = thrust::device_pointer_cast(d_cellIdxArray);
    //run an excludive scan on our arrays
    thrust::exclusive_scan(t_cellOccPtr,t_cellOccPtr+_size,t_cellIdxPtr);

    //DEBUG: uncomment to print out cell index buffer
    //thrust::copy(t_cellIdxPtr, t_cellIdxPtr+_size, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<std::endl;
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("createCellIdx CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void fluidSolver(cudaStream_t _stream, float3 *d_posArray, float3 *d_velArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _hashTableSize, int _hashResolution, unsigned int _maxNumThreads, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float _velCorrection, float _densKernConst, float _pressKernConst, float _viscKernConst){
    //std::cout<<"fluidSolver"<<std::endl;
    //printf("memory allocated: %d",_maxNumThreads*(sizeof(particleCellProp)));
    //fluidSolverKernal<<<_hashTableSize, 30>>>(d_posArray,d_velArray,d_cellOccArray,d_cellIndxArray,_smoothingLength,_timestep, _particleMass, _restDensity,_gasConstant,_visCoef, _densKernConst, _pressKernConst, _viscKernConst);

    int totalSamples = 500;
    fluidSolverPerCellKernal<<<_hashTableSize,totalSamples,totalSamples*sizeof(particleCellProp),_stream>>>(totalSamples,d_posArray,d_velArray,d_cellOccArray,d_cellIndxArray,_hashResolution,_hashTableSize,_smoothingLength,_timestep,_particleMass,_restDensity,_gasConstant,_visCoef,_velCorrection,_densKernConst,_pressKernConst,_viscKernConst);

    //std::cout<<std::endl;

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fluid solver CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}

//----------------------------------------------------------------------------------------------------------------------
void collisionDetectionSolver(cudaStream_t _stream, SimpleCuboidCollisionObject *d_collObjArray, unsigned int _numObjects, float3 *d_posArray, float3 *d_velArray, float _timeStep, unsigned int _numParticles, unsigned int _maxNumThreads){
    //calculate how many blocks we want
    int blocks = ceil(_numParticles/_maxNumThreads)+1;
    //launch collision solver
    collisionDetKernal<<<blocks,_maxNumThreads,_numObjects*sizeof(SimpleCuboidCollisionObject),_stream>>>(d_collObjArray,_numObjects,d_posArray,d_velArray,_numParticles,_timeStep);

}
