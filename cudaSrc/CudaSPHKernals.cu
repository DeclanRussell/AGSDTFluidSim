//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.cu
/// @author Declan Russell
/// @date 08/03/2015
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include <math.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "CudaSPHKernals.h"
#include "cutil_math.h"  //< some math operations with cuda types

#define pi 3.14159265359f
//----------------------------------------------------------------------------------------------------------------------
__global__ void fillKernel(float* array) {
    array[threadIdx.x] = threadIdx.x * 0.5;
}
//----------------------------------------------------------------------------------------------------------------------
//our kernal to update our particle postions
__global__ void updateParticles(float3 *d_pos, float timeStep, int numParticles){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //dont want to start accessing data that doesn't exist! Could be deadly!
    if(idx<numParticles){
        d_pos[idx].x += sin(timeStep);
        d_pos[idx].y += cos(timeStep*180);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void pointHash(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float resolution, int hashTableSize){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //dont want to start accessing data that doesn't exist! Could be deadly!
    if(idx<numParticles){
        //calculate our hash key and store it in our hash key array
        //would be better to do the divide before loading in, less compute time
        float3 pos = d_posArray[idx];
        unsigned int x = floor(pos.x*resolution);
        unsigned int y = floor(pos.y*resolution);
        unsigned int z = floor(pos.z*resolution);

        d_hashArray[idx] = (((x*73856093)^(y*19349663)^(z*83492791)) % hashTableSize);
    }
}
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
/// @param _currentPos - the postions of the particle we are solving for
/// @param _neighPos - the position of the neighbouring particle we wish to calculate the weighting for
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _densKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float densityWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength, float _densKernConst){
    float rLength = length(_currentPos - _neighPos);
    if(rLength==0.0||rLength==-0.0) return 0.0f;
    float smoothMinDist = (_smoothingLength - rLength);
//    printf("const %f\n",_densKernConst);
    float weighting = _densKernConst * smoothMinDist * smoothMinDist * smoothMinDist;
    //if length of r is larger than our smoothing length we want the weighting to be zero
    //However branching conditions are slow so here is a neat little trick so we dont need one
    //false also being 0 and true being 1 solves removes the need for branching
    return weighting * (float)(rLength<_smoothingLength);
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our desity weighting kernal used in our navier stokes equations
/// @param _currentPos - the postions of the particle we are solving for
/// @param _neighPos - the position of the neighbouring particle we wish to calculate the weighting for
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _pressKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float3 pressureWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength, float _pressKernConst){
    float3 r = _currentPos - _neighPos;
    float rLength = length(r);
    if(rLength==0.0||rLength==-0.0) return make_float3(0.0f,0.0f,0.0f);
    float weighting = _pressKernConst * (_smoothingLength-rLength) * (_smoothingLength-rLength);
    r /= rLength;
    r *= weighting;
    //if length of r is larger than our smoothing length we want the weighting to be zero
    //However branching conditions are slow so here is a neat little trick so we dont need one
    //false also being 0 and true being 1 solves removes the need for branching
    return r * (float)(rLength<_smoothingLength);
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our viscosty weighting kernal used in our navier stokes equations
/// @param _currentPos - the postions of the particle we are solving for
/// @param _neighPos - the position of the neighbouring particle we wish to calculate the weighting for
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _viscKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float3 viscosityWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength, float _viscKernConst){
    float3 r = _currentPos - _neighPos;
    float rLength = length(r);
    if(!(rLength>0)) return make_float3(0.0f,0.0f,0.0f);
    float weighting = _viscKernConst * (1.0/rLength) * (_smoothingLength - rLength) * (_smoothingLength - rLength);
    r *= weighting;
    //if length of r is larger than our smoothing length we want the weighting to be zero
    //However branching conditions are slow so here is a neat little trick so we dont need one
    //false also being 0 and true being 1 solves removes the need for branching
    return r * (float)(rLength<_smoothingLength);
}

//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverPerParticleKernal(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray,unsigned int _particleIdx, unsigned int _cellOcc, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float densKernConst, float pressKernConst, float viscKernConst){

    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //make sure that what were going to access is in our range
    if(idx<_cellOcc){
        int particleIdx = _particleIdx + idx;
        // In this solver we will be exploiting the shared memory of the this block
        // to store our neighbouring particles properties rather than loading it
        // from a buffer.
        // This gives us great speed advantages! So hold on to your seats!
        // Firstly lets declare our shared piece of memory
        __shared__ particleProp nParticleData[30];

        //lets load in our particles properties to our peice of shared memory
        //Due to limits on threads if we have more particles to this key than
        //Threads we may have to sacrifice some particles to sample for less
        //overhead but hopefully we can keep this under control by having a
        //good cell size (smoothing length) in our hash function.
        //While we're at it lets store our current particle position.
        float3 curPartPos = d_posArray[particleIdx];
        float3 curPartVel = d_velArray[particleIdx];
        int samples = min(_cellOcc,30);
        if(threadIdx.x<samples){
            nParticleData[threadIdx.x].pos = curPartPos;
            nParticleData[threadIdx.x].vel = curPartVel;
        }
        //sync our threads to make sure all our particle info has been copied
        //to shared memory
        __syncthreads();

        // Calculate the density of our particle
        // Possibly could optimise this with more dynamic parralism.
        // However for such a small loop this may not actually make
        // much difference. We also will have to take into count
        // kernal launch times & memory access.
        float density = 0.0;
        float3 nPartPosTemp;
        int i;
        for(i=0;i<samples; i++){
            nPartPosTemp = nParticleData[i].pos;
            density += _particleMass * densityWeighting(curPartPos,nPartPosTemp,_smoothingLength,densKernConst);
        }

        if(threadIdx.x<samples){
            nParticleData[threadIdx.x].density = density;
        }

        //Once this is done we can finally do some navier-stokes!!
        float3 pressureForce = make_float3(0,0,0);
        float3 viscosityForce = make_float3(0.0f,0.0f,0.0f);
        float3 pressWeightTemp, viscWeightTemp;
        float3 tensionSum = make_float3(0);
        float3 tensionSumTwo = make_float3(0) ;

        float nPartDenTemp;
        float massDivDen;
        float currPressTemp,nPressTemp,p1,p2;
        for(i=0;i<samples;i++){
            nPartPosTemp = nParticleData[i].pos;
            nPartDenTemp = nParticleData[i].density;


            //calculate the pressure force
            currPressTemp = (_gasConstant * (density - _restDensity));
            p1 = (currPressTemp/(currPressTemp*currPressTemp));
            nPressTemp = (_gasConstant * (nPartDenTemp - _restDensity));
            p2 = (nPressTemp/(nPressTemp*nPressTemp));
            pressWeightTemp = pressureWeighting(curPartPos,nPartPosTemp,_smoothingLength,pressKernConst);
            pressureForce += ( p1 + p2 ) * _particleMass * pressWeightTemp;

            //calculate our viscosity force
            //if the density is zero then we will get NAN's in our devide
            //when density is very small viscosity becomes very unstable so best to have a limiter
            if(nPartDenTemp>1){
                viscWeightTemp = viscosityWeighting(curPartPos,nPartPosTemp,_smoothingLength,viscKernConst);
                viscosityForce += (curPartVel - nParticleData[i].vel) * (_particleMass/nPartDenTemp) * viscWeightTemp;
                //this is needed for surface tension
                massDivDen = _particleMass/nPartDenTemp;
                tensionSum += massDivDen * pressWeightTemp;
                tensionSumTwo += massDivDen * viscWeightTemp;
            }


        }


        //finish our fource calculations
        pressureForce *= -density;
        //printf("visc: %f,%f,%f\n",viscosityForce.x,viscosityForce.y,viscosityForce.z);
        viscosityForce *= _visCoef;


        //calculate our surface tension
        float nLength = length(tensionSumTwo);
        float3 tensionForce = make_float3(0);
        //1.0 is currently our threshold as tension becomes very unstable as n approaches 0
        if(nLength>0.5){
            //this also needs to be multipied by our tension contant
            tensionForce = (tensionSumTwo/nLength) * tensionSum;
        }



        //calculate our acceleration
        float3 gravity = make_float3(0.0f,-9.8f,0.0f);
        float3 acc = gravity + pressureForce + viscosityForce - tensionForce;
        //calculate our new velocity

        //euler intergration (Rubbish over large time steps)
        //float3 newVel = curPartVel + (acc * _timestep);
        //float3 newPos = curPartPos + (newVel * _timestep);

        //leap frog integration
        float3 velHalfBack = curPartVel - 0.5f * _timestep * d_accArray[particleIdx];
        float3 velHalfFor = velHalfBack + _timestep * acc;
        float3 newVel = (velHalfBack + velHalfFor) * 0.5f;
        float3 newPos = curPartPos + (newVel * _timestep);

        //update our particle positin and velocity
        d_velArray[particleIdx] = newVel;
        d_posArray[particleIdx] = newPos;

        //printf("vel: %f,%f,%f\n",newVel.x,newVel.y,newVel.z);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverKernalDP(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _maxNumThreads,float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float densKernConst, float pressKernConst, float viscKernConst){
    // Read in our how many particles our cell holds
    unsigned int cellOcc = d_cellOccArray[blockIdx.x];
    // Calculate our index for these particles in our buffer
    unsigned int particleIdx = d_cellIndxArray[blockIdx.x];

    // Based on how many particles we have lets calculate how many threads
    // and blocks we need for our kernal launch
    int blocks = 1;
    int threads = cellOcc;
    if(cellOcc>_maxNumThreads){
        blocks = ceil((float)cellOcc/(float)_maxNumThreads)+1;
        threads = _maxNumThreads;
    }

    // Now lets use some dynamic parallism! *Gasps*
    // Lauching a new kernal means we can have as many or as little particles
    // in a cell as we like. However the accuracy of our calculations depends
    // on the ratio of the number of particles per cell and the number of
    // sameples in our SPH calculations. More particles than samples means
    // less accuracy. More samples means more computation
    if(cellOcc>0){
        fluidSolverPerParticleKernal<<<blocks,threads>>>(d_posArray,d_velArray,d_accArray,particleIdx,cellOcc,_smoothingLength, _timestep,_particleMass,_restDensity,_gasConstant,_visCoef,densKernConst,pressKernConst,viscKernConst);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void collisionDetKernal(planeProp *d_planeArray, unsigned int _numPlanes, float3 *d_posArray, float3 *d_velArray, unsigned int _numParticles, float _timeStep){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //Make sure we're in our range
    if(idx<_numParticles){
        // Exploit some shared memory for fast access of our plane information
        extern __shared__ planeProp planes[];
        if(threadIdx.x<_numPlanes){
            planes[threadIdx.x] = d_planeArray[threadIdx.x];
        }


        // start and end points of our line segement
        float3 vel = d_velArray[idx];
        float3 newVel = vel;
        float3 pos = d_posArray[idx];
        float t = 0;
        bool intersect = false;
        //iterate through planes
        for(int i=0; i<_numPlanes; i++){
            //if its on the wrong side of the plane move it back and reflect the velocity
            //this is not 100% accurate collision, but at least it works
            if(dot(pos-planes[i].pos,planes[i].normal)<=0.0f){
                t = dot(planes[i].pos,planes[i].normal) - dot(planes[i].normal,pos);
                t/= dot(planes[i].normal,vel);
                pos = pos + vel * t;
                newVel = newVel - (2.0f * dot(newVel,planes[i].normal) * planes[i].normal);
                newVel -= (1.0 - planes[i].restCoef) * newVel * planes[i].normal;
                intersect = true;
            }
        }

        //if intersect has occured move our particle back and change our velocity
        if(intersect==true){
            d_posArray[idx] = pos;
            d_velArray[idx] = newVel;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
void calcPositions(float3 *d_pos, int timeStep, int numParticles, int maxNumThreads){
    //calculate how many blocks we want
    int blocks = ceil(numParticles/maxNumThreads)+1;
    updateParticles<<<blocks,maxNumThreads>>>(d_pos,timeStep,numParticles);
}
//----------------------------------------------------------------------------------------------------------------------
void createHashTable(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float resolution, unsigned int hashTableSize, int maxNumThreads){
    //std::cout<<"createHashTable"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(numParticles/maxNumThreads)+1;
    pointHash<<<blocks,maxNumThreads>>>(d_hashArray,d_posArray,numParticles,resolution,hashTableSize);

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
void sortByKey(unsigned int *d_hashArray, float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int _numParticles){
    //std::cout<<"sortByKey"<<std::endl;
    //Turn our raw pointers into thrust pointers so we can use
    //thrusts sort algorithm
    thrust::device_ptr<unsigned int> t_hashPtr = thrust::device_pointer_cast(d_hashArray);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(d_posArray);
    thrust::device_ptr<float3> t_velPtr = thrust::device_pointer_cast(d_velArray);
    thrust::device_ptr<float3> t_accPtr = thrust::device_pointer_cast(d_accArray);


    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_velPtr,t_accPtr)));


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
void countCellOccupancy(unsigned int *d_hashArray, unsigned int *d_cellOccArray,unsigned int _hashTableSize, unsigned int _numPoints, unsigned int _maxNumThreads){
    //std::cout<<"countCellOccupancy"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(_numPoints/_maxNumThreads)+1;
    countCellOccKernal<<<blocks,_maxNumThreads>>>(d_hashArray,d_cellOccArray,_hashTableSize,_numPoints);


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
void fluidSolver(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _hashTableSize, unsigned int _maxNumThreads, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float _densKernConst, float _pressKernConst, float _viscKernConst){
    //std::cout<<"fluidSolver"<<std::endl;
    //printf("memory allocated: %d",_maxNumThreads*(sizeof(particleProp)));
    //fluidSolverKernal<<<_hashTableSize, 30>>>(d_posArray,d_velArray,d_cellOccArray,d_cellIndxArray,_smoothingLength,_timestep, _particleMass, _restDensity,_gasConstant,_visCoef, _densKernConst, _pressKernConst, _viscKernConst);


    fluidSolverKernalDP<<<_hashTableSize, 1>>>(d_posArray,d_velArray,d_accArray,d_cellOccArray,d_cellIndxArray,_maxNumThreads,_smoothingLength,_timestep, _particleMass, _restDensity,_gasConstant,_visCoef, _densKernConst, _pressKernConst, _viscKernConst);
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
void collisionDetectionSolver(planeProp *d_planeArray, unsigned int _numPlanes, float3 *d_posArray, float3 *d_velArray, float _timeStep, unsigned int _numParticles, unsigned int _maxNumThreads){
    //calculate how many blocks we want
    int blocks = ceil(_numParticles/_maxNumThreads)+1;
    //launch collision solver
    collisionDetKernal<<<blocks,_maxNumThreads,_numPlanes*sizeof(planeProp)>>>(d_planeArray,_numPlanes,d_posArray,d_velArray,_numParticles,_timeStep);

}
//----------------------------------------------------------------------------------------------------------------------
