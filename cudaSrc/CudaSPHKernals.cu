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
__device__ float densityWeighting(float _dst,float _smoothingLength, float _densKernConst){
    float weighting = 0;
    if(_dst>0 && _dst<=_smoothingLength){
        float temp = (_smoothingLength * _smoothingLength) - (_dst*_dst);
        weighting = _densKernConst * temp * temp * temp;
    }
    return weighting;
}
//----------------------------------------------------------------------------------------------------------------------
/// @brief This is our desity weighting kernal used in our navier stokes equations
/// @param _r - vector from our neighbour particle to our current particle
/// @param _dst - the distance between our particles
/// @param _smoothingLength - the smoothing length of our simulation. Can be thought of a hash cell size.
/// @param _pressKernCosnt - constant part of our kernal. Easier to calculate once on CPU and have loaded into device kernal.
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float3 pressureWeighting(float3 _r, float _dst,float _smoothingLength, float _pressKernConst){
    float weighting = 0.f;
    if(_dst>0 && _dst<=_smoothingLength){
        weighting = _pressKernConst * (_smoothingLength-_dst) * (_smoothingLength-_dst);
        _r/=_dst;
    }
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
__device__ float3 viscosityWeighting(float3 _r, float _dst,float _smoothingLength, float _viscKernConst){
    float weighting = 0;
    if(_dst>0 && _dst<=_smoothingLength){
        weighting = _viscKernConst * _smoothingLength - _dst;
    }
    _r *= weighting;
    return _r;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverPerCellKernal(int _maxSamples, float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, int _hashResolution, int _hashTableSize, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float densKernConst, float pressKernConst, float viscKernConst){


    // In this solver we will be exploiting the shared memory of the this block
    // to store our neighbouring particles properties rather than loading it
    // from a buffer.
    // This gives us great speed advantages! So hold on to your seats!
    // Firstly lets declare our shared piece of memory
    __shared__ int cellOcc;
    __shared__ unsigned short int sampleSum;
    __shared__ unsigned int cellStartIdx;

    if(threadIdx.x==0){
        sampleSum=0;
        // Read in our how many particles our cell holds
        cellOcc = d_cellOccArray[blockIdx.x];
        // Calculate our index for these particles in our buffer
        cellStartIdx = d_cellIndxArray[blockIdx.x];
    }
    __syncthreads();
    //If there is nothing in the cell its faster if we dont declare this shared
    //memory as just allocating it takes time!
    if(!cellOcc) return;
    extern __shared__ particleCellProp nParticleData[];
    sampleSum = min(_maxSamples,cellOcc);

    int i;
    if(threadIdx.x==0){
        //load in our neighbouring cells
        int resX = _hashResolution * _hashResolution;
        unsigned int x;
        unsigned int y;
        unsigned int nCellIdx;
        int j;
        for(i=0;i<27;i++){
            if(sampleSum>=_maxSamples) break;
            x = floor((float)i/(float)9);
            y = floor((float)(i -9*x)/(float)3);
            nCellIdx = blockIdx.x +  (x-1u) * resX + (y-1u) * _hashResolution + (i%3-1);
            if(nCellIdx>0&&nCellIdx<_hashTableSize){
                if(nCellIdx==blockIdx.x) continue;
                int nCellStart = d_cellIndxArray[nCellIdx];
                //see how much space we have left in our shared memory
                int nStart = sampleSum;
                int dif = min(_maxSamples-nStart,d_cellOccArray[nCellIdx]);
                sampleSum+=dif;
                //load in our neightbour data
                for(j=0; j<dif;j++){
                    nParticleData[nStart+j].idx = nCellStart+j;
                }
            }
        }
    }
    __syncthreads();
    float3 curPartPos;
    float3 curPartVel;
    int curPartIdx;
    if(threadIdx.x<sampleSum){
        if(threadIdx.x<cellOcc){
            curPartIdx = cellStartIdx+threadIdx.x;
            curPartPos = nParticleData[threadIdx.x].pos = d_posArray[curPartIdx];
            curPartVel = nParticleData[threadIdx.x].vel = d_velArray[curPartIdx];
        }
        else{
            curPartPos = nParticleData[threadIdx.x].pos = d_posArray[nParticleData[threadIdx.x].idx];
            curPartVel = nParticleData[threadIdx.x].vel = d_velArray[nParticleData[threadIdx.x].idx];
        }
    }

    // Calculate the density of our particle
    float density = 0.0;
    for(i=0;i<sampleSum; i++){
        density += _particleMass * densityWeighting(length(curPartPos-nParticleData[i].pos),_smoothingLength,densKernConst);
    }
    //load result into our shared memory
    if(threadIdx.x<sampleSum){
        nParticleData[threadIdx.x].density = density;
    }

    //make sure that what were going to access is in our range
    if(threadIdx.x<cellOcc){
        //multiplication faster than divide so lets just do it once
        float invDensity = 1.0f/density;
        __syncthreads();
        //Once this is done we can finally do some navier-stokes!!
        float3 viscosityForce = make_float3(0.0f,0.0f,0.0f);
        float3 tensionSum = make_float3(0);
        float3 tensionSumTwo = make_float3(0);
        float3 acc = make_float3(0.0f,0.0f,0.0f);
        {
            float massDivDen;
            float3 pressWeightTemp, viscWeightTemp, r;
            float currPressTemp,nPressTemp,p1,p2,dst;
            //calculate the pressure force of our current particle
            currPressTemp = (_gasConstant * (density - _restDensity));
            p1 = (currPressTemp*invDensity*invDensity);
            for(i=0;i<sampleSum;i++){
                //cuda seems to struggle when testing for >0 No idea why
                if(nParticleData[i].density>1 && density>1){
                //if(nParticleData[i].density>0 && density>0){
                    //calculate the pressure force of our neighbour particle
                    nPressTemp = (_gasConstant * (nParticleData[i].density - _restDensity));
                    p2 = (nPressTemp/(nParticleData[i].density*nParticleData[i].density));
                    r = curPartPos - nParticleData[i].pos;
                    dst = length(r);
                    pressWeightTemp = pressureWeighting(r,dst,_smoothingLength,pressKernConst);
                    acc -= ( p1 + p2 ) * pressWeightTemp;
                    //if(acc.x!=acc.x){
                    //    printf("balls acc %f,%f,%f cp %f cd %f cid %f np %f \nnd %f m %f pw %f,%f,%f cTn %f,%f,%f\n",acc.x,acc.y,acc.z,currPressTemp,density,invDensity,nPressTemp,nParticleData[i].density,_restDensity,_particleMass,pressWeightTemp.x,pressWeightTemp.y,pressWeightTemp.z,r.x,r.y,r.z);
                    //    if(density>0) printf("cd %f\n",density);
                    //}
                    //if(length(acc)>12)printf("acc %f,%f,%f  den %f nDen %f _dst %f w%f,%f,%f   cellOcc %d  nIdex %d\n",acc.x,acc.y,acc.z,density,nParticleData[i].density,dst,pressWeightTemp.x,pressWeightTemp.y,pressWeightTemp.z,cellOcc,i);
                    //calculate our viscosity force
                    viscWeightTemp = viscosityWeighting(r,dst,_smoothingLength,viscKernConst);
                    massDivDen = _particleMass/nParticleData[i].density;
                    viscosityForce += (curPartVel - nParticleData[i].vel) * massDivDen * viscWeightTemp;
                    //this is needed for surface tension
                    tensionSum += massDivDen * pressWeightTemp;
                    tensionSumTwo += massDivDen * viscWeightTemp;
                }
            }
        }


//        //calculate our surface tension
//        float3 tensionForce = make_float3(0);
//        {
//            float nLength = length(tensionSumTwo);
//            //1.0 is currently our threshold as tension becomes very unstable as n approaches 0
//            if(nLength>0.5){
//                //this also needs to be multipied by our tension contant
//                tensionForce = (tensionSumTwo/nLength) * tensionSum;
//            }
//        }

        acc*=_particleMass;
        acc+=make_float3(0.0f,-9.8f,0.0f);
        //if(density>0) acc+= (viscosityForce*_visCoef /*+ tensionForce*/)*invDensity;


        //calculate our new velocity
        //euler intergration (Rubbish over large time steps)
//        float3 newVel = curPartVel + (acc * _timestep);
//        float3 newPos = curPartPos + (newVel * _timestep);

        //leap frog integration
        //more stable if we move by half steps than full
        float3 velHalfBack = curPartVel - 0.5f * _timestep * d_accArray[curPartIdx];
        float3 velHalfFor = velHalfBack + _timestep * acc;
        //printf("pf %f,%f,%f vel %f,%f,%f\n",acc.x,acc.y,acc.z,velHalfFor.x,velHalfFor.y,velHalfFor.z);


        //XSPH velocity correction
        //can be found in Paiva, A., Petronetto, F., Lewiner, T. and Tavares, G. (2009).
        //Particle-based viscoplastic fluid/solid simulation,
        //To achieve this lets take advantage of our shared memory again
        nParticleData[threadIdx.x].vel = velHalfFor;
        __syncthreads();
        float3 newVel = make_float3(.0f);
        for(i=0;i<cellOcc;i++){
            if((density>0)&&(nParticleData[i].density>0))
            newVel += (2.0f*_particleMass/(density+nParticleData[i].density)) * (nParticleData[i].vel - velHalfBack) * densityWeighting(length(curPartPos-nParticleData[i].pos),_smoothingLength,densKernConst);
        }

        newVel = velHalfFor + 0.3f * newVel;
        float3 newPos = curPartPos + (newVel * _timestep);
        //printf("p %f,%f,%f\n",newPos.x,newPos.y,newPos.z);
//        if(newPos.x!=newPos.x){
//            printf("shit currentpos %f,%f,%f newPos %f,%f,%f samples %d\n",curPartPos.x,curPartPos.y,curPartPos.z,newPos.x,newPos.y,newPos.z,sampleSum);
//            printf("vel: %f,%f,%f\n",newVel.x,newVel.y,newVel.z);
//            printf("timeStep %f acc %f,%f,%f\n\n",_timestep,acc.x,acc.y,acc.z);
//        }
        //update our particle positin and velocity
        d_velArray[curPartIdx] = newVel;
        d_posArray[curPartIdx] = newPos;
        d_accArray[curPartIdx] = acc;


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
        float dst = 0;
        bool intersect = false;
        float3 pToO;
        int itersectPlane;
        //iterate through planes
        for(int i=0; i<_numPlanes; i++){
            //if its on the wrong side of the plane move it back and reflect the velocity
            //this is not 100% accurate collision, but at least it works!!
            if(dot(pos-planes[i].pos,planes[i].normal)<0.f){
                //move particle to closest point on plane
                pToO = pos-planes[i].pos;
                dst = dot(pToO,planes[i].normal);
                pos = pos-(dst*planes[i].normal);
                //reflect velocity arount normal
                newVel =  newVel - (2.0f * dot(newVel,planes[i].normal) * planes[i].normal);
                itersectPlane = i;
                intersect=true;
            }
        }

        //if intersect has occured move our particle back and change our velocity
        if(intersect==true){
            d_posArray[idx] = pos;
            d_velArray[idx] = newVel - (1.0 - planes[itersectPlane].restCoef) * newVel * fabs(planes[itersectPlane].normal);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
void createHashTable(unsigned int* d_hashArray, float3* d_posArray, unsigned int _numParticles, float _smoothingLength, float _gridSize, int _maxNumThreads){
    //std::cout<<"createHashTable"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(_numParticles/_maxNumThreads)+1;
    pointHash<<<_numParticles,1>>>(d_hashArray,d_posArray,_numParticles,_gridSize/_smoothingLength,1.0f/_gridSize);


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
void fluidSolver(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _hashTableSize, int _hashResolution, unsigned int _maxNumThreads, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float _densKernConst, float _pressKernConst, float _viscKernConst){
    //std::cout<<"fluidSolver"<<std::endl;
    //printf("memory allocated: %d",_maxNumThreads*(sizeof(particleCellProp)));
    //fluidSolverKernal<<<_hashTableSize, 30>>>(d_posArray,d_velArray,d_cellOccArray,d_cellIndxArray,_smoothingLength,_timestep, _particleMass, _restDensity,_gasConstant,_visCoef, _densKernConst, _pressKernConst, _viscKernConst);

    int totalSamples = 300;
    fluidSolverPerCellKernal<<<_hashTableSize,totalSamples,totalSamples*sizeof(particleCellProp)>>>(totalSamples,d_posArray,d_velArray,d_accArray,d_cellOccArray,d_cellIndxArray,_hashResolution,_hashTableSize,_smoothingLength,_timestep,_particleMass,_restDensity,_gasConstant,_visCoef,_densKernConst,_pressKernConst,_viscKernConst);


    //dynamic parrallelism version. Slow!
//    int blocks = ceil(10000/_maxNumThreads)+1;
//    fluidSolverKernalDP<<<blocks,_maxNumThreads>>>(d_posArray,d_velArray,d_accArray,d_cellOccArray,d_cellIndxArray,_hashResolution,_hashTableSize,_smoothingLength,_timestep,_particleMass,_restDensity,_gasConstant,_visCoef,_densKernConst,_pressKernConst,_viscKernConst);

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
