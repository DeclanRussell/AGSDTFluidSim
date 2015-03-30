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
__global__ void pointHash(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float smoothingLegnth, int hashTableSize){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //dont want to start accessing data that doesn't exist! Could be deadly!
    if(idx<numParticles){
        //calculate our hash key and store it in our hash key array
        unsigned int x = floor(d_posArray[idx].x/smoothingLegnth);
        unsigned int y = floor(d_posArray[idx].y/smoothingLegnth);
        unsigned int z = floor(d_posArray[idx].z/smoothingLegnth);

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
    float smoothMinDist = (_smoothingLength - rLength);
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
    float weighting = _viscKernConst * (1.0/rLength) * (_smoothingLength - rLength) * (_smoothingLength - rLength);
    r *= weighting;
    //if length of r is larger than our smoothing length we want the weighting to be zero
    //However branching conditions are slow so here is a neat little trick so we dont need one
    //false also being 0 and true being 1 solves removes the need for branching
    return r * (float)(rLength<_smoothingLength);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverKernal(float3 *d_posArray, float3 *d_velArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray,float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float densKernConst, float pressKernConst, float viscKernConst){

    // Read in our how many particles our cell holds
    unsigned int cellOcc = d_cellOccArray[blockIdx.x];
    // Calculate our index for these particles in our buffer
    unsigned int particleIdx = d_cellIndxArray[blockIdx.x] + threadIdx.x;
    // In this solver we will be exploiting the shared memory of the this block
    // to store our neighbouring particles properties rather than loading it
    // from a buffer.
    // This gives us great speed advantages! So hold on to your seats!
    // Firstly lets declare our shared piece of memory
    // The extern keyword means we can dynamically size our shared memory
    // in the third argument when we call our kernal.
    __shared__ particleProp nParticleData[30];


    //make sure we're not doing anything to particles that are not in our cell
    if(threadIdx.x<30/*cellOcc*/){
        //lets load in our particles properties to our peice of shared memory
        //Due to limits on threads if we have more particles to this key than
        //Threads we may have to sacrifice some particles to sample for less
        //overhead but hopefully we can keep this under control by having a
        //good cell size (smoothing length) in our hash function.
        //While we're at it lets store our current particle position.
        float3 curPartPos = d_posArray[particleIdx];
        nParticleData[threadIdx.x].pos = curPartPos;
        float3 curPartVel = d_velArray[particleIdx];
        nParticleData[threadIdx.x].vel = curPartVel;
        //sync our threads to make sure all our particle info has been copied
        //to shared memory
        __syncthreads();

        //calculate the density of our particle
        float density = 0;
        float sameCheck;
        float3 nPartPosTemp;
        int i;
        for(i=0;i<cellOcc&&i<30; i++){
            //if our neightbour particle is our current particle then we dont
            //want it to be effect our calculations. If we use this we can
            //discard any calculations with it without creating branching
            //conditions
            nPartPosTemp = nParticleData[i].pos;
            sameCheck = (float)!(threadIdx.x == i);
            density += _particleMass * densityWeighting(curPartPos,nPartPosTemp,_smoothingLength,densKernConst) * sameCheck;
        }
        nParticleData[threadIdx.x].density = density;

        //sync threads so we know that all our particles densitys have been calculated
        __syncthreads();

        //Once this is done we can finally do some navier-stokes!!
        float3 pressureForce = make_float3(0,0,0);
        float3 viscosityForce = make_float3(0,0,0);
        float nPartDenTemp;
        float currPressTemp,nPressTemp;
        for(i=0;i<cellOcc&&i<30;i++){
            nPartPosTemp = nParticleData[i].pos;
            nPartDenTemp = nParticleData[i].density;
            //if our particle is in exacly the same position as our current particle
            //then we dont want to do calculations, being not physically possible we can get errors
            if((curPartPos.x==nPartPosTemp.x)&&(curPartPos.y==nPartPosTemp.y)&&(curPartPos.z==nPartPosTemp.z)) continue;
            //calculate the pressure force
            currPressTemp = (_gasConstant * (density - _restDensity));
            nPressTemp = (_gasConstant * (nPartDenTemp - _restDensity));
            pressureForce += ( (currPressTemp/(currPressTemp*currPressTemp)) + (nPressTemp/(nPressTemp*nPressTemp)) ) * _particleMass * pressureWeighting(curPartPos,nPartPosTemp,_smoothingLength,pressKernConst);
            //calculate our viscosity force
            viscosityForce += (curPartVel - nParticleData[i].vel) * (_particleMass/nPartDenTemp) * viscosityWeighting(curPartPos,nPartPosTemp,_smoothingLength,viscKernConst);
        }
        pressureForce *= -density;
        viscosityForce *= _visCoef;

        //calculate our acceleration
        float3 gravity = make_float3(0.0,-9.8,0.0);
        float3 acc = gravity + pressureForce + viscosityForce;
        //calculate our new velocity

        //euler intergration
        float3 newVel = curPartVel + (acc * _timestep);
        float3 newPos = curPartPos + (newVel * _timestep);

        //leap frog integration
//        float3 velHalfBack = curPartVel - ( acc * _timestep * 0.5);
//        float3 velHalfForward = velHalfBack + (acc * _timestep);
//        float3 newVel = (velHalfBack + velHalfForward) * 0.5;
//        float3 newPos = curPartPos + (newVel * _timestep);

        //printf("vel: %f,%f,%f\n",newVel.x,newVel.y,newVel.z);

        //printf("vel: %f,%f,%f\n",newVel.x,newVel.y,newVel.z);


        //update our particle positin and velocity
        d_velArray[particleIdx] = newVel;
        d_posArray[particleIdx] = newPos;
        //d_posArray[particleIdx].y+=0.1;
    }
}

//----------------------------------------------------------------------------------------------------------------------
void fillGpuArray(float* array, int count) {
    fillKernel<<<1, count>>>(array);
}
//----------------------------------------------------------------------------------------------------------------------
void calcPositions(float3 *d_pos, int timeStep, int numParticles, int maxNumThreads){
    //calculate how many blocks we want
    int blocks = ceil(numParticles/maxNumThreads)+1;
    updateParticles<<<blocks,maxNumThreads>>>(d_pos,timeStep,numParticles);
}
//----------------------------------------------------------------------------------------------------------------------
void createHashTable(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float smoothingLegnth, unsigned int hashTableSize, int maxNumThreads){
    //std::cout<<"createHashTable"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(numParticles/maxNumThreads)+1;
    pointHash<<<blocks,maxNumThreads>>>(d_hashArray,d_posArray,numParticles,smoothingLegnth,hashTableSize);

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
void sortByKey(unsigned int *d_hashArray, float3 *d_posArray,float3 *d_velArray, unsigned int _numParticles){
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
void countCellOccupancy(unsigned int *d_hashArray, unsigned int *d_cellOccArray,unsigned int _hashTableSize, unsigned int _numPoints, unsigned int _maxNumThreads){
    //std::cout<<"countCellOccupancy"<<std::endl;
    //calculate how many blocks we want
    int blocks = ceil(_hashTableSize/_maxNumThreads)+1;
    countCellOccKernal<<<blocks,_maxNumThreads>>>(d_hashArray,d_cellOccArray,_hashTableSize,_numPoints);


    //DEBUG: uncomment to print out counted cell occupancy
    thrust::device_ptr<unsigned int> t_occPtr = thrust::device_pointer_cast(d_cellOccArray);
    thrust::copy(t_occPtr, t_occPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
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
void fluidSolver(float3 *d_posArray, float3 *d_velArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _hashTableSize, unsigned int _maxNumThreads, float _smoothingLength, float _timestep, float _particleMass, float _restDensity, float _gasConstant, float _visCoef, float _densKernConst, float _pressKernConst, float _viscKernConst){
    //std::cout<<"fluidSolver"<<std::endl;
    //printf("memory allocated: %d",_maxNumThreads*(sizeof(particleProp)));
    fluidSolverKernal<<<_hashTableSize, 30>>>(d_posArray,d_velArray,d_cellOccArray,d_cellIndxArray,_smoothingLength,_timestep, _particleMass, _restDensity,_gasConstant,_visCoef, _densKernConst, _pressKernConst, _viscKernConst);

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

