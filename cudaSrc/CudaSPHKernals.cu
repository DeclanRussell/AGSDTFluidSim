#include <math.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
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
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float densityWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength){
    float3 r = _currentPos - _neighPos;
    float rLength = length(r);
    float weighting = (315/(64*pi*pow(_smoothingLength,9))) * pow(((_smoothingLength*_smoothingLength) - (rLength*rLength)),3);
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
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float3 pressureWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength){
    float3 r = _currentPos - _neighPos;
    float rLength = length(r);
    float weighting = -(945/(32*pi*pow(_smoothingLength,9))) * pow(((_smoothingLength*_smoothingLength) - (rLength*rLength)),3);
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
/// @return return the weighting that our neighbouring particle has on our current particle
__device__ float3 viscosityWeighting(float3 _currentPos, float3 _neighPos,float _smoothingLength){
    float3 r = _currentPos - _neighPos;
    float rLength = length(r);
    float weighting = -(945/(32*pi*pow(_smoothingLength,9))) * pow(((_smoothingLength*_smoothingLength) - (rLength*rLength)),3) * ((3*(_smoothingLength*_smoothingLength)) - 7*(_smoothingLength*_smoothingLength));
    r *= weighting;
    //if length of r is larger than our smoothing length we want the weighting to be zero
    //However branching conditions are slow so here is a neat little trick so we dont need one
    //false also being 0 and true being 1 solves removes the need for branching
    return r * (float)(rLength<_smoothingLength);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fluidSolverKernal(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray,float _timestep){

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
    extern __shared__ float3 nParticlePos[];
    //now lets sycronise our threads so our memory is ready
    __syncthreads();


    //make sure we're not doing anything to particles that are not in our cell
    if(threadIdx.x<cellOcc){
        //lets load in our particles properties to our peice of shared memory
        //Due to limits on threads if we have more particles to this key than
        //Threads we may have to sacrifice some particles to sample for less
        //overhead but hopefully we can keep this under control by having a
        //good cell size (smoothing length) in our hash function.
        nParticlePos[threadIdx.x] = d_posArray[particleIdx];
        //Once this is done we can finally do some navier-stokes!!
        d_posArray[particleIdx].y += 0.01;
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
void sortByKey(unsigned int *d_hashArray, float3 *d_posArray, unsigned int _numParticles){
    //Turn our raw pointers into thrust pointers so we can use
    //thrusts sort algorithm
    thrust::device_ptr<unsigned int> t_hashPtr = thrust::device_pointer_cast(d_hashArray);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(d_posArray);
    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, t_posPtr);

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
    //calculate how many blocks we want
    int blocks = ceil(_hashTableSize/_maxNumThreads)+1;
    countCellOccKernal<<<blocks,_maxNumThreads>>>(d_hashArray,d_cellOccArray,_hashTableSize,_numPoints);


    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::device_ptr<unsigned int> t_occPtr = thrust::device_pointer_cast(d_cellOccArray);
    //thrust::copy(t_occPtr, t_occPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
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
void fluidSolver(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray,unsigned int _hashTableSize,unsigned int _maxNumThreads, float _timestep){
    /// @todo this is very basic at the moment, need to load cell of particles into shared block memory to do actual sph calculations
    /// @todo You can find an example of shared memory stuff in richards tesselation demo

    fluidSolverKernal<<<_hashTableSize, _maxNumThreads,_maxNumThreads*sizeof(float3)>>>(d_posArray,d_velArray,d_accArray,d_cellOccArray,d_cellIndxArray,_timestep);

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

