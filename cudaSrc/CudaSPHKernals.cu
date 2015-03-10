#include <math.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "CudaSPHKernals.h"
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
__global__ void pointHash(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float smoothingLegnth, unsigned int hashTableSize){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //dont want to start accessing data that doesn't exist! Could be deadly!
    if(idx<numParticles){
        //calculate our hash key and store it in our hash key array
        int x = floor(d_posArray[idx].x/smoothingLegnth);
        int y = floor(d_posArray[idx].y/smoothingLegnth);
        int z = floor(d_posArray[idx].z/smoothingLegnth);

        d_hashArray[idx] = (x*73856093^y*19349663^z*83492791) & hashTableSize;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void countCellOccKernal(unsigned int *d_hashArray, unsigned int *d_cellOccArray, unsigned int _hashTableSize, unsigned int _numPoints){
    //Create our idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Make sure our idx is valid and add the occupancy count to the relevant cell
    if ((idx < _numPoints) && (d_hashArray[idx] < _hashTableSize)) {
        atomicAdd(&(d_cellOccArray[d_hashArray[idx]]), 1);
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
}
//----------------------------------------------------------------------------------------------------------------------
void countCellOccuancy(unsigned int *d_hashArray, unsigned int *d_cellOccArray, unsigned int _hashTableSize, unsigned int _numPoints, unsigned int _maxNumThreads){
    //calculate how many blocks we want
    int blocks = ceil(_hashTableSize/_maxNumThreads)+1;
    countCellOccKernal<<<blocks,_maxNumThreads>>>(d_hashArray,d_cellOccArray,_hashTableSize,_numPoints);
}
//----------------------------------------------------------------------------------------------------------------------
//template <typename T>
//void thrustFill(T *_pointer, unsigned int _arraySize, T _fill){
//    //Turn our raw pointers into thrust pointers so we can use
//    //them in thrust fill
//    thrust::device_ptr<T> t_Ptr = thrust::device_pointer_cast(_pointer);
//    //fill our buffer
//    thrust::fill(t_Ptr, t_Ptr+_arraySize, _fill);
//}
//----------------------------------------------------------------------------------------------------------------------

