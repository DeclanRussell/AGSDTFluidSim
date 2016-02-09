//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolverCUDAKernals.cu
/// @author Declan Russell
/// @date 03/02/2016
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include "SPHSolverCUDAKernals.h"
#include <helper_math.h>  //< some math operations with cuda types
#include <iostream>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#define NULLHASH 4294967295
#define epsilon 0.0001f

//#define USE_SIZEFUNC
//#define DIFF_SF
//#define BLINN_SF

// Our simulation properties. These wont change much so lets load them into constant memory
__constant__ SimProps props;


//----------------------------------------------------------------------------------------------------------------------
__global__ void testKernal()
{
    printf("thread number %d\n",threadIdx.x);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fillIntZeroKernal(int *_bufferPtr,int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<size)
    {
        _bufferPtr[idx]=0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void createHashMapKernal(int _hashTableSize, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_hashTableSize)
    {
        int count =0;
        int key;
        cellInfo cell;
        for(int z=-1;z<2;z++)
        {
            for(int y=-1;y<2;y++)
            {
                for(int x=-1;x<2;x++)
                {
                    key = idx + x + (y*props.gridRes.x) + (z*props.gridRes.x*props.gridRes.y);
                    if(key>=0 && key<_hashTableSize && count < _hashTableSize && count < 27)
                    {
                        cell.cIdx[count] = key;
                        count++;
                    }
                }
            }
        }
        cell.cNum = count;
        _buff.hashMap[idx] = cell;
    }
}

//----------------------------------------------------------------------------------------------------------------------
__global__ void hashParticles(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 pos = _buff.posPtr[idx] - props.gridMin;
        // Make sure the point is within our hash table
        if(pos.x>0.f && pos.x<props.gridDim.x && pos.y>0.f && pos.y<props.gridDim.y && pos.z>0.f && pos.z<props.gridDim.z)
        {
            //Compute our hash key
            int key = floor((pos.x/props.gridDim.x)*props.gridRes.x) + (floor((pos.y/props.gridDim.y)*props.gridRes.y)*props.gridRes.x) + (floor((pos.z/props.gridDim.z)*props.gridRes.z)*props.gridRes.x*props.gridRes.y);
            _buff.hashKeys[idx] = key;

            //Increment our occumpancy of this hash cell
            atomicAdd(&(_buff.cellOccBuffer[key]), 1);

        }
        else
        {
            _buff.hashKeys[idx] = NULLHASH;
            printf("NULL HASH");
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calculatePressure(float _pi)
{
    return props.k*(_pi-props.restDensity);
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calcDensityWeighting(float _rLength)
{
    if(_rLength>0.f && _rLength<=props.h)
    {
        return props.dWConst * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength);
    }
    else
    {
        return 0.f;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcPressureWeighting(float3 _r, float _rLength)
{
    if(_rLength>0.f && _rLength<=props.h)
    {
        return props.pWConst * (_r) * (props.h - _rLength) * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcViscosityWeighting(float3 _r, float _rLength)
{
    if(_rLength>0.f && _rLength<=props.h)
    {
        return props.vWConst * _r * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calcCoheWeighting(float _rLength)
{
    float w = 0.f;
    if(((2.f*_rLength)>props.h)&&(_rLength<=props.h))
    {
        w = props.cWConst1*((props.h-_rLength)*(props.h-_rLength)*(props.h-_rLength)*_rLength*_rLength*_rLength);
    }
    else if((_rLength>0.f)&&(2.f*_rLength<=props.h))
    {
        w = props.cWConst1*(2.f*((props.h-_rLength)*(props.h-_rLength)*(props.h-_rLength)*_rLength*_rLength*_rLength) - props.cWConst2);
    }
    return w;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveDensityKernal(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pi = _buff.posPtr[idx];
        int key = floor((pi.x/props.gridDim.x)*props.gridRes.x) + (floor((pi.y/props.gridDim.y)*props.gridRes.y)*props.gridRes.x) + (floor((pi.z/props.gridDim.z)*props.gridRes.z)*props.gridRes.x*props.gridRes.y);

        // Get our neighbouring cell locations for this particle
        cellInfo nCells = _buff.hashMap[key];

        // Compute our density for all our particles
        int cellOcc;
        int cellIdx;
        int nIdx;
        float di = 0.f;
        float3 pj;
        float rLength;
        for(int c=0; c<nCells.cNum; c++)
        {
            // Get our cell occupancy total and start index
            cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
            cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                //Dont want to compare against same particle
                if(nIdx==idx) continue;
                // Get our neighbour position
                pj = _buff.posPtr[nIdx];
                //Calculate our arc length
                rLength = length(pi-pj);
                //Increment our density
                di+=props.mass*calcDensityWeighting(rLength);
            }
        }
        _buff.denPtr[idx] = di;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveForcesKernal(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {

        // Get our particle position and density
        float3 pi = _buff.posPtr[idx];
        float di = _buff.denPtr[idx];
        float3 acc = make_float3(0.f,0.f,0.f);
        float3 vel = _buff.velPtr[idx];

        // Put this in its own scope means we get some registers back at the end of it (I think)
        if(di>0.f)
        {
            // Calculate our hash key
            int key = floor((pi.x/props.gridDim.x)*props.gridRes.x) + (floor((pi.y/props.gridDim.y)*props.gridRes.y)*props.gridRes.x) + (floor((pi.z/props.gridDim.z)*props.gridRes.z)*props.gridRes.x*props.gridRes.y);

            // Get our neighbouring cell locations for this particle
            cellInfo nCells = _buff.hashMap[key];

            // Compute our fources for all our particles
            int cellOcc,cellIdx,nIdx;
            float dj,presi,presj,rLength,cWeight;
            float3 pj,r,w;
            float3 presForce,coheForce,viscForce;
            presForce = coheForce = viscForce = make_float3(0.f,0.f,0.f);
            for(int c=0; c<nCells.cNum; c++)
            {
                // Get our cell occupancy total and start index
                cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
                cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    nIdx = cellIdx+i;
                    //Dont want to compare against same particle
                    if(nIdx==idx) continue;
                    // Get our neighbour density
                    dj = _buff.denPtr[nIdx];
                    // Get our neighbour position
                    pj = _buff.posPtr[nIdx];
                    //Get our vector beteen points
                    r = pi - pj;
                    //Calculate our length
                    rLength = length(r);

                    // Normalise our differential
                    r/=rLength;

                    //Compute our particles pressure
                    presi = calculatePressure(di);
                    presj = calculatePressure(dj);


                    //Weighting
                    w = calcPressureWeighting(r,rLength);

                    // Accumilate our pressure force
                    presForce+= ((presi/(di*di)) + (presj/(dj*dj))) * props.mass * w;

                    // This shouldn't happen but for some reason it does :(
                    if(rLength>0){
                        // Calculate our cohesion weighting
                        cWeight = calcCoheWeighting(rLength);
                        // Accumilate our cohesion force
                        coheForce+=-props.tension*props.mass*props.mass*((2.f*props.restDensity)/(di+dj))*cWeight*r;

                        // Calculate the viscosity weighting
                        w = calcViscosityWeighting(r,rLength);
                        // Accumilate our viscosity force
                        viscForce += props.viscosity * (_buff.velPtr[nIdx] - vel) * (props.mass/dj) * w;
                    }
                }
            }
            // Complete our pressure force term
            presForce*=-1.f*props.mass;

            acc = (presForce + coheForce + viscForce + props.gravity)/props.mass;
        }

        // Now lets integerate our acceleration using leapfrog to get our new position
        float3 halfFwd,halfBwd;
        halfBwd = vel - 0.5f*props.timeStep*acc;
        halfFwd = halfBwd + props.timeStep*acc;

        // Update our velocity
        _buff.velPtr[idx] = halfFwd;
        // Integrate for our new position and update our particle
        pi+= props.timeStep * halfFwd;
        _buff.posPtr[idx] = pi;

    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveCollision(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 pos = _buff.posPtr[idx];
        float3 vel = _buff.velPtr[idx];
        float3 max = props.gridMin + props.gridDim - props.h;
        float3 min = props.gridMin + props.h;
        if(pos.x<min.x){
            pos.x = min.x;
            vel.x = 0.f;
        }
        if(pos.x>max.x){
            pos.x = max.x;
            vel.x = 0.f;
        }
        if(pos.y<min.y){
            pos.y = min.y;
            vel.y = 0.f;
        }
        if(pos.y>max.y){
            pos.y = max.y;
            vel.y = 0.f;
        }
        if(pos.z<min.z){
            pos.z = min.z;
            vel.z = 0.f;
        }
        if(pos.z>max.z){
            pos.z = max.z;
            vel.z = 0.f;
        }
        _buff.posPtr[idx] = pos;
        _buff.velPtr[idx] = vel;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void test(){
    printf("calling\n");
    testKernal<<<1,1000>>>();
    //make sure all our threads are done
    cudaThreadSynchronize();
    printf("called\n");
}
//----------------------------------------------------------------------------------------------------------------------
float computeAverageDensity(int _numParticles, fluidBuffers _buff)
{
    // Turn our density buffer pointer into a thrust iterater
    thrust::device_ptr<float> t_denPtr = thrust::device_pointer_cast(_buff.denPtr);

    // Use reduce to sum all our densities
    float sum = thrust::reduce(t_denPtr, t_denPtr+_numParticles, 0.f, thrust::plus<float>());

    // Return our average density
    return sum/(float)_numParticles;
}
//----------------------------------------------------------------------------------------------------------------------
void updateSimProps(SimProps *_props)
{
    #ifdef CUDA_42
        // Unlikely we will ever use CUDA 4.2 but nice to have it in anyway I guess?
        cudaMemcpyToSymbol ( "props", _props, sizeof(SimProps) );
    #else
        cudaMemcpyToSymbol ( props, _props, sizeof(SimProps) );
    #endif
}
//----------------------------------------------------------------------------------------------------------------------
void fillIntZero(cudaStream_t _stream, int _threadsPerBlock, int *_bufferPtr,int size)
{
    if(size>_threadsPerBlock)
    {
        //calculate how many blocks we want
        int blocks = ceil(size/_threadsPerBlock)+1;
        fillIntZeroKernal<<<blocks,_threadsPerBlock,0,_stream>>>(_bufferPtr,size);
    }
    else{
        fillIntZeroKernal<<<1,size,0,_stream>>>(_bufferPtr,size);
    }
    //make sure all our threads are done
    cudaThreadSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fill int zero: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void createHashMap(cudaStream_t _stream, int _threadsPerBlock, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _hashTableSize;
    if(_hashTableSize>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_hashTableSize/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    // Create ou hash map
    createHashMapKernal<<<blocks,threads,0,_stream>>>(_hashTableSize,_buff);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Create hash map error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();
}
//----------------------------------------------------------------------------------------------------------------------
void hashAndSort(cudaStream_t _stream,int _threadsPerBlock, int _numParticles, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Hash our partilces
    hashParticles<<<blocks,threads,0,_stream>>>(_numParticles,_buff);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Hash Particles error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

//    //Turn our raw pointers into thrust pointers so we can use
//    //thrusts sort algorithm
    thrust::device_ptr<int> t_hashPtr = thrust::device_pointer_cast(_buff.hashKeys);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(_buff.posPtr);
    thrust::device_ptr<float3> t_velPtr = thrust::device_pointer_cast(_buff.velPtr);
    thrust::device_ptr<float3> t_accPtr = thrust::device_pointer_cast(_buff.accPtr);
    thrust::device_ptr<int> t_cellOccPtr = thrust::device_pointer_cast(_buff.cellOccBuffer);
    thrust::device_ptr<int> t_cellIdxPtr = thrust::device_pointer_cast(_buff.cellIndexBuffer);

    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_velPtr,t_accPtr)));
    //make sure all our threads are done
    cudaThreadSynchronize();


    //Create our cell indexs
    //run an excludive scan on our arrays to do this
    thrust::exclusive_scan(t_cellOccPtr,t_cellOccPtr+_hashTableSize,t_cellIdxPtr);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::copy(t_cellOccPtr, t_cellOccPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
}
//----------------------------------------------------------------------------------------------------------------------
void initDensity(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Solve our particles
    solveDensityKernal<<<blocks,threads,0,_stream>>>(_numParticles,_buff);

    //make sure all our threads are done
    cudaThreadSynchronize();
}
//----------------------------------------------------------------------------------------------------------------------
void solve(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Solve for our new positions
    solveForcesKernal<<<blocks,threads,0,_stream>>>(_numParticles, _buff);

    //make sure all our threads are done
    cudaThreadSynchronize();

}
//----------------------------------------------------------------------------------------------------------------------
void collisionDetection(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Detect any particle collision
    solveCollision<<<blocks,threads,0,_stream>>>(_numParticles, _buff);

    //make sure all our threads are done
    cudaThreadSynchronize();

}
//----------------------------------------------------------------------------------------------------------------------
