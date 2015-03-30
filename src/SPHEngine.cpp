#include "SPHEngine.h"

#include <QString>
#include <vector>
#include <iostream>
#include <cmath>
#include "CudaSPHKernals.h"

#define pi 3.14159265359f
//----------------------------------------------------------------------------------------------------------------------
SPHEngine::SPHEngine(unsigned int _numParticles) : m_numParticles(_numParticles),
                                                   m_volume(m_numParticles),
                                                   m_density(1),
                                                   m_mass(1),
                                                   m_smoothingLength(1)
{
    init();
}

//----------------------------------------------------------------------------------------------------------------------
SPHEngine::~SPHEngine(){
    // Make sure we remember to unregister our cuda resource
    cudaGraphicsUnregisterResource(m_cudaBufferPtr);
    cudaFree(m_dhashKeys);
    cudaFree(m_dCellIndexBuffer);
    cudaFree(m_dCellOccBuffer);
    cudaFree(m_dVelBuffer);
    glDeleteBuffers(1,&m_VBO);
    glDeleteVertexArrays(1,&m_VAO);
}

void SPHEngine::init(){


    //create some points just for testing our instancing
    std::vector<float3> particles;
    float3 tempF3;
    float tx,ty,tz;
    tx=ty=tz=-5.0;
    for(unsigned int i=0; i<m_numParticles; i++){
        if(tx>5){ tx=-5.0; tz+=0.5;}
        if(tz>5){ tz=-5.0; ty+=0.5;}

        tempF3.x = tx;
        tempF3.y = ty;
        tempF3.z = tz;
        particles.push_back(tempF3);
        tx+=0.5;
    }


    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*particles.size(), &particles[0].x, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //register our particle postion buffer with cuda
    cudaGraphicsGLRegisterBuffer(&m_cudaBufferPtr, m_VBO, cudaGraphicsRegisterFlagsWriteDiscard);

    //set up our VAO
    // create a vao
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);

    // connect the data to the shader input
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    //set our point size
    glPointSize(10);

    //set the size of our hash table based on how many particles we have
    m_hashTableSize = nextPrimeNum(2*m_numParticles);
    std::cout<<"hash table size: "<<m_hashTableSize<<std::endl;
    //allocate space for our hash table,cell occupancy array and velocity array.
    cudaMalloc(&m_dhashKeys, m_numParticles*sizeof(unsigned int));
    cudaMalloc(&m_dCellOccBuffer, m_hashTableSize*sizeof(unsigned int));
    cudaMalloc(&m_dVelBuffer, m_numParticles*sizeof(float3));
    //initialize them with zeros
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);
    float3 x[m_numParticles];
    for(int i=0;i<m_numParticles;i++)x[i]=make_float3(0,0,0);
    cudaMemcpy(m_dVelBuffer, x, m_numParticles * sizeof(float3), cudaMemcpyHostToDevice);
    //allocate space for our cell index buffer
    cudaMalloc(&m_dCellIndexBuffer, m_hashTableSize*sizeof(unsigned int));
    //initialize it with zeros
    //not really that necesary but might reduce errors, nice to be safe
    fillUint(m_dCellIndexBuffer,m_hashTableSize,0);

    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found" << count << "CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip scrub!"<<std::endl;
        return;
    }
    for (int i=0; i < count; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        QString deviceString = QString("* %1, Compute capability: %2.%3").arg(prop.name).arg(prop.major).arg(prop.minor);
        QString propString1 = QString("  Global mem: %1M, Shared mem per block: %2k, Registers per block: %3").arg(prop.totalGlobalMem / 1024 / 1024)
                .arg(prop.sharedMemPerBlock / 1024).arg(prop.regsPerBlock);
        QString propString2 = QString("  Warp size: %1 threads, Max threads per block: %2, Multiprocessor count: %3 MaxBlocks: %4")
                .arg(prop.warpSize).arg(prop.maxThreadsPerBlock).arg(prop.multiProcessorCount).arg(prop.maxGridSize[0]);
        std::cout << deviceString.toStdString() << std::endl;
        std::cout << propString1.toStdString() << std::endl;
        std::cout << propString2.toStdString() << std::endl;
        m_numThreadsPerBlock = prop.maxThreadsPerBlock;
        m_maxNumBlocks = prop.maxGridSize[0];
    }

}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::update(float _timeStep){
    std::cout<<"update"<<std::endl;
    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_cudaBufferPtr);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_cudaBufferPtr);
    std::cout<<"d_posPointer is "<<d_posPtr<<std::endl;
    //calculate our hash keys
    createHashTable(m_dhashKeys,d_posPtr,m_numParticles,m_smoothingLength, m_hashTableSize, m_numThreadsPerBlock);

    //sort our particle postions based on there key to make
    //points of the same key occupy contiguous memory
    sortByKey(m_dhashKeys,d_posPtr,m_dVelBuffer,m_numParticles);

    //total up our cell occupancy
    countCellOccupancy(m_dhashKeys,m_dCellOccBuffer,m_hashTableSize,m_numParticles,m_numThreadsPerBlock);

    //Uses exclusive scan to count our cell occupancy and create our cell index buffer.
    createCellIdx(m_dCellOccBuffer,m_hashTableSize,m_dCellIndexBuffer);

    //update our particle positions with navier stokes equations
    fluidSolver(d_posPtr,m_dVelBuffer,m_dCellOccBuffer,m_dCellIndexBuffer,m_hashTableSize,m_numThreadsPerBlock,m_smoothingLength,_timeStep,m_mass,m_density,1,1,m_densWeightConst,m_pressWeightConst,m_viscWeightConst);

    //fill our occupancy buffer back up with zeros
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_cudaBufferPtr);
    //std::cout<<"update finished numParticles"<<m_numParticles<<" hash table size "<<m_hashTableSize<<std::endl;

}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::drawArrays(){
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, m_numParticles);
    glBindVertexArray(0);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::calcKernalConsts()
{
    m_densWeightConst = (15.0f/(pi*m_smoothingLength*m_smoothingLength*m_smoothingLength*m_smoothingLength*m_smoothingLength*m_smoothingLength));
    m_pressWeightConst = -(45.0f/(pi*pow(m_smoothingLength,6)));
    m_viscWeightConst = -(90/(pi*pow(m_smoothingLength,6)));
}
//----------------------------------------------------------------------------------------------------------------------
unsigned int SPHEngine::nextPrimeNum(int _x){
    int nextPrime = _x;
    bool Prime = false;
    if(_x<=0){
        std::cerr<<"The number input is less than or equal to zero"<<std::endl;
        return 1;
    }
    if(_x==2){
        return 2;
    }
    if((_x % 2 ) == 0){
        nextPrime+=1;
    }
    while(!Prime){
        Prime = true;
        for(int i = 3; i<sqrt(nextPrime); i+=2){
            if((nextPrime % i)==0){
                Prime = false;
            }
        }
        if(!Prime){
            nextPrime+=2;
        }
    }
    return nextPrime;
}

//----------------------------------------------------------------------------------------------------------------------
