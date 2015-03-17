#include "SPHEngine.h"

#include <ngl/Random.h>
#include <QString>
#include "CudaSPHKernals.h"

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
    glDeleteBuffers(1,&m_VBO);
    glDeleteVertexArrays(1,&m_VAO);
}

void SPHEngine::init(){


    //create some points just for testing our instancing
    std::vector<float3> particles;
    ngl::Random *rnd = ngl::Random::instance();
    ngl::Vec3 tempPoint;
    float3 tempF3;
    for(unsigned int i=0; i<m_numParticles; i++){
        tempPoint = rnd->getRandomPoint(5,5,5);
        tempF3.x = tempPoint.m_x;
        tempF3.y = tempPoint.m_y;
        tempF3.z = tempPoint.m_z;
        particles.push_back(tempF3);
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
    //allocate space for our hash table
    cudaMalloc(&m_dhashKeys, m_numParticles*sizeof(unsigned int));
    //allocate space for our cell occupancy array
    cudaMalloc(&m_dCellOccBuffer, m_hashTableSize*sizeof(unsigned int));
    //initialize it with zeros
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);
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
void SPHEngine::update(unsigned int _timeStep){
    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_cudaBufferPtr,0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_cudaBufferPtr);
    //calculate our hash keys
    createHashTable(m_dhashKeys,d_posPtr,m_numParticles,m_smoothingLength, m_hashTableSize, m_numThreadsPerBlock);
    // Wait for all the threads to be finished
    // if we are still reseting our occupancy buffer from a previous update this will also give it a
    // chance to finish
    cudaThreadSynchronize();
    //sort our particle postions based on there key to make
    //points of the same key occupy contiguous memory
    sortByKey(m_dhashKeys,d_posPtr,m_numParticles);
    // Wait for all the threads to be finished
    cudaThreadSynchronize();
    //total up our cell occupancy
    countCellOccupancy(m_dhashKeys,m_dCellOccBuffer,m_hashTableSize,m_numParticles,m_numThreadsPerBlock);
    // Make sure all threads have finished that calculations
    cudaThreadSynchronize();
    //count our cell occupancy to create our cell index buffer.
    //this being a serial operation will probably be quicker done by the CPU rather than the GPU
    //should try and think of a better way to do this because copying data to and from the device
    //is really slow! This will add overhead to the simulation.
    /// @todo Might be worth doing some tests to see what is faster, cpu with memCpy overhead or
    /// @todo just doing it all on the cpu even though its a serial operation.
    unsigned int cellOcc[m_hashTableSize];
    unsigned int cellIdx[m_hashTableSize];
    cudaMemcpy(cellOcc, m_dCellOccBuffer, m_hashTableSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int sum = 0;
    for(unsigned int i=0; i<m_hashTableSize;i++){
        cellIdx[i] = sum;
        sum+=cellOcc[i];
    }
//    if(sum==0){
//        std::cout<<"no particles ";
//        for(unsigned int i=0; i<m_hashTableSize;i++){
//            std::cout<<cellOcc[i]<<" ";
//        }
//        std::cout<<std::endl;
//    }
    //copy our index's to our device
    cudaMemcpy(m_dCellIndexBuffer, cellIdx, m_hashTableSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    /// @todo start the actual sph calculations for movement here


    float3* x,*y;
    fluidSolver(d_posPtr,x,y,m_dCellOccBuffer,m_dCellIndexBuffer,m_hashTableSize,m_numThreadsPerBlock,m_smoothingLength,0);

//    float3 pos[m_numParticles];
//    cudaMemcpy(pos, d_posPtr, m_numParticles * sizeof(float3), cudaMemcpyDeviceToHost);
//    for(unsigned int i=0; i<m_numParticles;i++){
//        printf("pos: %f,%f,%f\n",pos[i].x,pos[i].y,pos[i].z);
//    }
//    printf("\n");
//    //random test function
//    calcPositions(d_posPtr,time(NULL),m_numParticles, m_numThreadsPerBlock);
    // Make sure all threads have finished that calculations
    cudaThreadSynchronize();
    //now we've updated our positions lets set reset our occupancy buffer to zero ready for next update
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);


    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_cudaBufferPtr,0);

}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::drawArrays(){
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, m_numParticles);
    glBindVertexArray(0);
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
