
#include <QString>
#include "SPHEngine.h"
#include <vector>
#include <iostream>
#include <cmath>
#include "helper_math.h"  //< some math operations with cuda types


#define pi 3.14159265359f
//----------------------------------------------------------------------------------------------------------------------
SPHEngine::SPHEngine(unsigned int _numParticles, float _mass, float _density, float _contanerSize) : m_numParticles(_numParticles),
                                                                                         m_mass(_mass),
                                                                                         m_density(_density),
                                                                                         m_maxGridDim(_contanerSize),
                                                                                         m_smoothingLength(0.3f),
                                                                                         m_numCollisionObjects(0),
                                                                                         m_gasConstant(10.0f),
                                                                                         m_viscCoef(0.3f),
                                                                                         m_resetPending(false),
                                                                                         m_updating(false),
                                                                                         m_addParticlesPending(false),
                                                                                         m_velocityCorrectionCoef(0.3)

{
    std::cout<<"Particle Mass: "<<m_mass<<std::endl;
    calcKernalConsts();
    init();
}

//----------------------------------------------------------------------------------------------------------------------
SPHEngine::~SPHEngine(){
    // Make sure we remember to unregister our cuda resource
    cudaGraphicsUnregisterResource(m_cudaBufferPtr);
    // Free all our memory
    cudaFree(m_dhashKeys);
    cudaFree(m_dCellIndexBuffer);
    cudaFree(m_dCellOccBuffer);
    cudaFree(m_dVelBuffer);
    cudaFree(m_dCollisionObjectBuffer);
    cudaStreamDestroy(m_stream);
    glDeleteBuffers(1,&m_VBO);
    glDeleteVertexArrays(1,&m_VAO);
}

void SPHEngine::init(){

    //create our cuda stream
    cudaStreamCreate(&m_stream);

    //init our pointers to null
    m_dVelBuffer = 0;
    m_dhashKeys = 0;
    m_cudaBufferPtr = 0;
    m_dCellIndexBuffer = 0;
    m_dCellOccBuffer = 0;

    //add some walls to keep our particles in our grid
    addCollisionObject(make_float3(0),make_float3(m_maxGridDim-m_smoothingLength),make_float3(1.f,0.5f,1.f));
    //create our initial particles
    m_spawnParticlesPos = make_float3(2,0,2);
    m_spawnParticlesPos.y = m_smoothingLength;
    m_spawnBoxSize = 6;

    //create our buffer object
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    //add our particles
    signalAddParticles(m_numParticles);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //set up our VAO
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);

    // connect the data to the the appropriate places
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    //set the size of our hash table based on how many particles we have
    m_hashTableSize = pow(ceil((float)m_maxGridDim/(float)m_smoothingLength),3);
    std::cout<<"hash table size: "<<m_hashTableSize<<std::endl;
    //allocate space for our hash cell occupancy array
    cudaMalloc(&m_dCellOccBuffer, m_hashTableSize*sizeof(unsigned int));
    //initialize them with zeros
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
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
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

    //if no particles then theres no point in updating so just return
    if(!m_numParticles)return;

    //let our class know we are updating
    m_updating = true;

    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_cudaBufferPtr);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_cudaBufferPtr);

    //calculate our hash keys
    createHashTable(m_stream,m_dhashKeys,d_posPtr,m_numParticles,m_smoothingLength, m_maxGridDim, m_numThreadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //sort our particle postions based on there key to make
    //points of the same key occupy contiguous memory
    sortByKey(m_dhashKeys,d_posPtr,m_dVelBuffer,m_numParticles);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //total up our cell occupancy
    countCellOccupancy(m_stream,m_dhashKeys,m_dCellOccBuffer,m_hashTableSize,m_numParticles,m_numThreadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //Uses exclusive scan to count our cell occupancy and create our cell index buffer.
    createCellIdx(m_dCellOccBuffer,m_hashTableSize,m_dCellIndexBuffer);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //update our particle positions with navier stokes equations
    fluidSolver(m_stream,d_posPtr,m_dVelBuffer,m_dCellOccBuffer,m_dCellIndexBuffer,m_hashTableSize,m_maxGridDim/m_smoothingLength,m_numThreadsPerBlock,m_smoothingLength,_timeStep,m_mass,m_density,m_gasConstant,m_viscCoef,m_velocityCorrectionCoef,m_densWeightConst,m_pressWeightConst,m_viscWeightConst);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //Test our particles for collision with our walls
    collisionDetectionSolver(m_stream,m_dCollisionObjectBuffer,m_numCollisionObjects,d_posPtr,m_dVelBuffer,_timeStep,m_numParticles,m_numThreadsPerBlock);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //std::cout<<"\n\n"<<std::endl;

    //fill our occupancy buffer back up with zeros
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_cudaBufferPtr);

    m_updating = false;

    //if reset pending now we have finished using the buffers lets reset them
    if(m_resetPending)resetSimulation();
    //if we have add particles pending then lets add them now our update is finished
    if(m_addParticlesPending)addParticles();
    //if we need to resize our hash table lets do it now the our update has finished
    if(m_resizeHashPending)resizeHashTable();

}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::signalAddParticles(int _numParticles){
    m_numParticlesToAdd = _numParticles;
    if(m_updating){
        m_addParticlesPending = true;
    }
    else{
        addParticles();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::signalResizeHashTable()
{
    if(m_updating){
        m_resizeHashPending = true;
    }
    else{
        resizeHashTable();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::addParticles(){
    if(m_numParticlesToAdd>0){
        std::cerr<<"Adding "<<m_numParticlesToAdd<<" particles to simulation"<<std::endl;
        //get all our old particles of the GPU
        std::vector<float3> particles;

        //if we have not registered our cuda resource yet lets register it and
        //just stick the data strait into our opengl buffer
        if(!m_cudaBufferPtr){

            //create our new particles
            float3 tempF3;

            float tx,ty,tz;
            float increment =  m_spawnBoxSize/pow(m_numParticlesToAdd,1.f/3.f);
            float3 max = m_spawnParticlesPos + make_float3(m_spawnBoxSize);
            tx=m_spawnParticlesPos.x;
            ty=m_spawnParticlesPos.y;
            tz=m_spawnParticlesPos.z;
            for(unsigned int i=0; i<m_numParticlesToAdd; i++){
                if(tx>=(max.x)){ tx=m_spawnParticlesPos.x; tz+=increment;}
                if(tz>=(max.z)){ tz=m_spawnParticlesPos.z; ty+=increment;}
                tempF3.x = tx;
                tempF3.y = ty;
                tempF3.z = tz;
                particles.push_back(tempF3);
                tx+=increment;
            }
            //put our new data back into our vbo
            glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*particles.size(), &particles[0].x, GL_DYNAMIC_DRAW);

            //register our particle postion buffer with cuda
            cudaGraphicsGLRegisterBuffer(&m_cudaBufferPtr, m_VBO, cudaGraphicsRegisterFlagsWriteDiscard);

        }
        else{
            //map our buffer pointer
            float3* d_posPtr;
            size_t d_posSize;
            cudaGraphicsMapResources(1,&m_cudaBufferPtr);
            cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_cudaBufferPtr);
            particles.resize(m_numParticles+m_numParticlesToAdd);
            cudaMemcpy(&particles[0].x,d_posPtr,m_numParticles*sizeof(float3),cudaMemcpyDeviceToHost);
            //unmap our buffer pointer and set it free into the wild
            cudaGraphicsUnmapResources(1,&m_cudaBufferPtr);

            //create our new particles
            float3 tempF3;

            float tx,ty,tz;
            float increment =  m_spawnBoxSize/pow(m_numParticlesToAdd,1.f/3.f);
            float3 max = m_spawnParticlesPos + make_float3(m_spawnBoxSize);
            tx=m_spawnParticlesPos.x;
            ty=m_spawnParticlesPos.y;
            tz=m_spawnParticlesPos.z;
            for(unsigned int i=0; i<m_numParticlesToAdd; i++){
                if(tx>=(max.x)){ tx=m_spawnParticlesPos.x; tz+=increment;}
                if(tz>=(max.z)){ tz=m_spawnParticlesPos.z; ty+=increment;}
                tempF3.x = tx;
                tempF3.y = ty;
                tempF3.z = tz;
                particles[m_numParticles+i] = tempF3;
                tx+=increment;
            }
            //put our new data back into our vbo
            glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*particles.size(), &particles[0].x, GL_DYNAMIC_DRAW);

        }

        // extend our velocity and hash key buffers
        std::vector<float3> zeroVector;
        zeroVector.resize(m_numParticlesToAdd,make_float3(0.f));
        if(m_dVelBuffer){
            float3 *tempfPtr;
            cudaMalloc(&tempfPtr,(m_numParticles+m_numParticlesToAdd)*sizeof(float3));
            //copy our old data to our new buffer
            cudaMemcpy(tempfPtr,m_dVelBuffer,m_numParticles*sizeof(float3),cudaMemcpyDeviceToDevice);
            //copy our new data to our new buffer
            cudaMemcpy(tempfPtr+m_numParticles,&zeroVector[0].x,m_numParticlesToAdd*sizeof(float3),cudaMemcpyHostToDevice);
            cudaFree(m_dVelBuffer);
            m_dVelBuffer = tempfPtr;
        }
        else{
            cudaMalloc(&m_dVelBuffer,m_numParticlesToAdd*sizeof(float3));
            cudaMemcpy(m_dVelBuffer,&zeroVector[0].x,m_numParticlesToAdd*sizeof(float3),cudaMemcpyHostToDevice);
        }

        if(m_dhashKeys){
            unsigned int* tempuiPtr;
            cudaMalloc(&tempuiPtr,(m_numParticles+m_numParticlesToAdd)*sizeof(unsigned int));
            cudaFree(m_dhashKeys);
            m_dhashKeys = tempuiPtr;
        }
        else{
            cudaMalloc(&m_dhashKeys,(m_numParticlesToAdd)*sizeof(unsigned int));
        }

        //update how many particles we now how in our simulation
        m_numParticles+=m_numParticlesToAdd;

    }
    m_addParticlesPending = false;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::resizeHashTable(){
    std::cerr<<"Resizing hash table"<<std::endl;
    //set the size of our hash table based on how many particles we have
    m_hashTableSize = pow(ceil((float)m_maxGridDim/(float)m_smoothingLength),3);
    std::cout<<"hash table size: "<<m_hashTableSize<<std::endl;
    //allocate space for our hash cell occupancy array
    cudaFree(m_dCellOccBuffer);
    cudaMalloc(&m_dCellOccBuffer, m_hashTableSize*sizeof(unsigned int));
    //initialize it with zeros
    fillUint(m_dCellOccBuffer,m_hashTableSize,0);
    //allocate space for our cell index buffer
    cudaFree(m_dCellIndexBuffer);
    cudaMalloc(&m_dCellIndexBuffer, m_hashTableSize*sizeof(unsigned int));
    //initialize it with zeros
    //not really that necesary but might reduce errors, nice to be safe
    fillUint(m_dCellIndexBuffer,m_hashTableSize,0);
    m_resizeHashPending = false;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::resetSimulation(){
    //if we have something to remove
    if(!m_numParticles) return;
    //free all our buffers of there data
    cudaFree(m_dhashKeys);
    cudaFree(m_dVelBuffer);
    m_dhashKeys = 0;
    m_dVelBuffer = 0;

    //finally reset our particle count
    m_numParticles = 0;
    m_resetPending = false;
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
    m_densWeightConst = (315.0f/(64.f*pi*pow(m_smoothingLength,9)));
    m_pressWeightConst = (-45.0f/(pi*pow(m_smoothingLength,6)));
    m_viscWeightConst = (45.0f/(pi*pow(m_smoothingLength,6)));

    std::cout<<"Density Const: "<<m_densWeightConst<<std::endl;
    std::cout<<"Pressure Const: "<<m_pressWeightConst<<std::endl;
    std::cout<<"Viscosity Const: "<<m_viscWeightConst<<std::endl;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHEngine::addCollisionObject(float3 _min, float3 _max, float3 _restCoef){
    if(m_numCollisionObjects==0){
        //increment our count
        m_numCollisionObjects++;
        //allocate some space onto our device for our planes
        cudaMalloc(&m_dCollisionObjectBuffer, sizeof(SimpleCuboidCollisionObject));
        //create our plane
        SimpleCuboidCollisionObject c;
        c.p1 = make_float4(_min,1.0);
        c.p2 = make_float4(_max,1.0);
        c.restitution = _restCoef;
        //copy our data onto our device
        cudaMemcpy(m_dCollisionObjectBuffer, &c, sizeof(SimpleCuboidCollisionObject), cudaMemcpyHostToDevice);
    }
    else{
        //increment our count
        m_numCollisionObjects++;
        //create a new larger buffer
        SimpleCuboidCollisionObject *tempBuffer;
        cudaMalloc(&tempBuffer, m_numCollisionObjects * sizeof(SimpleCuboidCollisionObject));

        //can make this faster by copying across with a kernal but we're not
        //really going to be doing it that much so this is fine

        //copy our original data back to the host
        SimpleCuboidCollisionObject cArray[m_numCollisionObjects];
        cudaMemcpy(cArray, m_dCollisionObjectBuffer, (m_numCollisionObjects-1u) * sizeof(SimpleCuboidCollisionObject), cudaMemcpyDeviceToHost);

        //create our new wall
        cArray[m_numCollisionObjects-1].p1 = make_float4(_min,1.0);
        cArray[m_numCollisionObjects-1].p2 = make_float4(_max,1.0);
        cArray[m_numCollisionObjects-1].restitution = _restCoef;

        //copy our data onto our device
        cudaMemcpy(tempBuffer, cArray, m_numCollisionObjects * sizeof(SimpleCuboidCollisionObject), cudaMemcpyHostToDevice);

        //delete our old buffer
        cudaFree(m_dCollisionObjectBuffer);

        //set our pointer to our new buffer
        m_dCollisionObjectBuffer = tempBuffer;

    }
}
//----------------------------------------------------------------------------------------------------------------------
