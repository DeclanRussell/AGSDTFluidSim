#include "SPHSolverCUDA.h"
#include <iostream>
#define SpeedOfSound 34.29f
#include <helper_math.h>

//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::SPHSolverCUDA()
{
    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found" << count << "CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
        return;
    }
    for (int i=0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout<<prop.name<<", Compute capability:"<<prop.major<<"."<<prop.minor<<std::endl;;
        std::cout<<"  Global mem: "<<prop.totalGlobalMem/ 1024 / 1024<<"M, Shared mem per block: "<<prop.sharedMemPerBlock / 1024<<"k, Registers per block: "<<prop.regsPerBlock<<std::endl;
        std::cout<<"  Warp size: "<<prop.warpSize<<" threads, Max threads per block: "<<prop.maxThreadsPerBlock<<", Multiprocessor count: "<<prop.multiProcessorCount<<" MaxBlocks: "<<prop.maxGridSize[0]<<std::endl;
        m_threadsPerBlock = prop.maxThreadsPerBlock;
    }

    // Create our CUDA stream to run our kernals on. This helps with running kernals concurrently.
    // Check them out at http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
    checkCudaErrors(cudaStreamCreate(&m_cudaStream));

    // Make sure these are init to 0
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashKeys = 0;
    m_fluidBuffers.hashMap = 0;
    m_fluidBuffers.denPtr = 0;

    m_simProperties.gridDim = make_float3(0,0,0);
    setSmoothingLength(0.3f);
    setHashPosAndDim(make_float3(0.f,0.f,0.f),make_float3(10.f,10.f,10.f));
    m_simProperties.timeStep = 0.004f;
    m_simProperties.gravity = make_float3(0.f,-9.8f,0.f);
    m_simProperties.k = SpeedOfSound;//*SpeedOfSound;
    m_simProperties.tension = 1.f;
    m_simProperties.viscosity = 0.3f;
    m_simProperties.mass = 1.f;//0.0002f;
    m_simProperties.invMass = 1.f/m_simProperties.mass;
    m_simProperties.restDensity = 500.f;
    // Send these to the GPU
    updateGPUSimProps();

    // Create an OpenGL buffer for our position buffer
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_posVAO);
    glBindVertexArray(m_posVAO);

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    // We must alocate some space otherwise cuda cannot register it
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Unbind everything just in case
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);



}
//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::~SPHSolverCUDA()
{
    // Make sure we remember to unregister our cuda resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));

    // Delete our CUDA buffers
    if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
    if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
    // Delete our CUDA streams as well
    checkCudaErrors(cudaStreamDestroy(m_cudaStream));
    // Delete our openGL objects
    glDeleteBuffers(1,&m_posVBO);
    glDeleteVertexArrays(1,&m_posVAO);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setParticles(std::vector<float3> &_particles)
{
    // Set how many particles we have
    m_simProperties.numParticles = (int)_particles.size();

    // Unregister our resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));

    // Fill our buffer with our positions
    glBindVertexArray(m_posVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*_particles.size(), &_particles[0], GL_DYNAMIC_DRAW);

    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Delete our CUDA buffers fi they have anything in them
    if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
    if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
    if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
    if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.denPtr = 0;
    m_fluidBuffers.hashKeys = 0;

    // Fill them up with some blank data
    std::vector<float3> blankFloat3s;
    blankFloat3s.resize(_particles.size());
    for(unsigned int i=0;i<blankFloat3s.size();i++) blankFloat3s[i] = make_float3(0.f,0.f,0.f);

    // Send the data to the GPU
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.velPtr,blankFloat3s.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.velPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.accPtr,blankFloat3s.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.accPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashKeys,_particles.size()*sizeof(float3)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.denPtr,_particles.size()*sizeof(float)));
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.hashKeys,_particles.size());

}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float3> SPHSolverCUDA::getParticlePositions()
{
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));

    std::vector<float3> positions;
    positions.resize(m_simProperties.numParticles);

    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));

    return positions;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setSmoothingLength(float _h)
{
    m_simProperties.h = _h;
    m_simProperties.hSqrd = _h*_h;
    m_simProperties.dWConst = 315.f/(64.f*(float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.pWConst = -45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.vWConst = 45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.cWConst1 = 32.f/((float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.cWConst2 = (_h*_h*_h*_h*_h*_h)/64.f;

    setHashPosAndDim(m_simProperties.gridMin,m_simProperties.gridDim);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setHashPosAndDim(float3 _gridMin, float3 _gridDim)
{
    m_simProperties.gridMin = _gridMin;
    m_simProperties.gridDim = _gridDim;
    m_simProperties.invGridDim = make_float3(1.f/_gridDim.x,1.f/_gridDim.y,1.f/_gridDim.z);
    m_simProperties.gridRes.x = (int)ceil(_gridDim.x/m_simProperties.h);
    m_simProperties.gridRes.y = (int)ceil(_gridDim.y/m_simProperties.h);
    m_simProperties.gridRes.z = (int)ceil(_gridDim.z/m_simProperties.h);
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y *m_simProperties.gridRes.z);


    // No point in alocating a buffer size of zero so lets just return
    if(tableSize==0)return;

    std::cout<<"table size "<<tableSize<<std::endl;

    // Remove anything that is in our bufferes currently
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashMap = 0;
    // Send the data to our GPU buffers
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellIndexBuffer,tableSize*sizeof(int)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellOccBuffer,tableSize*sizeof(int)));
    // Fill with blank data
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

    // Update this our simulation properties on the GPU
    updateGPUSimProps();

    // Allocate memory for our hash map
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashMap,tableSize*sizeof(cellInfo)));
    // Compute our hash map
    createHashMap(m_cudaStream,m_threadsPerBlock,tableSize,m_fluidBuffers);


}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::update()
{
    //if no particles then theres no point in updating so just return
    if(!m_simProperties.numParticles)return;

    // Set our hash table values back to zero
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y *m_simProperties.gridRes.z);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellIndexBuffer,tableSize);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

    //Send our sim properties to the GPU
    updateSimProps(&m_simProperties);

    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));

    // Hash and sort our particles
    hashAndSort(m_cudaStream, m_threadsPerBlock, m_simProperties.numParticles, tableSize , m_fluidBuffers);

    // Compute our density
    initDensity(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);

    // Solve for our new positions
    solve(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);

    // Solve any collisions in our system
    collisionDetection(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
}
//----------------------------------------------------------------------------------------------------------------------
