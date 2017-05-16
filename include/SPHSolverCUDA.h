#ifndef SPHSOLVERCUDA_H
#define SPHSOLVERCUDA_H

//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolver.h
/// @brief Calculates and updates our new particle positions with navier-stokes equations using CUDA acceleration.
/// @author Declan Russell
/// @version 1.0
/// @date 03/02/2015
/// @class SPHSolverCUDA
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif

// Just another stupid quirk by windows -.-
// Without windows.h defined before cuda_gl_interop you get
// redefinition conflicts.
#ifdef WIN32
    #include <Windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>

#include "SPHSolverCUDAKernals.h"
#include <vector>

class SPHSolverCUDA
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our defualt constructor
    //----------------------------------------------------------------------------------------------------------------------
    SPHSolverCUDA();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~SPHSolverCUDA();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets the particles init positions in our simulation from an array. If set more than once old data will be removed.
    //----------------------------------------------------------------------------------------------------------------------
    void setParticles(std::vector<float3> &_particles);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves our particles from the GPU and returns them in a vector
    /// @return array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float3> getParticlePositions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the number of particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline int getNumParticles(){return m_simProperties.numParticles;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Returns our OpenGL VAO handle to our particle positions
    /// @return OpenGL VAO handle to our particle positions (GLuint)
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getPositionsVAO(){return m_posVAO;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the mass of our particles
    /// @param _m - mass of our particles (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMass(float _m){m_simProperties.mass = _m; m_simProperties.invMass = 1.f/_m; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the mass of our particles
    //----------------------------------------------------------------------------------------------------------------------
    inline float getMass(){return m_simProperties.mass;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator for the timestep of our simulation
    /// @param _t - desired timestep
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTimeStep(float _t){m_simProperties.timeStep = _t; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to k our gas/stiffness constant
    /// @param _k - desired gas/stiffness constant
    //----------------------------------------------------------------------------------------------------------------------
    inline void setKConst(float _k){m_simProperties.k = _k; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to k our gas/stiffness constant
    /// @return k our gas/stiffness constant (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getKConst(){return m_simProperties.k;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to tension constant
    /// @param _t - tension constant
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTensionConst(float _t){m_simProperties.tension = _t; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to tension constant
    /// @return tension constant (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getTensionConst(){return m_simProperties.tension;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to viscosity constant
    /// @param _v - viscosity constant
    //----------------------------------------------------------------------------------------------------------------------
    inline void setViscConst(float _v){m_simProperties.viscosity = _v; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to viscosity constant
    /// @return tension constant (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getViscConst(){return m_simProperties.tension;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our smoothing length h
    /// @param _h - desired smoothing length
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(float _h);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our rest/target density
    /// @param _d - desired rest/target density
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRestDensity(float _d){m_simProperties.restDensity = _d; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates the sim properties on our GPU
    //----------------------------------------------------------------------------------------------------------------------
    inline void updateGPUSimProps(){updateSimProps(&m_simProperties);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief function to set our hash grid position and dimensions
    /// @param _gridMin - minimum position of our grid
    /// @param _gridDim - grid dimentions
    //----------------------------------------------------------------------------------------------------------------------
    void setHashPosAndDim(float3 _gridMin, float3 _gridDim);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our update function to increment the step of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void update();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief compute the average density of our simulation
    /// @return average density of simulation (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getAverageDensity(){return computeAverageDensity(m_simProperties.numParticles,m_fluidBuffers);}
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our maximum threads per block.
    //----------------------------------------------------------------------------------------------------------------------
    int m_threadsPerBlock;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VAO handle of our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VBO handle to our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVBO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our cuda graphics resource for our particle positions OpenGL interop.
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourcePos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our fluid buffers on our device
    //----------------------------------------------------------------------------------------------------------------------
    fluidBuffers m_fluidBuffers;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Structure to hold all of our simulation properties so we can easily pass it to our CUDA kernal.
    //----------------------------------------------------------------------------------------------------------------------
    SimProps m_simProperties;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our CUDA stream to help run kernals concurrently
    //----------------------------------------------------------------------------------------------------------------------
    cudaStream_t m_cudaStream;
    //----------------------------------------------------------------------------------------------------------------------

};

#endif // SPHSOLVERKERNALS_H

