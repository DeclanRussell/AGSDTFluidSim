#ifndef SPHENGINE_H
#define SPHENGINE_H

//----------------------------------------------------------------------------------------------------------------------
/// @file SPHEngine.h
/// @class SPHEngine
/// @author Declan Russell
/// @date 09/03/2015
/// @version 1.0
/// @brief This class uses cuda and parralel algorithms to manage particles in a fluid simulation.
/// @brief Our fluid simulation uses smoothed particle hydrodynamics and true navier stokes equations to
/// @brief compute our particle positions in our simulation.
/// @brief This class also manages its own openGL buffer storing postions of particles in
/// @brief vertex attrip pointer 0. In case you wish to draw these particles.
//----------------------------------------------------------------------------------------------------------------------

#include <GL/glew.h>

/// @brief our cuda libraries
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
/// @brief our SPH kernals
#include "CudaSPHKernals.h"

class SPHEngine
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief defualt constructor
    /// @param _numParticles how many particles we want to have on simulation initialisation.
    /// @param _volume  - the volume of our fluid.
    /// @param _density - the density of our fluid
    /// @param _contanerSize - the contaner size of for our fluid. e.g. 1 is a cube of 1*1*1.
    //----------------------------------------------------------------------------------------------------------------------
    SPHEngine(unsigned int _numParticles = 0, unsigned int _volume = 1, float _density = 1000, float _contanerSize = 1);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief default destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~SPHEngine();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Initialises our class. This allocates all memory needed for the simulation.
    //----------------------------------------------------------------------------------------------------------------------
    void init();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates our particles positions with navier-stokes equations based on timestep of the update.
    /// @param _timeStep - the timestep we wish to update our particles with.
    //----------------------------------------------------------------------------------------------------------------------
    void update(float _timeStep);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Draws our VAO. This contatins a buffer of positions mapped to vertexAttriBArray 0
    //----------------------------------------------------------------------------------------------------------------------
    void drawArrays();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief returns the output postion openGL buffer of our particles
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getPositionBuffer(){return m_VAO;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mutator for the volume of our fluid. This in turn effects the mass of our particles.
    //----------------------------------------------------------------------------------------------------------------------
    inline void setVolume(float _volume){m_volume = _volume; calcMass();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query the volume of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getVolume(){return m_volume;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mutator for the desity of our fluid. This in turn effects the mass of our particles.
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDesity(float _density){m_density = _density; calcMass();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query the density of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getDensity(){return m_density;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mutator for the smoothing length of our simulation. Can also be thought of as hash cell size.
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(float _length){m_smoothingLength = _length; calcKernalConsts();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query the smoothing length of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getSmoothingLength(){return m_smoothingLength;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mutator for the gas constant of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setGasConstant(float _gasConst){m_gasConstant = _gasConst;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query the gas constant of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getGasConst(){return m_gasConstant;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mutator for our viscosity coeficient of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setViscCoef(float _viscCoef){m_viscCoef = _viscCoef;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query the viscosity coeficient of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getViscCoef(){return m_viscCoef;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutation for our tension coeficient of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTensionCoef(float _tensCoef){m_tensionCoef = _tensCoef;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to query our tension coeficient of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline float getTensionCoef(){return m_tensionCoef;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our velocity correction
    //----------------------------------------------------------------------------------------------------------------------
    inline void setVelCorCoef(float _velCorCoef){m_velocityCorrectionCoef = _velCorCoef;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator for our maximum number of samples per particle in our fluid solver
    /// @param _numSamples - desired maximum number of samples per particle in fluid solver
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMaxNumSamples(int _numSamples){m_maxNumSamples = _numSamples;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief adds a colllision plane to our simulation
    /// @param _pos - the position of our plane
    /// @param _norm - the normal of our plane
    /// @param _resrCoef - coeficient of restitution of our plane
    //----------------------------------------------------------------------------------------------------------------------
    void addWall(float3 _pos, float3 _norm, float _restCoef = 1.0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to number of particles in simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline int getNumParticles(){return m_numParticles;}
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the max dimention of our hash grid. E.g. 1 is a cube of dimention 1*1*1.
    //----------------------------------------------------------------------------------------------------------------------
    float m_maxGridDim;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the max number of samples of neighbouring particles in our fluid solving
    //----------------------------------------------------------------------------------------------------------------------
    int m_maxNumSamples;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of planes we have in our buffer
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int m_numPlanes;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our plane buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    planeProp* m_dPlaneBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief velocity correction used for XSPH in our fluid sovler
    //----------------------------------------------------------------------------------------------------------------------
    float m_velocityCorrectionCoef;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief calculate our kernal constance which depends on our hash cell size
    /// @brief smoothing kernals taken from M¨uller, M., Charypar, D. and Gross, M. (2003).
    /// @brief Particle-based fluid simulation for interactive applications,
    /// @brief SCA ’03: Proceedings of the 2003 ACM SIGGRAPH
    //----------------------------------------------------------------------------------------------------------------------
    void calcKernalConsts();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant to calculate our density kernal. Cheaper to calculate once on CPU.
    //----------------------------------------------------------------------------------------------------------------------
    float m_densWeightConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant to calculate our pressure kernal. Cheaper to calculate once on CPU.
    //----------------------------------------------------------------------------------------------------------------------
    float m_pressWeightConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constant to calculate our viscosity kernal. Cheaper to calculate once on CPU.
    //----------------------------------------------------------------------------------------------------------------------
    float m_viscWeightConst;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the smoothing length of our simulation.
    //----------------------------------------------------------------------------------------------------------------------
    float m_smoothingLength;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the cell size of our hash function
    //----------------------------------------------------------------------------------------------------------------------
    float m_cellSize;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the maximum number of particles per cell
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int m_numParticlePerCell;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a simple function to caluclate next prime number of the inpur variable
    /// @param _x - vaule you wish to calculate next prime number for
    /// @return the next lowest prime number higher than our input value _X
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int nextPrimeNum(int _x);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief A function to calculate the mass of our particles based on density, volume and number of particles in our simulation.
    /// @brief mass = density ( volume / number of particles )
    //----------------------------------------------------------------------------------------------------------------------
    inline void calcMass(){m_mass = m_density * (m_volume/(float)m_numParticles);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The mass of our fluid.
    /// @value Defaults to 1. User can modify density and volume to change this.
    //----------------------------------------------------------------------------------------------------------------------
    float m_mass;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the density of our fluid.
    /// @value Defaults to 1. User may modify this.
    //----------------------------------------------------------------------------------------------------------------------
    float m_density;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the volume of our fluid.
    /// @value Defaults to the number of particles in simulation. User may modify this.
    //----------------------------------------------------------------------------------------------------------------------
    float m_volume;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our gas constant for our fluid solving
    //----------------------------------------------------------------------------------------------------------------------
    float m_gasConstant;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our viscosity coeficient for our fluid solving
    //----------------------------------------------------------------------------------------------------------------------
    float m_viscCoef;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our tension coeficent for our fluid solving
    //----------------------------------------------------------------------------------------------------------------------
    float m_tensionCoef;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int m_numParticles;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our openGL VAO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief openGL buffer that hold the positions of our particles on the GPU
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_VBO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Cuda resource used to access our OpenGL position buffer by Cuda
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_cudaBufferPtr;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our particle velocity buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    float3 *m_dVelBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our hash keys buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int *m_dhashKeys;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the size of our hash table
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int m_hashTableSize;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief pointer to our cell occupancy buffer on our device
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int *m_dCellOccBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a buffer that holds the index's of where the points in our hash cell begin.
    //----------------------------------------------------------------------------------------------------------------------
    unsigned int *m_dCellIndexBuffer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of threads that our device will haveper block
    //----------------------------------------------------------------------------------------------------------------------
    int m_numThreadsPerBlock;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the max number of blocks that we can have
    //----------------------------------------------------------------------------------------------------------------------
    int m_maxNumBlocks;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // SPHENGINE_H
