#ifndef HELLOCUDA_H
#define HELLOCUDA_H
//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSPHKernals.h
/// @author Declan Russell
/// @date 08/03/2015
/// @version 1.0
/// @brief Used for prototyping our our functions used for our SPH simulation calculations.
/// @brief This file is linked with CudaSPHKernals.cu by gcc after CudaSPHKernals.cu has been
/// @brief compiled by nvcc.
//----------------------------------------------------------------------------------------------------------------------

#include <stdio.h>


//just a function that moves our particles with some wave functions
void calcPositions(float3* d_pos, int timeStep, int numParticles, int maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief a structure to hold the properties of our particle
//----------------------------------------------------------------------------------------------------------------------
struct particleProp {
    float3 pos;
    float3 vel;
    float density;
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief a structure to hold the properties of our planes
//----------------------------------------------------------------------------------------------------------------------
struct planeProp{
    float3 pos;
    float3 normal;
    float restCoef;
};
//----------------------------------------------------------------------------------------------------------------------
/// @brief Creates an index array for our cells using thrusts exclusive scan
/// @param d_cellOccArray - pointer to our hash table cell occupancy buffer on our device
/// @param _size - the size of our buffer on our divice
/// @param d_cellIdxArray - our output buffer on our device to store our index's
//----------------------------------------------------------------------------------------------------------------------
void createCellIdx(unsigned int* d_cellOccArray, unsigned int _size, unsigned int *d_cellIdxArray);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Creates a spatial hash key based on our particle postion
/// @brief This is taken from Optimized Spatial Hashing for Collision Detection of Deformable Objects.
/// @param d_hashArray - a pointer to the cuda buffer that we wish to store our hash keys
/// @param d_posArray - pointer to the cuda buffer that holds the particle postions we wish to hash
/// @param numParticles - the number of particles
/// @param smoothingLength - smoothing length of our hash. You can think of this as how many different hash keys availible.
/// @param hashTableSize - size of our hash table
/// @param maxNumThreads - the maximum number off threads we have in a block on our device. Can be found out with device query
//----------------------------------------------------------------------------------------------------------------------
void createHashTable(unsigned int* d_hashArray, float3* d_posArray, unsigned int numParticles, float smoothingLegnth,unsigned int hashTableSize, int maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Sorts our hash key buffer and postion buffer such that points of the same key occupy contiguous memory
/// @param d_hashArray - pointer to our hash key buffer
/// @param d_posArray - pointer to our particle position buffer
/// @param d_velArray - pointer to our particle velocity buffer
/// @param d_accArray - pointer to our particle acceleration buffer
/// @param _numParticles - the number of particels in our buffer
//----------------------------------------------------------------------------------------------------------------------
void sortByKey(unsigned int* d_hashArray, float3* d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int _numParticles);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Computes the particle occupancy of our hash cell
/// @param d_hashArray - pointer to our hash key buffer
/// @param d_cellOccArray - pointer to our cell occupancy array
/// @param _hashTableSize - size of our hash table
/// @param _numPoints - number of points in our hash table
/// @param _maxNumThreads - the maximum number of threads that we have per block on our device
//----------------------------------------------------------------------------------------------------------------------
void countCellOccupancy(unsigned int *d_hashArray, unsigned int *d_cellOccArray,unsigned int _hashTableSize, unsigned int _numPoints, unsigned int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
/// @brief simple function so that we can fill a buffer with unsigned ints.
/// @param _pointer - pointer to the buffer you wish to fill
/// @param _arraySize - size of the buffer you wish to fill
/// @param _fill - what you wish to fill it with
//----------------------------------------------------------------------------------------------------------------------
void fillUint(unsigned int *_pointer, unsigned int _arraySize, unsigned int _fill);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Solver function that will call our solver kernal to calculate the new positions of the particles in
/// @brief our fluid simulation.
/// @param d_posArray - pointer to our gpu buffer that holds the postions of our particles
/// @param d_velArray - pointer to our gpu buffer that holds the velocities of our particles
/// @param d_accArray - pointer to our gpu buffer that holds the accelleration of our particles
/// @param d_cellOccArray - pointer to our gpu buffer that holds the cell occupancy count of our hash table
/// @param d_cellIndxArray - pointer to our gpu buffer that holds the cell index's of our particles
/// @param _hashTableSize - the size of our hash table. This is used to calculate how many blocks we need to launch our kernal with
/// @param _maxNumThreads - the maximum nuber of threads we need to launch per block
/// @param _smoothingLength - smoothing length of our simulation. Can also be thought of as cell size.
/// @param _timeStep - the timestep that we want to increment our particles positions in our solver
/// @param _particleMass - the mass of each particle. Defaults to 1.
/// @param _restDensity - the density of each particle at rest. Defaults to 1.
/// @param _gasConstant - the gas constant of our fluid. Used for calculating pressure. Defaults to 1.
/// @param _visCoef - the coeficient of viscosity in our fluid simulation. Defaults to 1.
/// @param _densKernConst - constant part of the density kernal. Faster to compute once on CPU and load in.
/// @param _pressKernConst - constant part of the pressure kernal. Faster to compute once on CPU and load in.
/// @param _viscKernConst - constant part of the viscosity kernal. Faster to compute once on CPU and load in.
//----------------------------------------------------------------------------------------------------------------------
void fluidSolver(float3 *d_posArray, float3 *d_velArray, float3 *d_accArray, unsigned int *d_cellOccArray, unsigned int *d_cellIndxArray, unsigned int _hashTableSize, unsigned int _maxNumThreads, float _smoothingLength, float _timestep, float _particleMass = 1, float _restDensity = 1, float _gasConstant = 1, float _visCoef = 1, float _densKernConst = 1, float _pressKernConst = 1, float _viscKernConst = 1);
//----------------------------------------------------------------------------------------------------------------------
/// @brief Collision detection between particles and planes
/// @param d_PlaneArray - pointer to device buffer of our planes information
/// @param _numPlanes - number of planes in our array
/// @param d_posArray - pointer to device buffer of our particle positions
/// @param d_velArray - pointer to device buffer of our particle velocities
/// @param _timeStep - time step of our update
/// @param _numParticles - the number of particles in our scene
/// @param _maxNumThreads - the maximum nuber of threads we need to launch per block
//----------------------------------------------------------------------------------------------------------------------
void collisionDetectionSolver(planeProp* d_planeArray, unsigned int _numPlanes, float3 *d_posArray, float3 *d_velArray, float _timeStep, unsigned int _numParticles, unsigned int _maxNumThreads);
//----------------------------------------------------------------------------------------------------------------------
void test(float _const);

#endif // HELLOCUDA_H
