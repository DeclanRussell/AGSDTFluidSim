#ifndef SHADERLIB_H
#define SHADERLIB_H

//----------------------------------------------------------------------------------------------------------------------
/// @file ShaderLib.h
/// @class ShaderLib
/// @author Declan Russell
/// @date 08/02/1016
/// @version 1.0
/// @brief Singleton class for creating, storing OpenGL shaders in a library
//----------------------------------------------------------------------------------------------------------------------

#include <map>
#include "ShaderProgram.h"

class ShaderLib
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Accessor to the instance of our class
    //----------------------------------------------------------------------------------------------------------------------
    static ShaderLib *getInstance();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our default destructor.
    //----------------------------------------------------------------------------------------------------------------------
    ~ShaderLib();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief overloaded operators to access a shader program in our library
    /// @param _name - Name of shader program we want from our library
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram * operator[](const std::string &_name);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief overloaded operators to access a shader program in our library
    /// @param _name - Name of shader program we want from our library
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram * operator[](const char *_name);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates a shader program and adds it to our library
    /// @param _name - Desired name of shader
    //----------------------------------------------------------------------------------------------------------------------
    void createShaderProgram(std::string _name);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates a shader and adds it to our library
    /// @param _name - Desired name for our shader
    /// @param _type - Desired type of shader we wish to create
    //----------------------------------------------------------------------------------------------------------------------
    void attachShader(std::string _name, GLenum _type);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Loads a shader from source file
    /// @param _name - Desired name for our shader
    /// @param _loc - Location of shader source file
    //----------------------------------------------------------------------------------------------------------------------
    void loadShaderSource(std::string _name, std::string _loc);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Attaches a shader in our library to a shader program in our library
    /// @param _programName - name of the shader program in our library we wish to attach our shader to.
    /// @param _shaderName - name of the shader in our library we wish to attach to our program.
    //----------------------------------------------------------------------------------------------------------------------
    void attachShaderToProgram(std::string _programName, std::string _shaderName);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Links our shader program object from our library.
    /// @param _name - name of shader program we wish to link.
    //----------------------------------------------------------------------------------------------------------------------
    void linkProgramObject(std::string _name);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Use a shader program from our library.
    /// @param _name - name of shader program we wish to link.
    //----------------------------------------------------------------------------------------------------------------------
    void use(std::string _name);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Deletes our library
    //----------------------------------------------------------------------------------------------------------------------
    inline void destroy(){ if(m_instance) delete m_instance; }
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets float uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, float _x);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets float2 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, float _x, float _y);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets float3 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, float _x, float _y, float _z);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets float4 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, float _x, float _y, float _z, float _w);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets int uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, int _x);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets int2 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, int _x, int _y);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets int3 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, int _x, int _y, int _z);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets int4 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _x - Desired value for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, int _x, int _y, int _z, int _w);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets Mat2 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _m - Desired value matrix for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, glm::mat2 &_m);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets Mat3 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _m - Desired value matrix for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, glm::mat3 &_m);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets Mat4 uniform in current shader program
    /// @param _name - Uniform name
    /// @param _m - Desired value matrix for uniform
    //----------------------------------------------------------------------------------------------------------------------
    void setUniform(std::string _name, glm::mat4 &_m);
    //----------------------------------------------------------------------------------------------------------------------

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our default contructor. As this is a singleton class we dont want this to be availible to access.
    //----------------------------------------------------------------------------------------------------------------------
    ShaderLib();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our instance of our shader library
    //----------------------------------------------------------------------------------------------------------------------
    static ShaderLib *m_instance;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Map to store our Shader programs
    //----------------------------------------------------------------------------------------------------------------------
    std::map<std::string,ShaderProgram*> m_shaderPrograms;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Map to store our Shaders
    //----------------------------------------------------------------------------------------------------------------------
    std::map<std::string,Shader*> m_shaders;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our current selected shader program
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_currentProgram;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our current selected shader program name
    //----------------------------------------------------------------------------------------------------------------------
    std::string m_currentProgramName;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // SHADERLIB_H
