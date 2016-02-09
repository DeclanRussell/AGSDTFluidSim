#include "include/ShaderLib.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

// Declare our static variables
ShaderLib * ShaderLib::m_instance;

//----------------------------------------------------------------------------------------------------------------------
ShaderLib *ShaderLib::getInstance()
{
    if(!m_instance){
        m_instance = new ShaderLib();
    }
    return m_instance;
}
//----------------------------------------------------------------------------------------------------------------------
ShaderLib::~ShaderLib()
{
    //remove our shader programs
    std::map <std::string, ShaderProgram * >::const_iterator programs;
    for(programs=m_shaderPrograms.begin();programs!=m_shaderPrograms.end();programs++){
        std::cerr<<"Removing shader program "<<programs->first<<std::endl;
        delete programs->second;
    }
    //remove our shaders
    std::map <std::string, Shader * >::const_iterator shaders;
    for(shaders=m_shaders.begin();shaders!=m_shaders.end();shaders++){
        std::cerr<<"Removing shader "<<shaders->first<<std::endl;
        delete shaders->second;
    }
}
//----------------------------------------------------------------------------------------------------------------------
ShaderProgram *ShaderLib::operator[](const std::string &_name)
{
    //if its already active then just return the current texture
    if(_name==m_currentProgramName){
        return m_currentProgram;
    }

    //find our program in our library
    std::map <std::string, ShaderProgram * >::const_iterator program=m_shaderPrograms.find(_name);
    //make sure we have found something
    if(program!=m_shaderPrograms.end()){
        m_currentProgramName = _name;
        m_currentProgram = program->second;
        return m_currentProgram;
    }
    else{
        std::cerr<<"Cannot find Shader Program "<<_name<<std::endl;
        return 0;
    }
}

//----------------------------------------------------------------------------------------------------------------------
ShaderProgram *ShaderLib::operator[](const char *_name)
{
    //if its already active then just return the current texture
    if(_name==m_currentProgramName){
        return m_currentProgram;
    }

    //find our program in our library
    std::map <std::string, ShaderProgram * >::const_iterator program=m_shaderPrograms.find(_name);
    //make sure we have found something
    if(program!=m_shaderPrograms.end()){
        m_currentProgramName = _name;
        m_currentProgram = program->second;
        return m_currentProgram;
    }
    else{
        std::cerr<<"Cannot find Shader Program "<<_name<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::createShaderProgram(std::string _name)
{
    // See if our shader program already exists
    std::map <std::string, ShaderProgram * >::const_iterator program=m_shaderPrograms.find(_name);
    // If we find something then bail
    if(program!=m_shaderPrograms.end())
    {
        std::cerr<<"Shader Program already exists with name "<<_name<<std::endl;
        return;
    }
    std::cerr<<"Creating shader program "<<_name<<std::endl;
    // Otherwise lets create a shader program and add it to our library
    m_shaderPrograms[_name] = new ShaderProgram();
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::attachShader(std::string _name, GLenum _type)
{
    //see if a shader already exists with this name in our library
    std::map <std::string, Shader * >::const_iterator shader=m_shaders.find(_name);
    //If we have something found then we need to bail
    if(shader!=m_shaders.end()){
        std::cerr<<"Shader already exists with the name "<<_name<<std::endl;
        return;
    }

    std::string sType;
    switch (_type) {
    case GL_VERTEX_SHADER:
        sType = "vertex";
        break;
    case GL_GEOMETRY_SHADER:
        sType = "geometry";
        break;
    case GL_TESS_CONTROL_SHADER:
        sType = "tessalation";
        break;
    case GL_FRAGMENT_SHADER:
        sType = "fragment";
        break;
    default:
        sType = "other";
        break;
    }

    std::cerr<<"Creating "<<sType<<" shader "<<_name<<std::endl;
    // If all is good so far then lets make our shader
    Shader *s = new Shader(_type);
    // Add our shader to our map
    m_shaders[_name] = s;
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::loadShaderSource(std::string _name, std::string _loc)
{
    //see if a shader already exists with this name in our library
    std::map <std::string, Shader * >::const_iterator shader=m_shaders.find(_name);
    // If we have found our shader then load the source and compile our shader.
    if(shader!=m_shaders.end()){
        shader->second->loadFromSource(_loc);
    }
    else{
        //If we have cant find the shader of this name then we need to bail
        std::cerr<<"Cannot find Shader "<<_name<<std::endl;
        return;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::attachShaderToProgram(std::string _programName, std::string _shaderName)
{
    // Get our shader program
    ShaderProgram *p = (*this)[_programName];
    // If no program of this name can be found then we need to bail!
    if(!p){
        std::cerr<<"AttachShaderToProgram: Cannot find Shader Program "<<_programName<<std::endl;
        return;
    }
    // Otherwise if we have found our shader program then lets find our shader
    std::map <std::string, Shader * >::const_iterator shader=m_shaders.find(_shaderName);
    // If we have found our shader then lets attach is to our program.
    if(shader!=m_shaders.end()){
        std::cerr<<"Attaching shader "<<_shaderName<<" to program "<<_programName<<std::endl;
        p->attachShader(shader->second);
    }
    else{
        //If we have cant find the shader of this name then we need to bail!
        std::cerr<<"AttachShaderToProgram: Cannot find Shader "<<_shaderName<<std::endl;
        return;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::linkProgramObject(std::string _name)
{
    // Get our shader program
    ShaderProgram *p = (*this)[_name];
    // If no program of this name can be found then we need to bail!
    if(!p){
        std::cerr<<"linkProgramObject: Cannot find Shader Program "<<_name<<std::endl;
        return;
    }
    std::cerr<<"Linking shader program "<<_name<<std::endl;
    // Else lets link our program
    p->link();
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::use(std::string _name)
{
    // Get our shader program
    ShaderProgram *p = (*this)[_name];
    // If no program of this name can be found then we need to bail!
    if(!p){
        std::cerr<<"Use: Cannot find Shader Program "<<_name<<std::endl;
        return;
    }
    // Else lets use our program
    p->use();
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, float _x)
{
    glUniform1f(m_currentProgram->getUniformLoc(_name),_x);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, float _x, float _y)
{
    glUniform2f(m_currentProgram->getUniformLoc(_name),_x,_y);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, float _x, float _y, float _z)
{
    glUniform3f(m_currentProgram->getUniformLoc(_name),_x,_y,_z);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, float _x, float _y, float _z, float _w)
{
    glUniform4f(m_currentProgram->getUniformLoc(_name),_x,_y,_z,_w);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, int _x)
{
    glUniform1i(m_currentProgram->getUniformLoc(_name),_x);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, int _x, int _y)
{
    glUniform2i(m_currentProgram->getUniformLoc(_name),_x,_y);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, int _x, int _y, int _z)
{
    glUniform3i(m_currentProgram->getUniformLoc(_name),_x,_y,_z);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, int _x, int _y, int _z, int _w)
{
    glUniform4i(m_currentProgram->getUniformLoc(_name),_x,_y,_z,_w);
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, glm::mat2 &_m)
{
    glUniformMatrix2fv(m_currentProgram->getUniformLoc(_name), 1, GL_FALSE,glm::value_ptr(_m));
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, glm::mat3 &_m)
{
    glUniformMatrix3fv(m_currentProgram->getUniformLoc(_name), 1, GL_FALSE,glm::value_ptr(_m));
}
//----------------------------------------------------------------------------------------------------------------------
void ShaderLib::setUniform(std::string _name, glm::mat4 &_m)
{
    glUniformMatrix4fv(m_currentProgram->getUniformLoc(_name), 1, GL_FALSE,glm::value_ptr(_m));
}
//----------------------------------------------------------------------------------------------------------------------
ShaderLib::ShaderLib()
{
    m_currentProgram = 0;
    m_currentProgramName = "";
}
//----------------------------------------------------------------------------------------------------------------------
