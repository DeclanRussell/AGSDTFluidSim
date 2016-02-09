#include "Shader.h"
#include "ShaderUtils.h"

Shader::Shader(std::string _path, GLenum _type){
    m_shaderID = shaderUtils::createShaderFromFile(_path.c_str(), _type);
}

Shader::Shader(GLenum _type)
{
    m_shaderType = _type;
}

void Shader::loadFromSource(std::string _path)
{
    m_shaderID = shaderUtils::createShaderFromFile(_path.c_str(), m_shaderType);
}

Shader::~Shader(){
   glDeleteShader(m_shaderID);
}

GLuint Shader::getShaderID(){
   return m_shaderID;
}

