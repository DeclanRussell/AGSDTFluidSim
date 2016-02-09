#include "ShaderProgram.h"
#include "Shader.h"

ShaderProgram::ShaderProgram(){
#ifdef LINUX
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!!"<<std::endl;
    }
#endif
   m_programID = glCreateProgram();
}

ShaderProgram::~ShaderProgram(){
   glDeleteProgram(m_programID);
}

void ShaderProgram::attachShader(Shader* _shader){
   glAttachShader(m_programID, _shader->getShaderID());
}

void ShaderProgram::bindFragDataLocation(GLuint _colourAttachment, std::string _name){
   glBindFragDataLocation(m_programID, _colourAttachment, _name.c_str());
}

void ShaderProgram::link(){
   glLinkProgram(m_programID);

   GLint linkStatus;
   glGetProgramiv(m_programID, GL_LINK_STATUS, &linkStatus);
   if (linkStatus != GL_TRUE){
      std::cerr<<"Program link failed to compile: "<<std::endl;

      GLint infoLogLength;
      glGetProgramiv(m_programID, GL_INFO_LOG_LENGTH, &infoLogLength);
      GLchar* infoLog = new GLchar[infoLogLength + 1];
      glGetProgramInfoLog(m_programID, infoLogLength + 1, NULL, infoLog);
      std::cerr<<infoLog <<std::endl;
      delete infoLog;
   }
}

void ShaderProgram::use(){
   glUseProgram(m_programID);
}

GLint ShaderProgram::getAttribLoc(std::string _name){
   return glGetAttribLocation(m_programID, _name.c_str());
}

GLint ShaderProgram::getUniformLoc(std::string _name){
   return glGetUniformLocation(m_programID, _name.c_str());
}
