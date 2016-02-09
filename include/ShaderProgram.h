#ifndef __SHADERPROGRAM_H_
#define __SHADERPROGRAM_H_

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif

#include <iostream>
#include <glm/glm.hpp>
#include <vector>
#include "Shader.h"

class Shader;

class ShaderProgram{
public:
   ShaderProgram();
   virtual ~ShaderProgram();
   void attachShader(Shader* _shader);
   void bindFragDataLocation(GLuint _colourAttatchment, std::string _name);
   void link();
   void use();
   inline GLuint getProgramID(){return m_programID;}
   GLint getAttribLoc(std::string _name);
   GLint getUniformLoc(std::string _name);
private:
   GLuint m_programID;
};

#endif

