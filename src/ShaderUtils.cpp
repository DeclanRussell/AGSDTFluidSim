#include "ShaderUtils.h"

GLuint shaderUtils::createShaderFromFile(const GLchar *path, GLenum shaderType){
   GLuint shaderID = glCreateShader(shaderType);

   std::ifstream fin;
   fin.open(path);
   if (!fin.is_open()){
      fin.close();
      std::cerr<<"File Not Found! : "<<path<<std::endl;
      return 0;
   }
   std::string source((std::istreambuf_iterator<GLchar>(fin)), std::istreambuf_iterator<GLchar>());
   fin.close();

   const GLchar* shaderSource = source.c_str();

   glShaderSource(shaderID, 1, &shaderSource, NULL);

   glCompileShader(shaderID);

   GLint compileStatus;
   glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compileStatus);
   if (compileStatus != GL_TRUE){
      std::cerr<<"Shader failed to compile: "<< path<< " !"<<std::endl;

      GLint infoLogLength;
      glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

      GLchar* infoLog = new GLchar[infoLogLength + 1];
      glGetShaderInfoLog(shaderID, infoLogLength + 1, NULL, infoLog);
      std::cerr<<infoLog <<std::endl;
      delete infoLog;

      return 0;
   }

   return shaderID;
}
