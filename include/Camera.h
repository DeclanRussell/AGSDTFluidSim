#ifndef CAMERA_H
#define CAMERA_H

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif

#include <iostream>
#include <cmath>

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera{
public:
   Camera(glm::vec3 _pos);
   ~Camera();
   void setPosition(glm::vec3 _position);
   void lookAt(glm::vec3 _position, glm::vec3 _center, glm::vec3 _up);
   inline glm::mat4 getViewMatrix(){return m_viewMatrix;}
   //void setShape()
   void setProjectionMatrix(float _fov, float _aspect, float _near, float _far);
   inline glm::mat4 getProjectionMatrix(){return m_projectionMatrix;}
   void setShape(float _aspect);
   void setShape(float _w, float _h);


private:
   glm::vec3 m_position;
   glm::vec3 m_up;
   glm::mat4 m_projectionMatrix;
   glm::mat4 m_viewMatrix;
   float m_aspect;
   float m_fov;
};

#endif // CAMERA_H
