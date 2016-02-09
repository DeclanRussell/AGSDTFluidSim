#ifndef CAMERA_H
#define CAMERA_H

//----------------------------------------------------------------------------------------------------------------------
/// @class Camera
/// @file Camera.h
/// @brief OpenGL camera class for creating viewport and projection matricies
/// @todo Rewrite Tobies shitty Camera class. Its not even commented!! FFS TOBY! USELESS!!!
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif

#include <iostream>
#include <cmath>

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera{
public:
   Camera(){}
   Camera(glm::vec3 _pos);
   Camera(glm::vec3 _from, glm::vec3 _to, glm::vec3 _up);
   ~Camera();
   void setPosition(glm::vec3 _position);
   void lookAt(glm::vec3 _position, glm::vec3 _center, glm::vec3 _up);
   inline glm::mat4 getViewMatrix(){return m_viewMatrix;}
   inline glm::mat4 getVPMatrix(){return m_VPMatrix;}
   //void setShape()
   void setProjectionMatrix(float _fov, float _aspect, float _near, float _far);
   inline glm::mat4 getProjectionMatrix(){return m_projectionMatrix;}
   void setShape(float _aspect);
   void setShape(float _fov, float _w, float _h, float _near, float _far);
   inline float getNear(){return m_near;}
   inline float getFar(){return m_far;}
   inline float getAspect(){return m_aspect;}
   inline float getFOV(){return m_fov;}
   inline glm::vec3 getPos(){return m_position;}


private:
   glm::vec3 m_position;
   glm::vec3 m_up;
   glm::mat4 m_projectionMatrix;
   glm::mat4 m_viewMatrix;
   glm::mat4 m_VPMatrix;
   float m_aspect;
   float m_fov;
   float m_near;
   float m_far;
};

#endif // CAMERA_H
