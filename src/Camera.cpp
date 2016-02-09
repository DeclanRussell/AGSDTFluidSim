#include "Camera.h"

Camera::Camera(glm::vec3 _pos){
   m_position = _pos;
   m_up = glm::vec3(0.0, 1.0, 0.0);
   m_fov = 60.0;
   m_aspect = 720.0/576.0;
   m_projectionMatrix = glm::perspective(m_fov, m_aspect, 0.1f, 350.0f);
//   m_projectionMatrix = glm::ortho(-1,1,-1,1);
   lookAt(m_position, glm::vec3(0.0, 0.0, 0.0), m_up);
}

Camera::Camera(glm::vec3 _from, glm::vec3 _to, glm::vec3 _up){
    m_position = _from;
    m_up = _up;
    m_fov = 60.0;
    m_aspect = 720.0/576.0;
    m_projectionMatrix = glm::perspective(m_fov, m_aspect, 0.1f, 350.0f);
    lookAt(m_position,_to,_up);
}

Camera::~Camera(){
}

void Camera::lookAt(glm::vec3 _position, glm::vec3 _center, glm::vec3 _up){
    m_viewMatrix = glm::lookAt(_position, _center, _up);
    m_VPMatrix = m_projectionMatrix * m_viewMatrix;
}

void Camera::setPosition(glm::vec3 _position){
   m_position = _position;
}

void Camera::setProjectionMatrix(float _fov, float _aspect, float _near, float _far){
    m_projectionMatrix = glm::perspective(_fov, _aspect, _near, _far);
//    m_projectionMatrix = glm::ortho(-1.0, 1.0, -1.0 , 1.0);
}

void Camera::setShape(float aspect){
    setProjectionMatrix(m_fov, aspect, 0.1f, 350.0f);
}

void Camera::setShape(float _fov,float _w, float _h, float _near, float _far){
    m_fov = _fov;
    m_near = _near;
    m_far = _far;
    setProjectionMatrix(m_fov, _w/_h, m_near, m_far);
}
