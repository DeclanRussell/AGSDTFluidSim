#include "GLTexture.h"

//----------------------------------------------------------------------------------------------------------------------
GLTexture::GLTexture(GLenum _target, GLint _level, GLint _internalFormat, GLsizei _width, GLsizei _height, GLint _border, GLenum _format, GLenum _type, const GLvoid *_data) :AbstractOpenGlObject()
{
    glGenTextures(1,&m_handle);
    m_target = _target;
    m_level = _level;
    m_internalFormat = _internalFormat;
    m_width = _width;
    m_height = _height;
    m_border = _border;
    m_format = _format;
    m_type = _type;
    m_isCubeMap = false;
    bind();
    if(_height){
        glTexImage2D(_target,_level,_internalFormat,_width,_height,_border,_format,_type,_data);
    }
    else{
        glTexImage1D(_target,_level,_internalFormat,_width,_border,_format,_type,_data);
    }
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
GLTexture::GLTexture(GLenum _target, GLint _level, GLint _internalFormat, GLsizei _width, GLsizei _height, GLint _border, GLenum _format, GLenum _type, const GLvoid *_front, const GLvoid *_back, const GLvoid *_top, const GLvoid *_bottom, const GLvoid *_left, const GLvoid *_right)
{
    glGenTextures(1,&m_handle);
    m_target = _target;
    m_level = _level;
    m_internalFormat = _internalFormat;
    m_width = _width;
    m_height = _height;
    m_border = _border;
    m_format = _format;
    m_type = _type;
    m_isCubeMap = true;
    bind();
    // copy image data into 'target' side of cube map
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_Z, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _front);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _back);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _top);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_Y, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _bottom);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_X, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _left);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_X, m_level, m_internalFormat, _width, _height, m_border, m_format, m_type, _right);
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
GLTexture::~GLTexture(){
    if(m_isCubeMap){
        glDeleteTextures(6,&m_handle);
    }
    else{
        glDeleteTextures(1,&m_handle);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::setTexParamiteri(GLenum _pname, GLint _param){
    bind();
    glTexParameteri(m_target,_pname,_param);
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::setTexParamiterf(GLenum _pname, GLfloat _param){
    bind();
    glTexParameterf(m_target,_pname,_param);
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::resize(GLsizei _width, GLsizei _height){
    bind();
    if(m_height){
       glTexImage2D(m_target,m_level,m_internalFormat,_width,_height,m_border,m_format,m_type,0);
    }
    else{
       glTexImage1D(m_target,m_level,m_internalFormat,_width,m_border,m_format,m_type,0);
    }
    unbind();

}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::setData(const GLvoid *_data, GLsizei _width, GLsizei _height){
   bind();
    if(m_height){
       if(_width!=-1) m_width = _width;
       if(_height!=-1) m_height = _height;
       glTexImage2D(m_target,m_level,m_internalFormat,m_width,m_height,m_border,m_format,m_type,_data);
    }
    else{
       if(_width!=-1) m_width - _width;
       glTexImage1D(m_target,m_level,m_internalFormat,m_width,m_border,m_format,m_type,_data);
    }
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::setData(const GLvoid *_front, const GLvoid *_back, const GLvoid *_top, const GLvoid *_bottom, const GLvoid *_left, const GLvoid *_right, GLsizei _width, GLsizei _height){
    bind();
    if(_width!=-1) m_width = _width;
    if(_height!=-1) m_height = _height;
    // copy image data into 'target' side of cube map
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_Z, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _front);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _back);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _top);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_Y, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _bottom);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_NEGATIVE_X, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _left);
    glTexImage2D ( GL_TEXTURE_CUBE_MAP_POSITIVE_X, m_level, m_internalFormat, m_width, m_height, m_border, m_format, m_type, _right);
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::bind(){
    glBindTexture(m_target,m_handle);
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::bind(unsigned int _loc){
    glActiveTexture(GL_TEXTURE+_loc);
    glBindTexture(m_target,m_handle);
}
//----------------------------------------------------------------------------------------------------------------------
void GLTexture::unbind(){
    glBindTexture(m_target,0);
}
//----------------------------------------------------------------------------------------------------------------------
