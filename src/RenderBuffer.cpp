#include "RenderBuffer.h"

RenderBuffer::RenderBuffer(GLenum _internalformat, GLenum _attachment, GLsizei _width, GLsizei _height) : AbstractOpenGlObject(),
                                                                                      m_width(_width),
                                                                                      m_height(_height),
                                                                                      m_internalFormat(_internalformat)
{
    glGenRenderbuffers(1,&m_target);
    bind();
    glRenderbufferStorage(GL_RENDERBUFFER,m_internalFormat,m_width,m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,_attachment,GL_RENDERBUFFER,m_target);
}
//----------------------------------------------------------------------------------------------------------------------
RenderBuffer::~RenderBuffer(){
    glDeleteRenderbuffers(1,&m_handle);
}
//----------------------------------------------------------------------------------------------------------------------
void RenderBuffer::bind(){
    glBindRenderbuffer(GL_RENDERBUFFER, m_target);
}
//----------------------------------------------------------------------------------------------------------------------
void RenderBuffer::unbind(){
    glBindRenderbuffer(GL_RENDERBUFFER,0);
}
//----------------------------------------------------------------------------------------------------------------------
void RenderBuffer::resize(GLsizei _width, GLsizei _height){
    m_width = _width;
    m_height = _height;
    bind();
    glRenderbufferStorage(GL_RENDERBUFFER,m_internalFormat,m_width,m_height);
    unbind();
}
//----------------------------------------------------------------------------------------------------------------------
