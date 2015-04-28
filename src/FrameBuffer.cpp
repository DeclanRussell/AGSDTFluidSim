#include "FrameBuffer.h"

//----------------------------------------------------------------------------------------------------------------------
FrameBuffer::FrameBuffer() : AbstractOpenGlObject()
{
    glGenFramebuffers(1,&m_handle);
    m_target = GL_FRAMEBUFFER;
}
//----------------------------------------------------------------------------------------------------------------------
FrameBuffer::~FrameBuffer(){
    glDeleteFramebuffers(1,&m_handle);
}
//----------------------------------------------------------------------------------------------------------------------
void FrameBuffer::bind(){
    glBindFramebuffer(m_target, m_handle);
}
//----------------------------------------------------------------------------------------------------------------------
void FrameBuffer::unbind(){
    glBindFramebuffer(m_target,0);
}
//----------------------------------------------------------------------------------------------------------------------
void FrameBuffer::setFrameBufferTexture(GLenum _attachment, GLuint _texture, GLint _level){
    glFramebufferTexture(m_target,_attachment,_texture,_level);
}
//----------------------------------------------------------------------------------------------------------------------
void FrameBuffer::setDrawbuffers(GLsizei _n, const GLenum *_bufs){
    glDrawBuffers(_n,_bufs);
}
//----------------------------------------------------------------------------------------------------------------------
