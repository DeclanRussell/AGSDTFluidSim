#include "RenderTargetLib.h"
#include <iostream>

//declare our static instance of the library
RenderTargetLib * RenderTargetLib::m_instance;

//----------------------------------------------------------------------------------------------------------------------
RenderTargetLib *RenderTargetLib::getInstance(){
    if(!m_instance){
        m_instance = new RenderTargetLib();
    }
    return m_instance;
}
//----------------------------------------------------------------------------------------------------------------------
RenderTargetLib::~RenderTargetLib(){
    //remove our render targets
    std::map <std::string, RenderBuffer * >::const_iterator renderBuffer;
    for(renderBuffer=m_renderBuffers.begin();renderBuffer!=m_renderBuffers.end();renderBuffer++){
        std::cerr<<"Removing render buffer "<<renderBuffer->first<<std::endl;
        delete renderBuffer->second;
    }
    std::map <std::string, FrameBuffer * >::const_iterator frameBuffer;
    for(frameBuffer=m_frameBuffers.begin();frameBuffer!=m_frameBuffers.end();frameBuffer++){
        std::cerr<<"Removing frame buffer "<<frameBuffer->first<<std::endl;
        delete frameBuffer->second;
    }
}
//----------------------------------------------------------------------------------------------------------------------
FrameBuffer * RenderTargetLib::addFrameBuffer(std::string _name){
    std::cerr<<"Creating frame buffer "<<_name<<std::endl;
    FrameBuffer * newFramebuffer= new FrameBuffer();
    m_frameBuffers[_name] = newFramebuffer;
    return newFramebuffer;
}
//----------------------------------------------------------------------------------------------------------------------
RenderBuffer * RenderTargetLib::addRenderBuffer(std::string _name, GLenum _internalformat, GLenum _attachment, GLsizei _width, GLsizei _height){
    std::cerr<<"Creating render buffer "<<_name<<std::endl;
    RenderBuffer * newRenderbuffer= new RenderBuffer(_internalformat,_attachment,_width,_height);
    m_renderBuffers[_name] = newRenderbuffer;
    return newRenderbuffer;
}
//----------------------------------------------------------------------------------------------------------------------
FrameBuffer * RenderTargetLib::getFrameBuffer(const std::string &_name){
    //if its already active then just return the current render target
    if(_name==m_currentFramebufferName){
        return m_currentFramebuffer;
    }

    //find our texture in our library
    std::map <std::string, FrameBuffer * >::const_iterator frameBuffer=m_frameBuffers.find(_name);
    //make sure we have found something
    if(frameBuffer!=m_frameBuffers.end()){
        m_currentFramebufferName = _name;
        m_currentFramebuffer = frameBuffer->second;
        return m_currentFramebuffer;
    }
    else{
        std::cerr<<"no frame buffer of that name"<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
FrameBuffer * RenderTargetLib::getFrameBuffer(const char *_name){
    //if its already active then just return the current render target
    if(_name==m_currentFramebufferName){
        return m_currentFramebuffer;
    }

    //find our texture in our library
    std::map <std::string, FrameBuffer * >::const_iterator frameBuffer=m_frameBuffers.find(_name);
    //make sure we have found something
    if(frameBuffer!=m_frameBuffers.end()){
        m_currentFramebufferName = _name;
        m_currentFramebuffer = frameBuffer->second;
        return m_currentFramebuffer;
    }
    else{
        std::cerr<<"no frame buffer of that name"<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
RenderBuffer * RenderTargetLib::getRenderBuffer(const std::string &_name){
    //if its already active then just return the current render target
    if(_name==m_currentRenderbufferName){
        return m_currentRenderbuffer;
    }

    //find our texture in our library
    std::map <std::string, RenderBuffer * >::const_iterator renderBuffer=m_renderBuffers.find(_name);
    //make sure we have found something
    if(renderBuffer!=m_renderBuffers.end()){
        m_currentRenderbufferName = _name;
        m_currentRenderbuffer = renderBuffer->second;
        return m_currentRenderbuffer;
    }
    else{
        std::cerr<<"no render buffer of that name"<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
RenderBuffer * RenderTargetLib::getRenderBuffer(const char *_name){
    //if its already active then just return the current render target
    if(_name==m_currentRenderbufferName){
        return m_currentRenderbuffer;
    }

    //find our texture in our library
    std::map <std::string, RenderBuffer * >::const_iterator renderBuffer=m_renderBuffers.find(_name);
    //make sure we have found something
    if(renderBuffer!=m_renderBuffers.end()){
        m_currentRenderbufferName = _name;
        m_currentRenderbuffer = renderBuffer->second;
        return m_currentRenderbuffer;
    }
    else{
        std::cerr<<"no render buffer of that name"<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
