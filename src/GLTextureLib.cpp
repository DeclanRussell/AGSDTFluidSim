#include "GLTextureLib.h"
#include <iostream>

//declare our static instance of the library
GLTextureLib * GLTextureLib::m_instance;

//----------------------------------------------------------------------------------------------------------------------
GLTextureLib *GLTextureLib::getInstance(){
    if(!m_instance){
        m_instance = new GLTextureLib();
    }
    return m_instance;
}
//----------------------------------------------------------------------------------------------------------------------
GLTextureLib::~GLTextureLib(){
    //remove our textures
    std::map <std::string, GLTexture * >::const_iterator texures;
    for(texures=m_textures.begin();texures!=m_textures.end();texures++){
        std::cerr<<"Removing texture "<<texures->first<<std::endl;
        delete texures->second;
    }
}
//----------------------------------------------------------------------------------------------------------------------
GLTexture* GLTextureLib::addTexture(std::string _name, GLenum _target, GLint _level, GLint _internalFormat, GLsizei _width, GLsizei _height, GLint _border, GLenum _format, GLenum _type, const GLvoid *_data){
    std::cerr<<"Creating texture "<<_name<<std::endl;
    GLTexture * newTex = new GLTexture(_target,_level,_internalFormat,_width,_height,_border,_format,_type,_data);
    m_textures[_name] = newTex;
    return newTex;
}
//----------------------------------------------------------------------------------------------------------------------
GLTexture * GLTextureLib::addCubeMap(std::string _name,GLenum _target, GLint _level, GLint _internalFormat, GLsizei _width, GLsizei _height, GLint _border, GLenum _format, GLenum _type, const GLvoid *_front, const GLvoid *_back, const GLvoid *_top, const GLvoid *_bottom, const GLvoid *_left, const GLvoid *_right){
    std::cerr<<"Creating cube map texture "<<_name<<std::endl;
    GLTexture * newTex = new GLTexture(_target,_level,_internalFormat,_width,_height,_border,_format,_type,_front,_back,_top,_bottom,_left,_right);
    m_textures[_name] = newTex;
    return newTex;
}

//----------------------------------------------------------------------------------------------------------------------
GLTexture * GLTextureLib::operator [](const std::string &_name){

    //if its already active then just return the current texture
    if(_name==m_currentTextureName){
        return m_currentTexture;
    }

    //find our texture in our library
    std::map <std::string, GLTexture * >::const_iterator texure=m_textures.find(_name);
    //make sure we have found something
    if(texure!=m_textures.end()){
        m_currentTextureName = _name;
        m_currentTexture = texure->second;
        return m_currentTexture;
    }
    else{
        std::cerr<<"no texure of that name"<<std::endl;
        return 0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
GLTexture * GLTextureLib::operator[](const char *_name){
    //if its already active then just return the current texture
    if(_name==m_currentTextureName){
        return m_currentTexture;
    }

    //find our texture in our library
    std::map <std::string, GLTexture * >::const_iterator texure=m_textures.find(_name);
    //make sure we have found something
    if(texure!=m_textures.end()){
        m_currentTextureName = _name;
        m_currentTexture = texure->second;
        return m_currentTexture;
    }
    else{
        std::cerr<<"no texure of that name"<<std::endl;
        return 0;
    }
}

//----------------------------------------------------------------------------------------------------------------------
