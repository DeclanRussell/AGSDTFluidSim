#include "FluidShader.h"

#include <iostream>
#include <ngl/ShaderLib.h>
#include "RenderTargetLib.h"
#include "RenderBuffer.h"
#include "FrameBuffer.h"
#include "GLTextureLib.h"
#include "GLTexture.h"

//declare our static variable
int FluidShader::m_instanceCount;
bool FluidShader::m_cubeMapCreated;
GLuint FluidShader::m_billboardVAO;
int FluidShader::m_width;
int FluidShader::m_height;

//----------------------------------------------------------------------------------------------------------------------
FluidShader::FluidShader(int _width, int _height)
{
    m_width = _width;
    m_height = _height;
    //init our point size
    m_pointSize = 0.2f;
    //init refraction and fresnal powers
    setRefractionRatio(0.9f);
    m_fresnalPower = 3;
    m_pointThickness = 0.04f;
    //our blur shading init params
    m_blurFalloff = 10.f;
    m_blurRadius = 7.f;
    m_cubeMapCreated = false;
    //set our init fluid color to something nice
    m_fluidColor = ngl::Vec3(0,1,1);
    //set our instance number and increment
    m_instanceNo = m_instanceCount;
    if(m_instanceNo==0) m_cubeMapCreated=false;
    m_instanceCount++;
}
FluidShader::~FluidShader(){
    //clean up everything
    glDeleteVertexArrays(1,&m_billboardVAO);
    glDeleteVertexArrays(1,&m_cubeVAO);
}

//----------------------------------------------------------------------------------------------------------------------
void FluidShader::setCubeMap(int _width,int _height,const GLvoid *_front, const GLvoid *_back, const GLvoid *_top, const GLvoid *_bottom, const GLvoid *_left, const GLvoid *_right){
    GLTextureLib* tex = GLTextureLib::getInstance();
    if(!m_cubeMapCreated){
        tex->addCubeMap("cubeMap",GL_TEXTURE_CUBE_MAP, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE,_front,_back,_top,_bottom,_left,_right);
        // format cube map texture
        (*tex)["cubeMap"]->setTexParamiteri( GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        (*tex)["cubeMap"]->setTexParamiteri( GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        (*tex)["cubeMap"]->setTexParamiteri( GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        (*tex)["cubeMap"]->setTexParamiteri( GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (*tex)["cubeMap"]->setTexParamiteri(  GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        m_cubeMapCreated = true;
    }
    else{
        (*tex)["cubeMap"]->setData(_front,_back,_top,_bottom,_left,_right,_width,_height);
    }
}

//----------------------------------------------------------------------------------------------------------------------
void FluidShader::init(){
    //if the shader is already created lets not make it again
    if(m_instanceNo>0) return;
    if(!m_cubeMapCreated){
        std::cerr<<"Fluid Shader Error: no cube map set"<<std::endl;
        return;
    }

    //------------Set up our Fluid Shader---------------------------

    RenderTargetLib* renderTarget = RenderTargetLib::getInstance();
    //create our local frame buffer for our depth pass
    FrameBuffer* framebuffer = renderTarget->addFrameBuffer("depthFrameBuffer");
    framebuffer->bind();
    // The depth buffer so that we have a depth test when we render to texture
    renderTarget->addRenderBuffer("depthRenderBuffer",GL_DEPTH_COMPONENT,GL_DEPTH_ATTACHMENT,m_width,m_height);


    //Create our depth texture to render to on the GPU
    //note: For our shading technique it is important to have a high range in our colour values
    //      GL_RGBA WONT CUT IT!
    GLTexture *tex = GLTextureLib::getInstance()->addTexture("depthRender",GL_TEXTURE_2D, 0,GL_RGBA32F, m_width, m_height, 0,GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    //note poor filtering is needed to be accurate with pixels and have no smoothing
    tex->setTexParamiteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    tex->setTexParamiteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Set our render target to colour attachment 1
    framebuffer->setFrameBufferTexture(GL_COLOR_ATTACHMENT1, tex->getHandle(), 0);
    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT1};
    framebuffer->setDrawbuffers(1,DrawBuffers);

    //check to see if the frame buffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        std::cerr<<"Local framebuffer could not be created\n"<<std::endl;
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT){
            std::cerr<<"No texture attached"<<std::endl;
        }
        exit(-1);
    }

    //create our local frame buffer for our bilateral filter pass
    framebuffer = renderTarget->addFrameBuffer("bilateralFrameBuffer");
    framebuffer->bind();
    //create our bilateral texture on to render to on the GPU
    tex = GLTextureLib::getInstance()->addTexture("bilateralRender",GL_TEXTURE_2D, 0,GL_RGBA32F, m_width, m_height, 0,GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    tex->setTexParamiteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    tex->setTexParamiteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Set our render target to colour attachment 1
    framebuffer->setFrameBufferTexture(GL_COLOR_ATTACHMENT1, tex->getHandle(), 0);
    // Set the list of draw buffers.
    framebuffer->setDrawbuffers(1, DrawBuffers);

    //check to see if the frame buffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        std::cerr<<"Local framebuffer could not be created"<<std::endl;
        exit(-1);
    }

    //create our local frame buffer for our thickness render pass to
    framebuffer = renderTarget->addFrameBuffer("thicknessFrameBuffer");;
    framebuffer->bind();
    //create our bilateral texture on to render to on the GPU
    tex = GLTextureLib::getInstance()->addTexture("thicknessRender",GL_TEXTURE_2D, 0,GL_RGBA32F, m_width, m_height, 0,GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    tex->setTexParamiteri(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    tex->setTexParamiteri(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Set our render target to colour attachment 1
    framebuffer->setFrameBufferTexture( GL_COLOR_ATTACHMENT1, tex->getHandle(), 0);
    // Set the list of draw buffers.
    framebuffer->setDrawbuffers(1, DrawBuffers);

    //check to see if the frame buffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
        std::cerr<<"Local framebuffer could not be created"<<std::endl;
        exit(-1);
    }

    //create our cube geometry for our cube map
    float cubeVertex[] = {
      -10.0f,  10.0f, -10.0f,
      -10.0f, -10.0f, -10.0f,
       10.0f, -10.0f, -10.0f,
       10.0f, -10.0f, -10.0f,
       10.0f,  10.0f, -10.0f,
      -10.0f,  10.0f, -10.0f,

      -10.0f, -10.0f,  10.0f,
      -10.0f, -10.0f, -10.0f,
      -10.0f,  10.0f, -10.0f,
      -10.0f,  10.0f, -10.0f,
      -10.0f,  10.0f,  10.0f,
      -10.0f, -10.0f,  10.0f,

       10.0f, -10.0f, -10.0f,
       10.0f, -10.0f,  10.0f,
       10.0f,  10.0f,  10.0f,
       10.0f,  10.0f,  10.0f,
       10.0f,  10.0f, -10.0f,
       10.0f, -10.0f, -10.0f,

      -10.0f, -10.0f,  10.0f,
      -10.0f,  10.0f,  10.0f,
       10.0f,  10.0f,  10.0f,
       10.0f,  10.0f,  10.0f,
       10.0f, -10.0f,  10.0f,
      -10.0f, -10.0f,  10.0f,

      -10.0f,  10.0f, -10.0f,
       10.0f,  10.0f, -10.0f,
       10.0f,  10.0f,  10.0f,
       10.0f,  10.0f,  10.0f,
      -10.0f,  10.0f,  10.0f,
      -10.0f,  10.0f, -10.0f,

      -10.0f, -10.0f, -10.0f,
      -10.0f, -10.0f,  10.0f,
       10.0f, -10.0f, -10.0f,
       10.0f, -10.0f, -10.0f,
      -10.0f, -10.0f,  10.0f,
       10.0f, -10.0f,  10.0f
    };

    glGenVertexArrays (1, &m_cubeVAO);
    glBindVertexArray (m_cubeVAO);

    GLuint cubeVBO;
    glGenBuffers (1, &cubeVBO);
    glBindBuffer (GL_ARRAY_BUFFER, cubeVBO);
    glBufferData (GL_ARRAY_BUFFER, 3 * 36 * sizeof (float), &cubeVertex, GL_STATIC_DRAW);
    glBindBuffer (GL_ARRAY_BUFFER, cubeVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, 0, NULL);


    //create our billboard geomtry
    float vertex[]={
        //bottom left
        -1.0f,-1.0f,
        //top left
        -1.0f,1.0f,
        //bottom right
        1.0f,-1.0f,
        //top left
        -1.0f,1.0f,
        //top right
        1.0f,1.0f,
        //bottom right
        1.0f,-1.0f
    };
    float texCoords[]={
        //bottom left
        0.0,0.0f,
        //top left
        0.0f,1.0f,
        //bottom right
        1.0f,0.0f,
        //top left
        0.0f,1.0f,
        //top right
        1.0f,1.0f,
        //bottom right
        1.0f,0.0f
    };

    glGenVertexArrays(1,&m_billboardVAO);
    glBindVertexArray(m_billboardVAO);
    GLuint billboardVBO[2];
    glGenBuffers(2, billboardVBO);
    glBindBuffer(GL_ARRAY_BUFFER,billboardVBO[0]);
    glBufferData(GL_ARRAY_BUFFER,sizeof(vertex),vertex,GL_STATIC_DRAW);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2.0,(GLvoid*)(0*sizeof(GL_FLOAT)));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,billboardVBO[1]);
    glBufferData(GL_ARRAY_BUFFER,sizeof(texCoords),texCoords,GL_STATIC_DRAW);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*2.0,(GLvoid*)(0*sizeof(GL_FLOAT)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);


    //set up our particle shader to render depth information to a texture
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();

    //Create our sky box shader
    //create the program
    shader->createShaderProgram("SkyBoxShader");
    //add our shaders
    shader->attachShader("skyBoxVert",ngl::VERTEX);
    shader->attachShader("skyBoxFrag",ngl::FRAGMENT);
    //load the source
    shader->loadShaderSource("skyBoxVert","shaders/skyBoxVert.glsl");
    shader->loadShaderSource("skyBoxFrag","shaders/skyBoxFrag.glsl");
    //compile them
    shader->compileShader("skyBoxVert");
    shader->compileShader("skyBoxFrag");
    //attach them to our program
    shader->attachShaderToProgram("SkyBoxShader","skyBoxVert");
    shader->attachShaderToProgram("SkyBoxShader","skyBoxFrag");
    //link our shader to opengl
    shader->linkProgramObject("SkyBoxShader");

    //lets add out depth shader to our program
    //create the program
    shader->createShaderProgram("ParticleDepth");
    //add our shaders
    shader->attachShader("particleDepthVert",ngl::VERTEX);
    shader->attachShader("particleDepthFrag",ngl::FRAGMENT);
    //load the source
    shader->loadShaderSource("particleDepthVert","shaders/particleDepthVert.glsl");
    shader->loadShaderSource("particleDepthFrag","shaders/particleDepthFrag.glsl");
    //compile them
    shader->compileShader("particleDepthVert");
    shader->compileShader("particleDepthFrag");
    //attach them to our program
    shader->attachShaderToProgram("ParticleDepth","particleDepthVert");
    shader->attachShaderToProgram("ParticleDepth","particleDepthFrag");
    //link our shader to opengl
    shader->linkProgramObject("ParticleDepth");


    //lets add out thickness shader to our program
    //create the program
    shader->createShaderProgram("ThicknessShader");
    //add our shaders
    shader->attachShader("thicknessVert",ngl::VERTEX);
    shader->attachShader("thicknessFrag",ngl::FRAGMENT);
    //load the source
    shader->loadShaderSource("thicknessVert","shaders/thicknessVert.glsl");
    shader->loadShaderSource("thicknessFrag","shaders/thicknessFrag.glsl");
    //compile them
    shader->compileShader("thicknessVert");
    shader->compileShader("thicknessFrag");
    //attach them to our program
    shader->attachShaderToProgram("ThicknessShader","thicknessVert");
    shader->attachShaderToProgram("ThicknessShader","thicknessFrag");
    //link our shader to opengl
    shader->linkProgramObject("ThicknessShader");


    //create our bilateral filter shader
    //creat program
    shader->createShaderProgram("BilateralFilter");
    //ass shaders
    shader->attachShader("bilateralFilterVert",ngl::VERTEX);
    shader->attachShader("bilateralFilterFrag",ngl::FRAGMENT);
    //load source
    shader->loadShaderSource("bilateralFilterVert","shaders/bilateralFilterVert.glsl");
    shader->loadShaderSource("bilateralFilterFrag","shaders/bilateralFilterFrag.glsl");
    //compile them
    shader->compileShader("bilateralFilterVert");
    shader->compileShader("bilateralFilterFrag");
    //attach them to our program
    shader->attachShaderToProgram("BilateralFilter","bilateralFilterVert");
    shader->attachShaderToProgram("BilateralFilter","bilateralFilterFrag");
    //link our shader to openGL
    shader->linkProgramObject("BilateralFilter");

    //create our water shader
    //create program
    shader->createShaderProgram("FluidShader");
    //add shaders
    shader->attachShader("fluidShaderVert",ngl::VERTEX);
    shader->attachShader("fluidShaderFrag",ngl::FRAGMENT);
    //load source
    shader->loadShaderSource("fluidShaderVert","shaders/fluidShaderVert.glsl");
    shader->loadShaderSource("fluidShaderFrag","shaders/fluidShaderFrag.glsl");
    //compile them
    shader->compileShader("fluidShaderVert");
    shader->compileShader("fluidShaderFrag");
    //attach them to our program
    shader->attachShaderToProgram("FluidShader","fluidShaderVert");
    shader->attachShaderToProgram("FluidShader","fluidShaderFrag");
    //link our shader to openGL
    shader->linkProgramObject("FluidShader");



    //set some uniforms
    (*shader)["SkyBoxShader"]->use();
    shader->setUniform("cubeMapTex",2);

    (*shader)["ParticleDepth"]->use();
    shader->setUniform("screenWidth",m_width);

    (*shader)["ThicknessShader"]->use();
    shader->setUniform("screenWidth",m_width);
    shader->setUniform("thicknessScaler",m_pointThickness);

    (*shader)["BilateralFilter"]->use();
    shader->setUniform("depthTex",0);
    shader->setUniform("blurDepthFalloff",m_blurFalloff);
    shader->setUniform("texelSize",2.0f/((float)m_width+m_height));

    (*shader)["FluidShader"]->use();
    //set some uniforms
    shader->setUniform("depthTex",0);
    shader->setUniform("thicknessTex",1);
    shader->setUniform("cubeMapTex",2);
    shader->setUniform("fresnalPower",m_fresnalPower);
    shader->setUniform("refractRatio",m_refractionRatio);
    shader->setUniform("fresnalConst",m_fresnalConst);
    shader->setUniform("texelSizeX",1.0f/m_width);
    shader->setUniform("texelSizeY",1.0f/m_height);

    //to be used later with phong shading
    shader->setShaderParam3f("light.position",-1,-1,-1);
    shader->setShaderParam3f("light.intensity",0.8,0.8,0.8);
    shader->setShaderParam3f("Kd",0.5, 0.5, 0.5);
    shader->setShaderParam3f("Ka",0.5, 0.5, 0.5);
    shader->setShaderParam3f("Ks",1.0,1.0,1.0);
    shader->setShaderParam1f("shininess",1000.0);
}
//----------------------------------------------------------------------------------------------------------------------
void FluidShader::resize(int _w, int _h){
    m_width = _w;
    m_height = _h;
    //resize our render targets and depth buffer
    GLTextureLib *texLib = GLTextureLib::getInstance();
    (*texLib)["depthRender"]->resize(_w,_h);
    (*texLib)["bilateralRender"]->resize(_w,_h);
    (*texLib)["thicknessRender"]->resize(_w,_h);
    RenderBuffer* renderBuffer = RenderTargetLib::getInstance()->getRenderBuffer("depthRenderBuffer");
    renderBuffer->resize(_w,_h);
    //update our texel sizes
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();

    //set some uniforms
    (*shader)["ParticleDepth"]->use();
    shader->setUniform("screenWidth",_w);

    (*shader)["ThicknessShader"]->use();
    shader->setUniform("screenWidth",_w);

    (*shader)["FluidShader"]->use();;
    shader->setUniform("texelSizeX",1.0f/(float)_w);
    shader->setUniform("texelSizeY",1.0f/(float)_h);

    (*shader)["BilateralFilter"]->use();
    shader->setUniform("filterRadius",100.0f/*/width()*/);
    shader->setUniform("texelSize",2.0f/(_w+_h));
}
//----------------------------------------------------------------------------------------------------------------------
void FluidShader::draw(GLuint _positionVAO, int _numPoints, ngl::Mat4 _M, ngl::Mat4 _V, ngl::Mat4 _P,ngl::Mat4 _rotM, ngl::Vec4 _eyePos){
    ngl::Mat4 MV = _M * _V;
    ngl::Mat4 MVP = MV * _P;
    ngl::Mat4 Pinv = _P.inverse();
    ngl::Mat4 normalMatrix = _M.inverse();
    normalMatrix.transpose();

    //Here is where we will ultimately load our matricies to shader once written
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    (*shader)["ParticleDepth"]->use();

    //calculate the eyespace radius of our points
    ngl::Vec4 esr(m_pointSize,0,0,1.0);
    esr = Pinv * esr;
    //std::cout<<"real world size: "<<esr.m_x<<" "<<esr.m_y<<" "<<esr.m_y<<" "<<esr.m_w<<std::endl;

    shader->setUniform("screenWidth",m_width);
    shader->setUniform("pointSize",m_pointSize);
    shader->setUniform("pointRadius",esr.m_x/esr.m_w);
    shader->setUniform("P",_P);
    shader->setUniform("MV",MV);
    shader->setUniform("MVP",MVP);


    RenderTargetLib * renderTarget = RenderTargetLib::getInstance();
    // Render to our depth framebuffer
    renderTarget->getFrameBuffer("depthFrameBuffer")->bind();
    renderTarget->getRenderBuffer("depthRenderBuffer")->bind();
    //clear our frame buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(_positionVAO);
    glDrawArrays(GL_POINTS, 0, _numPoints);
    glBindVertexArray(0);


    //render our thickness pass
    renderTarget->getFrameBuffer("thicknessFrameBuffer")->bind();
    renderTarget->getRenderBuffer("depthRenderBuffer")->unbind();


    //bind our thickness shader
    (*shader)["ThicknessShader"]->use();
    shader->setUniform("thicknessScaler",m_pointThickness);
    shader->setUniform("screenWidth",m_width);
    shader->setUniform("pointSize",m_pointSize);
    shader->setUniform("P",_P);
    shader->setUniform("MV",MV);
    shader->setUniform("MVP",MVP);


    //clear our frame buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //disable our depth test
    glDisable(GL_DEPTH_TEST);
    //enable additive blending to accumilate thickness
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBindVertexArray(_positionVAO);
    glDrawArrays(GL_POINTS, 0, _numPoints);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    //resizeGL(width(),height());



    // Render to our bilateral filter framebuffer
    renderTarget->getFrameBuffer("bilateralFrameBuffer")->bind();
    renderTarget->getRenderBuffer("depthRenderBuffer")->bind();
    //clear our frame buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //bind our bilateral filter shader
    (*shader)["BilateralFilter"]->use();
    shader->setUniform("blurDepthFalloff",m_blurFalloff);
    float radius = (m_blurRadius/((m_height+m_width)*0.5));
    shader->setUniform("filterRadius",radius);
    //bind our billboard and texture
    glActiveTexture(GL_TEXTURE0);
    GLTextureLib* tex = GLTextureLib::getInstance();
    (*tex)["depthRender"]->bind();
    glBindVertexArray(m_billboardVAO);
    glDrawArrays(GL_TRIANGLES,0,6);


    //unbind our local static frame buffer so we now render to the screen
    renderTarget->getFrameBuffer("bilateralFrameBuffer")->unbind();
    renderTarget->getRenderBuffer("depthRenderBuffer")->unbind();


    if(m_instanceNo==0){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    //if this is our first instance of the shader
    //Draw our sky box
    //no point in drawing it more than once
    if(m_instanceNo==0){
        (*shader)["SkyBoxShader"]->use();
        //load our matricies to shader
        ngl::Mat4 MCube = _rotM;
        //move to where our camera is located
        MCube.m_m[3][0] = _eyePos.m_x;
        MCube.m_m[3][1] = _eyePos.m_y;
        MCube.m_m[3][2] = _eyePos.m_z;

        //set our MVP matrix
        ngl::Mat4 sbMVP = MCube*_V*_P;
        shader->setUniform("MVP",sbMVP);

        glDepthMask (GL_FALSE);
        glActiveTexture (GL_TEXTURE2);
        (*tex)["cubeMap"]->bind();
        glBindVertexArray (m_cubeVAO);
        //draw our cube
        glDrawArrays (GL_TRIANGLES, 0, 36);
        glDepthMask (GL_TRUE);
    }


    //bind our fluid shader
    (*shader)["FluidShader"]->use();
    shader->setUniform("PInv",Pinv);
    shader->setUniform("normalMatrix",normalMatrix);
    shader->setUniform("fresnalPower",m_fresnalPower);
    shader->setUniform("refractRatio",m_refractionRatio);
    shader->setUniform("fresnalConst",m_fresnalConst);
    shader->setUniform("color",(float)m_fluidColor.m_x,(float)m_fluidColor.m_y,(float)m_fluidColor.m_z);
    //bind our bilateral render texture
    glActiveTexture(GL_TEXTURE0);
    (*tex)["bilateralRender"]->bind();
    //bind our thickess texture
    glActiveTexture(GL_TEXTURE1);
    (*tex)["thicknessRender"]->bind();
    //draw our billboard
    glBindVertexArray (m_billboardVAO);
    glDrawArrays(GL_TRIANGLES,0,6);
}
//----------------------------------------------------------------------------------------------------------------------
