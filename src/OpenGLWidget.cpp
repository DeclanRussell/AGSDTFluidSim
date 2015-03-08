#include <QGuiApplication>

#include "OpenGLWidget.h"
#include <iostream>
#include <time.h>

#include <ngl/NGLInit.h>
#include <ngl/ShaderLib.h>
#include <ngl/Random.h>
#include <ngl/VertexArrayObject.h>
#include <ngl/ShaderLib.h>
#include <ngl/Random.h>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

#include "hellocuda.h"



//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for x/y translation with mouse movement
//----------------------------------------------------------------------------------------------------------------------
const static float INCREMENT=0.01;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for the wheel zoom
//----------------------------------------------------------------------------------------------------------------------
const static float ZOOM=0.1;

OpenGLWidget::OpenGLWidget(const QGLFormat _format, QWidget *_parent) : QGLWidget(_format,_parent){
    // set this widget to have the initial keyboard focus
    setFocus();
    setFocusPolicy( Qt::StrongFocus );
    //init some members
    m_rotate=false;
    // mouse rotation values set to 0
    m_spinXFace=0;
    m_spinYFace=0;
    m_modelPos=ngl::Vec3(0.0);
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    ngl::NGLInit *Init = ngl::NGLInit::instance();
    std::cout<<"Shutting down NGL, removing VAO's and Shaders\n";
    Init->NGLQuit();
    // Make sure we remember to unregister our cuda resource
    cudaGraphicsUnregisterResource(m_cudaBufferPtr);
    glDeleteBuffers(3,m_VBO);
    glDeleteVertexArrays(1,&m_VAO);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){

    // we must call this first before any other GL commands to load and link the
    // gl commands from the lib, if this is not done program will crash
    ngl::NGLInit::instance();

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    // enable depth testing for drawing
    glEnable(GL_DEPTH_TEST);
    // enable multisampling for smoother drawing
    //glEnable(GL_MULTISAMPLE);

    // as re-size is not explicitly called we need to do this.
    glViewport(0,0,width(),height());

    //used for drawing text later
    m_text = new ngl::Text(QFont("calibri",14));
    m_text->setColour(255,0,0);
    m_text->setScreenSize(width(),height());

    // Initialise the model matrix
    m_modelMatrix = ngl::Mat4(1.0);

    // Initialize the camera
    // Now we will create a basic Camera from the graphics library
    // This is a static camera so it only needs to be set once
    // First create Values for the camera position
    ngl::Vec3 from(0,0,40);
    ngl::Vec3 to(0,0,0);
    ngl::Vec3 up(0,1,0);
    m_cam= new ngl::Camera(from,to,up);
    // set the shape using FOV 45 Aspect Ratio based on Width and Height
    // The final two are near and far clipping planes of 0.5 and 10
    m_cam->setShape(45,(float)width()/height(),0.5,150);

    //set up our instancing shader
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    //first lets add out deformation shader to our program
    //create the program
    shader->createShaderProgram("InstancedPhong");
    //add our shaders
    shader->attachShader("InstancedPhongVert",ngl::VERTEX);
    shader->attachShader("InstancedPhongFrag",ngl::FRAGMENT);
    //load the source
    shader->loadShaderSource("InstancedPhongVert","shaders/instancedPhongVert.glsl");
    shader->loadShaderSource("InstancedPhongFrag","shaders/instancedPhongFrag.glsl");
    //compile them
    shader->compileShader("InstancedPhongVert");
    shader->compileShader("InstancedPhongFrag");
    //attach them to our program
    shader->attachShaderToProgram("InstancedPhong","InstancedPhongVert");
    shader->attachShaderToProgram("InstancedPhong","InstancedPhongFrag");
    //link our shader to opengl
    shader->linkProgramObject("InstancedPhong");

    //set some uniforms
    (*shader)["InstancedPhong"]->use();
    shader->setShaderParam3f("light.position",-1,-1,-1);
    shader->setShaderParam3f("light.intensity",0.8,0.8,0.8);
    shader->setShaderParam3f("Kd",0.5, 0.5, 0.5);
    shader->setShaderParam3f("Ka",0.5, 0.5, 0.5);
    shader->setShaderParam3f("Ks",1.0,1.0,1.0);
    shader->setShaderParam1f("shininess",100.0);
    shader->setShaderParam4f("color",0,1,1,1);


    //load in our sphere model to represent our particles
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile("models/sphere.obj", aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);
    aiMesh* mesh = scene->mMeshes[0];

    glGenBuffers(3, m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &mesh->mVertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(aiVector3D)*mesh->mNumVertices, &mesh->mNormals[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //create some points just for testing our instancing


    std::vector<float3> particles;
    ngl::Random *rnd = ngl::Random::instance();
    ngl::Vec3 tempPoint;
    float3 tempF3;
    for(int i=0; i<30000; i++){
        tempPoint = rnd->getRandomPoint(20,20,20);
        tempF3.x = tempPoint.m_x;
        tempF3.y = tempPoint.m_y;
        tempF3.z = tempPoint.m_z;
        particles.push_back(tempF3);
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[2]);
    //dynamic draw as we will probably be writting to this alot
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*particles.size(), &particles[0].x, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //register our particle postion buffer with cuda
    cudaGraphicsGLRegisterBuffer(&m_cudaBufferPtr, m_VBO[2], cudaGraphicsRegisterFlagsWriteDiscard);

    //set up our VAO
    // create a vao
    glGenVertexArrays(1,&m_VAO);
    glBindVertexArray(m_VAO);

    // connect the data to the shader input
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(aiVector3D), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(aiVector3D), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO[2]);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (GLvoid*)(0*sizeof(GL_FLOAT)));
    glVertexAttribDivisor(2, 1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);


    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found" << count << "CUDA device(s)" << std::endl;
    if(count == 0){
        std::cerr<<"Install an Nvidia chip scrub!"<<std::endl;
        return;
    }
    for (int i=0; i < count; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        QString deviceString = QString("* %1, Compute capability: %2.%3").arg(prop.name).arg(prop.major).arg(prop.minor);
        QString propString1 = QString("  Global mem: %1M, Shared mem per block: %2k, Registers per block: %3").arg(prop.totalGlobalMem / 1024 / 1024)
                .arg(prop.sharedMemPerBlock / 1024).arg(prop.regsPerBlock);
        QString propString2 = QString("  Warp size: %1 threads, Max threads per block: %2, Multiprocessor count: %3 MaxBlocks: %4")
                .arg(prop.warpSize).arg(prop.maxThreadsPerBlock).arg(prop.multiProcessorCount).arg(prop.maxGridSize[0]);
        std::cout << deviceString.toStdString() << std::endl;
        std::cout << propString1.toStdString() << std::endl;
        std::cout << propString2.toStdString() << std::endl;
        m_numThreadsPerBlock = prop.maxThreadsPerBlock;
        m_maxNumBlocks = prop.maxGridSize[0];
    }

    m_currentTime = m_currentTime.currentTime();
    startTimer(0);

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resizeGL(const int _w, const int _h){
    // set the viewport for openGL
    glViewport(0,0,_w,_h);
    m_cam->setShape(45,(float)_w/_h, 0.5,150);

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::timerEvent(QTimerEvent *){
    updateParticles();
    updateGL();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::updateParticles(){
    //map our buffer pointer
    float3* d_posPtr;
    size_t d_posSize;
    cudaGraphicsMapResources(1,&m_cudaBufferPtr,0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_posPtr,&d_posSize,m_cudaBufferPtr);

    calcPositions(d_posPtr,time(NULL),30000, m_numThreadsPerBlock);

    //unmap our buffer pointer and set it free into the wild
    cudaGraphicsUnmapResources(1,&m_cudaBufferPtr,0);

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::paintGL(){

    // calculate the framerate
    QTime newTime = m_currentTime.currentTime();
    int msecsPassed = m_currentTime.msecsTo(newTime);
    m_currentTime = m_currentTime.currentTime();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //Initialise the model matrix

    ngl::Mat4 rotX;
    ngl::Mat4 rotY;

    // create the rotation matrices
    rotX.rotateX(m_spinXFace);
    rotY.rotateY(m_spinYFace);
    // multiply the rotations
    m_mouseGlobalTX=rotY*rotX;
    // add the translations
    m_mouseGlobalTX.m_m[3][0] = m_modelPos.m_x;
    m_mouseGlobalTX.m_m[3][1] = m_modelPos.m_y;
    m_mouseGlobalTX.m_m[3][2] = m_modelPos.m_z;

    loadMatricesToShader();

    glBindVertexArray(m_VAO);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 2280, 30000);
    glBindVertexArray(0);

    QString text;
    if(msecsPassed==0){
        text.sprintf("framerate is faster than we can calculate lul :')");
    }
    else{
        text.sprintf("framerate is %f",(float)(1000.0/msecsPassed));
    }
    m_text->renderText(10,20,text);

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::loadMatricesToShader(){
    // Calculate MVP matricies
    ngl::Mat4 P = m_cam->getProjectionMatrix();
    ngl::Mat4 MV = m_mouseGlobalTX * m_cam->getViewMatrix();

    ngl::Mat3 normalMatrix = ngl::Mat3(MV);
    normalMatrix.inverse();
    normalMatrix.transpose();

    ngl::Mat4 MVP = MV * P;

    //Here is where we will ultimately load our matricies to shader once written
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    (*shader)["InstancedPhong"]->use();
    shader->setUniform("MV",MV);
    shader->setUniform("normalMatrix",normalMatrix);
    shader->setUniform("MVP",MVP);

}

//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    if(_event->key()==Qt::Key_Escape){
        QGuiApplication::exit();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseMoveEvent (QMouseEvent * _event)
{
    // note the method buttons() is the button state when event was called
    // this is different from button() which is used to check which button was
    // pressed when the mousePress/Release event is generated
    if(m_rotate && _event->buttons() == Qt::LeftButton){
        int diffx=_event->x()-m_origX;
        int diffy=_event->y()-m_origY;
        m_spinXFace += (float) 0.5f * diffy;
        m_spinYFace += (float) 0.5f * diffx;
        m_origX = _event->x();
        m_origY = _event->y();
        updateGL();
    }
    // right mouse translate code
    else if(m_translate && _event->buttons() == Qt::RightButton)
    {
        int diffX = (int)(_event->x() - m_origXPos);
        int diffY = (int)(_event->y() - m_origYPos);
        m_origXPos=_event->x();
        m_origYPos=_event->y();
        m_modelPos.m_x += INCREMENT * diffX;
        m_modelPos.m_y -= INCREMENT * diffY;
        updateGL();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mousePressEvent ( QMouseEvent * _event)
{
    // this method is called when the mouse button is pressed in this case we
    // store the value where the maouse was clicked (x,y) and set the Rotate flag to true
    if(_event->button() == Qt::LeftButton){
        m_origX = _event->x();
        m_origY = _event->y();
        m_rotate = true;
    }
    // right mouse translate mode
    else if(_event->button() == Qt::RightButton)
    {
        m_origXPos = _event->x();
        m_origYPos = _event->y();
        m_translate=true;
    }
}

//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::mouseReleaseEvent ( QMouseEvent * _event )
{
    // this event is called when the mouse button is released
    // we then set Rotate to false
    if(_event->button() == Qt::LeftButton){
        m_rotate = false;
    }
    // right mouse translate mode
    if (_event->button() == Qt::RightButton)
    {
        m_translate=false;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::wheelEvent(QWheelEvent *_event)
{
    // check the diff of the wheel position (0 means no change)
    if(_event->delta() > 0)
    {
        m_modelPos.m_z+=ZOOM;
    }
    else if(_event->delta() <0 )
    {
        m_modelPos.m_z-=ZOOM;
    }
    updateGL();
}
//-----------------------------------------------------------------------------------------------------------------------

