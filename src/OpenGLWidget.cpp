#include <QGuiApplication>

#include "OpenGLWidget.h"
#include <iostream>
#include <time.h>


#include <ngl/ShaderLib.h>

#include "GLTextureLib.h"
#include "RenderTargetLib.h"

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
    m_pan = false;
    m_modelPos=ngl::Vec3(0.0);
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    RenderTargetLib::getInstance()->destroy();
    GLTextureLib::getInstance()->destroy();
    ngl::NGLInit *Init = ngl::NGLInit::instance();
    std::cout<<"Shutting down NGL, removing VAO's and Shaders\n";
    Init->NGLQuit();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){

    // we must call this first before any other GL commands to load and link the
    // gl commands from the lib, if this is not done program will crash
    ngl::NGLInit::instance();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    // enable depth testing for drawing
    glEnable(GL_DEPTH_TEST);
    // enable multisampling for smoother drawing
    glEnable(GL_MULTISAMPLE);
    //enable point sprites
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);

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
    ngl::Vec3 from(5,0,15);
    ngl::Vec3 to(5,0,0);
    ngl::Vec3 up(0,1,0);
    m_cam= new ngl::Camera(from,to,up);
    // set the shape using FOV 45 Aspect Ratio based on Width and Height
    // The final two are near and far clipping planes of 0.1 and 100
    m_cam->setShape(45,(float)width()/height(),1,100);


    //get an instance of our shader library
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    //create our cuboid shader
    //Not the fastest way to draw cubes but could save some valuable global memory?
    //Also the geometry shader is really fun to use so why not!
    //create the program
    shader->createShaderProgram("CuboidShader");
    //add our shaders
    shader->attachShader("cuboidVert",ngl::VERTEX);
    shader->attachShader("cuboidGeom",ngl::GEOMETRY);
    shader->attachShader("cuboidFrag",ngl::FRAGMENT);
    //load the source
    shader->loadShaderSource("cuboidVert","shaders/cuboidVert.glsl");
    shader->loadShaderSource("cuboidGeom","shaders/cuboidGeom.glsl");
    shader->loadShaderSource("cuboidFrag","shaders/cuboidFrag.glsl");
    //compile them
    shader->compileShader("cuboidVert");
    shader->compileShader("cuboidGeom");
    shader->compileShader("cuboidFrag");
    //attach them to our program
    shader->attachShaderToProgram("CuboidShader","cuboidVert");
    shader->attachShaderToProgram("CuboidShader","cuboidGeom");
    shader->attachShaderToProgram("CuboidShader","cuboidFrag");
    //link our shader to opengl
    shader->linkProgramObject("CuboidShader");

    //create our dummy VAO for our shader
    glGenVertexArrays(1,&m_cubeVAO);
    glBindVertexArray(m_cubeVAO);
    GLuint cubeVBO;
    glGenBuffers(1, &cubeVBO);
    glBindBuffer(GL_ARRAY_BUFFER,cubeVBO);
    ngl::Vec3 nullpos(0,0,0);
    glBufferData(GL_ARRAY_BUFFER,sizeof(ngl::Vec3),&nullpos,GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(ngl::Vec3),(GLvoid*)(0*sizeof(GL_FLOAT)));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);


    m_currentTime = m_currentTime.currentTime();
    startTimer(0);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resizeGL(const int _w, const int _h){
    // set the viewport for openGL
    glViewport(0,0,_w,_h);
    m_cam->setShape(45,(float)_w/_h, m_cam->getNear(),m_cam->getFar());
    m_text->setScreenSize(_w,_h);
    for(unsigned int i=0;i<m_fluidShaders.size();i++){
        m_fluidShaders[i]->resize(_w,_h);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::timerEvent(QTimerEvent *){
    updateGL();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::paintGL(){

    // calculate the framerate
    QTime newTime = m_currentTime.currentTime();
    int msecsPassed = m_currentTime.msecsTo(newTime);
    m_currentTime = m_currentTime.currentTime();


    //update our fluid simulations with our time step
    for(unsigned int i=0;i<m_SPHEngines.size();i++){
        if(m_fluidSimProps[i].m_update){
            if(m_fluidSimProps[i].m_updateWithFixedTimeStep){
                m_SPHEngines[i]->update((m_fluidSimProps[i].m_timeStep) * m_fluidSimProps[i].m_playSpeed);
            }
            else{
                m_SPHEngines[i]->update(((float)msecsPassed/1000.f) * m_fluidSimProps[i].m_playSpeed);
            }
        }
    }


    // create the rotation matrices
    ngl::Mat4 rotX;
    ngl::Mat4 rotY;
    ngl::Mat4 rotXY;
    rotX.rotateX(m_spinXFace);
    rotY.rotateY(m_spinYFace);
    rotXY = rotX*rotY;

    //get an instance of our shader library
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    //draw our fluid
    ngl::Mat4 V = m_cam->getViewMatrix();
    ngl::Mat4 P = m_cam->getProjectionMatrix();
    for(unsigned int i=0;i<m_fluidShaders.size();i++){
        ngl::Mat4 M = m_mouseGlobalTX;
        M.m_m[3][0] = m_fluidSimProps[i].m_simPosition.m_x;
        M.m_m[3][1] = m_fluidSimProps[i].m_simPosition.m_y;
        M.m_m[3][2] = m_fluidSimProps[i].m_simPosition.m_z;
        M = M * rotXY;
        M.m_m[3][0] += m_modelPos.m_x;
        M.m_m[3][1] += m_modelPos.m_y;
        M.m_m[3][2] += m_modelPos.m_z;
        m_fluidShaders[i]->draw(m_SPHEngines[i]->getPositionBuffer(),m_SPHEngines[i]->getNumParticles(),M,V,P,rotXY,m_cam->getEye());
        if(m_fluidSimProps[i].m_displayHud){
            (*shader)["CuboidShader"]->use();
            shader->setUniform("color",1.f,0.f,0.f);
            shader->setUniform("cubeMin",0.f,0.f,0.f);
            float boxSize = m_SPHEngines[i]->getGridSize();
            shader->setUniform("cubeMax",boxSize,boxSize,boxSize);
            ngl::Mat4 MVP = M*m_cam->getVPMatrix();
            shader->setUniform("MVP",MVP);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(m_cubeVAO);
            glDrawArrays(GL_POINTS,0,1);
            shader->setUniform("color",0.f,1.f,0.f);
            float3 spawnPos = m_SPHEngines[i]->getSpawnBoxPos();
            shader->setUniform("cubeMin",spawnPos.x,spawnPos.y,spawnPos.z);
            float spawnboxSize = m_SPHEngines[i]->getSpawnBoxSize();
            shader->setUniform("cubeMax",spawnPos.x+spawnboxSize,spawnPos.y+spawnboxSize,spawnPos.z+spawnboxSize);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(m_cubeVAO);
            glDrawArrays(GL_POINTS,0,1);
            glEnable(GL_DEPTH_TEST);
        }

    }

    //write our framerate
    QString text;
    if(msecsPassed==0){
        text.sprintf("framerate is faster than we can calculate lul :')");
    }
    else{
        text.sprintf("framerate is %f",(float)(1000.0/msecsPassed));
    }
    m_text->renderText(10,20,text);
    int totalParticles = 0;
    for(unsigned int i=0;i<m_SPHEngines.size();i++){
        totalParticles+=m_SPHEngines[i]->getNumParticles();
    }
    text.sprintf("Number of particles: %d",totalParticles);
    m_text->renderText(10,40,text);

    //if we want our camera to pan the increment our rotation
    if(m_pan){ m_spinYFace-=INCREMENT*5;}

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::loadCubeMap(QString _loc){
    //seperate our cube map texture our into its 6 side textures
    QImage img(_loc);
    img = img.mirrored(false,true);
    //some error checking
    if(img.isNull()){
        std::cerr<<"Error: Environment map mage could not be loaded."<<std::endl;
        return;
    }

    int wStep = img.width()/4;
    int hStep = img.height()/3;

    QImage front = QGLWidget::convertToGLFormat( img.copy(wStep,hStep,wStep,hStep));
    QImage bottom = QGLWidget::convertToGLFormat( img.copy(wStep,2*hStep,wStep,hStep));
    QImage top = QGLWidget::convertToGLFormat( img.copy(wStep,0,wStep,hStep));
    QImage left = QGLWidget::convertToGLFormat( img.copy(0,hStep,wStep,hStep));
    QImage right = QGLWidget::convertToGLFormat( img.copy(2*wStep,hStep,wStep,hStep));
    QImage back = QGLWidget::convertToGLFormat( img.copy(3*wStep,hStep,wStep,hStep));

    m_fluidShaders[0]->setCubeMap(wStep,hStep,front.bits(),back.bits(),top.bits(),bottom.bits(),left.bits(),right.bits());

}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::addFluidSim(){
    //set up some inofrmation for when we update these simulations
    fluidSimProps props;
    props.m_simPosition = ngl::Vec3(0,0,0);
    props.m_update = false;
    props.m_timeStep = 0.004;
    props.m_updateWithFixedTimeStep = true;
    props.m_playSpeed = 1;
    props.m_displayHud = false;

    m_fluidSimProps.push_back(props);

    //create our fluid shader
    m_fluidShaders.push_back(new FluidShader(width(),height()));
    m_fluidShaders[m_fluidShaders.size()-1]->setScreenSize(width(),height());
    if(m_fluidShaders.size()==1){
        loadCubeMap("textures/skyCubeMap.png");
    }
    m_fluidShaders[m_fluidShaders.size()-1]->init();

    //allocate some space for our SPHEngine
    m_SPHEngines.push_back(new SPHEngine());
}


//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    if(_event->key()==Qt::Key_Escape){
        QGuiApplication::exit();
    }
    if(_event->key()==Qt::Key_E){
        for(unsigned int i=0;i<m_SPHEngines.size();i++){
            m_SPHEngines[i]->update(0.001);
        }
    }
    if(_event->key()==Qt::Key_R){
        m_pan = !m_pan;
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

