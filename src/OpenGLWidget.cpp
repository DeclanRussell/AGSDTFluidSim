#include <QGuiApplication>

#include "OpenGLWidget.h"
#include <iostream>
#include <time.h>

#include <ngl/NGLInit.h>
#include <ngl/Random.h>


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
    m_modelPos=ngl::Vec3(0.0);
    m_update = false;
    m_playBackSpeed = 1.f;
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    ngl::NGLInit *Init = ngl::NGLInit::instance();
    std::cout<<"Shutting down NGL, removing VAO's and Shaders\n";
    Init->NGLQuit();
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){

    // we must call this first before any other GL commands to load and link the
    // gl commands from the lib, if this is not done program will crash
    ngl::NGLInit::instance();

    glClearColor(0.5f, 0.5f, 0.5f, 0.0f);
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
    ngl::Vec3 from(5,15,15);
    ngl::Vec3 to(5,0,0);
    ngl::Vec3 up(0,1,0);
    m_cam= new ngl::Camera(from,to,up);
    // set the shape using FOV 45 Aspect Ratio based on Width and Height
    // The final two are near and far clipping planes of 0.1 and 100
    m_cam->setShape(45,(float)width()/height(),1,1000);

    //create our fluid shader
    m_fluidShader = new FluidShader(width(),height());
    m_fluidShader->setScreenSize(width(),height());
    loadCubeMap("textures/skyCubeMap.png");
    m_fluidShader->init();

    //allocate some space for our SPHEngine
    m_SPHEngine = new SPHEngine(10000,15.0f,998.2f,10.f);
    m_SPHEngine->setGasConstant(1);

    m_currentTime = m_currentTime.currentTime();
    startTimer(0);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::resizeGL(const int _w, const int _h){
    // set the viewport for openGL
    glViewport(0,0,_w,_h);
    m_cam->setShape(45,(float)_w/_h, m_cam->getNear(),m_cam->getFar());
    m_text->setScreenSize(_w,_h);
    m_fluidShader->resize(_w,_h);
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

    if(m_update){
        //update our fluid simulation with our time step
        m_SPHEngine->update((0.004f) * m_playBackSpeed);
        //m_SPHEngine->update(((float)msecsPassed/1000.f) * m_playBackSpeed);
    }

    // create the rotation matrices
    ngl::Mat4 rotX;
    ngl::Mat4 rotY;

    rotX.rotateX(m_spinXFace);
    rotY.rotateY(m_spinYFace);
    // multiply the rotations
    m_mouseGlobalTX=rotY*rotX;
    // add the translations
    m_mouseGlobalTX.m_m[3][0] = m_modelPos.m_x;
    m_mouseGlobalTX.m_m[3][1] = m_modelPos.m_y;
    m_mouseGlobalTX.m_m[3][2] = m_modelPos.m_z;

    //draw our fluid
    ngl::Mat4 V = m_cam->getViewMatrix();
    ngl::Mat4 P = m_cam->getProjectionMatrix();
    m_fluidShader->draw(m_SPHEngine->getPositionBuffer(),m_SPHEngine->getNumParticles(),m_mouseGlobalTX,V,P,m_cam->getEye());


    //write our framerate
    QString text;
    if(msecsPassed==0){
        text.sprintf("framerate is faster than we can calculate lul :')");
    }
    else{
        text.sprintf("framerate is %f",(float)(1000.0/msecsPassed));
    }
    m_text->renderText(10,20,text);
    text.sprintf("Number of particles: %d",m_SPHEngine->getNumParticles());
    m_text->renderText(10,40,text);

}
//----------------------------------------------------------------------------------------------------------------------
bool OpenGLWidget::loadCubeMap(QString _loc){
    //seperate our cube map texture our into its 6 side textures
    QImage img(_loc);
    img = img.mirrored(false,true);
    //some error checking
    if(img.isNull()){
        return false;
    }

    int wStep = img.width()/4;
    int hStep = img.height()/3;

    QImage front = QGLWidget::convertToGLFormat( img.copy(wStep,hStep,wStep,hStep));
    QImage bottom = QGLWidget::convertToGLFormat( img.copy(wStep,2*hStep,wStep,hStep));
    QImage top = QGLWidget::convertToGLFormat( img.copy(wStep,0,wStep,hStep));
    QImage left = QGLWidget::convertToGLFormat( img.copy(0,hStep,wStep,hStep));
    QImage right = QGLWidget::convertToGLFormat( img.copy(2*wStep,hStep,wStep,hStep));
    QImage back = QGLWidget::convertToGLFormat( img.copy(3*wStep,hStep,wStep,hStep));

    m_fluidShader->setCubeMap(wStep,hStep,front.bits(),back.bits(),top.bits(),bottom.bits(),left.bits(),right.bits());

}

//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    if(_event->key()==Qt::Key_Escape){
        QGuiApplication::exit();
    }
    if(_event->key()==Qt::Key_E){
        m_SPHEngine->update(0.001);
    }
    if(_event->key()==Qt::Key_Space){
        m_update = !m_update;
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

