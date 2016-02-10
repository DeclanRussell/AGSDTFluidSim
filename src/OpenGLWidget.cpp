#include <QGuiApplication>

#include "OpenGLWidget.h"
#include <iostream>
#include <time.h>
#include <math.h>

#include "GLTextureLib.h"
#include "RenderTargetLib.h"
#include "ShaderLib.h"

#define DtoR M_PI/180.f

//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for x/y translation with mouse movement
//----------------------------------------------------------------------------------------------------------------------
const static float INCREMENT=0.01f;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for the wheel zoom
//----------------------------------------------------------------------------------------------------------------------
const static float ZOOM=0.5f;

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
    m_modelPos=glm::vec3(0.0);
    // re-size the widget to that of the parent (in this case the GLFrame passed in on construction)
    this->resize(_parent->size());
}
//----------------------------------------------------------------------------------------------------------------------
OpenGLWidget::~OpenGLWidget(){
    // Remove all our
    RenderTargetLib::getInstance()->destroy();
    GLTextureLib::getInstance()->destroy();
    ShaderLib::getInstance()->destroy();
    std::cout<<"Shutting down NGL, removing VAO's and Shaders\n";
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::initializeGL(){

#ifndef DARWIN
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!! "<<std::endl;
    }
#endif

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
    m_text = new Text(QFont("calibri",14));
    m_text->setColour(255,0,0);
    m_text->setScreenSize(width(),height());

    // Initialise the model matrix
    m_modelMatrix = glm::mat4(1.0);

    // Initialize the camera
    // Now we will create a basic Camera from the graphics library
    // This is a static camera so it only needs to be set once
    // First create Values for the camera position
    glm::vec3 from(0,0,10);
    glm::vec3 to(0,0,0);
    glm::vec3 up(0,1,0);
    m_cam= new Camera(from,to,up);
    // set the shape using FOV 45 Aspect Ratio based on Width and Height
    // The final two are near and far clipping planes of 0.1 and 100
    m_cam->setShape(90.f,(float)width(),(float)height(),0.1f,100.f);

    //get an instance of our shader library
    ShaderLib *shader = ShaderLib::getInstance();
    //create our cuboid shader
    //Not the fastest way to draw cubes but could save some valuable global memory?
    //Also the geometry shader is really fun to use so why not!
    //create the program
    shader->createShaderProgram("CuboidShader");
    //add our shaders
    shader->attachShader("cuboidVert",GL_VERTEX_SHADER);
    shader->attachShader("cuboidGeom",GL_GEOMETRY_SHADER);
    shader->attachShader("cuboidFrag",GL_FRAGMENT_SHADER);
    //load the source
    shader->loadShaderSource("cuboidVert","shaders/cuboidVert.glsl");
    shader->loadShaderSource("cuboidGeom","shaders/cuboidGeom.glsl");
    shader->loadShaderSource("cuboidFrag","shaders/cuboidFrag.glsl");
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
    glm::vec3 nullpos(0,0,0);
    glBufferData(GL_ARRAY_BUFFER,sizeof(glm::vec3),&nullpos,GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(glm::vec3),(GLvoid*)(0*sizeof(GL_FLOAT)));
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
    m_cam->setShape(45,(float)_w,(float)_h, m_cam->getNear(),m_cam->getFar());
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
            m_SPHEngines[i]->update();
        }
    }

    // create the rotation matrices
    glm::mat4 rotX;
    glm::mat4 rotY;
    glm::mat4 rotXY;
    rotX = glm::rotate(rotX,(float)DtoR*m_spinXFace,glm::vec3(1,0,0));
    rotY = glm::rotate(rotY,(float)DtoR*m_spinYFace,glm::vec3(0,1,0));
    rotXY = rotY*rotX;

    //get an instance of our shader library
    ShaderLib *shader=ShaderLib::getInstance();
    //draw our fluid
    glm::mat4 V = m_cam->getViewMatrix();
    glm::mat4 P = m_cam->getProjectionMatrix();
    for(unsigned int i=0;i<m_fluidShaders.size();i++){
        glm::mat4 M = m_mouseGlobalTX;
        M[3][0] = m_fluidSimProps[i].m_simPosition.x;
        M[3][1] = m_fluidSimProps[i].m_simPosition.y;
        M[3][2] = m_fluidSimProps[i].m_simPosition.z;
        M = rotXY * M;
        M[3][0] += m_modelPos.x;
        M[3][1] += m_modelPos.y;
        M[3][2] += m_modelPos.z;
        m_fluidShaders[i]->draw(m_SPHEngines[i]->getPositionsVAO(),m_SPHEngines[i]->getNumParticles(),M,V,P,rotXY,glm::vec4(m_cam->getPos(),1.0f));
        if(m_fluidSimProps[i].m_displayHud){
            (*shader)["CuboidShader"]->use();
            shader->setUniform("color",1.f,0.f,0.f);
            shader->setUniform("cubeMin",m_fluidSimProps[i].m_simPosition.x,m_fluidSimProps[i].m_simPosition.y,m_fluidSimProps[i].m_simPosition.z);
            float3 boxMax = m_fluidSimProps[i].m_simPosition + m_fluidSimProps[i].m_simSize;
            shader->setUniform("cubeMax",boxMax.z,boxMax.y,boxMax.z);
            glm::mat4 MVP = m_cam->getVPMatrix()*M;
            shader->setUniform("MVP",MVP);
            glDisable(GL_DEPTH_TEST);
            glBindVertexArray(m_cubeVAO);
            glDrawArrays(GL_POINTS,0,1);
            shader->setUniform("color",0.f,1.f,0.f);
            shader->setUniform("cubeMin",m_fluidSimProps[i].m_spawnMin.x,m_fluidSimProps[i].m_spawnMin.y,m_fluidSimProps[i].m_spawnMin.z);
            shader->setUniform("cubeMax",m_fluidSimProps[i].m_spawnMin.x+m_fluidSimProps[i].m_spawnDim.x,m_fluidSimProps[i].m_spawnMin.y+m_fluidSimProps[i].m_spawnDim.y,m_fluidSimProps[i].m_spawnMin.z+m_fluidSimProps[i].m_spawnDim.z);
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
void OpenGLWidget::resizeGL(QResizeEvent *_event)
{
    // set the viewport for openGL
    glViewport(0,0,_event->size().width(),_event->size().height());
    m_cam->setShape(45.f,(float)_event->size().width(),(float)_event->size().height(), m_cam->getNear(),m_cam->getFar());
    m_text->setScreenSize(_event->size().width(),_event->size().height());
    for(unsigned int i=0;i<m_fluidShaders.size();i++){
        m_fluidShaders[i]->resize(_event->size().width(),_event->size().height());
    }
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
void OpenGLWidget::addParticlesToSim(int _numParticles, int _simNo)
{
    fluidSimProps props = m_fluidSimProps[_simNo];

    // Create our particle positions
    float tx,ty,tz;
    float increment = pow((props.m_spawnDim.x*props.m_spawnDim.y*props.m_spawnDim.z)/_numParticles,1.f/3.f);
    float3 min = props.m_spawnMin;
    float3 max = props.m_spawnMin + props.m_spawnDim;
    tx=1.f;
    ty=1.f;
    tz=1.f;
    float3 tempF3;
    std::vector<float3> particles;
    for(int i=0; i<_numParticles; i++){
        if(tx>=(max.x)){ tx=min.x; tz+=increment;}
        if(tz>=(max.z)){ tz=min.z; ty+=increment;}
        tempF3.x = tx;
        tempF3.y = ty;
        tempF3.z = tz;
        particles.push_back(tempF3);
        tx+=increment;
    }
    // Add particles to sim
    m_SPHEngines[_simNo]->setParticles(particles);
}
//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::addFluidSim(){
    //set up some inofrmation for when we update these simulations
    fluidSimProps props;
    props.m_simPosition = make_float3(0,0,0);
    props.m_simSize = make_float3(10,10,10);
    props.m_spawnMin = make_float3(1,1,1);
    props.m_spawnDim = make_float3(3,3,3);
    props.m_update = false;
    props.m_displayHud = false;

    m_fluidSimProps.push_back(props);

    //create our fluid shader
    m_fluidShaders.push_back(new FluidShader(width(),height()));
    m_fluidShaders[m_fluidShaders.size()-1]->setScreenSize(width(),height());
    if(m_fluidShaders.size()==1){
        loadCubeMap("textures/skyCubeMap.png");
    }
    m_fluidShaders[m_fluidShaders.size()-1]->init();

    SPHSolverCUDA *sim = new SPHSolverCUDA();
    sim->setHashPosAndDim(props.m_simPosition,props.m_simSize);
    sim->setTensionConst(1.f);

    //allocate some space for our SPHEngine
    m_SPHEngines.push_back(sim);
}


//----------------------------------------------------------------------------------------------------------------------
void OpenGLWidget::keyPressEvent(QKeyEvent *_event){
    if(_event->key()==Qt::Key_Escape){
        QGuiApplication::exit();
    }
    if(_event->key()==Qt::Key_E){
        for(unsigned int i=0;i<m_SPHEngines.size();i++){
            m_SPHEngines[i]->update();
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
        m_modelPos.x += INCREMENT * diffX;
        m_modelPos.y -= INCREMENT * diffY;
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
        m_modelPos.z+=ZOOM;
    }
    else if(_event->delta() <0 )
    {
        m_modelPos.z-=ZOOM;
    }
    updateGL();
}
//-----------------------------------------------------------------------------------------------------------------------

