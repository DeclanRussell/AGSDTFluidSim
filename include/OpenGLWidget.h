#ifndef OPENGLWIDGET_H
#define OPENGLWIDGET_H

//----------------------------------------------------------------------------------------------------------------------
/// @file OpenGLWidget.h
/// @class OpenGLWidget
/// @brief Basic Qt widget that holds a OpenGL context
/// @author Declan Russell
/// @version 1.0
/// @date 2/3/15 Initial version
//----------------------------------------------------------------------------------------------------------------------

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif


#include <glm/vec3.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "Camera.h"
#include "Text.h" //<-- for writting in GL
#include <QGLWidget>
#include <QEvent>
#include <QResizeEvent>
#include <QMessageBox>
#include <QString>
#include <QTime>
#include <QColor>

#include "SPHSolverCUDA.h"
#include "FluidShader.h"

class OpenGLWidget : public QGLWidget
{
    Q_OBJECT //must include to gain access to qt stuff

public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    explicit OpenGLWidget(const QGLFormat _format, QWidget *_parent=0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dtor must close down and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~OpenGLWidget();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the virtual initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Qt 5.5.1 must have this implemented and uses it
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(QResizeEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief called to resize the window
    //----------------------------------------------------------------------------------------------------------------------
    void resizeGL(int _w, int _h );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief keyboard press event
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse move
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse button release
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mouse button press
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent(QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a timer event function from the Q_object
    //----------------------------------------------------------------------------------------------------------------------
    void timerEvent(QTimerEvent *);
    //----------------------------------------------------------------------------------------------------------------------
public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set sim position
    /// @param _x - x position of our simulation
    /// @param _y - y position of our simulation
    /// @param _z - z position of our simualtion
    /// @param _simNo - simulation to set the position of
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSimPosition(float _x, float _y, float _z,int _simNo = 0){m_fluidSimProps[_simNo].m_simPosition = make_float3(_x,_y,_z);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to add a fluid simulation to our scene
    //----------------------------------------------------------------------------------------------------------------------
    void addFluidSim();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to load cube maps to for our enironment map
    /// @brief this function pressumes that all our cube map textures are in one texture
    /// @param _loc - the location of our cube map texture
    //----------------------------------------------------------------------------------------------------------------------
    void loadCubeMap(QString _loc);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to change our particle size
    /// @param _size - particle size
    /// @param _simNo - which simulation we want to change the particle size in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setParticleSize(float _size, int _simNo = 0){m_fluidShaders[_simNo]->setPointSize(_size);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to change our particle thickess
    /// @param _thickness - particle thickness
    /// @param _simNo - which simulation we want to change the thickness in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setParticleThickness(float _thickness, int _simNo = 0){m_fluidShaders[_simNo]->setThickness(_thickness);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to change our bilateral filter blur falloff
    /// @param _falloff - bilateral blur falloff
    /// @param _simNo - which simulation we want to change the blur fall off in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setBlurFalloff(float _falloff, int _simNo = 0){m_fluidShaders[_simNo]->setBlurFalloff(_falloff);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to change our bilateral filter blur radius
    /// @param _radius - bilateral blur radius
    /// @param _simNo - which simulation we want to change the blur radius in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setBlurRadius(float _radius, int _simNo = 0){m_fluidShaders[_simNo]->setBlurRadius( _radius);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to change the blur scale of our bilateral filter blur radius
    /// @param _scale - desired blur scale
    /// @param _simNo - which simulation we want to change the blur scale in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setBlurScale(float _scale, int _simNo = 0){m_fluidShaders[_simNo]->setBlurScale( _scale);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to set the refraction ratio of our fluid
    /// @param _eta - desired refraction ratio
    /// @param _simNo - which simulation we want to change the rafraction ratio in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRafractionRatio(float _eta, int _simNo = 0){m_fluidShaders[_simNo]->setRefractionRatio(_eta);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to set the fresnal power of our fluid
    /// @param _power - desired fresnal power of our fluid
    /// @param _simNo - which simulation we want to change the fresnal power in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setFresnalPower(float _power,int _simNo = 0){m_fluidShaders[_simNo]->setFresnalPower(_power);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief play/pause toggle slot
    /// @param _simNo - which simulation we want to toggle play
    //----------------------------------------------------------------------------------------------------------------------
    inline void playToggle(int _simNo = 0){m_fluidSimProps[_simNo].m_update = !m_fluidSimProps[_simNo].m_update;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set volume of our fluid slot
    /// @param _mass - desired mass of particles
    /// @param _simNo - which simulation we want to change
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMass(float _mass, int _simNo = 0){m_SPHEngines[_simNo]->setMass(_mass);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set density of our fluid slot
    /// @param _den - desired rest density
    /// @param _simNo - which simulation we want to change
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDensity(float _den, int _simNo = 0){m_SPHEngines[_simNo]->setRestDensity(_den);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set viscoty coeficient of our fluid slot
    /// @param _visc - viscosity coeficient
    /// @param _simNo - which simulation we want to change
    //----------------------------------------------------------------------------------------------------------------------
    inline void setViscCoef(float _visc, int _simNo = 0){m_SPHEngines[_simNo]->setViscConst(_visc);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set tension coeficient of our fluid slot
    /// @param _t - tension coeficient
    /// @param _simNo - which simulation we want to change
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTensionCoef(float _t, int _simNo = 0){m_SPHEngines[_simNo]->setTensionConst(_t);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set gas constant of our fluid slot
    /// @param _gasConst - gas constant
    /// @param _simNo - which simulation we want to change
    //----------------------------------------------------------------------------------------------------------------------
    inline void setGasConst(double _gasConst, int _simNo = 0){m_SPHEngines[_simNo]->setKConst(_gasConst);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set smoothing length of our fluid slot
    /// @param _len - smoothing length
    /// @param _simNo - which simulation we want to change our smoothing length in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(float _len, int _simNo = 0){m_SPHEngines[_simNo]->setSmoothingLength(_len);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the color of our fluid slot
    /// @param _col - desired color to set the fluid to
    /// @param _simNo - which simulation we want to change the color in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setFluidColor(QColor _col, int _simNo = 0){m_fluidShaders[_simNo]->setColor(_col.redF(),_col.greenF(),_col.blueF());}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to set the time step of our simulation
    /// @param _timeStep - the timeStep of our simulation
    /// @param _simNo - which simulation we want to change the play back speed in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSimTimeStep(float _timeStep, int _simNo = 0){m_SPHEngines[_simNo]->setTimeStep( _timeStep);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to reset our simulation and remove all particles
    /// @brief _simNo - simulation number to reset
    //----------------------------------------------------------------------------------------------------------------------
    inline void resetSim(int _simNo){addParticlesToSim(m_SPHEngines[_simNo]->getNumParticles(),_simNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to set the spawn box position in our simulation
    /// @param _x - x position of spawn box location
    /// @param _y - y position of spawn box location
    /// @param _z - z position of spawn box location
    /// @param _simNo - simulation number to set the spawn box location
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSpawnBoxPosition(float _x, float _y, float _z,int _simNo = 0){m_fluidSimProps[_simNo].m_spawnMin = make_float3(_x,_y,_z);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a slot to set the spawn box size in our simulation
    /// @param _size - desired spawn box size
    /// @param _simNo - simulation number to set the spawn box size
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSpawnBoxSize(float _size,int _simNo = 0){m_fluidSimProps[_simNo].m_spawnDim = make_float3(_size,_size,_size);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to add particles to our simulation
    /// @param _numParticles - number of particles to add to simulation
    /// @param _simNo - simulation to add particles to
    //----------------------------------------------------------------------------------------------------------------------
    void addParticlesToSim(int _numParticles,int _simNo = 0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the velocity correction of our simulation
    /// @param _val - value of velocity correction
    /// @param _simNo - simulation to set velocity correction in
    //----------------------------------------------------------------------------------------------------------------------
    inline void setVelCorrection(float _val, int _simNo = 0){}//m_SPHEngines[_simNo]->setVelocityCorrection(_val);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to toggle display of the hud of our simulatio
    /// @param _display - bool to indicate if we want to display it or not
    /// @param _simNo - simulation we wish to display Hud
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDisplayHud(bool _display, int _simNo = 0){m_fluidSimProps[_simNo].m_displayHud = _display;}
    //----------------------------------------------------------------------------------------------------------------------

private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dummy VAO to for drawing our VAO
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_cubeVAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief structure to hold some update information about our fluid simulations
    //----------------------------------------------------------------------------------------------------------------------
    struct fluidSimProps{
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief vector of our simulation positions
        //----------------------------------------------------------------------------------------------------------------------
        float3 m_simPosition;
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief vector of our simulation size
        //----------------------------------------------------------------------------------------------------------------------
        float3 m_simSize;
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief vector for spawn box minimum
        //----------------------------------------------------------------------------------------------------------------------
        float3 m_spawnMin;
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief vector for our spawn box dimension
        //----------------------------------------------------------------------------------------------------------------------
        float3 m_spawnDim;
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief a bool to tell us if we need to update our simulation
        //----------------------------------------------------------------------------------------------------------------------
        bool m_update;
        //----------------------------------------------------------------------------------------------------------------------
        /// @brief bool to indicate if we want to display the HUD
        //----------------------------------------------------------------------------------------------------------------------
        bool m_displayHud;
        //----------------------------------------------------------------------------------------------------------------------
    };
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief update information abour our simulations
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<fluidSimProps> m_fluidSimProps;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our SPHEngine that manages our particles.
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<SPHSolverCUDA *> m_SPHEngines;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our fluid shader
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<FluidShader *> m_fluidShaders;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used for calculating framerate
    //----------------------------------------------------------------------------------------------------------------------
    QTime m_currentTime;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used for drawing text in openGL
    //----------------------------------------------------------------------------------------------------------------------
    Text *m_text;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our Camera
    //----------------------------------------------------------------------------------------------------------------------
    Camera *m_cam;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Model matrix
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_modelMatrix;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Mouse transforms
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_mouseGlobalTX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief model pos
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_modelPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Spin face x
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinXFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sping face y
    //----------------------------------------------------------------------------------------------------------------------
    float m_spinYFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief rotate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_rotate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief translate bool
    //----------------------------------------------------------------------------------------------------------------------
    bool m_translate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origXPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief
    //----------------------------------------------------------------------------------------------------------------------
    int m_origYPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief bool to indicate if we want to pan our camera
    //----------------------------------------------------------------------------------------------------------------------
    bool m_pan;
    //----------------------------------------------------------------------------------------------------------------------

};

#endif // OPENGLWIDGET_H
