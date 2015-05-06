//----------------------------------------------------------------------------------------------------------------------
/// @file AbstractRenderTarget.h
/// @class AbstractRenderTarget
/// @author Declan Russell
/// @date 28/04/2015
/// @version 1.0
/// @brief Abstract base class for OpenGL Objects
//----------------------------------------------------------------------------------------------------------------------

#ifndef FLUIDPROPDOCKWIDGET_H
#define FLUIDPROPDOCKWIDGET_H

#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include "OpenGLWidget.h"

class FluidPropDockWidget : public QDockWidget
{
    Q_OBJECT
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief default constructor.
    /// @param _fluidWidget - OpenGLWidget used for our fluid sim
    //----------------------------------------------------------------------------------------------------------------------
    explicit FluidPropDockWidget(OpenGLWidget *_fluidWidget,QWidget *parent = 0);
    //----------------------------------------------------------------------------------------------------------------------
public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to change our environment map
    //----------------------------------------------------------------------------------------------------------------------
    void importEnvMap();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the simulation postion
    //----------------------------------------------------------------------------------------------------------------------
    void setSimPosition();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to toggle play of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void togglePlay(){m_fluidGLWidget->playToggle(m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the point size of our simulation
    /// @param _size - desired size of particles
    //----------------------------------------------------------------------------------------------------------------------
    inline void setPointSize(double _size){m_fluidGLWidget->setParticleSize(_size,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the thickness or the particles in our simulation
    /// @param _thickness - desired thickness
    //----------------------------------------------------------------------------------------------------------------------
    inline void setThickness(double _thickness){m_fluidGLWidget->setParticleThickness(_thickness,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set our bilateral filter blur fall off
    /// @param _falloff - desired blur fall off
    //----------------------------------------------------------------------------------------------------------------------
    inline void setBlurFalloff(double _falloff){m_fluidGLWidget->setBlurFalloff(_falloff,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set our bilateral filter blur radius
    /// @param _radius - desired blur radius
    //----------------------------------------------------------------------------------------------------------------------
    inline void setBlurRadius(double _radius){m_fluidGLWidget->setBlurRadius(_radius,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the fresnal power of our fluid
    /// @param _power - desired fresnal power
    //----------------------------------------------------------------------------------------------------------------------
    inline void setFresnalPower(double _power){m_fluidGLWidget->setFresnalPower(_power,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the refraction ratio of our fluid
    /// @param _eta - desired refraction ratio
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRefractionRatio(double _eta){m_fluidGLWidget->setRafractionRatio(_eta,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the mass of our fluid
    /// @param _mass - desired mass of fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMass(double _mass){m_fluidGLWidget->setMass(_mass,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the rest density of our fluid
    /// @param _density - desired rest density of of fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDensity(double _density){m_fluidGLWidget->setDensity(_density,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the viscosity coeficient of our fluid
    /// @param _visc - desired viscosity of fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setViscosity(double _visc){m_fluidGLWidget->setViscCoef(_visc,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the gas constant of our fluid
    /// @param _gconst - desired gas constant of fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setGasConst(double _gconst){m_fluidGLWidget->setGasConst(_gconst,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the smoothing length of our fluid simulation
    /// @param _len - desired smoothing length of fluid simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(double _len){m_fluidGLWidget->setSmoothingLength(_len,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the color of our fluid
    /// @param _col - desired color of fluid
    //----------------------------------------------------------------------------------------------------------------------
    inline void setFluidColor(QColor _col){m_fluidGLWidget->setFluidColor(_col,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the play back speed of our simulation
    /// @param _speed - desired speed of simulation as a absolute percentage
    //----------------------------------------------------------------------------------------------------------------------
    inline void setPlaybackSpeed(int _speed){m_fluidGLWidget->setPlaybackSpeed(_speed/100.f,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to reset our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void resetSim(){m_fluidGLWidget->resetSim(m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the spawn box size
    /// @param _size - desired size of spawn box
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSpawnBoxSize(double _size){m_fluidGLWidget->setSpawnBoxSize(_size,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to add particles to our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void addPartToSim(){m_fluidGLWidget->addParticlesToSim(m_numPartToAddSpn->value(),m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the spawn box position
    //----------------------------------------------------------------------------------------------------------------------
    void setSpawnBoxPos();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the time step our simulation
    /// @param _step - desired time step of simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTimeStep(double _step){m_fluidGLWidget->setSimTimeStep(_step,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the velocity correction of our simulatin
    /// @brief _val - value of velocity correciion
    //----------------------------------------------------------------------------------------------------------------------
    inline void setVelCorrection(double _val){m_fluidGLWidget->setVelCorrection(_val,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to toggle the hud of our simulation
    /// @param _display - bool to indicate if we want to display our hud
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDisplayHud(bool _display){m_fluidGLWidget->setDisplayHud(_display,m_instanceNo);}
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a pointer to our fluid sim openGlWidget
    //----------------------------------------------------------------------------------------------------------------------
    OpenGLWidget *m_fluidGLWidget;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a static member to keep track of how many instances we have
    //----------------------------------------------------------------------------------------------------------------------
    static int m_instanceCount;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief instance number
    //----------------------------------------------------------------------------------------------------------------------
    int m_instanceNo;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sim postion x
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spinPosX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sim postion y
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spinPosY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief sim postion z
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spinPosZ;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief spawn box position x
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spawnPosX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief spawn box position y
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spawnPosY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief spawn box position z
    //----------------------------------------------------------------------------------------------------------------------
    QDoubleSpinBox *m_spawnPosZ;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief spinbox for the number of particles to add to the simulation
    //----------------------------------------------------------------------------------------------------------------------
    QSpinBox *m_numPartToAddSpn;
    //----------------------------------------------------------------------------------------------------------------------



};

#endif // FLUIDPROPDOCKWIDGET_H
