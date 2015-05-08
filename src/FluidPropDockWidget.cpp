#include "FluidPropDockWidget.h"

#include <QScrollArea>
#include <QFileDialog>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QSpacerItem>
#include <QPushButton>
#include <QFileDialog>
#include <QColorDialog>
#include <QGroupBox>
#include <QSlider>
#include <QDesktopServices>
#include <QCheckBox>

//declare our static members
int FluidPropDockWidget::m_instanceCount;

FluidPropDockWidget::FluidPropDockWidget(OpenGLWidget *_fluidWidget, QWidget *parent) :
    QDockWidget(parent)
{
    //stop our widget being able to close, done want to loose it to unknown realms
    this->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable);
    this->setMinimumWidth(600);
    //install our widget
    m_fluidGLWidget = _fluidWidget;
    //tell our scene to add a new fluid simulation
    m_fluidGLWidget->addFluidSim();
    m_instanceNo = m_instanceCount;
    m_instanceCount++;
    //QObjects manage there children and will delete them when they are deleted
    //therefore no need to keep them all as members
    //Add a group box for our shader properties
    QScrollArea* scrollArea = new QScrollArea(this);
    this->setWidget(scrollArea);

    QGridLayout *dockGridLayout = new QGridLayout(this);
    scrollArea->setLayout(dockGridLayout);

    QGroupBox *shaderProperties = new QGroupBox("Shader Properties",this);
    dockGridLayout->addWidget(shaderProperties,0,1,1,1);
    QGridLayout *shadPropLayout = new QGridLayout(shaderProperties);
    shaderProperties->setLayout(shadPropLayout);

    //add a spacer to keep everything tidy
    QSpacerItem* shadPropSpcr = new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    dockGridLayout->addItem(shadPropSpcr,11,0,1,1);

    //add a field to edit particle size
    QLabel *partSizeLbl = new QLabel("Particle size:",shaderProperties);
    shadPropLayout->addWidget(partSizeLbl,0,0,1,1);
    QDoubleSpinBox *partSizeSpnBx = new QDoubleSpinBox(shaderProperties);
    partSizeSpnBx->setMinimum(0.0);
    partSizeSpnBx->setDecimals(3);
    partSizeSpnBx->setValue(0.2);
    partSizeSpnBx->setSingleStep(0.1);
    connect(partSizeSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setPointSize(double)));
    shadPropLayout->addWidget(partSizeSpnBx,0,1,1,1);

    //add a field to edit particle thickness
    QLabel *partThicknessLbl = new QLabel("Particle thickness:",shaderProperties);
    shadPropLayout->addWidget(partThicknessLbl,1,0,1,1);
    QDoubleSpinBox *partThicknessSpnBx = new QDoubleSpinBox(shaderProperties);
    partThicknessSpnBx->setMinimum(0.0);
    partThicknessSpnBx->setDecimals(6);
    partThicknessSpnBx->setValue(0.04);
    partThicknessSpnBx->setSingleStep(0.01);
    connect(partThicknessSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setThickness(double)));
    shadPropLayout->addWidget(partThicknessSpnBx,1,1,1,1);

    //add a field for our bilateral filter params
    QLabel *bilatFilLbl = new QLabel("Bilateral Filter Properties:",shaderProperties);
    shadPropLayout->addWidget(bilatFilLbl,2,0,1,1);

    //blur falloff
    QLabel *falloffLbl = new QLabel("Blur Falloff:",shaderProperties);
    shadPropLayout->addWidget(falloffLbl,3,0,1,1);
    QDoubleSpinBox *falloffSpnBx = new QDoubleSpinBox(shaderProperties);
    falloffSpnBx->setValue(10);
    falloffSpnBx->setMaximum(INFINITY);
    falloffSpnBx->setDecimals(3);
    falloffSpnBx->setSingleStep(0.1);
    connect(falloffSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setBlurFalloff(double)));
    shadPropLayout->addWidget(falloffSpnBx,3,1,1,1);

    //blur radius
    QLabel *blurRadLbl = new QLabel("Blur Radius:",shaderProperties);
    shadPropLayout->addWidget(blurRadLbl,4,0,1,1);
    QDoubleSpinBox *blurRadSpnBx = new QDoubleSpinBox(shaderProperties);
    blurRadSpnBx->setValue(7);
    blurRadSpnBx->setDecimals(2);
    blurRadSpnBx->setSingleStep(0.5);
    connect(blurRadSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setBlurRadius(double)));
    shadPropLayout->addWidget(blurRadSpnBx,4,1,1,1);

    //Fluid properties field
    QLabel *fresPropLbl = new QLabel("Fresnal Properties:",shaderProperties);
    shadPropLayout->addWidget(fresPropLbl,5,0,1,1);

    //Fresnal Power
    QLabel *fresPwLbl = new QLabel("Fresnal Power:",shaderProperties);
    shadPropLayout->addWidget(fresPwLbl,6,0,1,1);
    QDoubleSpinBox *fresPwSpnBx = new QDoubleSpinBox(shaderProperties);
    fresPwSpnBx->setValue(3);
    fresPwSpnBx->setDecimals(3);
    fresPwSpnBx->setSingleStep(0.1);
    connect(fresPwSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setFresnalPower(double)));
    shadPropLayout->addWidget(fresPwSpnBx,6,1,1,1);

    //Refraction Ratio
    QLabel *refRatLbl = new QLabel("Refraction Ratio:",shaderProperties);
    shadPropLayout->addWidget(refRatLbl,7,0,1,1);
    QDoubleSpinBox *refRatSpnBx = new QDoubleSpinBox(shaderProperties);
    refRatSpnBx->setValue(0.9);
    refRatSpnBx->setMaximum(1.0);
    refRatSpnBx->setMinimum(0.0);
    refRatSpnBx->setDecimals(3);
    refRatSpnBx->setSingleStep(0.01);
    connect(refRatSpnBx,SIGNAL(valueChanged(double)), this, SLOT(setRefractionRatio(double)));
    shadPropLayout->addWidget(refRatSpnBx,7,1,1,1);

    //color wheel
    QLabel *colorLbl = new QLabel("Set Fluid Color:",shaderProperties);
    shadPropLayout->addWidget(colorLbl,8,0,1,1);
    QColorDialog *colorWheel = new QColorDialog(QColor(0,255,255),shaderProperties);
    connect(colorWheel,SIGNAL(currentColorChanged(QColor)),this,SLOT(setFluidColor(QColor)));
    QPushButton *colorBtn = new QPushButton("Select Fluid Color",shaderProperties);
    connect(colorBtn,SIGNAL(clicked(bool)),colorWheel,SLOT(setHidden(bool)));
    shadPropLayout->addWidget(colorBtn,8,1,1,1);

    //Environment map button
    QLabel *envLbl = new QLabel("Set Environment Map:",shaderProperties);
    shadPropLayout->addWidget(envLbl,9,0,1,1);
    QPushButton *envBtn = new QPushButton("Import Environment Map",shaderProperties);
    connect(envBtn,SIGNAL(pressed()),this,SLOT(importEnvMap()));
    shadPropLayout->addWidget(envBtn,9,1,1,1);

    //Group box for our fluid simulation properties
    QGroupBox *fluidSimProp = new QGroupBox("Fluid Simulation Properties",this);
    dockGridLayout->addWidget(fluidSimProp,1,1,1,1);
    QGridLayout *fluidSimLayout = new QGridLayout(fluidSimProp);
    fluidSimProp->setLayout(fluidSimLayout);

    //set mass field
    QLabel *massLbl = new QLabel("Particle Mass:",fluidSimProp);
    fluidSimLayout->addWidget(massLbl,0,0,1,1);
    QDoubleSpinBox *massSpnBx = new QDoubleSpinBox(fluidSimProp);
    massSpnBx->setValue(10);
    massSpnBx->setMinimum(0);
    massSpnBx->setMaximum(INFINITY);
    massSpnBx->setSingleStep(0.01);
    massSpnBx->setDecimals(3);
    connect(massSpnBx,SIGNAL(valueChanged(double)),this,SLOT(setMass(double)));
    fluidSimLayout->addWidget(massSpnBx,0,1,1,1);

    //set density field
    QLabel *denLbl = new QLabel("Density:",fluidSimProp);
    fluidSimLayout->addWidget(denLbl,1,0,1,1);
    QDoubleSpinBox *denSpnBx = new QDoubleSpinBox(fluidSimProp);
    denSpnBx->setDecimals(4);
    denSpnBx->setMaximum(INFINITY);
    denSpnBx->setValue(998.2);
    denSpnBx->setMinimum(0);
    denSpnBx->setSingleStep(0.01);
    denSpnBx->setDecimals(3);
    connect(denSpnBx,SIGNAL(valueChanged(double)),this,SLOT(setDensity(double)));
    fluidSimLayout->addWidget(denSpnBx,1,1,1,1);

    //set viscosity coeficient field
    QLabel *viscLbl = new QLabel("Viscosity Coeficient:",fluidSimProp);
    fluidSimLayout->addWidget(viscLbl,2,0,1,1);
    QDoubleSpinBox *viscSpnBx = new QDoubleSpinBox(fluidSimProp);
    viscSpnBx->setValue(0.3f);
    viscSpnBx->setMinimum(0);
    viscSpnBx->setSingleStep(0.01);
    viscSpnBx->setDecimals(3);
    connect(viscSpnBx,SIGNAL(valueChanged(double)),this,SLOT(setViscosity(double)));
    fluidSimLayout->addWidget(viscSpnBx,2,1,1,1);

    //set gas constant field
    QLabel *gasConLbl = new QLabel("Gas Constant:",fluidSimProp);
    fluidSimLayout->addWidget(gasConLbl,3,0,1,1);
    QDoubleSpinBox *gasConSpnBx = new QDoubleSpinBox(fluidSimProp);
    gasConSpnBx->setValue(10.f);
    gasConSpnBx->setMinimum(0);
    gasConSpnBx->setMaximum(INFINITY);
    gasConSpnBx->setSingleStep(0.1);
    gasConSpnBx->setDecimals(3);
    connect(gasConSpnBx,SIGNAL(valueChanged(double)),this,SLOT(setGasConst(double)));
    fluidSimLayout->addWidget(gasConSpnBx,3,1,1,1);

    //set smoothing length field
    QLabel *smoothLenLbl = new QLabel("Smoothing Length:",fluidSimProp);
    fluidSimLayout->addWidget(smoothLenLbl,4,0,1,1);
    QDoubleSpinBox *smoothLenSpnBx = new QDoubleSpinBox(fluidSimProp);
    smoothLenSpnBx->setValue(0.3f);
    smoothLenSpnBx->setMinimum(0);
    smoothLenSpnBx->setSingleStep(0.01);
    smoothLenSpnBx->setDecimals(3);
    connect(smoothLenSpnBx,SIGNAL(valueChanged(double)),this,SLOT(setSmoothingLength(double)));
    fluidSimLayout->addWidget(smoothLenSpnBx,4,1,1,1);

    //Spinbox's to edit simulation postion
    QLabel *simPosLbl = new QLabel("Simulation Position:",fluidSimProp);
    fluidSimLayout->addWidget(simPosLbl,5,0,1,1);
    m_spinPosX = new QDoubleSpinBox(fluidSimProp);
    m_spinPosX->setMaximum(INFINITY);
    m_spinPosX->setMinimum(-INFINITY);
    m_spinPosX->setValue(0);
    connect(m_spinPosX,SIGNAL(valueChanged(double)),this,SLOT(setSimPosition()));
    fluidSimLayout->addWidget(m_spinPosX,5,1,1,1);
    m_spinPosY = new QDoubleSpinBox(fluidSimProp);
    m_spinPosY->setMaximum(INFINITY);
    m_spinPosY->setMinimum(-INFINITY);
    m_spinPosY->setValue(0);
    connect(m_spinPosY,SIGNAL(valueChanged(double)),this,SLOT(setSimPosition()));
    fluidSimLayout->addWidget(m_spinPosY,5,2,1,1);
    m_spinPosZ = new QDoubleSpinBox(fluidSimProp);
    m_spinPosZ->setMaximum(INFINITY);
    m_spinPosZ->setMinimum(-INFINITY);
    m_spinPosZ->setValue(0);
    connect(m_spinPosZ,SIGNAL(valueChanged(double)),this,SLOT(setSimPosition()));
    fluidSimLayout->addWidget(m_spinPosZ,5,3,1,1);

    //our veolicty correction field
    QLabel *velCorLbl = new QLabel("XSPH Velocity Correction",fluidSimProp);
    fluidSimLayout->addWidget(velCorLbl,6,0,1,1);
    QDoubleSpinBox *velCorSpn = new QDoubleSpinBox(fluidSimProp);
    velCorSpn->setDecimals(3);
    velCorSpn->setSingleStep(0.05);
    velCorSpn->setMaximum(1);
    velCorSpn->setValue(0.3);
    connect(velCorSpn,SIGNAL(valueChanged(double)),this,SLOT(setVelCorrection(double)));
    fluidSimLayout->addWidget(velCorSpn,6,1,1,1);


    //Spinbox's to edit spawn box postion
    QLabel *spwnPosLbl = new QLabel("Spawn Box Position:",fluidSimProp);
    fluidSimLayout->addWidget(spwnPosLbl,7,0,1,1);
    m_spawnPosX = new QDoubleSpinBox(fluidSimProp);
    m_spawnPosX->setMaximum(INFINITY);
    m_spawnPosX->setMinimum(-INFINITY);
    m_spawnPosX->setValue(2);
    connect(m_spawnPosX,SIGNAL(valueChanged(double)),this,SLOT(setSpawnBoxPos()));
    fluidSimLayout->addWidget(m_spawnPosX,7,1,1,1);
    m_spawnPosY = new QDoubleSpinBox(fluidSimProp);
    m_spawnPosY->setMaximum(INFINITY);
    m_spawnPosY->setMinimum(-INFINITY);
    m_spawnPosY->setValue(0);
    connect(m_spawnPosY,SIGNAL(valueChanged(double)),this,SLOT(setSpawnBoxPos()));
    fluidSimLayout->addWidget(m_spawnPosY,7,2,1,1);
    m_spawnPosZ = new QDoubleSpinBox(fluidSimProp);
    m_spawnPosZ->setMaximum(INFINITY);
    m_spawnPosZ->setMinimum(-INFINITY);
    m_spawnPosZ->setValue(2);
    connect(m_spawnPosZ,SIGNAL(valueChanged(double)),this,SLOT(setSpawnBoxPos()));
    fluidSimLayout->addWidget(m_spawnPosZ,7,3,1,1);
    QLabel *spwnSizeLbl = new QLabel("Spawn Box Size:",fluidSimProp);
    fluidSimLayout->addWidget(spwnSizeLbl,7,4,1,1);
    QDoubleSpinBox *spwnSizeSpn = new QDoubleSpinBox(fluidSimProp);
    spwnSizeSpn->setMaximum(INFINITY);
    spwnSizeSpn->setMinimum(-INFINITY);
    spwnSizeSpn->setValue(6);
    connect(spwnSizeSpn,SIGNAL(valueChanged(double)),this,SLOT(setSpawnBoxSize(double)));
    fluidSimLayout->addWidget(spwnSizeSpn,7,5,1,1);

    //add particles field
    QPushButton *addParticlesBtn = new QPushButton("Add Particles",fluidSimProp);
    connect(addParticlesBtn,SIGNAL(pressed()),this,SLOT(addPartToSim()));
    fluidSimLayout->addWidget(addParticlesBtn,8,0,1,1);

    m_numPartToAddSpn = new QSpinBox(fluidSimProp);
    m_numPartToAddSpn->setMaximum(INFINITY);
    m_numPartToAddSpn->setValue(50000);
    fluidSimLayout->addWidget(m_numPartToAddSpn,8,1,1,1);

    //add a check box for showing the hud
    QLabel *tglHudLbl = new QLabel("Toggle HUD:",fluidSimProp);
    fluidSimLayout->addWidget(tglHudLbl,9,0,1,1);
    QCheckBox *hudChkBox = new QCheckBox(fluidSimProp);
    hudChkBox->setChecked(false);
    connect(hudChkBox,SIGNAL(toggled(bool)),this,SLOT(setDisplayHud(bool)));
    fluidSimLayout->addWidget(hudChkBox,9,1,1,1);

    //Group box for our playback settings
    QGroupBox *playbackGrb = new QGroupBox("Playback:",this);
    dockGridLayout->addWidget(playbackGrb,2,1,1,1);
    QGridLayout *playBckLayout = new QGridLayout(playbackGrb);
    playbackGrb->setLayout(playBckLayout);

    //Play/pause button
    QPushButton *playPauseBtn = new QPushButton("Play/Pause",playbackGrb);
    connect(playPauseBtn,SIGNAL(pressed()),this,SLOT(togglePlay()));
    playBckLayout->addWidget(playPauseBtn,0,0,1,1);

    //time step spin box
    QDoubleSpinBox *timeStepSpn = new QDoubleSpinBox(playbackGrb);
    timeStepSpn->setDecimals(5);
    timeStepSpn->setValue(0.004);
    timeStepSpn->setSingleStep(0.001);
    connect(timeStepSpn,SIGNAL(valueChanged(double)),this,SLOT(setTimeStep(double)));
    playBckLayout->addWidget(timeStepSpn,0,1,1,1);

    //slider for play back speed
    QLabel *playSpeedLbl = new QLabel("Play back speed:",playbackGrb);
    playBckLayout->addWidget(playSpeedLbl,1,0,1,1);
    QSlider *playSpeedSld = new QSlider(Qt::Horizontal,playbackGrb);
    playSpeedSld->setMinimum(0);
    playSpeedSld->setMaximum(200);
    playSpeedSld->setValue(100);
    connect(playSpeedSld,SIGNAL(valueChanged(int)),this,SLOT(setPlaybackSpeed(int)));
    playBckLayout->addWidget(playSpeedSld,1,1,1,1);
    QSpinBox *playSpeedSpn = new QSpinBox(playbackGrb);
    playSpeedSpn->setMaximum(200);
    playSpeedSpn->setMinimum(0);
    playSpeedSpn->setValue(100);
    playSpeedSpn->setEnabled(false);
    connect(playSpeedSld,SIGNAL(sliderMoved(int)),playSpeedSpn,SLOT(setValue(int)));
    playBckLayout->addWidget(playSpeedSpn,1,2,1,1);

    //Reset button
    QPushButton *resetButton = new QPushButton("Reset Simulation",playbackGrb);
    connect(resetButton,SIGNAL(pressed()),this,SLOT(resetSim()));
    playBckLayout->addWidget(resetButton,2,0,1,1);

}
//----------------------------------------------------------------------------------------------------------------------
void FluidPropDockWidget::importEnvMap(){
    //let the user select a environment map to load in
    QString location = QFileDialog::getOpenFileName(0,QString("Import Environment Map"), QString("textures/"));
    //if nothing selected then we dont want to do anything
    if(location.isEmpty()) return;

    m_fluidGLWidget->loadCubeMap(location);
}
//----------------------------------------------------------------------------------------------------------------------
void FluidPropDockWidget::setSimPosition(){
    m_fluidGLWidget->setSimPosition(m_spinPosX->value(),m_spinPosY->value(),m_spinPosZ->value(),m_instanceNo);
}
//----------------------------------------------------------------------------------------------------------------------
void FluidPropDockWidget::setSpawnBoxPos(){
    m_fluidGLWidget->setSpawnBoxPosition(m_spawnPosX->value(),m_spawnPosY->value(),m_spawnPosZ->value(),m_instanceNo);
}
//----------------------------------------------------------------------------------------------------------------------
