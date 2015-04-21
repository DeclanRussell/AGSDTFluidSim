#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpacerItem>
#include <QPushButton>
#include <QFileDialog>
#include <QColorDialog>
#include <QDesktopServices>


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //do this so everything isnt so bunched up
    this->setMinimumHeight(600);

    //add our openGL context to our scene
    m_openGLWidget = new OpenGLWidget(format,this);
    ui->gridLayout->addWidget(m_openGLWidget,0,0,4,1);

    //QObjects manage there children and will delete them when they are deleted
    //therefore no need to keep them all as members
    //Add a group box for our shader properties
    QGroupBox *shaderProperties = new QGroupBox("Shader Properties",this);
    ui->gridLayout->addWidget(shaderProperties,0,1,1,1);
    QGridLayout *shadPropLayout = new QGridLayout(shaderProperties);
    shaderProperties->setLayout(shadPropLayout);

    //add a spacer to keep everything tidy
    QSpacerItem* shadPropSpcr = new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    shadPropLayout->addItem(shadPropSpcr,11,0,1,1);

    //add a field to edit particle size
    QLabel *partSizeLbl = new QLabel("Particle size:",shaderProperties);
    shadPropLayout->addWidget(partSizeLbl,0,0,1,1);
    QDoubleSpinBox *partSizeSpnBx = new QDoubleSpinBox(shaderProperties);
    partSizeSpnBx->setMinimum(0.0);
    partSizeSpnBx->setDecimals(3);
    partSizeSpnBx->setValue(0.02);
    partSizeSpnBx->setSingleStep(0.1);
    connect(partSizeSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setParticleSize(double)));
    shadPropLayout->addWidget(partSizeSpnBx,0,1,1,1);

    //add a field to edit particle thickness
    QLabel *partThicknessLbl = new QLabel("Particle thickness:",shaderProperties);
    shadPropLayout->addWidget(partThicknessLbl,1,0,1,1);
    QDoubleSpinBox *partThicknessSpnBx = new QDoubleSpinBox(shaderProperties);
    partThicknessSpnBx->setMinimum(0.0);
    partThicknessSpnBx->setDecimals(4);
    partThicknessSpnBx->setValue(0.02);
    partThicknessSpnBx->setSingleStep(0.01);
    connect(partThicknessSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setParticleThickness(double)));
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
    connect(falloffSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setBlurFalloff(double)));
    shadPropLayout->addWidget(falloffSpnBx,3,1,1,1);

    //blur radius
    QLabel *blurRadLbl = new QLabel("Blur Radius:",shaderProperties);
    shadPropLayout->addWidget(blurRadLbl,4,0,1,1);
    QDoubleSpinBox *blurRadSpnBx = new QDoubleSpinBox(shaderProperties);
    blurRadSpnBx->setValue(10);
    blurRadSpnBx->setDecimals(2);
    blurRadSpnBx->setSingleStep(0.5);
    connect(blurRadSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setBlurRadius(double)));
    shadPropLayout->addWidget(blurRadSpnBx,4,1,1,1);

    //Fluid properties field
    QLabel *fresPropLbl = new QLabel("Fresnal Properties:",shaderProperties);
    shadPropLayout->addWidget(fresPropLbl,5,0,1,1);

    //Fresnal Power
    QLabel *fresPwLbl = new QLabel("Fresnal Power:",shaderProperties);
    shadPropLayout->addWidget(fresPwLbl,6,0,1,1);
    QDoubleSpinBox *fresPwSpnBx = new QDoubleSpinBox(shaderProperties);
    fresPwSpnBx->setValue(1);
    fresPwSpnBx->setDecimals(3);
    fresPwSpnBx->setSingleStep(0.1);
    connect(fresPwSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setFresnalPower(double)));
    shadPropLayout->addWidget(fresPwSpnBx,6,1,1,1);

    //Refraction Ratio
    QLabel *refRatLbl = new QLabel("Refraction Ratio:",shaderProperties);
    shadPropLayout->addWidget(refRatLbl,7,0,1,1);
    QDoubleSpinBox *refRatSpnBx = new QDoubleSpinBox(shaderProperties);
    refRatSpnBx->setValue(0.2);
    refRatSpnBx->setMaximum(1.0);
    refRatSpnBx->setMinimum(0.0);
    refRatSpnBx->setDecimals(3);
    refRatSpnBx->setSingleStep(0.01);
    connect(refRatSpnBx,SIGNAL(valueChanged(double)), m_openGLWidget, SLOT(setRefractionRatio(double)));
    shadPropLayout->addWidget(refRatSpnBx,7,1,1,1);

    //color wheel
    QLabel *colorLbl = new QLabel("Set Fluid Color:",shaderProperties);
    shadPropLayout->addWidget(colorLbl,8,0,1,1);
    QColorDialog *colorWheel = new QColorDialog(QColor(0,255,255),shaderProperties);
    connect(colorWheel,SIGNAL(currentColorChanged(QColor)),m_openGLWidget,SLOT(setFluidColor(QColor)));
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
    ui->gridLayout->addWidget(fluidSimProp,1,1,1,1);
    QGridLayout *fluidSimLayout = new QGridLayout(fluidSimProp);
    fluidSimProp->setLayout(fluidSimLayout);

    //set volume field
    QLabel *volLbl = new QLabel("Volume:",fluidSimProp);
    fluidSimLayout->addWidget(volLbl,0,0,1,1);
    QDoubleSpinBox *volSpnBx = new QDoubleSpinBox(fluidSimProp);
    volSpnBx->setValue(2);
    volSpnBx->setMinimum(0);
    volSpnBx->setSingleStep(0.01);
    volSpnBx->setDecimals(3);
    connect(volSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setVolume(double)));
    fluidSimLayout->addWidget(volSpnBx,0,1,1,1);

    //set density field
    QLabel *denLbl = new QLabel("Density:",fluidSimProp);
    fluidSimLayout->addWidget(denLbl,1,0,1,1);
    QDoubleSpinBox *denSpnBx = new QDoubleSpinBox(fluidSimProp);
    denSpnBx->setValue(998.2);
    denSpnBx->setMinimum(0);
    denSpnBx->setSingleStep(0.01);
    denSpnBx->setDecimals(3);
    connect(denSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setDensity(double)));
    fluidSimLayout->addWidget(denSpnBx,1,1,1,1);

    //set viscosity coeficient field
    QLabel *viscLbl = new QLabel("Viscosity Coeficient:",fluidSimProp);
    fluidSimLayout->addWidget(viscLbl,2,0,1,1);
    QDoubleSpinBox *viscSpnBx = new QDoubleSpinBox(fluidSimProp);
    viscSpnBx->setValue(0.003f);
    viscSpnBx->setMinimum(0);
    viscSpnBx->setSingleStep(0.01);
    viscSpnBx->setDecimals(3);
    connect(viscSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setViscCoef(double)));
    fluidSimLayout->addWidget(viscSpnBx,2,1,1,1);

    //set gas constant field
    QLabel *gasConLbl = new QLabel("Gas Constant:",fluidSimProp);
    fluidSimLayout->addWidget(gasConLbl,3,0,1,1);
    QDoubleSpinBox *gasConSpnBx = new QDoubleSpinBox(fluidSimProp);
    gasConSpnBx->setValue(10.f);
    gasConSpnBx->setMinimum(0);
    gasConSpnBx->setSingleStep(0.1);
    gasConSpnBx->setDecimals(3);
    connect(gasConSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setGasConst(double)));
    fluidSimLayout->addWidget(gasConSpnBx,3,1,1,1);

    //set smoothing length field
    QLabel *smoothLenLbl = new QLabel("Smoothing Length:",fluidSimProp);
    fluidSimLayout->addWidget(smoothLenLbl,4,0,1,1);
    QDoubleSpinBox *smoothLenSpnBx = new QDoubleSpinBox(fluidSimProp);
    smoothLenSpnBx->setValue(1.2f);
    smoothLenSpnBx->setMinimum(0);
    smoothLenSpnBx->setSingleStep(0.01);
    smoothLenSpnBx->setDecimals(3);
    connect(smoothLenSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setSmoothingLength(double)));
    fluidSimLayout->addWidget(smoothLenSpnBx,4,1,1,1);


    //Group box for our playback settings
    QGroupBox *playbackGrb = new QGroupBox("Playback:",this);
    ui->gridLayout->addWidget(playbackGrb,2,1,1,1);
    QGridLayout *playBckLayout = new QGridLayout(playbackGrb);
    playbackGrb->setLayout(playBckLayout);

    //Play/pause button
    QPushButton *playPauseBtn = new QPushButton("Play/Pause",playbackGrb);
    connect(playPauseBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(playToggle()));
    playBckLayout->addWidget(playPauseBtn,0,0,1,1);

    //Group box for our documentation
    QGroupBox *docGrb = new QGroupBox("Documentation:",this);
    ui->gridLayout->addWidget(docGrb,3,1,1,1);
    QGridLayout *docLayout = new QGridLayout(docGrb);
    docGrb->setLayout(docLayout);

    //open Documation button
    QPushButton *openDocBtn = new QPushButton("Open Documentation",docGrb);
    connect(openDocBtn,SIGNAL(pressed()),this,SLOT(openDoc()));
    docLayout->addWidget(openDocBtn,0,0,1,1);
}

void MainWindow::importEnvMap(){
    //let the user select a environment map to load in
    QString location = QFileDialog::getOpenFileName(0,QString("Import Environment Map"), QString("textures/"));
    //if nothing selected then we dont want to do anything
    if(location.isEmpty()) return;

    m_openGLWidget->loadCubeMap(location);

}

void MainWindow::openDoc(){
    QDesktopServices::openUrl(QUrl(QDir::currentPath() + "/doc/html/index.html"));
}

MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;

}
