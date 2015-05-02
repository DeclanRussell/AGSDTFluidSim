#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QPushButton>
#include <QDesktopServices>
#include "FluidPropDockWidget.h"


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //do this so everything isnt so bunched up
    this->setMinimumHeight(600);

    //add our openGL context to our scene
    m_openGLWidget = new OpenGLWidget(format,this);
    m_openGLWidget->hide();
    ui->gridLayout->addWidget(m_openGLWidget,0,0,7,1);

    //Group box for our general UI buttons
    QGroupBox *docGrb = new QGroupBox("General:",this);
    ui->gridLayout->addWidget(docGrb,8,0,1,1);
    QGridLayout *docLayout = new QGridLayout(docGrb);
    docGrb->setLayout(docLayout);

    //button to add fluid simulations
    QPushButton *addFluidSimBtn = new QPushButton("Add Fluid Simulation",docGrb);
    connect(addFluidSimBtn,SIGNAL(clicked()),this,SLOT(addFluidSim()));
    docLayout->addWidget(addFluidSimBtn,0,0,1,1);

    //open Documation button
    QPushButton *openDocBtn = new QPushButton("Open Documentation",docGrb);
    connect(openDocBtn,SIGNAL(pressed()),this,SLOT(openDoc()));
    docLayout->addWidget(openDocBtn,1,0,1,1);

}



void MainWindow::openDoc(){
    QDesktopServices::openUrl(QUrl(QDir::currentPath() + "/doc/html/index.html"));
}

void MainWindow::addFluidSim(){
    m_openGLWidget->show();
    this->addDockWidget(Qt::RightDockWidgetArea, new FluidPropDockWidget(m_openGLWidget,this));
}

MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;

}
