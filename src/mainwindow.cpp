#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //add our openGL context to our scene
    m_openGLWidget = new OpenGLWidget(format,this);
    ui->gridLayout->addWidget(m_openGLWidget,0,0,1,2);

}

MainWindow::~MainWindow(){
    delete ui;
    delete m_openGLWidget;

}
