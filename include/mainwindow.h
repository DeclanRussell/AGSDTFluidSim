#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QGroupBox>
#include <QGridLayout>
#include <QSpacerItem>
#include "OpenGLWidget.h"
#include <iostream>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief open documentation slot
    //----------------------------------------------------------------------------------------------------------------------
    void openDoc();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief adds a fluid sim to our opengl scene
    //----------------------------------------------------------------------------------------------------------------------
    void addFluidSim();
    //----------------------------------------------------------------------------------------------------------------------

private:
    QGridLayout *m_gridLayout;
    OpenGLWidget *m_openGLWidget;





};

#endif // MAINWINDOW_H
