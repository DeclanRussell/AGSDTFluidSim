#include <QApplication>
#include "mainwindow.h"

int main(int argc, char **argv)
{
    QApplication app(argc,argv);
    MainWindow w;
    w.setWindowTitle(QString("Arbiturary Fluid Sim"));
    w.show();
    app.exec();
}
