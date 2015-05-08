TARGET=FluidSim
OBJECTS_DIR=obj

#Enter your gencode here!
GENCODE = arch=compute_52,code=sm_52

# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
UI_HEADERS_DIR=ui
MOC_DIR=moc

CONFIG-=app_bundle
QT+=gui opengl core
SOURCES += \
    src/main.cpp \
    src/mainwindow.cpp \
    src/OpenGLWidget.cpp \
    src/SPHEngine.cpp \
    cudaSrc/*.cu \
    src/GLTexture.cpp \
    src/GLTextureLib.cpp \
    src/FrameBuffer.cpp \
    src/RenderBuffer.cpp \
    src/RenderTargetLib.cpp \
    src/FluidShader.cpp \
    src/FluidPropDockWidget.cpp

SOURCES -= cudaSrc/*.cu

HEADERS += \
    include/mainwindow.h \
    include/OpenGLWidget.h \
    include/ui_mainwindow.h \
    include/CudaSPHKernals.h \
    include/SPHEngine.h \
    include/GLTexture.h \
    include/GLTextureLib.h \
    include/FrameBuffer.h \
    include/RenderBuffer.h \
    include/RenderTargetLib.h \
    include/AbstractOpenGLObject.h \
    include/FluidShader.h \
    include/FluidPropDockWidget.h

FORMS += \
    ui/mainwindow.ui

OTHER_FILES += shaders/*glsl \
    shaders/fluidShaderFrag.glsl \
    shaders/fluidShaderVert.glsl \
    shaders/bilateralFilterFrag.glsl \
    shaders/bilateralFilterVert.glsl \
    shaders/thicknessFrag.glsl \
    shaders/thicknessVert.glsl \
    shaders/skyBoxFrag.glsl \
    shaders/skyBoxVert.glsl \
    mainpage.dox \
    shaders/cuboidVert.glsl \
    shaders/cuboidGeom.glsl \
    shaders/cuboidFrag.glsl


INCLUDEPATH +=./include /opt/local/include $$(HOME)/NGL/include/
LIBS += -L/opt/local/lib -lGLEW
DESTDIR=./

CONFIG += console
CONFIG -= app_bundle

# use this to suppress some warning from boost
QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter -Wno-reorder"
QMAKE_CXXFLAGS+= -msse -msse2 -msse3
macx:QMAKE_CXXFLAGS+= -arch x86_64
macx:INCLUDEPATH+=/usr/local/include/
# define the _DEBUG flag for the graphics lib
DEFINES +=NGL_DEBUG

unix:LIBS += -L/usr/local/lib
# add the ngl lib
unix:LIBS +=  -L/$(HOME)/NGL/lib -lNGL

# now if we are under unix and not on a Mac (i.e. linux) define GLEW
linux-*{
                linux-*:QMAKE_CXXFLAGS +=  -march=native
                linux-*:DEFINES+=GL42
                DEFINES += LINUX
}
DEPENDPATH+=include
# if we are on a mac define DARWIN
macx:DEFINES += DARWIN

#----------------------------------------------------------------
#-------------------------Cuda setup-----------------------------
#----------------------------------------------------------------

#set out cuda sources
CUDA_SOURCES += cudaSrc/*.cu

# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-6.5
# Path to cuda toolkit install
macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
linux:CUDA_SDK = /usr/local/cuda-6.5/samples

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/common/inc/
INCLUDEPATH += $$CUDA_DIR/../shared/inc/


#cuda libs
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20


#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o

## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr


# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

