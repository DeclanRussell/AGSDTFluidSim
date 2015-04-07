TARGET=FluidSim
OBJECTS_DIR=obj

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
    src/SPHEngine.cpp


HEADERS += \
    include/mainwindow.h \
    include/OpenGLWidget.h \
    include/ui_mainwindow.h \
    include/CudaSPHKernals.h \
    include/SPHEngine.h \
    include/cutil_math.h

FORMS += \
    ui/mainwindow.ui

OTHER_FILES += shaders/*glsl \
    shaders/fluidShaderFrag.glsl \
    shaders/fluidShaderVert.glsl


INCLUDEPATH +=./include /opt/local/include $$(HOME)/NGL/include/
LIBS += -L/opt/local/lib -lGLEW
DESTDIR=./

CONFIG += console
CONFIG -= app_bundle

# use this to suppress some warning from boost
QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter"
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
LIBS += -lcudart

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_52,code=sm_52 -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

