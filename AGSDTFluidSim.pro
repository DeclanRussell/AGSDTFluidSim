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
MOC_DIR=moc

CONFIG-=app_bundle
QT+=gui opengl core
VPATH += ./src
SOURCES += \
    src/main.cpp \
    src/mainwindow.cpp \
    src/OpenGLWidget.cpp \
    #src/SPHEngine.cpp \
    src/GLTexture.cpp \
    src/GLTextureLib.cpp \
    src/FrameBuffer.cpp \
    src/RenderBuffer.cpp \
    src/RenderTargetLib.cpp \
    src/FluidShader.cpp \
    src/FluidPropDockWidget.cpp \
    src/Camera.cpp \
    src/Text.cpp \
    src/ShaderLib.cpp \
    src/Shader.cpp \
    src/ShaderProgram.cpp \
    src/ShaderUtils.cpp \
    src/SPHSolverCUDA.cpp

HEADERS += \
    include/mainwindow.h \
    include/OpenGLWidget.h \
    #include/CudaSPHKernals.h \
    #include/SPHEngine.h \
    include/GLTexture.h \
    include/GLTextureLib.h \
    include/FrameBuffer.h \
    include/RenderBuffer.h \
    include/RenderTargetLib.h \
    include/AbstractOpenGLObject.h \
    include/FluidShader.h \
    include/FluidPropDockWidget.h \
    include/Camera.h \
    include/Text.h \
    include/ShaderLib.h \
    include/Shader.h \
    include/ShaderProgram.h \
    include/ShaderUtils.h \
    include/SPHSolverCUDAKernals.h \
    include/SPHSolverCUDA.h

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
    shaders/cuboidFrag.glsl \
    shaders/TextFrag.glsl \
    shaders/TextVert.gls


INCLUDEPATH +=./include
!win32:{
    INCLUDEPATH+= /opt/local/include
    LIBS += -L/opt/local/lib -lGLEW
}
DESTDIR=./

CONFIG += console

DEFINES += _USE_MATH_DEFINES
#in on mac define DARWIN
macx:DEFINES+=DARWIN
win32:{
    DEFINES+=WIN32
    DEFINES+=_WIN32
    DEFINES += GLEW_STATIC
    INCLUDEPATH+=C:/boost
    LIBS+= -lopengl32 -lglew32s
}
# basic compiler flags (not all appropriate for all platforms)
QMAKE_CXXFLAGS+= -msse -msse2 -msse3
# use this to suppress some warning from boost
unix*:QMAKE_CXXFLAGS_WARN_ON += "-Wno-unused-parameter"


#----------------------------------------------------------------
#-------------------------Cuda setup-----------------------------
#----------------------------------------------------------------

#Enter your gencode here!
GENCODE = arch=compute_52,code=sm_52

#We must define this as we get some confilcs in minwindef.h and helper_math.h
DEFINES += NOMINMAX

#set out cuda sources
CUDA_SOURCES = "$$PWD"/cudaSrc/SPHSolverCUDAKernals.cu

#This is to add our .cu files to our file browser in Qt
SOURCES+=cudaSrc/SPHSolverCUDAKernals.cu
SOURCES-=cudaSrc/SPHSolverCUDAKernals.cu

# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-6.5
win32:CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5"
# Path to cuda toolkit install
macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
linux:CUDA_SDK = /usr/local/cuda-6.5/samples
win32:CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5"

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
#To get some prewritten helper functions from NVIDIA
win32:INCLUDEPATH += $$CUDA_SDK\common\inc


#cuda libs
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\x64
linux|macx:QMAKE_LIBDIR += $$CUDA_SDK/common/lib
win32:QMAKE_LIBDIR +=$$CUDA_SDK\common\lib\x64
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20

#On windows we must define if we are in debug mode or not
CONFIG(debug, debug|release) {
#DEBUG
    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    win32:NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
}
else{
#Release UNTESTED!!!
    win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
    win32:NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
}

#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
#So in windows object files have to be named with the .obj suffix instead of just .o
#God I hate you windows!!
win32:cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
win32:cudaIntr.clean = cudaIntrObj/*.obj

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr


# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

