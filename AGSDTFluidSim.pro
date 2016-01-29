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


INCLUDEPATH +=./include
!win32:{
    INCLUDEPATH+= /opt/local/include
    LIBS += -L/opt/local/lib -lGLEW
}
DESTDIR=./

CONFIG += console

# note each command you add needs a ; as it will be run as a single line
# first check if we are shadow building or not easiest way is to check out against current
!equals(PWD, $${OUT_PWD}){
        copydata.commands = echo "creating destination dirs" ;
        # now make a dir
        copydata.commands += mkdir -p $$OUT_PWD/shaders ;
        copydata.commands += echo "copying files" ;
        # then copy the files
        copydata.commands += $(COPY_DIR) $$PWD/shaders/* $$OUT_PWD/shaders/ ;
        # now make sure the first target is built before copy
        first.depends = $(first) copydata
        export(first.depends)
        export(copydata.commands)
        # now add it as an extra target
        QMAKE_EXTRA_TARGETS += first copydata
}
NGLPATH=$$(NGLDIR)
isEmpty(NGLPATH){ # note brace must be here
        message("including $HOME/NGL")
        linux:include($(HOME)/NGL/UseNGL.pri)
        macx:include($(HOME)/NGL/UseNGL.pri)
        win32:include(C:/NGL/UseNGL.pri)
}
else{ # note brace must be here
        message("Using custom NGL location")
        include($(NGLDIR)/UseNGL.pri)
}

#----------------------------------------------------------------
#-------------------------Cuda setup-----------------------------
#----------------------------------------------------------------

#set out cuda sources
CUDA_SOURCES = "$$PWD"/cudaSrc/*.cu

message($$CUDA_SOURCES)

# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-6.5
win32:CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0"
# Path to cuda toolkit install
macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
linux:CUDA_SDK = /usr/local/cuda-6.5/samples
win32:CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.0"

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/


#cuda libs
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\Win32
linux|macx:QMAKE_LIBDIR += $$CUDA_SDK/common/lib
win32:QMAKE_LIBDIR +=$$CUDA_SDK/common/lib/x64
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v #-maxrregcount 20


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

