#version 400

layout (location = 0) in vec2 vertexPosition;
layout (location = 1) in vec2 vertexTexCoord;

out vec2 VTexCoord;

void main(void)
{
    VTexCoord = vertexTexCoord;
    gl_Position = vec4(vertexPosition,0,1);
}


