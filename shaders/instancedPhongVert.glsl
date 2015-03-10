#version 400

layout (location = 0) in vec3 vertexPosition;


out vec3 position;

uniform mat4 MV;
uniform mat4 MVP;

void main(){
    position = vec3(MV * vec4(vertexPosition, 1.0) );
    gl_Position = vec4(MVP * vec4(vertexPosition, 1.0));
}
