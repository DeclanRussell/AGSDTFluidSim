#version 400

layout (location = 0) in vec3 vertexPosition;
uniform mat4 MVP;
out vec3 texcoords;

void main () {
  texcoords = vertexPosition;
  gl_Position = MVP * vec4 (vertexPosition, 1.0);

}
