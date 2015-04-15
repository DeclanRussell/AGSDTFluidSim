#version 400

in vec3 texcoords;
uniform samplerCube cubeMapTex;
out vec4 fragcolour;

void main () {
  fragcolour = texture (cubeMapTex, texcoords);
  fragcolour.a = 0;
}
