#version 400

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexNormal;
layout (location = 2) in vec3 vertexColor;
//layout (location = 2) in vec2 texCoord;

out vec3 VPosition;
out vec3 VNormal;
out vec3 VColor;
//out vec2 TexCoords;

uniform mat4 MV;
uniform mat3 normalMatrix;
uniform mat4 MVP;


void main(){
   //TexCoords = texCoord;
   VColor = vertexColor;
   VNormal = normalize(normalMatrix * vertexNormal);
   VPosition = vec3(MV * vec4(vertexPosition,1.0));
   gl_Position = MVP * vec4(vertexPosition,1.0);
}
