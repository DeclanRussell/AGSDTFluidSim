#version 400

layout (location = 0) in vec3 vertexPosition;

out vec3 position;

uniform mat4 P;
uniform mat4 MV;
uniform mat4 MVP;
uniform int screenWidth;
uniform float pointSize;


void main(){


    //scale the point sprite based on our projection matrix
    vec4 eyePos = MV * vec4(vertexPosition,1.0);
    position = vec3(eyePos);
    vec4 projCorner = P * vec4(0.5*pointSize, 0.5*pointSize, eyePos.z, eyePos.w);
    gl_PointSize = screenWidth * projCorner.x / projCorner.w;

    gl_Position = vec4(MVP * vec4(vertexPosition, 1.0));
}
