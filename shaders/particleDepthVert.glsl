#version 400

layout (location = 0) in vec3 vertexPosition;

out vec3 position;

uniform mat4 P;
uniform mat4 MV;
uniform mat4 MVP;
uniform int screenWidth;
uniform float pointRadius;


void main(){

//    vec4 scrnCoord = MVP * vec4(vertexPosition,1.0);
//    float projRadius = pointRadius / (1.0 - scrnCoord.w);
//    gl_PointSize = projRadius * projRadius;
    vec4 eyePos = MV * vec4(vertexPosition,1.0);
    position = vec3(eyePos);
    vec4 projCorner = P * vec4(0.5*pointRadius, 0.5*pointRadius, eyePos.z, eyePos.w);

    gl_PointSize = screenWidth * projCorner.x / projCorner.w;
    //gl_PointSize = pointRadius * pointRadius;

    gl_Position = vec4(MVP * vec4(vertexPosition, 1.0));
}
