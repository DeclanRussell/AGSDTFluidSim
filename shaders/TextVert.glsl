#version 400
in vec2 inVert;
in vec2 inUV;
out vec2 vertUV;
uniform vec3 textColour;
uniform float scaleX;
uniform float scaleY;
uniform float xpos;
uniform float ypos;
uniform vec2 transform;
void main()
{
    vertUV=inUV;
    gl_Position=vec4( ( transform.x*(xpos+inVert.x)*scaleX)-1.0,(transform.y*(ypos+inVert.y)*scaleY)+1.0,0.0,1.0);
}
