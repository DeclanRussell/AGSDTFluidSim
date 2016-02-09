#version 400
uniform sampler2D tex;
in vec2 vertUV;
out vec4 fragColour;
uniform vec3 textColour;

void main()
{
    vec4 text=texture(tex,vertUV.st);
    fragColour.rgb=textColour.rgb;
    fragColour.a=text.a;
}
