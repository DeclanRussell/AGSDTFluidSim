#version 400

//texture coordinates of billboard
in vec2 VTexCoord;

//texture for holding the depth pass
uniform sampler2D depthTex;
uniform vec2 blurDir;
uniform float blurDepthFalloff;
uniform float filterRadius;
uniform float texelSize;

//output
out vec4 fragout;

void main(void)
{
    float depth = texture(depthTex, VTexCoord).x;
    float alpha = texture(depthTex, VTexCoord).a;
    //if outside our viewing range discard fragment
    if (alpha==0) {
        discard;
        return;
    }

    float sum = 0;
    float wsum = 0;
    float r,w,r2,g;
    for(float x=-filterRadius; x<=filterRadius; x+=texelSize) {
        float depSample = texture(depthTex, VTexCoord + x*blurDir).x;
        // spatial domain
        r = x * 2.0f;
        w = exp(-r*r);
        // range domain
        r2 = (depSample - depth) * blurDepthFalloff;;
        g = exp(-r2*r2);
        sum += depSample * w * g;
        wsum += w * g;
    }
    if (wsum > 0.0f) {
        sum /= wsum;
    }
    fragout = vec4(sum,sum,sum,1.0);

}
