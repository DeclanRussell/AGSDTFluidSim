#version 400

//texture coordinates of billboard
in vec2 VTexCoord;

//texture for holding the depth pass
uniform sampler2D depthTex;
uniform float blurDepthFalloff;
uniform float filterRadius;
uniform float texelSize;

//output
out vec4 fragout;

float filter(vec2 blurDir, float depth){
    float sum = 0;
    float wsum = 0;
    float r,w,r2,g;
    for(float x=-filterRadius; x<=filterRadius; x+=texelSize) {
        float depSample = texture(depthTex, VTexCoord + x*blurDir).x;
        // spatial domain
        r = x * 2.0f;
        w = exp(-r*r);
        // range domain
        r2 = (depSample - depth) * blurDepthFalloff;
        g = exp(-r2*r2);
        sum += depSample * w * g;
        wsum += w * g;
    }
    if (wsum > 0.0f) {
        sum /= wsum;
    }
    return sum;
}


void main(void)
{
    float curDepth = texture(depthTex, VTexCoord).x;
    float alpha = texture(depthTex, VTexCoord).a;
    //if outside our viewing range discard fragment
    if (alpha==0) {
        discard;
        return;
    }

    float result = filter(vec2(1.0f,0.0f),curDepth);
    result += filter(vec2(0.0f,1.0f),curDepth);
    result *= 0.5f;

    fragout = vec4(result,result,result,1.0);

    //fragout = texture(depthTex, VTexCoord);

}
