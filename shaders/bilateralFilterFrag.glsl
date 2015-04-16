//----------------------------------------------------------------------------------------------------------------------
/// @file bilateralFilterFrag.glsl
/// @author Declan Russell
/// @date 14/03/15
/// @version 2.0
/// @brief Fragment shader to apply bilateral filter blur on an input
//----------------------------------------------------------------------------------------------------------------------

#version 400

//----------------------------------------------------------------------------------------------------------------------
/// @brief in variable of the texture coordinates
//----------------------------------------------------------------------------------------------------------------------
in vec2 VTexCoord;
//----------------------------------------------------------------------------------------------------------------------
/// @brienf texture for holding the depth pass
//----------------------------------------------------------------------------------------------------------------------
uniform sampler2D depthTex;
//----------------------------------------------------------------------------------------------------------------------
/// @brief unifrom to set the blur falloff
//----------------------------------------------------------------------------------------------------------------------
uniform float blurDepthFalloff;
//----------------------------------------------------------------------------------------------------------------------
/// @brief uniform to set the blur radius
//----------------------------------------------------------------------------------------------------------------------
uniform float filterRadius;
//----------------------------------------------------------------------------------------------------------------------
/// @brief uniform for the size of our texel
//----------------------------------------------------------------------------------------------------------------------
uniform float texelSize;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our out our fragment
//----------------------------------------------------------------------------------------------------------------------
out vec4 fragout;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our bilateral filter function
/// @param blurDir - the direction of our blur
/// @param depth  - the depth of our current input fragment
//----------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------
/// @brief our main shader function
//----------------------------------------------------------------------------------------------------------------------
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
