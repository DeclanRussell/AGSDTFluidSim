//----------------------------------------------------------------------------------------------------------------------
/// @file particleDepthFrag.glsl
/// @author Declan Russell
/// @date 8/03/15
/// @version 1.0
/// @namepsace GLSL
/// @class particleDepthFrag
/// @brief Fragment shader for drawing point sprites as spheres. The output fragment will be the depth.
//----------------------------------------------------------------------------------------------------------------------

#version 400

//----------------------------------------------------------------------------------------------------------------------
/// @brief eye space postion from vertex shader
//----------------------------------------------------------------------------------------------------------------------
in vec3 position;
//----------------------------------------------------------------------------------------------------------------------
/// @brief radius of our points
//----------------------------------------------------------------------------------------------------------------------
uniform float pointRadius;
//----------------------------------------------------------------------------------------------------------------------
/// @brief projection matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 P;

//----------------------------------------------------------------------------------------------------------------------
/// @brief output fragment. This will be the depth of our particles
//----------------------------------------------------------------------------------------------------------------------
out vec4 fragout;

//----------------------------------------------------------------------------------------------------------------------
/// @brief fragment main. Draws point sprites as spheres, calculates and outputs there depth.
//----------------------------------------------------------------------------------------------------------------------
void  main(){
    //normal of our fragment
    vec3 normal;
    // calculate eye-space sphere normal from texture coordinates
    normal.xy = (gl_PointCoord*2.0f) - 1.0f;
    normal.y*=-1.0;
    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) discard; // kill pixels outside circle
    normal.z = -sqrt(r2) * 2;



    // calculate depth
    // point radius calculated from inverse projection * 0.5*pointSize
    vec4 pixelPos = vec4(position + (normal * pointRadius * 2), 1.0);
    pixelPos.z -= 1.5;
    vec4 clipSpacePos = P * pixelPos;
    vec3 depth = vec3(clipSpacePos.z / clipSpacePos.w);

    fragout = vec4(depth,1.0);
    //fragout = vec4(normal,1.0);

}
