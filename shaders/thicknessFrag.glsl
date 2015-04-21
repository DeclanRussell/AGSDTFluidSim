//----------------------------------------------------------------------------------------------------------------------
/// @file thicknessFrag.glsl
/// @author Declan Russell
/// @date 15/03/15
/// @version 1.0
/// @namepsace GLSL
/// @class thicknessFrag
/// @brief Fragment shader to output the thickness of our fluid. It is important for you to run this with
/// @brief additive blending and no depth test for it to work. The more items that are drawn on top of each
/// @brief other then the thicker the fluid.
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
/// @brief scaler for our thickness
//----------------------------------------------------------------------------------------------------------------------
uniform float thicknessScaler;


//----------------------------------------------------------------------------------------------------------------------
/// @brief the output fragment
//----------------------------------------------------------------------------------------------------------------------
out vec4 fragout;


//----------------------------------------------------------------------------------------------------------------------
/// @brief fragment main. Draws spheres from point sprites colored by our thickness scaler
//----------------------------------------------------------------------------------------------------------------------
void  main(){
    //normal of our fragment
    vec3 normal;
    // calculate eye-space sphere normal from texture coordinates
    normal.xy = (gl_PointCoord*vec2(2.0)) - vec2(1.0);
    normal.y*=-1.0;
    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) {
        discard; // kill pixels outside circle
        return;
    }

    fragout = vec4(thicknessScaler,thicknessScaler,thicknessScaler,1.0);
}
