#version 400

//eye space postion
in vec3 position;
//radius of our points
uniform float pointRadius;
//projection matrix
uniform mat4 P;

//scaler for our thickness
uniform float thicknessScaler;

out vec4 fragout;


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
