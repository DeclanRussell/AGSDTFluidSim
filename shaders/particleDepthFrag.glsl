#version 400


//eye space postion
in vec3 position;
//radius of our points
uniform float pointRadius;
//projection matrix
uniform mat4 P;

out vec4 fragout;


void  main(){
    //normal of our fragment
    vec3 normal;
    // calculate eye-space sphere normal from texture coordinates
    normal.xy = (gl_PointCoord*2.0f) - 1.0f;
    normal.y*=-1.0;
    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) discard; // kill pixels outside circle
    normal.z = sqrt(1.0 - r2*0.2);


    // calculate depth
    // point radius calculated from inverse projection * 0.5*pointSize
    vec4 pixelPos = vec4(position + (normal * pointRadius), 1.0);
    vec4 clipSpacePos = P *pixelPos;
    vec3 depth = vec3(clipSpacePos.z / clipSpacePos.w);

    fragout = vec4(depth,1.0);
    //fragout = vec4(normal,1.0);

}
