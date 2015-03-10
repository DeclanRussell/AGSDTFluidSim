#version 400

in vec3 position;
vec3 normal;

struct lightInfo{
   vec4 position;
   vec3 intensity;
};

uniform lightInfo light;
uniform vec3 Kd;
uniform vec3 Ka;
uniform vec3 Ks;
uniform float shininess;
uniform vec4 color;

out vec4 fragColour;

vec3 ads(){
   vec3 n = normalize(normal);
   vec3 s = normalize(vec3(light.position) - position);
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

void  main(){
    // calculate eye-space sphere normal from texture coordinates
    normal.xy = gl_PointCoord*2.0-1.0;
    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) discard; // kill pixels outside circle
    normal.z = sqrt(1.0 - r2);
    fragColour = vec4(ads(),1.0) * color;//vec4(normal,1);
}
