#version 400

//texture for holding the depth pass
uniform sampler2D depthTex;
//the size of each texel
uniform float texelSizeX;
uniform float texelSizeY;
//inverse projection matrix'
uniform mat4 PInv;
//texture coordinates of billboard
in vec2 VTexCoord;

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

//to be used later with phong shading
vec3 ads(vec3 position, vec3 normal){
   vec3 n = normalize(normal);
   vec3 s = normalize(vec3(light.position) - position);
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

//output
out vec4 FragColor;

//converts uv coords and depth to eye space coodinates
vec3 uvToEye(vec2 _uv, float _depth){
    vec2 normUV = (_uv*2) - 1.0;

    //float d = -1 / (-0 + _depth * ((1000.0-1)/1000.0));
    //return vec3(normUV,_depth) * d;
    return vec3(PInv * vec4(normUV,_depth,1.0));

}


void main(void)
{
    float depth = texture(depthTex, VTexCoord).x;
    float alpha = texture(depthTex, VTexCoord).a;
    //if outside our viewing range discard fragment
    if (alpha==0) {
        discard;
        return;
    }

    vec3 posEye = vec3(uvToEye(VTexCoord,depth));

    // calculate differences
    vec2 tempTextCoord = VTexCoord + vec2(texelSizeX*3, 0);
    vec3 ddx = uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x) - posEye;
    tempTextCoord = VTexCoord + vec2(-texelSizeX*3, 0);
    vec3 ddx2 = posEye - uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x);

    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2;
    }

    tempTextCoord = VTexCoord + vec2(0,texelSizeY*3);
    vec3 ddy = uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x) - posEye;
    tempTextCoord = VTexCoord + vec2(0,-texelSizeY*3);
    vec3 ddy2 = posEye - uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x);

    if (abs(ddy2.z) < abs(ddy.z)) {
        ddy = ddy2;
    }

    // calculate normal
    vec3 n1 = cross(ddx, ddy);
    vec3 n = normalize(n1);


    //phong shading
    //FragColor = vec4(ads(posEye,n)*vec3(0,1,1),1.0);

    //normals shading
    FragColor = vec4(n,1.0);

    //raw input generally depth
    //FragColor = texture(depthTex, VTexCoord);
}