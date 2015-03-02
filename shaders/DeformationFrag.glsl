#version 400

in vec3 GPosition;
smooth in vec3 GNormal;
smooth in vec3 GColor;
noperspective in vec3 GEdgeDistance;
//in vec2 TexCoords;

struct lightInfo{
   vec3 position;
   vec3 intensity;
};

uniform lightInfo light;

uniform vec3 Kd;
uniform vec3 Ka;
uniform vec3 Ks;
uniform float shininess;
//uniform sampler2D tex;

out vec4 fragColour;

vec3 ads(){
   vec3 n = normalize(GNormal);
   vec3 s = normalize(light.position - GPosition);
   vec3 v = normalize(-GPosition);
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

void  main(){
    vec3 color = ads()*GColor;

    //find smallest distance
    float d = min(GEdgeDistance.x,GEdgeDistance.y);
    d = min(d, GEdgeDistance.z);

    //determin mix factor with line
    float lineWidth = 0.002;
    float mixVal = smoothstep(lineWidth-(lineWidth/2), lineWidth+(lineWidth/2), d);

    //mix with our line colour in this case black
   fragColour = mix(vec4(0,0.7,0,1.0), vec4(color,1.0),mixVal);// * texture(tex, TexCoords);
//    fragColour = vec4(color,1.0);
}
