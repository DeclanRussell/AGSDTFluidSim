//----------------------------------------------------------------------------------------------------------------------
/// @file fluidShaderFrag.glsl
/// @author Declan Russell
/// @date 14/03/15
/// @version 1.0
/// @namepsace GLSL
/// @class fluidShaderFrag
/// @brief Fragment shader to apply the final shading of our fluid. This calculates the fragment normals
/// @brief based on a depth pass texture. It also adds reflection and refraction based on a cube map.
//----------------------------------------------------------------------------------------------------------------------

#version 400

//----------------------------------------------------------------------------------------------------------------------
/// @brief texture for holding the depth pass
//----------------------------------------------------------------------------------------------------------------------
uniform sampler2D depthTex;
//----------------------------------------------------------------------------------------------------------------------
/// @brief texture for holding the thickness pass
//----------------------------------------------------------------------------------------------------------------------
uniform sampler2D thicknessTex;
//----------------------------------------------------------------------------------------------------------------------
/// @brief texture for holding the cube map
//----------------------------------------------------------------------------------------------------------------------
uniform samplerCube cubeMapTex;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the size of each texel in the x direction
//----------------------------------------------------------------------------------------------------------------------
uniform float texelSizeX;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the size of each texel in the y direction
//----------------------------------------------------------------------------------------------------------------------
uniform float texelSizeY;
//----------------------------------------------------------------------------------------------------------------------
/// @brief inverse projection matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 PInv;
//----------------------------------------------------------------------------------------------------------------------
/// @brief normal matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 normalMatrix;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our fresnal power
//----------------------------------------------------------------------------------------------------------------------
uniform float fresnalPower;
//----------------------------------------------------------------------------------------------------------------------
/// @brief refraction ratio
//----------------------------------------------------------------------------------------------------------------------
uniform float refractRatio;
//----------------------------------------------------------------------------------------------------------------------
/// @brief fresnal constant
//----------------------------------------------------------------------------------------------------------------------
uniform float fresnalConst;
//----------------------------------------------------------------------------------------------------------------------
/// @brief color of our fluid
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 color;
//----------------------------------------------------------------------------------------------------------------------
/// @brief texture coordinates of billboard
//----------------------------------------------------------------------------------------------------------------------
in vec2 VTexCoord;

//----------------------------------------------------------------------------------------------------------------------
/// @brief a structure to hold our light information
//----------------------------------------------------------------------------------------------------------------------
struct lightInfo{
   vec4 position;
   vec3 intensity;
};

//----------------------------------------------------------------------------------------------------------------------
/// @brief our scene light
//----------------------------------------------------------------------------------------------------------------------
uniform lightInfo light;
//----------------------------------------------------------------------------------------------------------------------
/// @breif phong shading diffuse
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 Kd;
//----------------------------------------------------------------------------------------------------------------------
/// @brief phong shading ambient
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 Ka;
//----------------------------------------------------------------------------------------------------------------------
/// @brief phong shading specular
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 Ks;
//----------------------------------------------------------------------------------------------------------------------
/// @brief phong shading shininess
//----------------------------------------------------------------------------------------------------------------------
uniform float shininess;

//----------------------------------------------------------------------------------------------------------------------
/// @brief calculates our phong shading
/// @param position of surface
/// @param normal of surface
//----------------------------------------------------------------------------------------------------------------------
vec3 ads(vec3 position, vec3 normal){
   vec3 n = normalize(normal);
   vec3 s = normalize(vec3(light.position) - position);
   vec3 v = normalize(vec3(-position));
   vec3 r = reflect(-s, n);
   vec3 h = normalize(v + s);
   return light.intensity * (Ka + Kd * max(dot(s,n),0.0)+ Ks * pow(max(dot(h, n), 0.0), shininess));
}

//----------------------------------------------------------------------------------------------------------------------
/// @brief output fragment
//----------------------------------------------------------------------------------------------------------------------
out vec4 FragColor;

//----------------------------------------------------------------------------------------------------------------------
/// @brief converts uv coords and depth to eye space coodinates
/// @param _uv - texture coordinate
/// @param _depth - depth at location
/// @return eye space coordinate
//----------------------------------------------------------------------------------------------------------------------
vec3 uvToEye(vec2 _uv, float _depth){
    vec3 screenCoord = (vec3(_uv,_depth) * vec3(2.0)) - vec3(1.0);

    //float d = -1 / (-0 + _depth * ((1000.0-1)/1000.0));
    //return vec3(normUV,_depth) * d;
    vec4 eyeSpace = PInv * vec4(screenCoord,1.0);
    eyeSpace.xyz / eyeSpace.w;
    return vec3(eyeSpace);

}

//----------------------------------------------------------------------------------------------------------------------
/// @brief our fragment shader main
//----------------------------------------------------------------------------------------------------------------------
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
    vec2 tempTextCoord = VTexCoord + vec2(texelSizeX, 0);
    vec3 ddx = uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x) - posEye;
    tempTextCoord = VTexCoord + vec2(-texelSizeX, 0);
    vec3 ddx2 = posEye - uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x);

    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2;
    }

    tempTextCoord = VTexCoord + vec2(0,texelSizeY);
    vec3 ddy = uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x) - posEye;
    tempTextCoord = VTexCoord + vec2(0,-texelSizeY);
    vec3 ddy2 = posEye - uvToEye(tempTextCoord,texture(depthTex,tempTextCoord).x);

    if (abs(ddy2.z) < abs(ddy.z)) {
        ddy = ddy2;
    }

    // calculate normal
    vec3 n1 = cross(ddx, ddy);
    vec3 n = normalize(n1);



    float thickness = texture(thicknessTex,VTexCoord).x;

    vec3 i = normalize(posEye);
    float fresnalRatio = fresnalConst + (1.0 - fresnalConst) * pow((1.0 - dot(-i, n)), fresnalPower);
    if(fresnalRatio>1)fresnalRatio= 1;
    if(fresnalRatio<0)fresnalRatio= 0;


    vec3 normN = vec3(normalMatrix * vec4(n,1.0));
    vec3 Refract = refract(i, normN,refractRatio);

    vec3 Reflect = reflect(i, normN);

    vec3 refractColor = vec3(texture(cubeMapTex, Refract));
    vec3 reflectColor = vec3(texture(cubeMapTex, Reflect));


    //our final colour
    //Phong shading to find out color due to our light source
    vec3 phong = ads(posEye,n)*color;
    //mix our refraction color with our phong due to our thick
    //our fluid is.
    //texture can be more than value of 1 so if it is lets clamp it to 1
    refractColor = mix(refractColor,phong,(thickness>1)?1:thickness);
    FragColor  = vec4(mix(refractColor, reflectColor, fresnalRatio),1.0);


    //Debug:
    //position shading
    //FragColor = vec4(posEye,1.0);

    //normals shading
    //FragColor = vec4(n,1.0);

    //raw input generally depth
    //FragColor = texture(depthTex, VTexCoord);
    //FragColor = texture(thicknessTex, VTexCoord);

}
