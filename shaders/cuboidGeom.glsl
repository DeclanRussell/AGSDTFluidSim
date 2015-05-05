//----------------------------------------------------------------------------------------------------------------------
/// @file cuboidGeom.glsl
/// @class cuboidGeom
/// @author Declan Russell
/// @date 2/05/15
/// @version 1.0
/// @namepsace GLSL
/// @brief Geometry shader for our cuboid shader. Turns one point into a cube drawn with GL_LINES
//----------------------------------------------------------------------------------------------------------------------
#version 400

//----------------------------------------------------------------------------------------------------------------------
/// @brief the input to our geometry shader
//----------------------------------------------------------------------------------------------------------------------
layout(points) in;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the output of our geometry shader
//----------------------------------------------------------------------------------------------------------------------
layout(line_strip, max_vertices = 18) out;
//----------------------------------------------------------------------------------------------------------------------
/// @brief minimum point of our cube
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 cubeMin;
//----------------------------------------------------------------------------------------------------------------------
/// @brief maximum point of our cube
//----------------------------------------------------------------------------------------------------------------------
uniform vec3 cubeMax;
//----------------------------------------------------------------------------------------------------------------------
/// @brief our MVP matrix
//----------------------------------------------------------------------------------------------------------------------
uniform mat4 MVP;
//----------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------
/// @brief our main function of our geomtry shader. creates a GL_LINEs cube from input uniforms
//----------------------------------------------------------------------------------------------------------------------
void main(){
       vec4 min4 = vec4(cubeMin,1.0);
       vec4 max4 = vec4(cubeMax,1.0);

       //back sqaure
       gl_Position = MVP*min4;
       EmitVertex();
       vec4 temp = min4;
       temp.x=max4.x;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.y=max4.y;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.x=min4.x;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.y=min4.y;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();

       //front square
       //back sqaure
       min4.z = cubeMax.z;
       gl_Position = MVP*min4;
       EmitVertex();
       temp = min4;
       temp.x=max4.x;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.y=max4.y;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.x=min4.x;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.y=min4.y;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();


       //join the 2 squares up
       temp = vec4(cubeMin,1.0);
       gl_Position = MVP*temp;
       EmitVertex();
       temp.z = cubeMax.z;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();
       temp.y = cubeMax.y;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.z = cubeMin.z;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();
       temp.x = cubeMax.x;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.z = cubeMax.z;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();
       temp.y = cubeMin.y;
       gl_Position = MVP*temp;
       EmitVertex();
       temp.z = cubeMin.z;
       gl_Position = MVP*temp;
       EmitVertex();
       EndPrimitive();
}
//----------------------------------------------------------------------------------------------------------------------
