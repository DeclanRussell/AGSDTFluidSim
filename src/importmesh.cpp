#include "importmesh.h"
#include <iostream>
#include <ngl/ShaderLib.h>
#include <ngl/Transformation.h>
#include <ngl/VAOPrimitives.h>


ImportMesh::ImportMesh()
{
}
//----------------------------------------------------------------------------------------------------------------------
ImportMesh::~ImportMesh(){
}

//----------------------------------------------------------------------------------------------------------------------
ImportMesh::ImportMesh(std::string _loc){

    OpenMesh::IO::Options ropt;
    if ( ! OpenMesh::IO::read_mesh(m_mesh, _loc,ropt))
    {
      std::cerr << "Error loading mesh from file " << _loc << std::endl;
    }
    //check to see if Open mesh has behaved and imported useful things
    bool hasNormals = ropt.check(OpenMesh::IO::Options::VertexNormal);
    bool hasTexCoords = ropt.check(OpenMesh::IO::Options::VertexTexCoord);
    bool hasColors = ropt.check(OpenMesh::IO::Options::VertexColor);
    //if not let dynamically add it, not much we can do about texture coords sadly :(
    //need colors though as we are going to use them to store our weights!
    if(!hasNormals) m_mesh.request_vertex_normals();
    if(!hasColors) m_mesh.request_vertex_colors();
    std::cout << "imported VertexNormal" << ( hasNormals ? ": yes\n":": no\n");
    std::cout << "provides VertexTexCoord" << ( hasTexCoords ? ": yes\n":": no\n");

    // mesh structure
    (m_mesh.is_trimesh()) ? std::cout<<"It's triangulated wooo!"<<std::endl :std::cout<<"Triangluate this fool!"<<std::endl;
    // mesh stats
    std::cout << "# Vertices: " << m_mesh.n_vertices() << std::endl;
    std::cout << "# Edges   : " << m_mesh.n_edges() << std::endl;
    std::cout << "# Faces   : " << m_mesh.n_faces() << std::endl;
    // mesh capabilities
    std::cout << "Mesh supports\n";
    std::cout<<"vertex normals: ";(m_mesh.has_vertex_normals()) ? std::cout<<"true"<<std::endl : std::cout<<"false"<<std::endl;
    std::cout<<"vertex colors: ";(m_mesh.has_vertex_colors()) ? std::cout<<"true"<<std::endl : std::cout<<"false"<<std::endl;
    std::cout<<"vertex texcoords: ";(m_mesh.has_vertex_texcoords2D()) ? std::cout<<"true"<<std::endl : std::cout<<"false"<<std::endl;
    std::cout<<"face normals: ";(m_mesh.has_face_normals()) ? std::cout<<"true"<<std::endl : std::cout<<"false"<<std::endl;
    std::cout<<"face colors: ";(m_mesh.has_face_colors()) ? std::cout<<"true"<<std::endl : std::cout<<"false"<<std::endl;

    // put our verticies into our VBO
    MyMesh::Point currentPoint;
    MyMesh::Normal currentNormal;
    MyMesh::Color currentColour(1.0,1.0,0.0);
    std::vector<ngl::Vec3> normals;
    std::vector<ngl::Vec3> colors;
    std::vector<ngl::Vec3> positions;
    ngl::Vec3 e1,e2,fn;
    //iterate over all faces
    for(MyMesh::FaceIter f_it=m_mesh.faces_begin(); f_it!=m_mesh.faces_end(); ++f_it){
        // iterate over every vertex in the face
        for(MyMesh::FaceVertexIter fv_it=m_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it){
            currentPoint = m_mesh.point(*fv_it);
            positions.push_back(ngl::Vec3(currentPoint[0],currentPoint[1],currentPoint[2]));
            //initialize all our colors aswell
            m_mesh.set_color(*fv_it,currentColour);
            colors.push_back(ngl::Vec3(currentColour[0],currentColour[1],currentColour[2]));
            //if open mesh is being nice and it has normals lets push them back
            if(hasNormals){
                currentNormal = m_mesh.normal(*fv_it);
                normals.push_back(ngl::Vec3(currentNormal[0],currentNormal[1],currentNormal[2]));
            }

        }
        // If open mesh is not being nice and importing normals lets give it some face normals
        if(!hasNormals){
            e1 = positions[positions.size()-2] - positions[positions.size()-1];
            e2 = positions[positions.size()-3] - positions[positions.size()-1];
            fn.cross(e2,e1);
            normals.push_back(fn);
            normals.push_back(fn);
            normals.push_back(fn);
            currentNormal[0] = (float)fn.m_x;
            currentNormal[1] = (float)fn.m_y;
            currentNormal[2] = (float)fn.m_z;
            // stick this information into our half edge structure aswell
            for(MyMesh::FaceVertexIter fv_it=m_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it){
                m_mesh.set_normal(*fv_it,currentNormal);
            }
        }
    }
    //push the data into our vbo
    std::vector<ngl::Vec3> meshVBO;
    for(unsigned int i=0; i<positions.size();i++){
        meshVBO.push_back(positions[i]);
        meshVBO.push_back(normals[i]);
        meshVBO.push_back(colors[i]);
    }
    //add our data to our VAO
    m_VAO = ngl::VertexArrayObject::createVOA(GL_TRIANGLES);
    m_VAO->bind();
    m_VAO->setData(meshVBO.size()*sizeof(ngl::Vec3), meshVBO[0].m_x);

    //set up our attribute pointer
    m_VAO->setVertexAttributePointer(0,3,GL_FLOAT,sizeof(ngl::Vec3)*3,0);
    m_VAO->setVertexAttributePointer(1,3,GL_FLOAT,sizeof(ngl::Vec3)*3,3);
    m_VAO->setVertexAttributePointer(2,3,GL_FLOAT,sizeof(ngl::Vec3)*3,6);

    //set our indices
    m_VAO->setNumIndices(positions.size());
    //set our VAO free into the wild
    m_VAO->unbind();

    // now lets fill up our array of actual verticies for use with our selectables
    for(MyMesh::VertexIter v_it=m_mesh.vertices_begin(); v_it!=m_mesh.vertices_end();++v_it){
        currentPoint = m_mesh.point(*v_it);
        m_vertPositions.push_back(ngl::Vec3(currentPoint[0],currentPoint[1],currentPoint[2]));
    }
}
//----------------------------------------------------------------------------------------------------------------------
void ImportMesh::update(){
    // Update our VAO
    m_VAO->bind();
    // get a pointer to our buffer
    ngl::Real* ptr = m_VAO->getDataPointer(0,GL_WRITE_ONLY);
    MyMesh::Point currentPoint;
    MyMesh::Normal currentNormal;
    MyMesh::Color currentColor;
    int step = 0;
    // iterate through every face
    for(MyMesh::FaceIter f_it=m_mesh.faces_begin(); f_it!=m_mesh.faces_end(); ++f_it){
        // iterate over every vertex in the face
        for(MyMesh::FaceVertexIter fv_it=m_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it){
            //get our updated values
            currentPoint = m_mesh.point(*fv_it);
            currentNormal = m_mesh.normal(*fv_it);
            currentColor = m_mesh.color(*fv_it);
            //write to our buffer
            ptr[step] = currentPoint[0];
            ptr[step+1] = currentPoint[1];
            ptr[step+2] = currentPoint[2];
            ptr[step+3] = currentNormal[0];
            ptr[step+4] = currentNormal[1];
            ptr[step+5] = currentNormal[2];
            ptr[step+6] = currentColor[0];
            ptr[step+7] = currentColor[1];
            ptr[step+8] = currentColor[2];
            step+=9;
        }
    }
    // release our pointer and VAO free into the wild
    m_VAO->freeDataPointer();
    m_VAO->bind();
}
//----------------------------------------------------------------------------------------------------------------------
void ImportMesh::draw(ngl::Mat4 _mouseGlobalTX, ngl::Camera *_cam){
    //make our shader active
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    (*shader)["DeformationShader"]->use();

    loadMatricesToShader(_mouseGlobalTX,_cam);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    m_VAO->bind();
    m_VAO->draw();
    m_VAO->unbind();
}
//----------------------------------------------------------------------------------------------------------------------
void ImportMesh::loadMatricesToShader(ngl::Mat4 _mouseGlobalTX, ngl::Camera *_cam){

    // Calculate MVP matricies
    ngl::Mat4 P = _cam->getProjectionMatrix();
    ngl::Mat4 MV = _mouseGlobalTX * _cam->getViewMatrix();

    ngl::Mat3 normalMatrix = ngl::Mat3(MV);
    normalMatrix.inverse();
    normalMatrix.transpose();

    ngl::Mat4 MVP = MV * P;

    //set our uniforms
    ngl::ShaderLib *shader=ngl::ShaderLib::instance();
    (*shader)["DeformationShader"]->use();
    shader->setUniform("normalMatrix",normalMatrix);
    shader->setUniform("MV",MV);
    shader->setUniform("MVP",MVP);
}

//----------------------------------------------------------------------------------------------------------------------
void ImportMesh::tetrahedralizeMesh(){
//    tetgenio in,out;
//    tetgenio::facet *f;
//    tetgenio::polygon *p;

//    // All indices start from 1.
//    in.firstnumber = 1;

//    //declare how many points we have
//    in.numberofpoints = m_mesh->mNumVertices;
//    //create our point list
//    in.pointlist = new REAL[in.numberofpoints * 3];

//    //add all our vertex positions
//    for(unsigned int i=0; i<m_mesh->mNumVertices;i+=3){
//        in.pointlist[i] = m_mesh->mVertices[i].x * 10.0;
//        in.pointlist[i+1] = m_mesh->mVertices[i].y * 10.0;
//        in.pointlist[i+2] = m_mesh->mVertices[i].z * 10.0;
//    }

//    //declare how many faces we have
//    in.numberoffacets = m_mesh->mNumFaces;
//    in.facetlist = new tetgenio::facet[in.numberoffacets];
//    in.facetmarkerlist = new int[in.numberoffacets];

//    //declare all of our faces
//    int sum=0;
//    for(unsigned int i=0; i<m_mesh->mNumFaces;i++){
//        f = &in.facetlist[i];
//        f->numberofpolygons = 1;
//        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
//        //let pressume that there are not any holes
//        f->numberofholes = 0;
//        f->holelist = NULL;
//        p = &f->polygonlist[0];
//        //lets now add our face indicies
//        p->numberofvertices = m_mesh->mFaces[i].mNumIndices;
//        sum+=p->numberofvertices;
//        p->vertexlist = new int[p->numberofvertices];
//        for(int j=0;j<p->numberofvertices;j++){
//            p->vertexlist[j] = m_mesh->mFaces[i].mIndices[j];

////            std::cerr<<m_mesh->mFaces[i].mIndices[j]<<std::endl;
//        }

//    }
//    std::cerr<<"num verticies = "<<m_mesh->mNumVertices<<" sum indicies = "<<sum<<" num faces * 3 = "<<m_mesh->mNumFaces*3<<std::endl;

//    //set face marker list, not too clear what this is atm
//    for(int i=0;i<m_mesh->mNumFaces;i++){
//        in.facetmarkerlist[i] = 0;
//    }

//    // Output the PLC to files ’mesh.node’ and ’mesh.poly’.
//    in.save_nodes("meshin");
//    in.save_poly("meshin");

//    // Tetrahedralize the PLC. Switches are chosen to read a PLC (p),
//    // do quality mesh generation (q) with a specified quality bound
//    // (1.414), and apply a maximum volume constraint (a0.1).

//    tetrahedralize("pq1.414a0.1-M", &in, &out);

//    // Output mesh to files ’meshout.node’, ’meshout.ele’ and ’meshout.face’.
//    out.save_nodes("meshout");
//    out.save_elements("meshout");
//    out.save_faces("meshout");


}
//----------------------------------------------------------------------------------------------------------------------
