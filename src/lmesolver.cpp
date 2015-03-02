#include "lmesolver.h"
#include <iostream>
#include <eigen3/Eigen/SparseCholesky>
#include <cmath>

LMESolver::LMESolver()
{
}
//----------------------------------------------------------------------------------------------------------------------
LMESolver::LMESolver(std::vector<ngl::Vec3> _points){
    m_OGVertex = _points;
    m_laplaceMatrix.resize(_points.size(),_points.size());
    m_delta.resize(_points.size(),3);
    createMatricies(_points);
}
//----------------------------------------------------------------------------------------------------------------------
LMESolver::LMESolver(MyMesh &_mesh){
    m_laplaceMatrix.resize(_mesh.n_vertices(),_mesh.n_vertices());
    m_delta.resize(_mesh.n_vertices(),3);
    createMatricies(_mesh);
}
//----------------------------------------------------------------------------------------------------------------------
void LMESolver::createMatricies(MyMesh &_mesh){

    //the center point of our traingle "umbrella"
    MyMesh::Point centerPoint;
    //our neighbours for doing our cotagent weights
    MyMesh::Point currentN, nextN, prevN, e1, e2;
    //our two angles to be calcuated in our cotangent weights
    float alpha,beta;
    //our calcuated weight
    float w;
    //weight * vj for calcuation of delta at the end
    MyMesh::Point weightedSum;
    //our delta vector
    MyMesh::Point delta;
    //how many neighbours make up our "umbrella"
    int numNeighbours;
    //index of our center vertex
    int idx=0;
    //index of our neightbour vertex
    int nIdx;
    for (MyMesh::VertexIter v_it=_mesh.vertices_begin(); v_it!=_mesh.vertices_end(); ++v_it)
    {

        //std::cout<<"mesh id "<<v_it.handle().idx()<<std::endl;
        //The diagonal of this matrix will always be 1
        m_laplaceMatrix.coeffRef(idx,idx) = 1.0;
        //total up how many neighbours we have
        numNeighbours = 0;
        for (MyMesh::VertexVertexIter vv_it=_mesh.vv_iter( *v_it ); vv_it.is_valid(); ++vv_it){
            numNeighbours++;
        }

        centerPoint = _mesh.point( *v_it );
        //set sum to 0
        weightedSum = MyMesh::Point(0,0,0);
        for (MyMesh::VertexVertexIter vv_it=_mesh.vv_iter( *v_it ); vv_it.is_valid(); ++vv_it){
            //increment our neightbour count
            numNeighbours++;
            nIdx = vv_it.handle().idx();
            //get the neightbour points we need for our cotangent calcuations
            currentN = _mesh.point( *vv_it );
            nextN = _mesh.point( *++vv_it );
            prevN = _mesh.point( *----vv_it );
            //return iterator to original potision
            ++vv_it;

            //create our 2 edges to work out angle alpha
            e1 = currentN-nextN;
            e2 = centerPoint-nextN;
            //dot product to work out the alpha
            alpha = acos(OpenMesh::dot(e1,e2)/(e1.length()*e2.length()));

            //now do the same to calculate beta
            e1 = currentN-prevN;
            e2 = centerPoint-prevN;
            //dot product to work out the alpha
            beta = acos(OpenMesh::dot(e1,e2)/(e1.length()*e2.length()));

            w = 0.5 * (1.0/tan(alpha) + 1.0/tan(beta));
            w/=numNeighbours;

            //bit of a work around
            //need to talk to richard about this URGENT!!!!
            if(w>1) w=1.0;
            if(w<0) w=0;

            m_laplaceMatrix.coeffRef(idx,nIdx) = -w;

            weightedSum+=w*currentN;
        }

        //calculate our delta vector
        delta = centerPoint - weightedSum;
        m_delta.coeffRef(idx,0) = delta[0];
        m_delta.coeffRef(idx,1) = delta[1];
        m_delta.coeffRef(idx,2) = delta[2];
        idx++;
    }
}

//----------------------------------------------------------------------------------------------------------------------
void LMESolver::createMatricies(std::vector<ngl::Vec3> _points){
    ngl::Vec3 current,prev,next,delta;
    int prevLoc, nextLoc;

    for(unsigned int i=0; i<_points.size();i++){
        std::cout<<"OG points "<<_points[i].m_x<<","<<_points[i].m_y<<","<<_points[i].m_z<<std::endl;
    }
    for(unsigned int i=0; i<_points.size();i++){
        //do some bourndary checks
        if(i==0){
            prevLoc = _points.size()-1;
        }
        else{
            prevLoc = i-1;
        }
        if(i==(_points.size()-1)){
            nextLoc = 0;
        }
        else{
            nextLoc = i+1;
        }

        current = _points[i];
        prev = _points[prevLoc];
        next = _points[nextLoc];

        delta = current - 0.5*(prev+next);

        m_delta.coeffRef(i,0) = delta.m_x;
        m_delta.coeffRef(i,1) = delta.m_y;
        m_delta.coeffRef(i,2) = delta.m_z;

        //some lame hard coded mathematics!
        //to be improoved at a later date
        m_laplaceMatrix.coeffRef(i,i) = 1.0;
        m_laplaceMatrix.coeffRef(i,prevLoc) = -0.5;
        m_laplaceMatrix.coeffRef(i,nextLoc) = -0.5;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void LMESolver::addAnchor(int _vertex, MyMesh &_mesh){
    //add the handle to our laplace matrix
    m_laplaceMatrix.conservativeResize(m_laplaceMatrix.rows()+1,m_laplaceMatrix.cols());
    m_laplaceMatrix.coeffRef(m_laplaceMatrix.rows()-1,_vertex) = 1.0;
    //add the index to our anchor list so we know where it is
    anchorInfo aInfo;
    aInfo.matIdx = m_laplaceMatrix.rows()-1;
    aInfo.vertIdx = _vertex;
    m_anchorList.push_back(aInfo);

    MyMesh::VertexIter v_it=_mesh.vertices_begin();
    //iterate to our vertex in our mesh
    for(int i=0;i<_vertex;i++)
        v_it++;

    MyMesh::Point vertPos = _mesh.point( *v_it );
//    std::cout<<"vert position "<<vertPos[0]<<","<<vertPos[1]<<","<<vertPos[2]<<std::endl;
    //add the handle to our delta matrix
    m_delta.conservativeResize(m_delta.rows()+1,m_delta.cols());
    m_delta.coeffRef(m_delta.rows()-1,0) = vertPos[0];
    m_delta.coeffRef(m_delta.rows()-1,1) = vertPos[1];
    m_delta.coeffRef(m_delta.rows()-1,2) = vertPos[2];

    std::cout<<"anchor added, vert no "<<_vertex<<" pos "<<vertPos[0]<<","<<vertPos[1]<<","<<vertPos[2]<<std::endl;
    //change the color of our vertex to match the weight
    MyMesh::Color vertColor = _mesh.color( *v_it );
    //make our anchored verts red
    vertColor[1] = 0.0;
    _mesh.set_color(*v_it, vertColor );

}

//----------------------------------------------------------------------------------------------------------------------
int LMESolver::addHandle(int _vertex, MyMesh &_mesh){
    //make some more room in out matricies
    m_laplaceMatrix.conservativeResize(m_laplaceMatrix.rows()+1,m_laplaceMatrix.cols());
    m_delta.conservativeResize(m_delta.rows()+1,m_delta.cols()+1);
    //add to our handle list so we know what its idx is
    handleInfo hInfo;
    hInfo.matIdx = m_laplaceMatrix.rows()-1;
    hInfo.vertIdx = _vertex;
    m_handleList.push_back(hInfo);

    //add our handle to our matrix
    m_laplaceMatrix.coeffRef(m_laplaceMatrix.rows()-1, _vertex) = 1.0;

    MyMesh::VertexIter v_it=_mesh.vertices_begin();
    //iterate to our vertex in our mesh
    for(int i=0;i<_vertex;i++)
        v_it++;

    //add this to our
    MyMesh::Point vertPos = _mesh.point( *v_it );

    std::cout<<"vertPos "<<vertPos[0]<<","<<vertPos[1]<<","<<vertPos<<std::endl;

    m_delta.coeffRef(m_delta.rows()-1,0) = vertPos[0];
    m_delta.coeffRef(m_delta.rows()-1,1) = vertPos[1];
    m_delta.coeffRef(m_delta.rows()-1,2) = vertPos[2];

    //change the color of our vertex to match the weight
    MyMesh::Color vertColor = _mesh.color( *v_it );
    vertColor[2] = 1.0;
    _mesh.set_color(*v_it, vertColor );

    std::cout<<"handle list size "<<m_handleList.size()<<std::endl;
    return (m_handleList.size()-1);
}


//----------------------------------------------------------------------------------------------------------------------
void LMESolver::moveHandle(int _handleNo, ngl::Vec3 _trans){
    int handleIdx;
    //find the location of our handle in our delta matrix
    handleIdx = m_handleList[_handleNo].matIdx;

    //change the posision of our handle in our delta matrix
    m_delta.coeffRef(handleIdx,0) += _trans.m_x;
    m_delta.coeffRef(handleIdx,1) += _trans.m_y;
    m_delta.coeffRef(handleIdx,2) += _trans.m_z;

}

//----------------------------------------------------------------------------------------------------------------------
std::vector<ngl::Vec3> LMESolver::calculatePoints(MyMesh &_mesh){


//    Eigen::SimplicialCholeskyLDLT<Eigen::SparseMatrix<double> > solver;


    Eigen::SparseMatrix<double> At(m_laplaceMatrix);
    At.transpose();
    Eigen::SparseMatrix<double> A(m_laplaceMatrix);
    Eigen::SparseMatrix<double> b(m_delta);
//    std::cout<<"A our laplace matrix"<<std::endl;
//    std::cout<<A<<std::endl;
//    std::cout<<"B our delta matrix"<<std::endl;
//    std::cout<<b<<std::endl;
//    std::cout<<"calculated matrix "<<std::endl;

    //solve At*A*x = At*b
    Eigen::SparseMatrix<double> AtA = A.transpose() * A;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>, Eigen::Upper > solver(AtA);

    Eigen::SparseMatrix<double> Atb = A.transpose()*b;

    Eigen::SparseMatrix<double> final = solver.solve(Atb);
    if(solver.info()!=Eigen::Success){
        std::cout<<"oh balls it failed"<<std::endl;
    }
    //std::cout<<final<<std::endl;
    MyMesh::VertexIter v_it=_mesh.vertices_begin();
    MyMesh::Point currentPoint;
    std::vector<ngl::Vec3> returnPoints;
    returnPoints.resize(final.rows());
    for(int i=0;i<final.rows();i++){
        returnPoints[i] = ngl::Vec3(final.coeff(i,0),final.coeff(i,1),final.coeff(i,2));
        currentPoint[0] = returnPoints[i].m_x;
        currentPoint[1] = returnPoints[i].m_y;
        currentPoint[2] = returnPoints[i].m_z;
        _mesh.set_point(*v_it,currentPoint);
        ++v_it;
    }
    return returnPoints;
}

//----------------------------------------------------------------------------------------------------------------------
