#ifndef SELECTABLE_H
#define SELECTABLE_H

//----------------------------------------------------------------------------------------------------------------------
/// @file selectable.h
/// @brief A class to add selectable objects to a scene
/// @author Declan Russell
/// @version 1.0
/// @date 12/12/14 Initial version
//----------------------------------------------------------------------------------------------------------------------

#include <ngl/Mat4.h>
#include <ngl/Camera.h>
#include <ngl/VertexArrayObject.h>

class selectable
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our defalut constructor
    //----------------------------------------------------------------------------------------------------------------------
    selectable();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief constructor to create our selectable with a postion
    //----------------------------------------------------------------------------------------------------------------------
    selectable(ngl::Vec3 _pos, int _vertexID);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief contructor to create our selectable with a postion and radius
    //----------------------------------------------------------------------------------------------------------------------
    selectable(ngl::Vec3 _pos, int _vertexID, float _radius);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief default destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~selectable();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our draw funciton
    //----------------------------------------------------------------------------------------------------------------------
    void draw(ngl::Mat4 _mouseGlobalTX, ngl::Camera *_cam);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a function to test if our selectable has been seleted
    //----------------------------------------------------------------------------------------------------------------------
    bool testSelection(ngl::Vec3 _ray, ngl::Mat4 _mouseGlobalTX, ngl::Camera *_cam);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a mutator to set selected
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSelected(bool _selected){m_seleted = _selected;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a accesor to see if our selectable is selected
    //----------------------------------------------------------------------------------------------------------------------
    inline bool isSelected(){return m_seleted;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a function to move our selectable
    //----------------------------------------------------------------------------------------------------------------------
    inline void move(ngl::Vec3 _dir){if(m_isMovable) m_pos+=_dir;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the position of our selectable
    //----------------------------------------------------------------------------------------------------------------------
    inline void setPos(ngl::Vec3 _pos){m_pos = _pos;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief an accesor to our position
    //----------------------------------------------------------------------------------------------------------------------
    inline ngl::Vec3 getPos(){return m_pos;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief return our vertex ID
    //----------------------------------------------------------------------------------------------------------------------
    inline int getID(){return m_vertexID;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set if our selectable is selectable
    //----------------------------------------------------------------------------------------------------------------------
    inline void isSelectable(bool _isSelectable){m_isSelectable=_isSelectable;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set is handle
    //----------------------------------------------------------------------------------------------------------------------
    inline void isHandle(bool _isHandle){m_isHandle = _isHandle;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set if our handle is movable
    //----------------------------------------------------------------------------------------------------------------------
    inline void isMovable(bool _isMovable){m_isMovable = _isMovable;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief an array of our owned handled
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<int> m_handles;
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief load our MVP matricies to our shader
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShader(ngl::Mat4 _mouseGlobalTX, ngl::Camera *_cam);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the postion of our selectable
    //----------------------------------------------------------------------------------------------------------------------
    ngl::Vec3 m_pos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the radiuse of our selectable, default 0.5.
    //----------------------------------------------------------------------------------------------------------------------
    float m_radius;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a bool to let us know if our selectable has been seleted
    //----------------------------------------------------------------------------------------------------------------------
    bool m_seleted;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a variable to keep track of which vertex we are moving
    //----------------------------------------------------------------------------------------------------------------------
    int m_vertexID;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set if our selectable is selectable
    //----------------------------------------------------------------------------------------------------------------------
    bool m_isSelectable;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set if our selectable is a handle, basically just changes the colour to green
    //----------------------------------------------------------------------------------------------------------------------
    bool m_isHandle;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a bool to clarify if our selectable is movable
    //----------------------------------------------------------------------------------------------------------------------
    bool m_isMovable;
    //----------------------------------------------------------------------------------------------------------------------

};

#endif // SELECTABLE_H
