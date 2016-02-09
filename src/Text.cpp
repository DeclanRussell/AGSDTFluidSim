/*
  Copyright (C) 2011 Jon Macey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
//---------------------------------------------------------------------------

#include "Text.h"
#include <iostream>
#include <QtGui/QImage>
#include <QFontMetrics>
#include <QPainter>


//---------------------------------------------------------------------------
/// @brief code taken from here http://jeffreystedfast.blogspot.com/2008/06/calculating-nearest-power-of-2.html
/// @param _num the number we wish to get the nearest power from
// OpenGL needs textures to be in powers of two, this function will get the
// nearest power of two to the current value passed in
//---------------------------------------------------------------------------
unsigned int nearestPowerOfTwo ( unsigned int _num )
{
    unsigned int j, k;
    (j = _num & 0xFFFF0000) || (j = _num);
    (k = j & 0xFF00FF00) || (k = j);
    (j = k & 0xF0F0F0F0) || (j = k);
    (k = j & 0xCCCCCCCC) || (k = j);
    (j = k & 0xAAAAAAAA) || (j = k);
    return j << 1;
}
// end citation

//---------------------------------------------------------------------------
Text::Text( const QFont &_f)
{

  //Create our text shader
  m_textShader = new ShaderProgram();
  Shader vert("shaders/TextVert.glsl",GL_VERTEX_SHADER);
  Shader frag("shaders/TextFrag.glsl",GL_FRAGMENT_SHADER);
  m_textShader->attachShader(&vert);
  m_textShader->attachShader(&frag);
  m_textShader->bindFragDataLocation(0, "fragColour");
  m_textShader->link();
  m_textShader->use();

  // so first we grab the font metric of the font being used
  QFontMetrics metric(_f);
  // this allows us to get the height which should be the same for all
  // fonts of the same class as this is the total glyph height
  float fontHeight=metric.height();

  // loop for all basic keyboard chars we will use space to ~
  // should really change this to unicode at some stage
  const static char startChar=' ';
  const static char endChar='~';
  // Most OpenGL cards need textures to be in powers of 2 (128x512 1024X1024 etc etc) so
  // to be safe we will conform to this and calculate the nearest power of 2 for the glyph height
  // we will do the same for each width of the font below
  int heightPow2=nearestPowerOfTwo(fontHeight);

  // we are now going to create a texture / billboard for each font
  // they will be the same height but will possibly have different widths
  // as some of the fonts will be the same width, to save VAO space we will only create
  // a vao if we don't have one of the set width. To do this we use the has below
  QHash <int,GLuint> widthVAO;

  for(char c=startChar; c<=endChar; ++c)
  {
    QChar ch(c);
    FontChar fc;
    // get the width of the font and calculate the ^2 size
    int width=metric.width(c);
    int widthPow2=nearestPowerOfTwo(width);
    // now we set the texture co-ords for our quad it is a simple
    // triangle billboard with tex-cords as shown
    //  s0/t0  ---- s1,t0
    //         |\ |
    //         | \|
    //  s0,t1  ---- s1,t1
    // each quad will have the same s0 and the range s0-s1 == 0.0 -> 1.0
    float s0=0.0;
    // we now need to scale the tex cord to it ranges from 0-1 based on the coverage
    // of the glyph and not the power of 2 texture size. This will ensure that kerns
    // / ligatures match
    float s1=width*1.0/widthPow2;
    // t0 will always be the same
    float t0=0.0;
    // this will scale the height so we only get coverage of the glyph as above
    float t1=metric.height()*-1.0/heightPow2;
    // we need to store the font width for later drawing
    fc.width=width;
    // now we will create a QImage to store the texture, basically we are going to draw
    // into the qimage then save this in OpenGL format and load as a texture.
    // This is relativly quick but should be done as early as possible for max performance when drawing
    QImage finalImage(nearestPowerOfTwo(width),nearestPowerOfTwo(fontHeight),QImage::Format_ARGB32);
    // set the background for transparent so we can avoid any areas which don't have text in them
    finalImage.fill(Qt::transparent);
    // we now use the QPainter class to draw into the image and create our billboards
    QPainter painter;
    painter.begin(&finalImage);
    // try and use high quality text rendering (works well on the mac not as good on linux)
    painter.setRenderHints(QPainter::HighQualityAntialiasing
                   | QPainter::TextAntialiasing);
    // set the font to draw with
    painter.setFont(_f);
    // we set the glyph to be drawn in black the shader will override the actual colour later
    // see TextShader.h in src/shaders/
    painter.setPen(Qt::black);
    // finally we draw the text to the Image
    painter.drawText(0, metric.ascent(), QString(c));
    painter.end();
    // for debug purposes we can save the files as .png and view them
    // not needed just useful when developing the class/
    /*
    QString filename=".png";
    filename.prepend(c);
    finalImage.save(filename);
    */

    // now we create the OpenGL texture ID and bind to make it active
    glGenTextures(1, &fc.textureID);
    glBindTexture(GL_TEXTURE_2D, fc.textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // QImage has a method to convert itself to a format suitable for OpenGL
    finalImage=finalImage.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    // set rgba image data
    int widthTexture=finalImage.width();
    int heightTexture=finalImage.height();
    unsigned char *data = new unsigned char[ widthTexture*heightTexture * 4];
    unsigned int index=0;
    QRgb colour;
    for(int y=heightTexture-1; y>0; --y)
    {
      for(int x=0; x<widthTexture; ++x)
      {
        colour=finalImage.pixel(x,y);
        data[index++]=qRed(colour);
        data[index++]=qGreen(colour);
        data[index++]=qBlue(colour);
        data[index++]=qAlpha(colour);
        }
    }



    // the image in in RGBA format and unsigned byte load it ready for later
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, widthTexture, heightTexture,0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    delete [] data;
    // see if we have a Billboard of this width already
    if (!widthVAO.contains(width))
    {
        // this structure is used by the VAO to store the data to be uploaded
        // for drawing the quad
        struct textVertData
        {
        float x;
        float y;
        float u;
        float v;
        };
        // we are creating a billboard with two triangles so we only need the
        // 6 verts, (could use index and save some space but shouldn't be too much of an
        // issue
        textVertData d[6];
        // load values for triangle 1
        d[0].x=0;
        d[0].y=0;
        d[0].u=s0;
        d[0].v=t0;

        d[1].x=fc.width;
        d[1].y=0;
        d[1].u=s1;
        d[1].v=t0;

        d[2].x=0;
        d[2].y=fontHeight;
        d[2].u=s0;
        d[2].v=t1;
        // load values for triangle two
        d[3].x=0;
        d[3].y=0+fontHeight;
        d[3].u=s0;
        d[3].v=t1;


        d[4].x=fc.width;
        d[4].y=0;
        d[4].u=s1;
        d[4].v=t0;


        d[5].x=fc.width;
        d[5].y=fontHeight;
        d[5].u=s1;
        d[5].v=t1;


        // now we create a VAO to store the data
        GLuint vao;//=VertexArrayObject::createVOA(GL_TRIANGLES);
        glGenVertexArrays(1,&vao);
        // bind it so we can set values
        glBindVertexArray(vao);
        // set the vertex data (2 for x,y 2 for u,v)
        GLuint vbo;
        glGenBuffers(1,&vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,6*sizeof(textVertData),&d[0].x,GL_STATIC_DRAW);
        // now we set the attribute pointer to be 0 (as this matches vertIn in our shader)
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(textVertData),(GLvoid*)(0*sizeof(GL_FLOAT)));
        glEnableVertexAttribArray(0);
        // We can now create another set of data (which will be added to the VAO)
        // in this case the UV co-ords
        // now we set this as the 2nd attribute pointer (1) to match inUV in the shader
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(textVertData),((float *)NULL + (2)));
        glEnableVertexAttribArray(1);
        // say how many indecis to be rendered
        //vao->setNumIndices(6);

        // now unbind
        glBindBuffer(GL_ARRAY_BUFFER,0);
        glBindVertexArray(0);
        // store the vao pointer for later use in the draw method
        fc.vao=vao;
        widthVAO[width]=vao;
    }
    else
    {
      fc.vao=widthVAO[width];
    }
    // finally add the element to the map, this must be the last
    // thing we do
    m_characters[c]=fc;
  }
  std::cout<<"created "<<widthVAO.size()<<" unique billboards\n";
  // set a default colour (black) incase user forgets
  this->setColour(0,0,0);
  this->setTransform(1.0,1.0);
}


//---------------------------------------------------------------------------
Text::~Text()
{
  // our dtor should clear out the textures and remove the VAO's
  foreach( FontChar m, m_characters)
  {
    glDeleteTextures(1,&m.textureID);
    glDeleteBuffers(1,&m.vao);
  }

}




//---------------------------------------------------------------------------
void Text::renderText( float _x, float _y,  const QString &text ) const
{
  //activate ouf shader
  m_textShader->use();
  // make sure we are in texture unit 0 as this is what the
  // shader expects
  //set our texture uniform
  glActiveTexture(GL_TEXTURE0);
  GLuint texLoc = m_textShader->getUniformLoc("tex");
  glUniform1f(texLoc,0);


  GLuint yPosLoc = m_textShader->getUniformLoc("ypos");
  // the y pos will always be the same so set it once for each
  // string we are rendering
  glUniform1f(yPosLoc,_y);
  // now enable blending and disable depth sorting so the font renders
  // correctly
  glEnable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //get the xPos location in our shader
  GLuint xPosLoc = m_textShader->getUniformLoc("xpos");
  // now loop for each of the char and draw our billboard

  unsigned int textLength=text.length();

  for (unsigned int i = 0; i < textLength; ++i)
  {
    // set the shader x position this will change each time
    // we render a glyph by the width of the char
    glUniform1f(xPosLoc,_x);
    // so find the FontChar data for our current char
//    FontChar f = m_characters[text[i].toAscii()];
    FontChar f = m_characters[text[i].toLatin1()];

    // bind the pre-generated texture
    glBindTexture(GL_TEXTURE_2D, f.textureID);
    // bind the vao
    glBindVertexArray(f.vao);
    // draw
    glDrawArrays(GL_TRIANGLES,0,6);
    // now unbind the vao
    glBindVertexArray(0);
    // finally move to the next glyph x position by incrementing
    // by the width of the char just drawn
    _x+=f.width;

  }
  // finally disable the blend and re-enable depth sort
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

}

//---------------------------------------------------------------------------
void Text::setScreenSize(int _w, int _h )
{

  float scaleX=2.0/_w;
  float scaleY=-2.0/_h;
  // in shader we do the following code to transform from
  // x,y to NDC
  // gl_Position=vec4( ((xpos+inVert.x)*scaleX)-1,((ypos+inVert.y)*scaleY)+1.0,0.0,1.0); "
  // so all we need to do is calculate the scale above and pass to shader every time the
  // screen dimensions change
  m_textShader->use();
  GLuint scaleXLoc = m_textShader->getUniformLoc("scaleX");
  GLuint scaleYLoc = m_textShader->getUniformLoc("scaleY");
  glUniform1f(scaleXLoc,scaleX);
  glUniform1f(scaleYLoc,scaleY);

}

//---------------------------------------------------------------------------
// our text shader uses the alpha of the texture to modulate visibility
// when we render the text we use this colour passed to the shader
// it is default to black but this will change it
// the shader uses the following code
// vec4 text=texture(tex,vertUV.st);
// fragColour.rgb=textColour.rgb;
// fragColour.a=text.a;


//---------------------------------------------------------------------------
void Text::setColour(float _r,  float _g, float _b)
{


  m_textShader->use();
  GLuint colLoc = m_textShader->getUniformLoc("textColour");
  glUniform3f(colLoc,_r,_g,_b);

}

void Text::setTransform(float _x, float _y)
{

  m_textShader->use();
  GLuint transLoc = m_textShader->getUniformLoc("transform");
  glUniform2f(transLoc,_x,_y);
}


//---------------------------------------------------------------------------

