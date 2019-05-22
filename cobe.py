# functions required to rasterize COBE Quadrilaterialized Spherical cube data
import numpy as np

def ecliptic_lat_lon_to_pix(e_lat,e_lon,pix_index):

    """
    Function: returns pixel indices of quadrilateralized spherical cube, given e_lat and e_lon.
        See https://e_lon.gsfc.nasa.gov/product/cobe/skymap_info_new.cfm for data format.
        e_lat and e_lon are defined by J2000 ecliptic coordinate system
            
        Arguments
        ---------
        
        e_lat[...]: float
            ecliptic latitude (radians)
        
        e_lon[...]: float
            ecliptic longitude (radians)
            
        pix_index: int
            qs resolution index


        Result
        ------
        
        qs_index[...]: (int,)
            indices of qs array
    
    """

    # convert e_lat and e_lon into Cartesian unit vector
    sin_e_lat=np.sin(e_lat)
    cos_e_lat=np.cos(e_lat)
    sin_e_lon=np.sin(e_lon)
    cos_e_lon=np.cos(e_lon)
    
    x=cos_e_lat*cos_e_lon
    y=cos_e_lat*sin_e_lon
    z=sin_e_lat
    
    # get face index
    face_index=np.argmax(np.array((z,x,y,-x,-y,-z)),axis=0)
    
    # set up face coordinates
    x_face=np.zeros(face_index.shape,dtype=np.float)
    y_face=np.zeros(face_index.shape,dtype=np.float)
    
    mask=face_index==0
    x_face[mask]=y[mask]/z[mask]
    y_face[mask]=-x[mask]/z[mask]
    
    mask=face_index==1
    x_face[mask]=y[mask]/x[mask]
    y_face[mask]=z[mask]/x[mask]
    
    mask=face_index==2
    x_face[mask]=-x[mask]/y[mask]
    y_face[mask]=z[mask]/y[mask]
    
    mask=face_index==3
    x_face[mask]=y[mask]/x[mask]
    y_face[mask]=-z[mask]/x[mask]
    
    mask=face_index==4
    x_face[mask]=-x[mask]/y[mask]
    y_face[mask]=-z[mask]/y[mask]
    
    mask=face_index==5
    x_face[mask]=-y[mask]/z[mask]
    y_face[mask]=-x[mask]/z[mask]
    
    # convert tangent plane x y into curvilinear X Y
    X,Y=tangent_to_curvilinear(x_face,y_face)    
    
    # get indices
    qs_index=face_coord_to_pix(X,Y,face_index,pix_index)
    
    return qs_index

def raster_unfolded_pix(pix_index,blank=-1):
    
    """
    
    Function: returns array of pixel indces of unfolded cube
    See https://e_lon.gsfc.nasa.gov/product/cobe/skymap_info_new.cfm for data format. for layout details
    
    Arguments
    ---------
    
    pix_index: int
        qs resolution indexs
        
    blank: int
        index for empty areas of map
        
    Result
    ------
    
    qs_index[:,:]: int
        indices of unfolded cube
        
    """
    
    # set face edge length
    n_axis=1<<(pix_index-1)
    
    # initialise qs_index
    qs_index=np.full((4*n_axis,3*n_axis),blank)
    
    # set face X,Y coordinates
    pixel_centres=np.arange(n_axis,dtype=np.float)/(n_axis)+1./(2.*n_axis)
    pixel_centres=pixel_centres*2-1.
    X,Y=np.meshgrid(pixel_centres,pixel_centres,indexing="ij")
    
    # set face 0
    qs_index[:n_axis,2*n_axis:]=face_coord_to_pix(X,Y,0,pix_index)
    
    # set face 1
    qs_index[:n_axis,n_axis:2*n_axis]=face_coord_to_pix(X,Y,1,pix_index)
    
    # set face 2
    qs_index[n_axis:2*n_axis,n_axis:2*n_axis]=face_coord_to_pix(X,Y,2,pix_index)
    
    # set face 3
    qs_index[2*n_axis:3*n_axis,n_axis:2*n_axis]=face_coord_to_pix(X,Y,3,pix_index)
    
    # set face 4
    qs_index[3*n_axis:,n_axis:2*n_axis]=face_coord_to_pix(X,Y,4,pix_index)
    
    # set face 5
    qs_index[:n_axis,:n_axis]=face_coord_to_pix(X,Y,5,pix_index)
    
    # reverse x axis and transpose to follow data format convention
    qs_index=np.flip(qs_index,axis=0).T
    
    return qs_index


def face_coord_to_pix(X,Y,face_index,pix_index):
    
    """
    
    Function: returns the pixel indices of curvilinear coordinates on a face
    
    Arguments
    ---------
    
    X[...]: float
        x positions on face (between -1 and 1)
        
    Y[...]: float
        y postions on face (between -1 and 1) 
        
    face_index: int
        index of face
        
    pix_index: int
        qs resolution index
        
    Result
    ------
    
    qs_index[...]: int
        indices of X,Y positions
        
    
    """
    
    # find qs_index
    x_wall=np.zeros(X.shape,dtype=np.float)
    y_wall=np.zeros(X.shape,dtype=np.float)
    
    # initialise qs_index
    qs_index=np.zeros(x_wall.shape,dtype=np.int)
    
    # set most significant bits of qs_index
    qs_index+=face_index<<(2*(pix_index-1))
    
    # set remaining significant bits
    for i in range(pix_index-1):

        i_shift=pix_index-i-2
        dx=0.5**(i+1)
        
        mask=X>x_wall
        qs_index[mask]^=1<<(2*i_shift)
        x_wall[mask]+=dx
        x_wall[np.logical_not(mask)]-=dx
        
        mask=Y>y_wall
        qs_index[mask]^=1<<(2*i_shift+1)
        y_wall[mask]+=dx
        y_wall[np.logical_not(mask)]-=dx
        
    return qs_index
    

def tangent_to_curvilinear(x,y):
        
    """
    
    Function: converts tangent plane x-y coordinates to curvilinear X, Y coordinates
    
    Note: this a conversion of some *ancient* FORTRAN code:
        https://e_lona.gsfc.nasa.gov/data/cobe/cobe_analysis_software/cgis-for.tar
        incube.for
        
    
    Arguments
    ---------
    
    x,y: float
        tangent plane x y coords
        
    Result
    ------
    X,Y: float
        curilinear X Y coords
        
    """
    
    # set params
    gstar=1.37484847732
    g=-0.13161671474
    m=0.004869491981
    w1=-0.159596235474
    c00=0.141189631152
    c10=0.0809701286525
    c01=-0.281528535557
    c11=0.15384112876
    c20=-0.178251207466
    c02=0.106959469314
    d0=0.0759196200467
    d1=-0.0217762490699
    
    aa=x**2
    bb=y**2
    a4=aa**2
    b4=bb**2
    onmaa=1.-aa
    onmbb=1.-bb
    
    gstar_1=1.-gstar
    m_g=m-g
    c_comb=c00+c11*aa*bb
    
    X=x*(gstar+aa*gstar_1+onmaa
         *(bb*(g+m_g*aa+onmbb*(c_comb+c10*aa+c01*bb+c20*a4+c02*b4))
         +aa*(w1-onmaa*(d0+d1*aa))))
    
    Y=y*(gstar+bb*gstar_1+onmbb
         *(aa*(g+m_g*bb+onmaa*(c_comb+c10*bb+c01*aa+c20*b4+c02*a4))
         +bb*(w1-onmbb*(d0+d1*bb))))
    
    return X,Y