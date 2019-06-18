# functions required to rasterize COBE Quadrilaterialized Spherical cube data
import numpy as np

def ecliptic_lat_lon_to_pix(e_lat,e_lon,pix_index):

    """
    Function: returns pixel indices of quadrilateralized spherical cube, given e_lat and e_lon.
        See https://lambda.gsfc.nasa.gov/product/cobe/skymap_info_new.cfm for data format.
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

    # get tangent plane x y
    x,y,face_index=ecliptic_lat_lon_to_tangent(e_lat,e_lon)
    
    # convert tangent plane x y into curvilinear X Y
    X,Y=tangent_to_curvilinear(x,y)    
    
    # get indices
    qs_index=curvilinear_to_pix(X,Y,face_index,pix_index)
    
    return qs_index


def pix_to_ecliptic_lat_lon(qs_index,pix_index):

    """
    Function: returns e_lat and e_lon given pixel indices of quadrilateralized spherical cube
        See https://lambda.gsfc.nasa.gov/product/cobe/skymap_info_new.cfm for data format.
        e_lat and e_lon are defined by J2000 ecliptic coordinate system
            
        Arguments
        ---------
        
        qs_index[...]: (int,)
            indices of qs array
            
        pix_index: int
            qs resolution index

        Result
        ------
        
        e_lat[...]: float
            ecliptic latitude (radians)
        
        e_lon[...]: float
            ecliptic longitude (radians)
    
    """

    # get curvilinear coordinates
    X,Y,face_index=pix_to_curvilinear(qs_index,pix_index)
    
    # convert curvilinear X Y to tangent plane x y
    x,y=curvilinear_to_tangent(X,Y)
    
    # get latitude and logngitude
    e_lat,e_lon=tangent_to_ecliptic_lat_lon(x,y,face_index)
    
    return e_lat,e_lon


def raster_unfolded_pix(pix_index,blank=-1):
    
    """
    
    Function: returns array of pixel indces of unfolded cube
    See https://lambda.gsfc.nasa.gov/product/cobe/skymap_info_new.cfm for data format. for layout details
    
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
    qs_index[:n_axis,2*n_axis:]=curvilinear_to_pix(X,Y,0,pix_index)
    
    # set face 1
    qs_index[:n_axis,n_axis:2*n_axis]=curvilinear_to_pix(X,Y,1,pix_index)
    
    # set face 2
    qs_index[n_axis:2*n_axis,n_axis:2*n_axis]=curvilinear_to_pix(X,Y,2,pix_index)
    
    # set face 3
    qs_index[2*n_axis:3*n_axis,n_axis:2*n_axis]=curvilinear_to_pix(X,Y,3,pix_index)
    
    # set face 4
    qs_index[3*n_axis:,n_axis:2*n_axis]=curvilinear_to_pix(X,Y,4,pix_index)
    
    # set face 5
    qs_index[:n_axis,:n_axis]=curvilinear_to_pix(X,Y,5,pix_index)
    
    # reverse x axis and transpose to follow data format convention
    qs_index=np.flip(qs_index,axis=0).T
    
    return qs_index


def ecliptic_lat_lon_to_tangent(e_lat,e_lon):

    """
    Function: returns tangential coordinates for a given latitude and longitude
            
        Arguments
        ---------
        
        e_lat[...]: float
            ecliptic latitude (radians)
        
        e_lon[...]: float
            ecliptic longitude (radians)

        Result
        ------
        
        x_tan[...]: float
            tangential x coordinate
            
        y_tan[...]: float
            tangential y coordinate
            
        face_index: int
            index designating face of cube
    
    """

    # convert e_lat and e_lon into Cartesian unit vector
    x=np.cos(e_lat)*np.cos(e_lon)  
    y=np.cos(e_lat)*np.sin(e_lon)
    z=np.sin(e_lat)
    
    # get face index from largest unit vector component
    face_index=np.argmax(np.array((z,x,y,-x,-y,-z)),axis=0)
    
    # set up tangential coordinates
    x_tan=np.zeros(face_index.shape,dtype=np.float)
    y_tan=np.zeros(face_index.shape,dtype=np.float)
    
    # project coordinates onto tangential plane
    mask=face_index==0
    x_tan[mask]=y[mask]/z[mask]
    y_tan[mask]=-x[mask]/z[mask]
    
    mask=face_index==1
    x_tan[mask]=y[mask]/x[mask]
    y_tan[mask]=z[mask]/x[mask]
    
    mask=face_index==2
    x_tan[mask]=-x[mask]/y[mask]
    y_tan[mask]=z[mask]/y[mask]
    
    mask=face_index==3
    x_tan[mask]=y[mask]/x[mask]
    y_tan[mask]=-z[mask]/x[mask]
    
    mask=face_index==4
    x_tan[mask]=-x[mask]/y[mask]
    y_tan[mask]=-z[mask]/y[mask]
    
    mask=face_index==5
    x_tan[mask]=-y[mask]/z[mask]
    y_tan[mask]=-x[mask]/z[mask]
    
    return x_tan,y_tan,face_index


def tangent_to_ecliptic_lat_lon(x_tan,y_tan,face_index):

    """
    Function: returns tangential coordinates for a given latitude and longitude
            
        Arguments
        ---------
        
        x_tan[...]: float
            tangential x coordinate
            
        y_tan[...]: float
            tangential y coordinate
            
        face_index: int
            index designating face of cube
        

        Result
        ------
        
        e_lat[...]: float
            ecliptic latitude (radians)
        
        e_lon[...]: float
            ecliptic longitude (radians)
        

    
    """

    # convert x_tan and y_tan to xyz coordinatse
    x=np.zeros(x_tan.shape,dtype=np.float)
    y=np.zeros(x_tan.shape,dtype=np.float)
    z=np.zeros(x_tan.shape,dtype=np.float)
    
    # de-project coordinates from tangential plane
    mask=face_index==0
    x[mask]=-y_tan[mask]
    y[mask]=x_tan[mask]
    z[mask]=1.
    
    mask=face_index==1
    x[mask]=1.
    y[mask]=x_tan[mask]
    z[mask]=y_tan[mask]
    
    mask=face_index==2
    x[mask]=-x_tan[mask]
    y[mask]=1.
    z[mask]=y_tan[mask]
    
    mask=face_index==3
    x[mask]=-1.
    y[mask]=-x_tan[mask]
    z[mask]=y_tan[mask]
    
    mask=face_index==4
    x[mask]=x_tan[mask]
    y[mask]=-1.
    z[mask]=y_tan[mask]
    
    mask=face_index==5
    x[mask]=y_tan[mask]
    y[mask]=x_tan[mask]
    z[mask]=-1.

    # Convert Cartesian vectors into polar coordinates
    r=np.sqrt(x**2+y**2+z**2)
    e_lat=np.arcsin(z/r)
    e_lon=np.arctan2(y,x)

    return e_lat,e_lon


def curvilinear_to_pix(X,Y,face_index,pix_index):
    
    """
    
    Function: returns the pixel indices of curvilinear coordinates on a face
    
    Arguments
    ---------
    
    X[...]: float
        curvilinear X positions on face (between -1 and 1)
        
    Y[...]: float
        curvilinear Y postions on face (between -1 and 1) 
        
    face_index: int
        index of face
        
    pix_index: int
        qs resolution index
        
    Result
    ------
    
    qs_index[...]: int
        indices of X,Y positions
        
    
    """
    
    # convert coordinates to indices between 0 and 2**pix_index-1
    i_max=1<<(pix_index-1)
    x_index=np.floor(i_max*0.5*(X+1.)).astype(np.int)
    y_index=np.floor(i_max*0.5*(Y+1.)).astype(np.int)

    # make sure indices are within bounds
    x_index=np.maximum(np.minimum(x_index,i_max-1),0)
    y_index=np.maximum(np.minimum(y_index,i_max-1),0)
    
    # interleave bits of x_index and y_index to make qs_index
    qs_index=np.zeros(X.shape,dtype=np.int)
    for i in range(pix_index-1):
        
        # filter bit value from indices
        bit_filter=1<<i
        x_bit=x_index&bit_filter
        y_bit=y_index&bit_filter
        
        # get bitshift dispacements
        x_shift=i
        y_shift=i+1
    
        # write bits to qs_index
        qs_index^=x_bit<<x_shift
        qs_index^=y_bit<<y_shift
        
    # set most significant bits of qs_index to face index
    qs_index^=face_index<<(2*(pix_index-1))
        
    return qs_index


def pix_to_curvilinear(qs_index,pix_index):
    
    """
    
    Function: returns the pixel indices of curvilinear coordinates on a face
    
    Arguments
    ---------
    
    qs_index[...]: int
        indices of X,Y positions
        
    pix_index: int
        qs resolution index
        
    Result
    ------
    
    X[...]: float
        curvilinear X positions on face (between -1 and 1)
        
    Y[...]: float
        curvilinear Y postions on face (between -1 and 1) 
        
    face_index: int
        index of face    
    
    """
    
    # de-interleave bits of qs_index to get x_index and y_index
    x_index=np.zeros(qs_index.shape,dtype=np.int)
    y_index=np.zeros(qs_index.shape,dtype=np.int)
    for i in range(pix_index-1):
        
        # fliter bit value from qs_index
        bit_filter=1<<(2*i)
        x_bit=qs_index&bit_filter
        bit_filter=1<<(2*i+1)
        y_bit=qs_index&bit_filter
        
        # get bitshift dispacements
        x_shift=i
        y_shift=i+1
    
        # write bits to x_index and y_index
        x_index^=x_bit>>x_shift
        y_index^=y_bit>>y_shift
    
    # get face_index from most significant bits of qs_index
    face_index=qs_index>>(2*(pix_index-1))
    
    # convert indices to coordinates between -1 and 1
    dx=2./(1<<(pix_index-1))
    X=(x_index+0.5)*dx-1.
    Y=(y_index+0.5)*dx-1.    
    
    return X,Y,face_index

def tangent_to_curvilinear(x,y):
        
    """
    
    Function: converts tangent plane x-y coordinates to curvilinear X-Y coordinates
    
    Note: this a conversion of some *ancient* FORTRAN code:
        https://lambda.gsfc.nasa.gov/data/cobe/cobe_analysis_software/cgis-for.tar/incube.for
        
    
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


def curvilinear_to_tangent(X,Y):
        
    """
    
    Function: converts curvilinear X-Y coordinates to tangent plane x-y coordinates
    
    Note: this a conversion of some *ancient* FORTRAN code:
        https://lambda.gsfc.nasa.gov/data/cobe/cobe_analysis_software/cgis-for.tar/forward_cube.for
        
    
    Arguments
    ---------
    
    X,Y: float
        curilinear X Y coords
        
    Result
    ------
    
    x,y: float
        tangent plane x y coords
    
        
    """
    
    # set params
    P=(-0.27292696,-0.07629969,-0.02819452,-0.22797056,
       -0.01471565, 0.27058160, 0.54852384, 0.48051509,
       -0.56800938,-0.60441560,-0.62930065,-1.74114454,
        0.30803317, 1.50880086, 0.93412077, 0.25795794,
        1.71547508, 0.98938102,-0.93678576,-1.41601920,
       -0.63915306, 0.02584375,-0.53022337,-0.83180469,
        0.08693841, 0.33887446, 0.52032238, 0.14381585)
    
    XX=X*X
    YY=Y*Y
    
    x=X*(1.+(1.-XX)*(
        P[0]+XX*(P[1]+XX*(P[3]+XX*(P[6]+XX*(P[10]+XX*(P[15]+XX*P[21])))))+
        YY*(P[2]+XX*(P[4]+XX*(P[7]+XX*(P[11]+XX*(P[16]+XX*P[22]))))+
        YY*(P[5]+XX*(P[8]+XX*(P[12]+XX*(P[17]+XX*P[23])))+
        YY*(P[9]+XX*(P[13]+XX*(P[18]+XX*P[24]))+
        YY*(P[14]+XX*(P[19]+XX*P[25])+
        YY*(P[20]+XX*P[26]+YY*P[27])))))))
    
    y=Y*(1.+(1.-YY)*(
        P[0]+YY*(P[1]+YY*(P[3]+YY*(P[6]+YY*(P[10]+YY*(P[15]+YY*P[21])))))+
        XX*(P[2]+YY*(P[4]+YY*(P[7]+YY*(P[11]+YY*(P[16]+YY*P[22]))))+
        XX*(P[5]+YY*(P[8]+YY*(P[12]+YY*(P[17]+YY*P[23])))+
        XX*(P[9]+YY*(P[13]+YY*(P[18]+YY*P[24]))+
        XX*(P[14]+YY*(P[19]+YY*P[25])+
        XX*(P[20]+YY*P[26]+XX*P[27])))))))
    
    return x,y
    
    