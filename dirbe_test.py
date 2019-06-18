from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cobe import ecliptic_lat_lon_to_pix,raster_unfolded_pix,pix_to_ecliptic_lat_lon

# load DIRBE data
dirbe_fits=fits.open("./DIRBE_BAND3A_ZSMA.FITS")
dirbe_brightness=dirbe_fits[1].data["Resid"]
pix_index=dirbe_fits[0].header["PIXRESOL"]



# make galactic lat-lon values
n_lat=513
n_lon=1025
g_lat,g_lon=np.meshgrid(np.linspace(-0.5*np.pi,0.5*np.pi,n_lat),
                        np.linspace(-np.pi,np.pi,n_lon),indexing="ij")

# convert to ecliptic lat-lon values
gc=SkyCoord(b=g_lat*units.radian,l=g_lon*units.radian,frame="galactic")
e_lat=gc.geocentrictrueecliptic.lat.to("radian").value
e_lon=gc.geocentrictrueecliptic.lon.to("radian").value

# make unfolded box projection
indices=raster_unfolded_pix(pix_index)
box_proj=np.full(indices.shape,np.nan)
box_proj[indices>-1]=dirbe_brightness[indices[indices>-1]]

# plot map
fig,ax=plt.subplots(2,2,figsize=(16,8))
im=ax[0,0].imshow(np.log10(box_proj),cmap="inferno",origin="lower")
ax[0,0].set_ylabel("$y$ [pix]")
ax[0,0].set_xticklabels([])

# plot indices
im=ax[1,0].imshow(np.where(indices>-1,indices,np.nan),cmap="viridis",origin="lower")
ax[1,0].set_xlabel("$x$ [pix]")
ax[1,0].set_ylabel("$y$ [pix]")

# make Mercator projection
indices=ecliptic_lat_lon_to_pix(e_lat,e_lon,pix_index)
mercator_proj=dirbe_brightness[indices]

#plot map
im=ax[0,1].imshow(np.log10(mercator_proj),extent=(g_lon[0,0],g_lon[0,-1],g_lat[0,0],g_lat[-1,0]),cmap="inferno",origin="lower")
cbar=fig.colorbar(im,ax=ax[0,1])
cbar.set_label("log($S$/[MJy/sr])")
ax[0,1].tick_params(top=True,left=True,bottom=True,right=True)
ax[0,1].set_ylabel("$b$ [rad]")
ax[0,1].set_xticklabels([])

# plot indices
im=ax[1,1].imshow(indices,extent=(g_lon[0,0],g_lon[0,-1],g_lat[0,0],g_lat[-1,0]),cmap="viridis",origin="lower")
cbar=fig.colorbar(im,ax=ax[1,1])
cbar.set_label("$i$")
ax[1,1].tick_params(top=True,left=True,bottom=True,right=True)
ax[1,1].set_xlabel("$l$ [rad]")
ax[1,1].set_ylabel("$b$ [rad]")
fig.tight_layout()
fig.subplots_adjust(hspace=0,wspace=-0.15,left=-0.0725,right=1.0)
plt.savefig("mercator_cube.png",dpi=200)

# make 3d plots
pix_index=6
indices=np.arange((4**(pix_index-1))*6)
e_lat,e_lon=pix_to_ecliptic_lat_lon(indices,pix_index)
x=np.cos(e_lat)*np.cos(e_lon)  
y=np.cos(e_lat)*np.sin(e_lon)
z=np.sin(e_lat)

fig=plt.figure(figsize=(12,9))
ax=fig.add_subplot(111, projection='3d')

# get plot viewing angles
ux=np.cos(ax.elev)*np.cos(ax.azim)  
uy=np.cos(ax.elev)*np.sin(ax.azim)
uz=np.sin(ax.elev)

# order points by viewing angle
order=np.argsort(-(ux*x+uy*y+uz*z))

# plot indices
im=ax.scatter(x[order],y[order],z[order],c=indices[order],cmap="viridis")
cbar=fig.colorbar(im,ax=ax)
cbar.set_label("$i$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
plt.savefig("indices_3d.png",dpi=200)
