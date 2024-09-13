
# Make a Volume from a numpy object
#
import numpy as np
from vedo import *

settings.default_backend = '2d' # or k3d, 2d, or vtk

X, Y, Z = np.mgrid[:30, :30, :30]

# scaled distance from the center at (15, 15, 15)
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2)/225/3
print('scalar min, max =', np.min(scalar_field), np.max(scalar_field))

vol = Volume(scalar_field)
lego = vol.legosurface(vmin=.3, vmax=.6)
lego.cmap("hot_r")

show(lego, axes=1, zoom=1.1, viewup='z')
