import random
import tarfile

import numpy as np
import pyvista as pv
import pynoddy.output
import pynoddy.history
import os
import gzip
import time
from tqdm import tqdm
import pickle
import copy

# path = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\test'
# model_name = '20-09-04-16-00-26-664297926'
#
# output_dir = path
# his_file = os.path.join(path, model_name) + '.his'
# output_path = os.path.join(output_dir, model_name)
# pynoddy.compute_model(his_file, output_path)
# pynoddy.compute_model(his_file, output_path, sim_type='GEOPHYSICS')
# nout = pynoddy.output.NoddyOutput(output_path)
# nout_geophysics = pynoddy.output.NoddyGeophysics(output_path)
# # nout.export_to_vtk()
# nout_geophysics.grv_data
# vtr_path = os.path.join(path, model_name) + '.vtr'
# vtr_model = pv.read(vtr_path)
# vtr_model.plot()

from pyvista import set_plot_theme
set_plot_theme('document')
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvq
dataset = examples.load_uniform()
dataset.set_active_scalars("Spatial Point Data")
dataset.plot()

result = dataset.threshold().elevation().clip(normal="z").slice_orthogonal()
result_1 = dataset.threshold().elevation().clip(normal='z')

result_1.plot()
p = pv.Plotter()
p.add_mesh(dataset.outline(), color="k")
p.add_mesh(result, scalars="Elevation")
p.view_isometric()
p.show()

print('dataset:', dataset)
# Apply a threshold over a data range
threshed = dataset.threshold([100, 500])
threshed.plot()
outline = dataset.outline()
outline.plot()
p = pv.Plotter()
p.add_mesh(outline, color="k")
p.add_mesh(threshed)
p.camera_position = [-2, 5, 3]
p.show()

contours = dataset.contour()
contours.plot()

slices = dataset.slice_orthogonal()
slices.plot()
glyphs = dataset.glyph(factor=1e-3, geom=pv.Sphere())

p = pv.Plotter(shape=(2, 2))
# Show the threshold
p.add_mesh(outline, color="k")
p.add_mesh(threshed, show_scalar_bar=False)
p.camera_position = [-2, 5, 3]
# Show the contour
p.subplot(0, 1)
p.add_mesh(outline, color="k")
p.add_mesh(contours, show_scalar_bar=False)
p.camera_position = [-2, 5, 3]
# Show the slices
p.subplot(1, 0)
p.add_mesh(outline, color="k")
p.add_mesh(slices, show_scalar_bar=False)
p.camera_position = [-2, 5, 3]
# Show the glyphs
p.subplot(1, 1)
p.add_mesh(outline, color="k")
p.add_mesh(glyphs, show_scalar_bar=False)
p.camera_position = [-2, 5, 3]

p.link_views()
p.show()

# def f(x, y, z):
#     return 2 * x ** 3 + 3 * y ** 2 - z
#
# x = np.linspace(1, 4, 11)
# y = np.linspace(4, 7, 22)
# z = np.linspace(7, 9, 33)
# xg, yg, zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
# data = f(xg, yg, zg)
# pnt_xyz = (x, y, z)
# interp = RegularGridInterpolator(pnt_xyz, data)
# pts = np.array([[2.1, 6.2, 8.3],
#                 [3.3, 5.2, 7.1]])
# values = interp(pts)
# rng = np.random.default_rng()
# xobs = 2 * Halton(2, seed=rng).random(100) - 1
# yobs = np.sum(xobs, axis=1) * np.exp(-6 * np.sum(xobs ** 2, axis=1))
# xgrid = np.mgrid[-1:1:50j, -1:1:50j]
# xflat = xgrid.reshape(2, -1).T
# yflat = RBFInterpolator(xobs, yobs)(xflat)
# ygrid = yflat.reshape(50, 50)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
# p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
# fig.colorbar(p)
# plt.show()
