import tetgen
import pyvista as pv
import numpy as np
import pytetgen

if __name__ == '__main__':
    # sphere = pv.Sphere()
    # tet = tetgen.TetGen(sphere)
    # tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    # grid = tet.grid
    # grid.plot(show_edges=True)

    N = 50
    points = np.random.random(3 * N).reshape(N, 3)
    points_data = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(points_data, render_points_as_spheres=True)
    plotter.show()
