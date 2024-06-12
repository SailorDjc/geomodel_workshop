import pyvista as pv
import tetgen
import numpy as np

## Prepare Your PLC and Background Mesh:
# Load or create your PLC
sphere = pv.Sphere(theta_resolution=10, phi_resolution=10)

# Generate a background mesh with desired resolution
def generate_background_mesh(bounds, resolution=20, eps=1e-6):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(x_min - eps, x_max + eps, resolution),
        np.linspace(x_min - eps, y_max + eps, resolution),
        np.linspace(z_min - eps, z_max + eps, resolution),
        indexing="ij",
    )
    return pv.StructuredGrid(grid_x, grid_y, grid_z).triangulate()

bg_mesh = generate_background_mesh(sphere.bounds)

## Define the Sizing Function and Write to Disk:
# Define sizing function based on proximity to a point of interest
def sizing_function(points, focus_point=np.array([0, 0, 0]), max_size=1.0, min_size=0.1):
    distances = np.linalg.norm(points - focus_point, axis=1)
    return np.clip(max_size - distances, min_size, max_size)

bg_mesh.point_data['target_size'] = sizing_function(bg_mesh.points)

# Optionally write out the background mesh
def write_background_mesh(background_mesh, out_stem):
    """Write a background mesh to a file.

    This writes the mesh in tetgen format (X.b.node, X.b.ele) and a X.b.mtr file
    containing the target size for each node in the background mesh.
    """
    mtr_content = [f"{background_mesh.n_points} 1"]
    target_size = background_mesh.point_data["target_size"]
    for i in range(background_mesh.n_points):
        mtr_content.append(f"{target_size[i]:.8f}")

write_background_mesh(bg_mesh, 'bgmesh.b')

## Use TetGen with the Background Mesh:
# Directly pass the background mesh from PyVista to tetgen:
tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
tet = tetgen.TetGen(mesh)
tet.tetrahedralize(bgmesh=bg_mesh, **tet_kwargs)
refined_mesh = tet.grid
# Alternatively, use the background mesh files.
tet = tetgen.TetGen(sphere)
tet.tetrahedralize(bgmeshfilename='bgmesh.b', **tet_kwargs)
refined_mesh = tet.grid
# This example demonstrates generating a background mesh,
# defining a spatially varying sizing function, and using
# this background mesh to guide TetGen in refining a PLC.
# By following these steps, you can achieve adaptive mesh
# refinement tailored to your specific simulation requirements.