from data_structure.terrain import TerrainData
from data_structure.reader import ReadExportFile
from pyvista import examples
import pyvista as pv
import numpy as np
from utils.vtk_utils import add_terrain_to_base_grid, create_vtk_grid_by_rect_bounds
import os

if __name__ == '__main__':
    mask_bounds = [555000, 556000, 3501000, 3502000, -100, 579]
    tiff_path = r"E:\MyDataset\ASTGTMV003_N31E117\ASTGTMV003_N31E117_dem.tif"
    save_path = os.path.join(tiff_path, '../')
    terrain = TerrainData()
    terrain.set_input_tiff_file(file_path=tiff_path)
    terrain.execute(mask_bounds=mask_bounds)
    # bounds = terrain.bounds
    # bounds[4] = -100
    # terrain.extend_mesh_from_surface_by_bounds(bounds=bounds)
    # terrain.vtk_data.plot()
    # terrain.save(dir_path=save_path, out_name='terrain')
    grid_bounds = [555000, 556000, 3501000, 3502000, -100, 559]
    base_grid = create_vtk_grid_by_rect_bounds(dim=np.array([100, 100, 80]), bounds=np.array(grid_bounds))
    add_terrain_to_base_grid(terrain=terrain, base_grid=base_grid)
    # aa = 1
    # bb = aa + 1

    # reader = ReadExportFile()
    # file_path = r"E:\MyDataset\ASTGTMV003_N31E117\terrain"
    # terrain = reader.read_geodata(file_path=file_path)
    #
    # print(terrain)
    # bounds = terrain.bounds()
    # bounds[4] = -100
    # terrain.extend_mesh_from_surface(bounds=bounds)
    # aa = 1
    # bb = aa + 1
    # terrain.vtk_data.plot()

    # pImg = vtkImageData()
    # pImg.SetDimensions(z_matrix.shape[0] + 1, z_matrix.shape[1] + 1, 1)
    # vtk_arr = numpy_support.numpy_to_vtk(z_matrix)
    # pImg.GetPointData().SetScalars(vtk_arr)
    # pimage = pv.wrap(pImg)
    # pimage.plot()