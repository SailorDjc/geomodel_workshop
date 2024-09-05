from data_structure.reader import ReadExportFile
from utils.plot_utils import *
from utils.vtk_utils import *

root_dir = os.path.abspath('..')
if __name__ == '__main__':
    reader = ReadExportFile()
    surf_file_path = os.path.join(root_dir, 'data', 'error_surf.vtk')
    surf = reader.read_vtk_data(file_path=surf_file_path)
    grid_data_path = os.path.join(root_dir, 'data', 'grid_data', 'gme_base_grid', 'gme_base_grid.dat')
    grid_data = reader.read_geodata(file_path=grid_data_path)
    cell_ids = poly_surf_intersect_with_grid(poly_surf=surf, grid=grid_data.vtk_data,
                                             check_level=0)