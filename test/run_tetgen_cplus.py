from data_structure.reader import ReadExportFile, WriteExportFile
from data_structure.geodata import GeodataSet, Grid, BoreholeSet, Borehole
from dgl_geodataset import DglGeoDataset
from geomodel_analysis import GmeModelGraphList, GeoDataMLClassifier
from gme_trainer import GmeTrainer, GmeTrainerConfig, GraphTransConfig
from models.model import GraphTransfomer
from data_structure.terrain import TerrainData
import numpy as np
import os


root_path = os.path.abspath('..')

reader = ReadExportFile()

file_sec_path = os.path.join(root_path, 'data', '测试样本数据_钻孔.xlsx')
sec_boreholes = reader.tmp_read_boreholes(excel_path=file_sec_path)
# bounds_seg_1 = [506250, 507210, 4299149.58, 4303055.14, 306.088, 960.074]
# boreholes_select = sec_boreholes.search_by_rect2d(rect2d=bounds_seg_1)
geodataset = GeodataSet()
# geod.append(boreholes)
geodataset.append(sec_boreholes)
label_dict = {'1-1': 0, '1-2': 1, '1-3': 2, '2-41': 3, '2-122': 4, '2-153': 5, '4-41': 6, '4-133': 7, '5-41': 8,
              '5-122': 9, '5-161': 10, '10-113': 11, '23-21': 12, '23-12': 13, '23-13': 14, '87-12': 15,
              '87-13': 16, '118-11': 17, '118-12': 18, '118-13': 19, '118-22': 20, '118-23': 21, '118-32': 22,
              '118-33': 23, '119-11': 24, '119-12': 25, '119-13': 26, '119-22': 27, '119-23': 28, '119-32': 29,
              '119-33': 30, '119-52': 31, '119-53': 32, '123-12': 33, '123-13': 34, '123-23': 35, '124-11': 36,
              '124-12': 37, '124-13': 38, '124-23': 39}
geodataset.standardize_labels(label_dict=label_dict)
top_points_seg = sec_boreholes.get_top_points()
bot_points_seg = sec_boreholes.get_bottom_points()
terr_seg = TerrainData()
# 范围约束，按照剖面线缓冲，生成线状建模区域
terr_seg.set_boundary_from_line_buffer(trajectory_line_xy=top_points_seg, buffer_dist=100)
terr_seg.set_control_points(top_points_seg)
terr_seg.execute(clip_reconstruct=True, simplify=0.005, top_surf_offset=3)
z_min = min(bot_points_seg[:, 2])
out_bounds = geodataset.geodata_list[0].get_points_data().bounds
grid_poly = terr_seg.create_grid_from_terrain_surface(z_min=z_min-2, only_closed_poly_surface=True)
# grid_poly.plot(show_edges=True)
# grid_poly.triangulate().plot(show_edges=True)
grid_poly = grid_poly
# grid_poly.plot(show_edges=True)
grid_poly = grid_poly.triangulate()  # .plot(show_edges=True)

import pyvista as pv
boreholes_points_data = sec_boreholes.get_points_data()
# plotter = pv.Plotter()
# plotter.add_mesh(grid_poly)
pots = pv.PolyData(boreholes_points_data.points)
# plotter.add_mesh(pots, color='red')
# plotter.show()

selected = pots.select_enclosed_points(grid_poly, tolerance=0.000000001)
cell_indices = selected.point_data['SelectedPoints']
insert_cell_indices = np.argwhere(cell_indices <= 0).flatten()

grid_poly.save(os.path.join(root_path, 'output', 'boundary_poly.vtk'), binary=False)

writer = WriteExportFile()
writer.write_nodes(points_data=boreholes_points_data, file_path=os.path.join(root_path, 'output', 'sec_nodes.node'))

