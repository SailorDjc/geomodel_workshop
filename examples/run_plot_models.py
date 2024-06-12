from utils.plot_utils import control_visibility_with_layer_label
from data_structure.reader import ReadExportFile
from data_structure.geodata import Grid, GeodataSet, Section
from utils.vtk_utils import get_bounds_from_coords, exaggerate_vtk_object
import os
import numpy as np

if __name__ == '__main__':
    reader = ReadExportFile()
    grid_data_0 = reader.read_geodata(file_path=r"E:\11-22-GeoSci\geomodel_workshop-main\output\vtk_sec_model_0")
    grid_data_0.vtk_data.plot()
    # aa = grid_data_0.vtk_data['stratum']
    # bb = np.add(aa, 1)
    # grid_data_0.vtk_data['stratum'] = bb
    # grid_data_0.vtk_data = exaggerate_vtk_object(vtk_object=grid_data_0.vtk_data, horizontal_x_exaggeration=2,
    #                                              horizontal_y_exaggeration=5, vertical_exaggeration=1)
    label_dict = {'1-1': 0, '1-2': 1, '1-3': 2, '2-41': 3, '2-122': 4, '2-153': 5, '4-41': 6, '4-133': 7, '5-41': 8,
                  '5-122': 9, '5-161': 10, '10-113': 11, '23-21': 12, '23-12': 13, '23-13': 14, '87-12': 15,
                  '87-13': 16, '118-11': 17, '118-12': 18, '118-13': 19, '118-22': 20, '118-23': 21, '118-32': 22,
                  '118-33': 23, '119-11': 24, '119-12': 25, '119-13': 26, '119-22': 27, '119-23': 28, '119-32': 29,
                  '119-33': 30, '119-52': 31, '119-53': 32, '123-12': 33, '123-13': 34, '123-23': 35, '124-11': 36,
                  '124-12': 37, '124-13': 38, '124-23': 39}
    text_info = {'1-1': '杂填土', '1-2': '素填土', '1-3': '填筑土', '2-41': '粉质黏土', '2-122': '细角砾土',
                 '2-153': '粗圆砾土'
        , '4-41': '粉质黏土', '4-133': '细圆砾土', '5-41': '粉质黏土',
                 '5-122': '细角砾土', '5-161': '碎石土', '10-113': '细圆砾土', '23-21': '断层角砾', '23-12': '强风化石英闪长岩'
        , '23-13': '弱风化石英闪长岩', '87-12': '强风化煌斑岩',
                 '87-13': '弱风化煌斑岩', '118-11': '全风化黑云斜长片麻岩', '118-12': '强风化黑云斜长片麻岩'
        , '118-13': '弱风化黑云斜长片麻岩', '118-22': '强风化斜长角闪岩', '118-23': '弱风化斜长角闪岩'
        , '118-32': '强风化变粒岩', '118-33': '弱风化变粒岩', '119-11': '全风化黑云斜长片麻岩'
        , '119-12': '强风化黑云斜长片麻岩', '119-13': '弱风化黑云斜长片麻岩', '119-22': '强风化斜长角闪岩'
        , '119-23': '弱风化黑云斜长角闪岩', '119-32': '强风化变粒岩', '119-33': '弱风化变粒岩', '119-52': '强风化石英岩'
        , '119-53': '弱风化麻粒岩', '123-12': '强风化黑云二长片麻岩', '123-13': '弱风化黑云二长片麻岩'
        , '123-23': '弱风化斜长角闪岩', '124-11': '黑云斜长片麻岩', '124-12': '强风化黑云斜长片麻岩'
        , '124-13': '弱风化黑云斜长片麻岩', '124-23': '弱风化斜长角闪岩'}
    root_path = os.path.abspath('..')
    # # 这是一个剖面的钻孔密采样
    file_boreholes_path = os.path.join(root_path, 'data', '20m_VirtualDrill.xlsx')
    sec_boreholes = reader.tmp_read_virtual_boreholes(dat_file_path=file_boreholes_path)
    bounds = [507200, 508500, 4299149.58, 4303055.14, 306.088, 960.074]
    sec_boreholes_seg1 = sec_boreholes.search_by_rect2d(rect2d=bounds)
    #
    top_points = sec_boreholes_seg1.get_top_points()

    line_points_sort_ind = np.lexsort((top_points[:, 1 - 0]
                                       , top_points[:, 0]))
    line_points_sorted = top_points[line_points_sort_ind]

    section = Section()
    grid_bounds = get_bounds_from_coords(sec_boreholes_seg1.get_points_data().points)
    vtk_sec = section.create_surface_by_sweepline(trajectory_line_xy=line_points_sorted
                                                  , grid_bounds=grid_bounds
                                                  , resolution_xy=2
                                                  , resolution_z=0.5)
    out = section.prob_volume(grid=grid_data_0, surf=vtk_sec)


    # secobj = Grid(grid_vtk=out)
    # plot = control_visibility_with_layer_label(geo_object_list=[secobj]
    #                                            , font_file=None)
    #     plot.show()
    # plot.show()
    cc = out['stratum']
    dd = np.subtract(cc, 1)
    out['stratum'] = dd
    sec_grid = Grid(grid_vtk=out)
    geod = GeodataSet()
    # geod.append(boreholes)
    geod.append(sec_boreholes_seg1)
    geod.standardize_labels(label_dict=label_dict)
    label_dict_reverse = {}
    for k, v in geod.label_dict.items():
        label_dict_reverse[v] = k
    boreholes = geod.geodata_list[0]
    font_file_path = os.path.join(root_path, 'utils', 'SanJiZiHaiSongGBK-2.ttf')
    plot = control_visibility_with_layer_label([sec_grid, boreholes]
                                               , labels_info=label_dict_reverse
                                               , font_file=font_file_path
                                               , text_info=text_info)   # , boreholes
    plot.show()

# if __name__ == '__main__':
#     reader = ReadExportFile()
#     grid_data_0 = reader.read_geodata(file_path=r"E:\11-22-GeoSci\geomodel_workshop-main\output\vtk_sec_model_0")
#
#     # grid_data_0.vtk_data = exaggerate_vtk_object(vtk_object=grid_data_0.vtk_data, horizontal_x_exaggeration=2
#     #                                              , horizontal_y_exaggeration=5, vertical_exaggeration=1)
#
#
#     label_dict = {'1-1': 0, '1-2': 1, '1-3': 2, '2-41': 3, '2-122': 4, '2-153': 5, '4-41': 6, '4-133': 7, '5-41': 8,
#                   '5-122': 9, '5-161': 10, '10-113': 11, '23-21': 12, '23-12': 13, '23-13': 14, '87-12': 15,
#                   '87-13': 16, '118-11': 17, '118-12': 18, '118-13': 19, '118-22': 20, '118-23': 21, '118-32': 22,
#                   '118-33': 23, '119-11': 24, '119-12': 25, '119-13': 26, '119-22': 27, '119-23': 28, '119-32': 29,
#                   '119-33': 30, '119-52': 31, '119-53': 32, '123-12': 33, '123-13': 34, '123-23': 35, '124-11': 36,
#                   '124-12': 37, '124-13': 38, '124-23': 39}
#     text_info = {'1-1': '杂填土', '1-2': '素填土', '1-3': '填筑土', '2-41': '粉质黏土', '2-122': '细角砾土',
#                  '2-153': '粗圆砾土'
#         , '4-41': '粉质黏土', '4-133': '细圆砾土', '5-41': '粉质黏土',
#                  '5-122': '细角砾土', '5-161': '碎石土', '10-113': '细圆砾土', '23-21': '断层角砾', '23-12': '强风化石英闪长岩'
#         , '23-13': '弱风化石英闪长岩', '87-12': '强风化煌斑岩',
#                  '87-13': '弱风化煌斑岩', '118-11': '全风化黑云斜长片麻岩', '118-12': '强风化黑云斜长片麻岩'
#         , '118-13': '弱风化黑云斜长片麻岩', '118-22': '强风化斜长角闪岩', '118-23': '弱风化斜长角闪岩'
#         , '118-32': '强风化变粒岩', '118-33': '弱风化变粒岩', '119-11': '全风化黑云斜长片麻岩'
#         , '119-12': '强风化黑云斜长片麻岩', '119-13': '弱风化黑云斜长片麻岩', '119-22': '强风化斜长角闪岩'
#         , '119-23': '弱风化黑云斜长角闪岩', '119-32': '强风化变粒岩', '119-33': '弱风化变粒岩', '119-52': '强风化石英岩'
#         , '119-53': '弱风化麻粒岩', '123-12': '强风化黑云二长片麻岩', '123-13': '弱风化黑云二长片麻岩'
#         , '123-23': '弱风化斜长角闪岩', '124-11': '黑云斜长片麻岩', '124-12': '强风化黑云斜长片麻岩'
#         , '124-13': '弱风化黑云斜长片麻岩', '124-23': '弱风化斜长角闪岩'}
#     root_path = os.path.abspath('..')
#     # 这是一个剖面的钻孔密采样
#     file_boreholes_path = os.path.join(root_path, 'data', '20m_VirtualDrill.xlsx')
#     sec_boreholes = reader.tmp_read_virtual_boreholes(dat_file_path=file_boreholes_path)
#     bounds = [507200, 508500, 4299149.58, 4303055.14, 306.088, 960.074]
#     sec_boreholes_seg1 = sec_boreholes.search_by_rect2d(rect2d=bounds)
#     geod = GeodataSet()
#     # geod.append(boreholes)
#     geod.append(sec_boreholes_seg1)
#     geod.standardize_labels(label_dict=label_dict)
#     label_dict_reverse = {}
#     for k, v in geod.label_dict.items():
#         label_dict_reverse[v] = k
#     boreholes = geod.geodata_list[0]
#     font_file_path = os.path.join(root_path, 'utils', 'SanJiZiHaiSongGBK-2.ttf')
#
#     # top_points = sec_boreholes_seg1.get_top_points()
#     #
#     # line_points_sort_ind = np.lexsort((top_points[:, 1 - 0]
#     #                                    , top_points[:, 0]))
#     # line_points_sorted = top_points[line_points_sort_ind]
#     #
#     # section = Section()
#     # grid_bounds = get_bounds_from_coords(sec_boreholes_seg1.get_points_data().points)
#     # vtk_sec = section.create_surface_by_sweepline(trajectory_line_xy=line_points_sorted
#     #                                               , grid_bounds=grid_bounds
#     #                                               , resolution_xy=2
#     #                                               , resolution_z=0.5)
#     # out = section.prob_volume(grid=grid_data_0, surf=vtk_sec)
#     # out.save(filename=os.path.join(root_path, 'output', 'sec.vtk'))
#     # import pyvista as pv
#     # plotter = pv.Plotter()
#     # plotter.add_mesh(out)
#     # plotter.add_mesh(grid_data_0.vtk_data)
#     # plotter.show()
#     sec = reader.read_vtk_data(file_path=os.path.join(root_path, 'output', 'sec.vtk'))
#     secobj = Grid(grid_vtk=sec)
#     plot = control_visibility_with_layer_label(geo_object_list=[secobj]
#                                                , labels_info=label_dict_reverse
#                                                , font_file=None
#                                                , text_info=text_info)
#     plot.show()
