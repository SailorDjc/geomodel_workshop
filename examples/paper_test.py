import copy
import os.path

from geomodel_analysis import *
from data_structure.reader import ReadExportFile
from utils.plot_utils import *
from utils.vtk_utils import *
from geomodel_analysis import *
from data_structure.geodata import *
from data_structure.data_sampler import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

root_dir = os.path.abspath('..')
reader = ReadExportFile()


def get_acc_pic():
    loss_data_dir = os.path.join(root_dir, '虚拟数据2', '训练过程记录')
    loss_data_path_1 = os.path.join(loss_data_dir, 'train_loss_log_1.txt')
    loss_data_path_2 = os.path.join(loss_data_dir, 'train_loss_log_2.txt')
    loss_data_path_3 = os.path.join(loss_data_dir, 'train_loss_log_3.txt')
    loss_data_path_4 = os.path.join(loss_data_dir, 'train_loss_log_4.txt')
    loss_data_1 = reader.read_train_loss_log(txt_file_path=loss_data_path_1, sep=',')
    loss_data_2 = reader.read_train_loss_log(txt_file_path=loss_data_path_2, sep=',')
    loss_data_3 = reader.read_train_loss_log(txt_file_path=loss_data_path_3, sep=',')
    loss_data_4 = reader.read_train_loss_log(txt_file_path=loss_data_path_4, sep=',')
    plt.figure(figsize=[14, 5])
    # valid accuracy
    plt.plot(loss_data_1['epochs'], loss_data_1['valid_accuracy'], "red", label="spatial correlation")  # bs-
    plt.plot(loss_data_2['epochs'], loss_data_2['valid_accuracy'], "green", label="gaussian kernel")
    plt.plot(loss_data_3['epochs'], loss_data_3['valid_accuracy'], "yellow", label="pseudo coordinate")
    plt.plot(loss_data_4['epochs'], loss_data_4['valid_accuracy'], "blue", label="base SAGEConv mean")
    plt.title(label='Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()
    pic_path = os.path.join(loss_data_dir, 'acc_pic.jpg')
    plt.savefig(pic_path)


def get_loss_pic():
    loss_data_dir = os.path.join(root_dir, '虚拟数据2', '训练过程记录')
    loss_data_path_1 = os.path.join(loss_data_dir, 'train_loss_log_1.txt')
    loss_data_path_2 = os.path.join(loss_data_dir, 'train_loss_log_2.txt')
    loss_data_path_3 = os.path.join(loss_data_dir, 'train_loss_log_3.txt')
    loss_data_path_4 = os.path.join(loss_data_dir, 'train_loss_log_4.txt')
    loss_data_1 = reader.read_train_loss_log(txt_file_path=loss_data_path_1, sep=',')
    loss_data_2 = reader.read_train_loss_log(txt_file_path=loss_data_path_2, sep=',')
    loss_data_3 = reader.read_train_loss_log(txt_file_path=loss_data_path_3, sep=',')
    loss_data_4 = reader.read_train_loss_log(txt_file_path=loss_data_path_4, sep=',')
    plt.figure(figsize=[14, 5])
    # valid loss
    plt.plot(loss_data_1['epochs'], loss_data_1['valid_loss'], "red", label="spatial correlation")  # bs-
    plt.plot(loss_data_2['epochs'], loss_data_2['valid_loss'], "green", label="gaussian kernel")
    plt.plot(loss_data_3['epochs'], loss_data_3['valid_loss'], "yellow", label="pseudo coordinate")
    plt.plot(loss_data_4['epochs'], loss_data_4['valid_loss'], "blue", label="base SAGEConv mean")
    plt.title(label='Validation Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    pic_path = os.path.join(loss_data_dir, 'loss_pic.jpg')
    plt.savefig(pic_path)


def get_boreholes_distribution_pic():
    base_grid = reader.read_geodata(
        file_path=r"E:\PyCode\geomodel_workshop\小论文实验\虚拟数据2\训练过程记录\tmp_graph1723723298\GeoGrid\GeoGrid.dat")
    boreholes_path = os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录'
                                  , 'tmp_graph1723723298', 'tmp_hole1723723298'
                                  , 'tmp_hole1723723298.dat')
    boreholes = reader.read_geodata(file_path=boreholes_path)
    # boreholes.update_holelayers()
    # boreholes.generate_vtk_data_as_tube(borehole_radius=10)
    boreholes.vtk_data.plot()
    # boreholes.save(dir_path=r"E:\PyCode\geomodel_workshop\小论文实验\虚拟数据2\训练过程记录\tmp_graph1723723298", replace=True)
    # boreholes.generate_vtk_data_as_tube(borehole_radius=10, is_tube=True)
    base_grid.vtk_data.plot()

    grid_bounds = base_grid.bounds
    axis_z = (grid_bounds[5] - grid_bounds[4]) * 0.999 + grid_bounds[4]
    center = base_grid.center
    center[2] = axis_z
    top_sec = base_grid.vtk_data.slice(normal='z', origin=center)
    plotter_0 = pv.Plotter()
    actor = plotter_0.add_mesh(base_grid.vtk_data)
    lookup_table = actor.mapper.lookup_table
    plotter_1 = pv.Plotter()
    plotter_1.add_mesh(top_sec, cmap=lookup_table)
    plotter_1.add_mesh(boreholes.vtk_data)
    plotter_1.show()

    boreholes_pos = boreholes.get_top_points()
    root_path = os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录')
    gme_models = GmeModelGraphList('gme_model', root=root_dir, processed_dir=root_path)
    gme_models.load_geograph(graph_id=0, dir_path=os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录'))
    geograph = gme_models.geograph[0]
    borehole_split_ids = geograph.data_sampler.geo_sample_data_val_map
    borehole_train_ids = borehole_split_ids[0]['train']
    borehole_val_ids = borehole_split_ids[0]['val']
    borehole_test_ids = borehole_split_ids[0]['test']

    plotter = pv.Plotter()
    train_points = pv.PolyData(boreholes_pos[borehole_train_ids, :])
    plotter.add_mesh(train_points, color='red', render_points_as_spheres=True, point_size=10)
    val_points = pv.PolyData(boreholes_pos[borehole_val_ids, :])
    plotter.add_mesh(val_points, color='green', render_points_as_spheres=True, point_size=10)
    test_points = pv.PolyData(boreholes_pos[borehole_test_ids, :])
    plotter.add_mesh(test_points, color='blue', render_points_as_spheres=True, point_size=10)
    plotter.add_mesh(top_sec, cmap=lookup_table)
    plotter.add_mesh(boreholes.vtk_data)
    plotter.add_mesh(base_grid.vtk_data.outline(), color='black')
    plotter.set_background('white')
    plotter.camera_position = [0, 0, 1]
    # pic_path = os.path.join(root_dir, '虚拟数据2', '训练过程记录', 'boreholes_2d_pic.jpg')
    # plotter.save_graphic(filename=pic_path)
    plotter.show()


def get_boreholes_ids():
    boreholes_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\tmp_graph1725936461\tmp_geo1725936463\tmp_hole1725936463\tmp_hole1725936463.dat"
    boreholes = reader.read_geodata(file_path=boreholes_path)
    boreholes_pos = boreholes.get_top_points()
    root_path = os.path.join(root_dir, '真实数据1', 'plain1')
    gme_models = GmeModelGraphList('gme_model', root=root_dir, processed_dir=root_path)
    gme_models.load_geograph(graph_id=0, dir_path=root_path)
    geograph = gme_models.geograph[0]
    borehole_split_ids = geograph.data_sampler.geo_sample_data_val_map
    borehole_train_ids = borehole_split_ids[0]['train']
    borehole_val_ids = borehole_split_ids[0]['val']
    borehole_test_ids = borehole_split_ids[0]['test']
    # plotter = pv.Plotter()

    grid_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\vtk_model\vtk_model.dat"
    borehole_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\tmp_graph1725936461\tmp_geo1725936463\tmp_hole1725936463\tmp_hole1725936463.dat"
    grid_data = reader.read_geodata(file_path=grid_path)
    boreholes = reader.read_geodata(file_path=borehole_path)
    plotter = control_visibility_with_layer_label([grid_data, boreholes])


    train_points = pv.PolyData(boreholes_pos[borehole_train_ids[0], :])
    plotter.add_point_labels(train_points, borehole_train_ids[0], render_points_as_spheres=True, point_size=10
                             , font_size=10, point_color='red')
    # val_points = pv.PolyData(boreholes_pos[borehole_val_ids, :])
    # plotter.add_point_labels(val_points, borehole_val_ids, point_color='green'
    #                          , render_points_as_spheres=True, point_size=10, font_size=10)
    test_points = pv.PolyData(boreholes_pos[borehole_test_ids[0], :])
    plotter.add_point_labels(test_points, borehole_test_ids[0], point_color='blue'
                             , render_points_as_spheres=True, point_size=10, font_size=10)

    # grid_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\vtk_model\vtk_model.dat"
    # output_grid = reader.read_geodata(grid_path)
    # plotter.add_mesh(output_grid.vtk_data)
    # plotter.set_background('white')
    plotter.show()


base_grid_path = os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录', 'tmp_graph1723723298', 'GeoGrid', 'GeoGrid.dat')
base_grid_path_1 = os.path.join(root_dir, '小论文实验', '虚拟数据2的输出模型', 'output_model_1', 'output_model_1.dat')
base_grid_path_2 = os.path.join(root_dir, '小论文实验', '虚拟数据2的输出模型', 'output_model_2', 'output_model_1.dat')
base_grid_path_3 = os.path.join(root_dir, '小论文实验', '虚拟数据2的输出模型', 'output_model_3', 'output_model_1.dat')
base_grid_path_4 = os.path.join(root_dir, '小论文实验', '虚拟数据2的输出模型', 'output_model_4',
                                'output_model_1_tmp_grid1723790917.dat')


def get_section(grid_data_path, scalar_name='stratum'):
    # boreholes_path = os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录'
    #                               , 'tmp_graph1723723298', 'tmp_hole1723723298'
    #                               , 'tmp_hole1723723298.dat')
    boreholes_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\tmp_graph1725936461\tmp_geo1725936463\tmp_hole1725936463\tmp_hole1725936463.dat"
    boreholes = reader.read_geodata(file_path=boreholes_path)
    # boreholes_select = boreholes.get_boreholes(idx=[47, 38, 85, 70, 74, 55, 61, 88, 53, 60, 5, 80, 12, 77])
    # train : 47, 38, 85, 74, 55, 60, 5, 80, 77
    # val: 70, 61, 53
    # test: 88, 12

    # train : 35, 42, 31, 83, 17
    # val: 72, 99
    # test: 10, 86, 87, 76
    plain_sec_1_ids = [32, 50, 68, 54, 102, 63, 20, 65, 72, 70, 8]
    # test: 102, 72
    plain_sec_2_ids = [104, 103, 6, 13, 106, 17, 107, 70, 92]
    # test: 17
    boreholes_select = boreholes.get_boreholes(idx=plain_sec_1_ids)
    section = Section()
    grid_bounds = get_bounds_from_coords(boreholes_select.get_points_data().points)
    top_points = boreholes_select.get_top_points()
    # line_points_sort_ind = np.lexsort((top_points[:, 1]
    #                                    , top_points[:, 0]))
    # line_points_sorted = top_points[line_points_sort_ind]
    x_length = grid_bounds[1] - grid_bounds[0]
    y_length = grid_bounds[3] - grid_bounds[2]
    z_length = grid_bounds[5] - grid_bounds[4]
    resolution_xy = (x_length / 200 + y_length / 200) * 0.5
    resolution_z = z_length / 200
    vtk_section = section.create_surface_by_sweepline(trajectory_line_xy=top_points
                                                      , grid_bounds=grid_bounds
                                                      , resolution_xy=resolution_xy
                                                      , resolution_z=resolution_z)
    grid_data = reader.read_geodata(
        file_path=grid_data_path)
    vtk_sec = section.prob_volume(grid=grid_data, surf=vtk_section, scalar_name=scalar_name)
    plotter = pv.Plotter()
    plotter.add_mesh(vtk_sec)
    boreholes_select.generate_vtk_data_as_tube(borehole_radius=10)
    plotter.add_mesh(boreholes_select.vtk_data)
    plotter.add_point_labels(top_points, [str(id) for id in plain_sec_1_ids], font_size=10)
    plotter.show()


def get_svm_model_result():
    root_path = os.path.join(root_dir, '小论文实验', '虚拟数据2', '训练过程记录')
    classifier = GeoDataMLClassifier(method='svm', is_grid_search=True)
    gme_models = GmeModelGraphList('gme_model', root=root_dir, processed_dir=root_path)
    gme_models.load_geograph(graph_id=0, dir_path=root_path)
    geograph = gme_models.geograph[0]
    train_pts = geograph.train_data_indexes
    val_pts = geograph.val_data_indexes
    test_pts = geograph.test_data_indexes
    grid_points = geograph.base_grid.grid_points
    grid_labels = geograph.base_grid.grid_points_series
    train_data = PointSet(points=grid_points[train_pts], point_labels=grid_labels[train_pts])
    classifier.append_data(data=train_data)
    classifier.execute_train()

    grid_data = reader.read_geodata(
        file_path=base_grid_path)
    grid_result = classifier.predict(grid=grid_data)
    pred_labels = grid_result.labels
    pred_labels_test = pred_labels[test_pts]
    test_labels = grid_labels[test_pts]
    if grid_result is not None:
        grid_result.save(dir_path=os.path.join(root_path), out_name='svm_result')
    acc = accuracy_score(test_labels, pred_labels_test, normalize=False, average='micro')
    f1 = f1_score(test_labels, pred_labels_test, normalize=False, average='micro')
    rec = recall_score(test_labels, pred_labels_test, normalize=False, average='micro')
    print(f'accuracy: {acc}, f1-score: {f1}, recall-score:{rec}')


def plot_grid_with_legend():
    path = r"E:\geomodel_workshop-main\代码备份\geomodel_workshop\小论文实验\虚拟数据2\训练过程记录\svm_result\svm_result.dat"
    grid_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\vtk_model\vtk_model.dat"
    grid = reader.read_geodata(file_path=grid_path)
    grid.vtk_data.set_active_scalars(name='stratum')
    plotter_0 = pv.Plotter()
    actor = plotter_0.add_mesh(grid.vtk_data)
    lookup_table = actor.mapper.lookup_table
    plotter = pv.Plotter()
    plotter = control_visibility_with_layer_label([grid])
    # plotter.add_mesh(grid.vtk_data, show_scalar_bar=False)
    label_list = np.unique(grid.labels)
    label_list = [(str(label), np.array(lookup_table.map_value(label)[0:3], dtype=float)) for label in label_list]
    plotter.add_legend(label_list, loc='lower right', face='rectangle', bcolor='white')
    plotter.add_axes()
    plotter.show()


if __name__ == '__main__':
    get_boreholes_ids()
    grid_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\vtk_model\vtk_model.dat"
    borehole_path = r"E:\gitHouse_new\geomodel_workshop\真实数据1\plain1\tmp_graph1725936461\tmp_geo1725936463\tmp_hole1725936463\tmp_hole1725936463.dat"
    grid_data = reader.read_geodata(file_path=grid_path)
    boreholes = reader.read_geodata(file_path=borehole_path)
    plot = control_visibility_with_layer_label([grid_data, boreholes])
    plot.show()
    get_section(grid_data_path=grid_path)
    # get_boreholes_ids()
    # plot_grid_with_legend()
    # vtk_data = reader.read_vtk_data(file_path=r"C:\Users\Sailor\Desktop\vtk_model.vtk")
    # vtk_data.plot()
    get_boreholes_ids()

    # get_section(
    #     grid_data_path=r"E:\geomodel_workshop-main\代码备份\geomodel_workshop\小论文实验\虚拟数据2\训练过程记录\svm_result\svm_result.dat"
    #     , scalar_name='label')
    # get_svm_model_result()
