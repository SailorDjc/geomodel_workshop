import copy

from data_structure.reader import ReadExportFile
from utils.plot_utils import *
from utils.vtk_utils import *
from geomodel_analysis import *

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


if __name__ == '__main__':
    get_boreholes_distribution_pic()
    a = 1
