import pyvista
import pyvista as pv
import pyvistaqt as pvq
import torch
import vtkmodules.all as vtk
import numpy as np
import os
from pyvistaqt import MultiPlotter
from retrieve_noddy_files import NoddyModelData
import copy
import matplotlib.pyplot as plt
import pandas as pd
import scipy.spatial as spt
import sklearn
import scipy.spatial as spt
from geograph_parse import GeoMeshGraphParse


# 生成基于规则网格的模型，规则网格已经在geodata中指定
def generate_model_on_base_grid(geodata, cell_values, extent=None, save_path=None):
    if isinstance(cell_values, torch.Tensor):
        cell_values = cell_values.cpu().numpy()
    if cell_values.ndim > 1:
        scalars = np.argmax(cell_values, axis=1)
    else:
        scalars = np.float32(cell_values)
    # if geodata.is_regular_grid:
    # if geodata.sample_grid is not None:
    #     gen_mesh = copy.deepcopy(geodata.sample_grid)
    # else:
    gen_mesh, _ = geodata.create_base_grid(only_out_put=True, extent=extent)
    gen_mesh.cell_data['stratum'] = scalars
    # else:
    #      gen_mesh, _ = geodata.match_unregular_grid_to_regular_grid(cell_density=1, predict_point_label=scalars)
    if save_path is not None:
        gen_mesh.save(filename=save_path)
    return gen_mesh

# cell_field_mesh 外部输入的场模型（cell有属性值）， cell_labels cell的属性值， iso_list：地层列表， stratum：相应属性值取值对应的地层
# return cell_stratum： cell的地层类别，contour：地层列表对应的等值面， sample_grid，场模型，是点渲染， cell_mesh转成cell渲染的模型
def process_point_field_model(cell_field_mesh, cell_labels, iso_list, stratum_match=None, save_path=None):
    points = cell_field_mesh.points
    sample_grid = copy.deepcopy(cell_field_mesh)
    cell_idx = sample_grid.find_containing_cell(points)
    point_label = cell_labels[cell_idx]
    sample_grid.point_data['spatial_points'] = point_label
    sample_grid.set_active_scalars('spatial_points')
    if iso_list is None:
        iso_list = sorted(np.unique(np.round(cell_labels)))
    contour = sample_grid.contour(isosurfaces=iso_list)
    cell_mesh = copy.deepcopy(cell_field_mesh)
    if stratum_match is None:
        cell_labels = np.round(cell_labels)
        for it, scalar in enumerate(cell_labels):
            if cell_labels[it] < min(iso_list):
                cell_labels[it] = min(iso_list)
            if cell_labels[it] > max(iso_list):
                cell_labels[it] = max(iso_list)
        cell_mesh.cell_data['stratum'] = cell_labels
    cell_stratum = copy.deepcopy(cell_labels)
    if stratum_match is not None:
        for cit, cell_label in enumerate(cell_labels):
            sit = 0
            flag = 0
            for it, item in enumerate(iso_list):
                if cell_label < item:
                    flag = 1
                    sit = it
                    break
            if flag == 0:
                sit = len(stratum_match) - 1
            cell_stratum[cit] = stratum_match[sit]
        cell_mesh.cell_data['stratum'] = cell_stratum
    if save_path is not None:
        cell_mesh.save(filename=save_path)
    return sample_grid, contour, cell_mesh, cell_stratum


# 将相应的 key-values表格进行文件导出，例如 x, y, z, label表格，每一列以相应的key-values写入，如 x=np.array,y=...,z=...,label=...
# 默认空格作为分隔符
def export_points_labels_dat_file(file_path=None, **kwargs):
    out_path = file_path
    pd_list = []
    for k, v in kwargs.items():
        pd_n = pd.DataFrame(v)
        pd_list.append(pd_n)
    pd_file = pd.concat(pd_list, axis=1)
    pd_file.dropna(axis=0, how='any')
    pd_file.to_csv(out_path, index=False, header=False, sep='\t')





def visual_bar_picture(labels):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
    plt.barh()


def read_train_loss_log_txt(file_path, **kwargs):
    df = pd.read_table(file_path, sep=',', comment='=', header=None)
    epoch_slice = df.iloc[:, 0].str.split(' ').tolist()
    epoch = [int(item[-1]) for item in epoch_slice]
    train_loss_slice = df.iloc[:, 1].str.split(' ').tolist()
    train_loss = [float(item[-1]) for item in train_loss_slice]
    train_acc_slice = df.iloc[:, 2].str.split(' ').tolist()
    train_acc = [float(item[-1]) for item in train_acc_slice]
    val_loss_slice = df.iloc[:, 3].str.split(' ').tolist()
    val_loss = [float(item[-1]) for item in val_loss_slice]
    val_acc_slice = df.iloc[:, 4].str.split(' ').tolist()
    val_acc = [float(item[-1]) for item in val_acc_slice]
    return epoch, train_loss, val_loss, train_acc, val_acc





def display_noddy_datasets():
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=50, sample_random=False)
    # for item, model_name in enumerate(noddy_models):
    #     model = noddyData.get_grid_model(model_name)
    pre_train_model_list = []
    for item in [0, 1, 6, 9, 42, 7, 16, 21, 24,
                 34]:  # [0, 1, 6, 7, 9, 15, 16, 21, 24, 33, 34, 39, 42, 44]:  [0, 1, 7, 9, 16]
        pre_train_model_list.append(noddy_models[item])
    mesh_list = []
    for item in pre_train_model_list[0:10]:
        mesh = noddyData.get_grid_model(item)  # 1, 6
        mesh_list.append(mesh)
    plotter = pv.Plotter(shape=(2, 5))
    plotter.remove_legend()
    plotter.add_mesh(mesh_list[0].outline(), color="k")
    plotter.add_mesh(mesh_list[0])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_list[1].outline(), color="k")
    plotter.add_mesh(mesh_list[1])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh_list[2].outline(), color="k")
    plotter.add_mesh(mesh_list[2])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(0, 3)
    plotter.add_mesh(mesh_list[3].outline(), color="k")
    plotter.add_mesh(mesh_list[3])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(0, 4)
    plotter.add_mesh(mesh_list[4].outline(), color="k")
    plotter.add_mesh(mesh_list[4])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(1, 0)
    plotter.add_mesh(mesh_list[5].outline(), color="k")
    plotter.add_mesh(mesh_list[5])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(1, 1)
    plotter.add_mesh(mesh_list[6].outline(), color="k")
    plotter.add_mesh(mesh_list[6])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(1, 2)
    plotter.add_mesh(mesh_list[7].outline(), color="k")
    plotter.add_mesh(mesh_list[7])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(1, 3)
    plotter.add_mesh(mesh_list[8].outline(), color="k")
    plotter.add_mesh(mesh_list[8])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.subplot(1, 4)
    plotter.add_mesh(mesh_list[9].outline(), color="k")
    plotter.add_mesh(mesh_list[9])
    plotter.add_axes()
    plotter.camera_position = [-2, 5, 3]
    plotter.show()


def test_0(geodata):
    edge_list = copy.deepcopy(geodata.edge_list)
    new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
    point_idx = geodata.train_idx
    point_label = geodata.unregular_grid_point_label[point_idx]
    visual_edge_list(edge_list=np.array(new_edge_list), edge_points=geodata.unregular_grid_points, e_color='gray',
                     points_idx=point_idx, points_label=point_label, add_points=True, p_size=3)


def test_1(geodata):
    edge_list = copy.deepcopy(geodata.edge_list)
    new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
    point_idx = geodata.train_idx
    point_label = geodata.unregular_grid_point_label[point_idx]
    visual_edge_list(edge_list=np.array(new_edge_list), edge_points=geodata.unregular_grid_points,
                     e_color='gray', points_idx=point_idx, points_label=point_label, add_points=True, p_size=3)


def test_2():
    mesh_list = visual_separate_stratum(
        vtk_file=r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\vtk_model.vtk",
        color_map=[[254, 228, 193, 1], [254, 228, 193, 1], [255, 255, 153], [255, 217, 192, 1], [255, 217, 192, 1],
                   [255, 204, 153, 1], [243, 255, 155, 1], [255, 153, 255, 1], [255, 51, 255, 1], [255, 64, 25, 1]],
        is_explode=True, explode_dist=[0, 0, 20])
    label_map = {0: '素填土', 1: '耕植土', 2: '淤泥质土', 3: '粉砂、细砂', 4: '中砂、粗砂、砾砂', 5: '粘土、粉质粘土、砂质粘土',
                 6: '残积砂质粘土/残积粘性土', 7: '全风化花岗岩', 8: '土状强风化花岗岩', 9: '中风化花岗岩'}


def test_3(geodata):
    _, grid_outline = geodata.get_unregular_grid_points_convexhull_surface(geodata.ori_points)

    drills = visual_sample_data(geodata, is_show=False, drill_radius=1)
    visual_separte_sample_data(drills[0], color_map=[[254, 228, 193, 1], [254, 228, 193, 1], [255, 255, 153],
                                                     [255, 217, 192, 1], [255, 217, 192, 1],
                                                     [255, 204, 153, 1], [243, 255, 155, 1], [255, 153, 255, 1],
                                                     [255, 51, 255, 1], [255, 64, 25, 1]],
                               unique_value_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], box=grid_outline)


def test_4():
    geomodel = get_vtk_mesh_from_file(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\real_world_data\3_12\vtk_model.vtk")
    bound = geomodel.bounds
    slices_x_0 = clip_section_along_axis(geomodel, sample_axis='x', scroll_scale=0.07)
    slices_x_1 = clip_section_along_axis(geomodel, sample_axis='x', scroll_scale=0.5)
    slices_x_2 = clip_section_along_axis(geomodel, sample_axis='x', scroll_scale=0.8)
    slices_y_0 = clip_section_along_axis(geomodel, sample_axis='y', scroll_scale=0.2)
    slices_y_1 = clip_section_along_axis(geomodel, sample_axis='y', scroll_scale=0.5)
    slices_y_2 = clip_section_along_axis(geomodel, sample_axis='y', scroll_scale=0.8)

    visual_separte_sample_data([slices_x_0, slices_x_1, slices_x_2, slices_y_0, slices_y_1, slices_y_2],
                               color_map=[[254, 228, 193, 1], [254, 228, 193, 1], [255, 255, 153],
                                          [255, 217, 192, 1], [255, 217, 192, 1],
                                          [255, 204, 153, 1], [243, 255, 155, 1], [255, 153, 255, 1],
                                          [255, 51, 255, 1], [255, 64, 25, 1]],
                               unique_value_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    visual_multiple_model([slices_x_0, slices_x_1, slices_x_2, slices_y_0, slices_y_1, slices_y_2])
    # _, sample_grid_outline = geodata.match_unregular_grid_to_regular_grid(cell_density=1)


class VisualKit(object):
    def __init__(self, geodata, predict):
        pass


if __name__ == '__main__':
    display_noddy_datasets()
    # display_noddy_datasets()
    mesh_path = r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\interface.vtp"
    mesh = get_vtk_mesh_from_file(mesh_path)
    mesh.plot()
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=50, sample_random=False)
    pre_train_model_list = []
    # for item in [0, 1, 6, 9, 42, 7, 16, 21, 24,
    #              34]:  # [0, 1, 6, 7, 9, 15, 16, 21, 24, 33, 34, 39, 42, 44]:  [0, 1, 7, 9, 16]
    mesh = noddyData.get_grid_model(noddy_models[0])  # 1, 6
    mesh.plot()
    geodata = GeoMeshParse(mesh=mesh)
    geodata.execute(extent=[100, 100, 50])
    train_idx = geodata.extract_interface_points(rarefy_ratio=0.1)
    geodata.export_points_data_to_xml_polydata_file(points=geodata.grid_points[train_idx],
                                                    labels=geodata.grid_point_label[train_idx],
                                                    save_path=r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\interface.vtp')

    test_4()

    # root_path = os.path.abspath('.')
    out_path = r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\dataset"

    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=50, sample_random=False)
    model_list = []
    for item in [0, 1, 6, 9, 42, 7, 16, 21, 24, 34]:
        model_list.append(noddy_models[item])
    for item in model_list[0:10]:
        mesh = noddyData.get_grid_model(item)
        for si in np.arange(3):
            mesh_file_name = item + '_section_' + str(si) + '.data'
            section = clip_section_along_axis(mesh=mesh, scroll_scale=(si + 1) * 0.2)
            out_file_path = os.path.join(out_path, mesh_file_name)
            export_vtk_mesh_to_xyz_label_data(out_file_path=out_file_path, mesh=section)

    path_pr = r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\tran_loss_log.txt"
    _, train_loss_1, val_loss_1, train_acc_1, val_acc_1 = read_train_loss_log_txt(file_path=path_pr)
    plt.figure(figsize=[14, 5])
    plt.plot(train_acc_1, 'bs-', label='train')
    plt.plot(val_acc_1, 'rs-', label='validate')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    path = r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\output"
    # 读取dat 钻孔文件
    path_1 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\复杂模型_model_6_(batch_1024_50drills)\SageModel\tran_loss_log.txt'
    path_2 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\复杂模型_model_6_(batch_1024_50drills)\GraphModel\tran_loss_log.txt'
    path_3 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\复杂模型_model_6_(batch_1024_50drills)\SageTransformer\tran_loss_log.txt'
    path_4 = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\复杂模型_model_6_(batch_1024_50drills)\GraphTransformer\tran_loss_log.txt'
    _, train_loss_1, val_loss_1, train_acc_1, val_acc_1 = read_train_loss_log_txt(file_path=path_1)
    _, train_loss_2, val_loss_2, train_acc_2, val_acc_2 = read_train_loss_log_txt(file_path=path_2)
    _, train_loss_3, val_loss_3, train_acc_3, val_acc_3 = read_train_loss_log_txt(file_path=path_3)
    _, train_loss_4, val_loss_4, train_acc_4, val_acc_4 = read_train_loss_log_txt(file_path=path_4)
    plt.figure(figsize=[14, 5])
    plt.plot(val_acc_1, "ys-", label="GraphSAGE")
    plt.plot(val_acc_2, "bs-", label="SpatialGraph")
    plt.plot(val_acc_3, "gs-", label="GraphSAGE+Transformer")
    plt.plot(val_acc_4, "rs-", label="SpatialGraph+Transformer")
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.legend()
    max_train_acc_1_value = max(val_acc_1)  # 求列表最大值
    max_train_acc_1_idx = val_acc_1.index(max_train_acc_1_value)  # 求最大值对应索引
    max_train_acc_2_value = max(val_acc_2)
    max_train_acc_2_idx = val_acc_2.index(max_train_acc_2_value)
    max_train_acc_3_value = max(val_acc_3)
    max_train_acc_3_idx = val_acc_3.index(max_train_acc_3_value)
    max_train_acc_4_value = max(val_acc_4)
    max_train_acc_4_idx = val_acc_4.index(max_train_acc_4_value)
    print('The {}th epoch, val_1 acc reached the highest value: {}.'.format(max_train_acc_1_idx, max_train_acc_1_value))
    print('The {}th epoch, val_2 acc reached the highest value: {}.'.format(max_train_acc_2_idx, max_train_acc_2_value))
    print('The {}th epoch, val_3 acc reached the highest value: {}.'.format(max_train_acc_3_idx, max_train_acc_3_value))
    print('The {}th epoch, val_4 acc reached the highest value: {}.'.format(max_train_acc_4_idx, max_train_acc_4_value))
    pic_name = 'Val_Acc_pic.jpg'
    save_path = os.path.join(path, pic_name)
    plt.savefig(save_path)
    #
    # # test_grid_regular()
