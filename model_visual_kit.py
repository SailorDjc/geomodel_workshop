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


# 根据给定采样轴向与采样位置，在给定模型中采样虚拟剖面
def clip_section_along_axis(mesh, sample_axis='x', scroll_scale=0.5):
    center = mesh.center
    if scroll_scale <= 0 or scroll_scale >= 1:
        raise ValueError('scroll must be larger than 0 and less than 1.')
    axis_labels = ['x', 'y', 'z']
    label_to_index = {label: index for index, label in enumerate(axis_labels)}
    ax_index = label_to_index[sample_axis]
    axis_label = sample_axis.lower()
    bounds = mesh.bounds
    pos_axis = bounds[ax_index * 2] + (bounds[ax_index * 2 + 1] - bounds[ax_index * 2]) * scroll_scale
    center[ax_index] = pos_axis
    slice = mesh.slice(normal=axis_label, origin=center)
    return slice


# pos_scale_list  pos_xy_scale_list: ndarray
# 根据给定坐标，在给定模型中采样虚拟钻孔
def sample_virtual_drills(mesh, pos_xy_scale_list, drill_radius=1):
    if max(pos_xy_scale_list[:, 0]) > 1 or max(pos_xy_scale_list[:, 1]) > 1:
        pos_xy_scale_list = sklearn.preprocessing.MinMaxScaler().fit_transform(pos_xy_scale_list)
        pos_xy_scale_list = list(pos_xy_scale_list)
    bounds = mesh.bounds
    pos_x_base = bounds[0] + (bounds[1] - bounds[0])
    pos_y_base = bounds[2] + (bounds[3] - bounds[2])
    z_min = bounds[4]
    z_max = bounds[5]
    grid_points = mesh.cell_centers().points
    grid_scalars = mesh.active_scalars
    drill_list = pv.MultiBlock()
    for scroll_scale in pos_xy_scale_list:
        pos_x = pos_x_base * scroll_scale[0]
        pos_y = pos_y_base * scroll_scale[1]
        pos_a = [pos_x, pos_y, z_max]
        pos_b = [pos_x, pos_y, z_min]
        drill_cells = mesh.find_cells_along_line(pointa=pos_a, pointb=pos_b)
        drill = mesh.sample_over_line(pointa=pos_a, pointb=pos_b, resolution=len(drill_cells))
        drill_point_idx = mesh.find_containing_cell(drill.points)
        drill_scalars = grid_scalars[drill_point_idx]
        drill['stratum'] = drill_scalars
        drill = drill.tube(radius=drill_radius)
        drill.plot()
        drill_list.append(drill)
    return drill_list


# 获取vtk模型文件
def get_vtk_mesh_from_file(file):
    vtk_mesh = pv.read(file)
    return vtk_mesh


# 将vtk格式的模型数据导出为 x, y, z, label 的采样点数据
def export_vtk_mesh_to_xyz_label_data(out_file_path, vtk_file=None, mesh=None, drill_num=0):
    if vtk_file is not None:
        mesh = get_vtk_mesh_from_file(vtk_file)
    if mesh is None:
        raise ValueError('Mesh data is empty.')
    mesh_points = mesh.cell_centers().points
    mesh_point_labels = mesh.active_scalars
    export_points_labels_dat_file(file_path=out_file_path, mesh_points=mesh_points, mesh_point_labels=mesh_point_labels)
    if drill_num > 0:
        pass


# 输入模型的标签必须是自然数, label_name 是用于制作图例，将编码匹配到地层名称上
def visual_separate_stratum(vtk_file=None, mesh=None, label_map=None, color_map=None, max_label_num=15, is_show=True,
                            is_save=False, out_file_path=None, is_explode=False, explode_dist=None):
    geo_model = mesh
    if vtk_file is not None:
        if geo_model is None:
            geo_model = get_vtk_mesh_from_file(vtk_file)
    labels = geo_model.active_scalars
    unique_labels = sorted(np.unique(labels))
    # 若模型的地层类别小于设置的最大地层类别数，则模型的地层标签是正常的，否则视为场模型，将不进行分地层处理
    epsilon = 0.1
    if len(unique_labels) <= max_label_num:
        if color_map is None or len(color_map) != len(unique_labels):
            color_map = plt.cm.get_cmap('viridis', len(unique_labels))
            color_list = color_map.colors
        else:
            color_list = color_map
        plotter = pv.Plotter()
        geo_model_stratum_list = []
        stratum_num = len(unique_labels)
        for sit in np.arange(stratum_num):
            stratum_mesh = geo_model.threshold([unique_labels[sit] - epsilon, unique_labels[sit] + epsilon])
            if is_explode:
                if explode_dist is None:
                    explode_dist = [0, 0, 10]
                dist = np.float32(explode_dist) * np.array(stratum_num - sit)
                stratum_mesh.points = stratum_mesh.points.__add__(dist)
            geo_model_stratum_list.append(stratum_mesh)
        for sit in np.arange(len(geo_model_stratum_list)):
            stratum_name = str(int(unique_labels[sit]))
            if label_map is not None:
                if int(unique_labels[sit]) in label_map.keys():
                    stratum_name = label_map[int(unique_labels[sit])]
            plotter.add_mesh(geo_model_stratum_list[sit], label=stratum_name, color=color_list[sit])
            plotter.add_axes()
        # plotter.add_legend(loc='lower right', border=False, bcolor='w', face='rectangle', size=[0.4, 0.4])
        if is_show:
            plotter.show()
        if is_save and out_file_path is not None:
            geo_model.save(filename=out_file_path)
        return geo_model_stratum_list


# section 和 drill
def visual_separte_sample_data(mesh, unique_value_list, color_map, box=None, is_show=True):
    epsilon = 0.1
    plotter = pv.Plotter()
    if isinstance(mesh, pv.PolyData) or isinstance(mesh, pv.RectilinearGrid):
        for sit in np.arange(len(unique_value_list)):
            stratum_mesh = mesh.threshold([unique_value_list[sit] - epsilon, unique_value_list[sit] + epsilon])
            plotter.add_mesh(stratum_mesh, label=str(unique_value_list[sit]), color=color_map[sit])
    elif isinstance(mesh, list):
        for poly in mesh:
            for sit in np.arange(len(unique_value_list)):
                stratum_mesh = poly.threshold([unique_value_list[sit] - epsilon, unique_value_list[sit] + epsilon])
                if len(stratum_mesh.points) == 0:
                    continue
                plotter.add_mesh(stratum_mesh, label=str(unique_value_list[sit]), color=color_map[sit])
    elif isinstance(mesh, pv.MultiBlock):
        for key in mesh.keys():
            poly = mesh[key]
            for sit in np.arange(len(unique_value_list)):
                stratum_mesh = poly.threshold([unique_value_list[sit] - epsilon, unique_value_list[sit] + epsilon])
                if len(stratum_mesh.points) == 0:
                    continue
                plotter.add_mesh(stratum_mesh, label=str(unique_value_list[sit]), color=color_map[sit])
    plotter.add_axes()
    if box is not None:
        plotter.add_mesh(box, color='w')
    if is_show:
        plotter.show()


# drills pv.MutiBlock()
# 可视化钻孔柱子
def drill_construct_tube(drills, drill_radius=1.0):
    drill_list = pv.MultiBlock()
    for drill in drills:
        if isinstance(drill, pv.PolyData):
            # 将drill的顶点渲染改为cell渲染，cell_data
            drill_lines = drill.lines
            if drill.n_cells == 1 and len(drill_lines) > 2:
                drill_points = drill.points
                drill_lines = drill.lines
                drill_points_label = drill.active_scalars
                lines = []
                labels = []
                for l_it in np.arange(1, len(drill_lines) - 1):
                    # 每一个cell依次设置两个点
                    one_line = [2, drill_lines[l_it], drill_lines[l_it + 1]]
                    lines.append(np.array(one_line, dtype=int))
                    labels.append(drill_points_label[drill_lines[l_it]])
                labels = np.array(labels)
                lines = np.array(lines)
                drill = pv.PolyData(drill_points, lines=lines)
                drill.cell_data['stratum'] = labels
        else:
            # pv.RectilinearGrid
            drill = drill.cell_data_to_point_data()
            drill_points = drill.points
            drill_points_label = drill.active_scalars
            lines = []
            labels = []
            # 遍历每个cell,一个cell对应一个line
            for l_it in np.arange(len(drill_points) - 1):
                one_line = [2, l_it, l_it + 1]
                lines.append(np.array(one_line, dtype=int))
                labels.append(drill_points_label[l_it + 1])
            labels = np.array(labels)
            lines = np.array(lines)
            drill = pv.PolyData(drill_points, lines=lines)
            drill.cell_data['stratum'] = labels
        drill = drill.tube(radius=drill_radius)
        drill_list.append(drill)
    return drill_list


# 可视化geodata中的采样数据
def visual_sample_data(geodata, is_show=True, camera=None, drill_radius=1, plot_points=False):
    grid = None
    if geodata.sample_grid is not None:
        grid = copy.deepcopy(geodata.sample_grid)
    sample_data = []
    train_plot_data_type = copy.deepcopy(geodata.train_plot_data_type)
    for it, plot_data_type in enumerate(train_plot_data_type):
        if plot_data_type == 'section':
            # 剖面
            sections = pv.MultiBlock()
            tmp_slice_param = geodata.train_plot_data[it]
            for axis_label in tmp_slice_param.keys():
                if len(tmp_slice_param[axis_label]) > 1:
                    for id, center in enumerate(tmp_slice_param[axis_label]):
                        if id == 0:
                            ns = center
                        else:
                            sl = grid.slice(normal=axis_label, origin=center)
                            sections.append(sl)
                else:
                    ns = tmp_slice_param[axis_label][0]
                    secs = grid.slice_along_axis(axis=axis_label, n=ns)
                    sections.append(secs)
            sample_data.append(sections)
        elif plot_data_type == 'drill':
            # 钻孔
            tmp_drill_param = geodata.train_plot_data[it]
            if isinstance(tmp_drill_param, tuple):
                drill_pos, drill_num = tmp_drill_param
                drills, _, _, _ = geodata.sample_with_drills(drill_pos=drill_pos)
                drills = drill_construct_tube(drills, drill_radius=drill_radius)
                sample_data.append(drills)
            elif isinstance(tmp_drill_param, dict):
                drills = pv.MultiBlock()
                for drill_key in tmp_drill_param.keys():
                    drill_points_labels = tmp_drill_param[drill_key]
                    drill_points = drill_points_labels[:, 0:3]
                    drill_label = drill_points_labels[:, 3]
                    lines = np.hstack([len(drill_points), list(np.arange(0,
                                                                         len(
                                                                             drill_points)))])  # [[len(drill_points)], list(np.arange(0, len(drill_points)))]
                    drill = pv.PolyData(drill_points, lines=lines)
                    drill.point_data['stratum'] = drill_label
                    drills.append(drill)
                drills = drill_construct_tube(drills, drill_radius=drill_radius)
                sample_data.append(drills)
    if is_show:
        if plot_points is True:
            plot_num = len(sample_data) + 1 + 1
        else:
            plot_num = len(sample_data)
        if plot_num < 3:
            column_num, row_num = plot_num, 1
        else:
            column_num, row_num = 3, plot_num // 3
        plotter = pv.Plotter(shape=(row_num, column_num))
        row_it, col_it, sit = 0, 0, 0
        for it, sample_it in enumerate(sample_data):
            row_it = it // 3
            col_it = it % 3
            sit = it
            if camera is not None and isinstance(camera[0], list):
                if len(camera) == plot_num:
                    plotter.camera_position = camera[it]
                else:
                    plotter.camera_position = camera[0]
            if row_it == 0 and col_it == 0:
                plotter.add_mesh(sample_it, show_scalar_bar=True)
                plotter.add_axes()
            else:
                plotter.subplot(row_it, col_it)
                plotter.add_mesh(sample_it, show_scalar_bar=False)
                plotter.add_axes()
            if grid is not None:
                _ = plotter.add_mesh(grid.outline(), color="k")
        if plot_points:
            row_it = (sit + 1) // 3
            col_it = (sit + 1) % 3
            plotter.subplot(row_it, col_it)
            base_grid = copy.deepcopy(geodata.sample_grid)
            geomap = clip_section_along_axis(mesh=base_grid, sample_axis='z', scroll_scale=0.999)
            drills = sample_data[0]
            drill_points_2d = get_drills_scatters_2d(drills, geodata.bound)
            pdata = pv.PolyData(drill_points_2d)
            plotter.add_mesh(geomap)
            plotter.add_mesh(grid.outline(), color="k")
            plotter.add_mesh(pdata, point_size=5, render_points_as_spheres=True, color='r')
            section_1 = clip_section_along_axis(mesh=base_grid, sample_axis='x', scroll_scale=0.001)
            section_2 = clip_section_along_axis(mesh=base_grid, sample_axis='y', scroll_scale=0.001)
            plotter.add_mesh(drills)
            plotter.add_mesh(section_1)
            plotter.add_mesh(section_2)
            plotter.camera_position = [0, 0, 1]
            plotter.add_axes()
        plotter.show()
    return sample_data


# 获取钻孔在地表的二维分布点，这里默认所有钻孔都可以接触到地表，返回二维坐标
def get_drills_scatters_2d(drills, bounds):
    drills_points_2d = []
    for drill in drills:
        drill_points = drill.cell_centers().points
        if len(drill_points) > 0:
            drill_xy = drill_points[0]
            drill_xy[2] = bounds[5]
            drills_points_2d.append(drill_xy)
    return np.array(drills_points_2d)


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


# 可视化输入的多个pyvista模型，给几个模型就分几个subPlotter显示，若输入是[]列表，则列表中的模型在一个subPlotter中显示
def visual_multiple_model(*visual_mesh_args, camera=None, box=None, is_show=True):
    plot_num = len(visual_mesh_args)
    plotter = pv.Plotter(shape=(1, plot_num))
    for pit, visual_mesh in enumerate(visual_mesh_args):
        if pit == 0:
            if isinstance(visual_mesh, list):
                for model in visual_mesh:
                    plotter.add_mesh(model)
            else:
                plotter.add_mesh(visual_mesh, show_scalar_bar=True)
            plotter.add_axes()
            if box is not None:
                plotter.add_mesh(box, color='k')
            if camera is not None:
                plotter.camera_position = camera
        else:
            plotter.subplot(0, pit)
            if isinstance(visual_mesh, list):
                for model in visual_mesh:
                    plotter.add_mesh(model)
            else:
                plotter.add_mesh(visual_mesh, show_scalar_bar=False)
            plotter.add_axes()
            if box is not None:
                plotter.add_mesh(box, color='k')
            if camera is not None:
                plotter.camera_position = camera
    if is_show:
        plotter.show()


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


# 'b'蓝色  'g'绿色 'r'红色 'c'青色 'm'品红 'y'黄色 'k'黑色 'w'白色
def visual_loss_picture(train_loss, test_loss, title=None, x_label='epoch', y_label='Loss', save_path=None):
    plt.figure(figsize=[14, 5])
    plt.plot(train_loss, "ro-", label="Train Loss")
    plt.plot(test_loss, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if save_path is not None:
        pic_name = 'loss_pic.jpg'
        save_path = os.path.join(save_path, pic_name)
        plt.savefig(save_path)


def visual_acc_picture(train_acc, test_acc, title=None, x_label='epoch', y_label='Acc', save_path=None):
    plt.figure(figsize=[14, 5])
    plt.plot(train_acc, "ro-", label="Train Acc")
    plt.plot(test_acc, "bs-", label="Test Acc")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    max_train_acc_value = max(train_acc)  # 求列表最大值
    max_train_acc_idx = train_acc.index(max_train_acc_value)  # 求最大值对应索引
    max_test_acc_value = max(test_acc)
    max_test_acc_idx = test_acc.index(max_test_acc_value)
    print('The {}th epoch, train acc reached the highest value: {}.'.format(max_train_acc_idx, max_train_acc_value))
    print('The {}th epoch, test acc reached the highest value: {}.'.format(max_test_acc_idx, max_test_acc_value))
    if title is not None:
        plt.title(title)
    if save_path is not None:
        pic_name = 'acc_pic.jpg'
        save_path = os.path.join(save_path, pic_name)
        plt.savefig(save_path)


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


def visual_confusion_matrix(y_pred, y_true, labels=None, is_show=True, is_normalize=True):
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = np.float32(y_pred)
    C = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    if is_normalize:
        con_mat_norm = C.astype('float') / C.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        plt.figure(figsize=(8, 8))


# edge_list: [num_edge, 2]
# 可视化边集，将三角网处理成边集 edge_list，可以用该函数可视化
def visual_edge_list(edge_list, edge_points, is_show=True, e_color='gray', add_points=False, points_idx=None,
                     points_label=None, p_color=None, p_size=2.5):
    # 去除重复边
    new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
    lines = []
    for sub in new_edge_list:
        one_line = [2, sub[0], sub[1]]
        lines.append(np.array(one_line, dtype=int))
    edges = pv.PolyData(edge_points, lines=lines)
    if is_show:
        plotter = pv.Plotter()
        plotter.add_mesh(edges, color=e_color)
        if add_points:
            if points_idx is not None:
                edge_points = copy.deepcopy(edge_points)[points_idx]
            point_data = pv.PolyData(edge_points)
            if points_label is not None:
                point_data.point_data['stratum'] = np.array(points_label)
            if p_color is not None:
                plotter.add_mesh(point_data, render_points_as_spheres=True, point_size=p_size, color=p_color)
            else:
                plotter.add_mesh(point_data, render_points_as_spheres=True, point_size=p_size)
        plotter.add_axes()
        plotter.show()
    return edges


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
