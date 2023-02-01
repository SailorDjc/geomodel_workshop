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


def clip_section(mesh, norm='x', origin=None, num=1):
    if origin is None:
        center = mesh.center
    if isinstance(norm, list):
        single_slice = mesh.slice(normal=norm, origin=origin)
    elif isinstance(norm, str):
        if norm == 'x':
            single_slice = mesh.slice_along_axis(axis='x', center=origin, n=num)
        if norm == 'y':
            single_slice = mesh.slice_along_axis(axis='y', center=origin, n=num)
        if norm == 'z':
            single_slice = mesh.slice_along_axis(axis='z', center=origin, n=num)

    plotter = pvq.BackgroundPlotter()
    plotter.add_mesh(single_slice)
    plotter.add_mesh(mesh.outline())
    plotter.app.exec()
    # print('keys:', single_slice.keys())
    return single_slice


def visual_sample_data(geodata, plotter=None, camera=None, plot_points=False):
    sample_data = []
    for it, plot_data_type in enumerate(geodata.train_plot_data_type):
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
                            sl = geodata.sample_grid.slice(normal=axis_label, origin=center)
                            sections.append(sl)
                else:
                    ns = tmp_slice_param[axis_label][0]
                    secs = geodata.sample_grid.slice_along_axis(axis=axis_label, n=ns)
                    sections.append(secs)
            sample_data.append(sections)
        elif plot_data_type == 'drill':
            # 钻孔
            drills = pv.MultiBlock()
            tmp_drill_param = geodata.train_plot_data[it]
            drill_pos, drill_num = tmp_drill_param
            for pos in drill_pos:
                pos_a = copy.deepcopy(pos)
                pos_a[2] = geodata.bound[5]
                pos_b = copy.deepcopy(pos)
                pos_b[2] = geodata.bound[4]
                drill = geodata.sample_grid.sample_over_line(pointa=pos_a, pointb=pos_b,
                                                             resolution=geodata.output_grid_param[2])
                drills.append(drill)
            sample_data.append(drills)
    if plotter is None:
        if plot_points is True:
            plot_num = len(sample_data) + 1
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
                _ = plotter.add_mesh(geodata.sample_grid.outline(), color="k")
                plotter.add_mesh(sample_it, show_scalar_bar=True)
                plotter.add_axes()
            else:
                plotter.subplot(row_it, col_it)
                _ = plotter.add_mesh(geodata.sample_grid.outline(), color="k")
                plotter.add_mesh(sample_it, show_scalar_bar=False)
                plotter.add_axes()
            #
        if plot_points is True:
            row_it = (sit + 1) // 3
            col_it = (sit + 1) % 3
            plotter.subplot(row_it, col_it)
            centers = geodata.sample_grid.cell_centers().points
            pdata = pv.PolyData(centers)
            pdata['scalars'] = np.array(geodata.sample_label)
            grid_tri = pdata.delaunay_3d()
            edges = grid_tri.extract_all_edges()
            plotter.add_mesh(edges)
            _ = plotter.add_mesh(geodata.sample_grid.outline(), color="k")
            plotter.add_mesh(pdata, point_size=10.0, render_points_as_spheres=True)
            plotter.add_axes()
        plotter.show()


def visual_comparison_mesh(geodata, prediction, label, plotter=None, is_show=True, save_path=None):
    gen_mesh, _ = geodata.create_base_grid(extent=geodata.output_grid_param)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    scalars = np.argmax(prediction, axis=1)
    unique = np.unique(scalars)
    print(unique)

    grid = geodata.sample_grid
    grid.cell_data['scalars'] = label.cpu().numpy()

    if plotter is None:

        plotter = pv.Plotter(shape=(1, 2))
        plotter.add_mesh(grid.outline(), color="k")
        plotter.add_mesh(grid, show_scalar_bar=True)
        plotter.add_axes()
        plotter.camera_position = [-2, 5, 3]
        plotter.subplot(0, 1)
        gen_mesh.cell_data['scalars'] = scalars
        plotter.add_mesh(gen_mesh.outline(), color="k")
        plotter.add_mesh(gen_mesh, show_scalar_bar=True)
        plotter.add_axes()
        plotter.camera_position = [-2, 5, 3]
    else:
        gen_mesh.cell_data['scalars'] = geodata.ori_label[geodata.sample_idx]
        plotter[0, 0].add_mesh(gen_mesh, show_scalar_bar=True)
        plotter[0, 0].add_mesh(gen_mesh.outline(), color="k")
        plotter[0, 0].add_axes()
        gen_mesh.cell_data['scalars'] = scalars
        plotter[0, 1].add_mesh(gen_mesh, show_scalar_bar=True)
        plotter[0, 1].add_mesh(gen_mesh.outline(), color="k")
        plotter[0, 1].add_axes()
    if is_show:
        plotter.show()
    if save_path is not None:
        if geodata.name is not None:
            file_name = geodata.name + 'vtk'
        else:
            file_name = 'grid_model' + 'vtk'
        gen_mesh.save(os.path.join(save_path, file_name))


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


class VisualKit(object):
    def __init__(self, geodata, predict):
        pass


if __name__ == '__main__':
    root_path = os.path.abspath('.')

    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=30)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=30, sample_random=False)
    for item, model_name in enumerate(noddy_models):
        model = noddyData.get_grid_model(model_name)
        model.plot()
    # path = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\backup\2_1\tran_loss_log.txt'
    # epoch, train_loss, val_loss, train_acc, val_acc = read_train_loss_log_txt(file_path=path)
    # path = r'E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\backup\2_1'
    # visual_acc_picture(train_acc=train_acc, test_acc=val_acc, save_path=path)
    # visual_loss_picture(train_loss=train_loss, test_loss=val_loss, save_path=path)
