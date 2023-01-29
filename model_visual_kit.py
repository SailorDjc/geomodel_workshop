import pyvista as pv
import pyvistaqt as pvq
import torch
import vtkmodules.all as vtk
import numpy as np
import os
from pyvistaqt import MultiPlotter
from retrieve_noddy_files import NoddyModelData
import copy


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


def visual_comparison_mesh(geodata, prediction, label, plotter=None):
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
    plotter.show()


def visual_loss_picture():
    pass


def visual_acc_picture():
    pass


def generate_model_with_points():
    pass


class VisualKit(object):
    def __init__(self, geodata, predict):
        pass


if __name__ == '__main__':
    root_path = os.path.abspath('.')
    noddyData = NoddyModelData(root=r'F:\NoddyDataset', max_model_num=10)
    path_list = noddyData.get_noddy_model_list_names(model_num=10)

    model = noddyData.get_grid_model(path_list[1])
    model.plot()
