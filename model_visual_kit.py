import pyvista as pv
import pyvistaqt as pvq
import torch
import vtkmodules.all as vtk
import numpy as np
import os
from pyvistaqt import MultiPlotter
from retrieve_noddy_files import NoddyModelData


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


def visual_sample_data(geodata):
    for it, plot_data_type in enumerate(geodata.train_plot_data_type):
        if plot_data_type == 'section':
            # 剖面
            sections = pv.MultiBlock()
            tmp_slice_param = geodata.train_plot_data[it]
            for axis_label in tmp_slice_param.keys():
                if len(tmp_slice_param[axis_label]) > 1:
                    for center in tmp_slice_param[axis_label]:
                        sl = geodata.sample_grid.slice(normal=axis_label, origin=center)
                        sections.append(sl)
                else:
                    ns = tmp_slice_param[axis_label][0]
                    secs = geodata.sample_grid.slice_along_axis(axis=axis_label, n=ns)
                    sections.append(secs)
        elif plot_data_type == 'drill':
            # 钻孔
            drills = pv.MultiBlock()
            drill_pos, drill_num = geodata.train_plot_data[0]
            for pos in drill_pos:
                pass


def visual_comparison_mesh(geodata, prediction, label, plotter=None):

    gen_mesh, _ = geodata.create_empty_grid(extent=geodata.output_grid_param)
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


def generate_model_with_points():
    pass


class VisualKit(object):
    def __init__(self, geodata, predict):
        pass


if __name__ == '__main__':
    root_path = os.path.abspath('.')
    noddyData = NoddyModelData(root=r'F:\NoddyDataset', max_model_num=10)
    path_list = noddyData.get_noddy_model_list_paths(model_num=10)
    for mesh_path in path_list:
        model = noddyData.get_grid_model(mesh_path)
        model.plot()
