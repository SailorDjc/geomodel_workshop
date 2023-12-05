from retrieve_noddy_files import NoddyModelData
from data_structure.grids import Grid
import numpy as np
from data_structure.data_sampler import GeoGridDataSampler
from data_structure.boreholes import BoreholeSet, Borehole
from utils.plot_utils import contorl_visibility_with_layer_label, control_threshold_with_scalars \
    , control_clip_with_plane, control_clip_with_spline

if __name__ == '__main__':
    noddyData = NoddyModelData(root=r'F:\NoddyDataset', dataset_list=['FOLD_FOLD_FOLD'], max_model_num=10,
                               update_grid=False)
    noddy_grid_list = noddyData.get_grid_model_by_idx(dataset='FOLD_FOLD_FOLD', idx=[0])  # 1 6
    grid_list = []
    for noddy_grid in noddy_grid_list:
        grid = Grid(grid_vtk=noddy_grid, name='GeoGrid')
        grid.resample_regular_grid(dim=np.array([150, 150, 120]))
        grid_list.append(grid)

    # 地质采样，从地质模型中随机采样钻孔，钻孔采样数为25
    geodata_drills = GeoGridDataSampler(grid=grid_list[0], sample_operator=['rand_drills'], drill_num=25
                                        , sample_data_names=['drills'])
    geodata_drills.execute()
    # 钻孔数据
    boreholes_data = geodata_drills.sample_data_list[0]
    boreholes_data.show()

    # 将钻孔数据映射到空网格上
    geodata_sample = GeoGridDataSampler()
    geodata_sample.set_base_grid_by_boreholes(boreholes=boreholes_data, dims=np.array([150, 150, 120]))
    geodata_sample.execute()

    new_grid = geodata_sample.grid
    # plotter_1 = control_clip_with_plane(grid=grid_list[0], only_section=True)
    # plotter_1.show()
    plotter_2 = contorl_visibility_with_layer_label(geo_object_list=[boreholes_data, new_grid], grid_smooth=False
                                                    , show_edge=False)
    plotter_2.show()
