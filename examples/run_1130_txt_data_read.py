import copy
import os.path

import numpy as np
from data_structure.reader import ReadExportFile

# 本实例演示了如何通过ReadExportFile类读取钻孔文件，并展示钻孔模型。
if __name__ == '__main__':
    borehole_file = r"E:\drills_data_0110.txt"
    reader_0 = ReadExportFile()
    borehole_0 = reader_0.read_boreholes_data_from_text_file(dat_file_path=borehole_file)
    from utils.plot_utils import control_visibility_with_layer_label

    # label_map = False 不进行标签标准化处理
    plotter_2 = control_visibility_with_layer_label(geo_object_list=[borehole_0], grid_smooth=False
                                                    , show_edge=False)
    plotter_2.show()

    root_path = os.path.abspath('..')
    file_path = os.path.join(root_path, 'data', 'origin_borehole_data.dat')
    # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes = reader.read_boreholes_data_from_text_file(dat_file_path=file_path)
    boreholes.show(borehole_radius=10, is_tube=True)
    # 以下接口使用了use_cols参数，是因为点数据文件中 x,y,z,label字段在第1，2，3，4列。
    points_data = reader.read_points_data_from_text_file(dat_file_path=file_path, use_cols=[1, 2, 3, 4])
    points_data.show()
