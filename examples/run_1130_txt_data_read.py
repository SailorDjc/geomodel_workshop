import copy
import os.path

import numpy as np
from data_structure.reader import ReadExportFile

if __name__ == '__main__':
    root_path = os.path.abspath('..')
    file_path = os.path.join(root_path, 'data', 'origin_borehole_data.dat')
    # 从外部数据文本中加载钻孔数据
    reader = ReadExportFile()
    boreholes = reader.read_boreholes_data_from_text_file(dat_file_path=file_path)
    boreholes.show(borehole_radius=10, is_tube=True)
    points_data = reader.read_points_data_from_text_file(dat_file_path=file_path, use_cols=[1, 2, 3, 4])
    points_data.show()
