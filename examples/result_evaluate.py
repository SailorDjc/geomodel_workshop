from data_structure.geodata import *
from data_structure.reader import *
root_path = os.path.abspath('..')

xm_file_path = os.path.join(root_path, 'processed', 'tmp_grid1722297929', 'tmp_grid1722297929.dat')
# # # 从外部数据文本中加载钻孔数据
reader = ReadExportFile()

grid_data = reader.read_geodata(file_path=os.path.join(root_path, 'data', 'out_model', 'out_model.dat'))
grid_data.plot(activate_scalars='stratum')
