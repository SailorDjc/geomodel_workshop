from dgl_geodataset import DglGeoDataset
from gme_model_generate import GmeModelList
from retrieve_noddy_files import NoddyModelData
import model_visual_kit as mvk
import os

if __name__ == '__main__':
    print('Loading data')
    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')
    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    noddy_models = noddyData.get_noddy_model_list_names(model_num=50, sample_random=False)
    pre_train_model_list = []
    for item in [6, 42, 0, 1, 7, 9, 16]:
        pre_train_model_list.append(noddy_models[item])
    model_idx = 0
    # 只有第一次输入的noddy_model可以用到，之后代码会自动加载数据缓存
    gme_models = GmeModelList('gme_model', root=root_path, pre_train_model_list=None, model_extern=[120, 120, 50],
                              noddy_data=noddyData,  # train_model_list=pre_train_model_list[3:4],
                              sample_operator=['rand_drills'],  # ['axis_sections'],
                              add_inverse_edge=True,
                              data_type='Noddy', drill_num=50)  # # 'Wells',  # 'Points'
    gme_models.predict_with_machine_learning_method(model_idx=model_idx, method='xgboost')
    geodata = gme_models.geodata[model_idx]
    drills = mvk.visual_sample_data(geodata, drill_radius=25)