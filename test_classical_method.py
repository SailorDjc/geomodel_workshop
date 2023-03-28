import torch
import pyvista as pv
import model_visual_kit as mvk
from gme_model_generate import GmeModelList
import os
from retrieve_noddy_files import NoddyModelData

if __name__ == '__main__':

    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')

    # svm
    svm_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\svm.vtk")
    rbf_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\rbf.vtk")
    idw_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\idw.vtk")
    gnn_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\gnn.vtk")
    ori_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\ori.vtk")
    rf_model = pv.read(r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\rf.vtk")
    # mvk.visual_multiple_model(svm_model, rbf_model, idw_model, gnn_model, ori_model, rf_model)
    svm_slice = mvk.clip_section_along_axis(svm_model, scroll_scale=0.3)
    print('slices: ', svm_slice)

    pred_true = ori_model.active_scalars
    pred = gnn_model.active_scalars
    mvk.visual_confusion_matrix(y_pred=pred, y_true=pred_true)



    rbf_slice_x = mvk.clip_section_along_axis(rbf_model, sample_axis='x')
    idw_slice_x = mvk.clip_section_along_axis(idw_model, sample_axis='x')
    gnn_slice_x = mvk.clip_section_along_axis(gnn_model, sample_axis='x')
    ori_slice_x = mvk.clip_section_along_axis(ori_model, sample_axis='x')
    svm_slice_x = mvk.clip_section_along_axis(svm_model, sample_axis='x')
    rf_slice_x = mvk.clip_section_along_axis(rf_model, sample_axis='x')

    rbf_slice_y = mvk.clip_section_along_axis(rbf_model, sample_axis='y')
    idw_slice_y = mvk.clip_section_along_axis(idw_model, sample_axis='y')
    gnn_slice_y = mvk.clip_section_along_axis(gnn_model, sample_axis='y')
    ori_slice_y = mvk.clip_section_along_axis(ori_model, sample_axis='y')
    svm_slice_y = mvk.clip_section_along_axis(svm_model, sample_axis='y')
    rf_slice_y = mvk.clip_section_along_axis(rf_model, sample_axis='y')
    # ori_slice.plot()
    # ori_slice_0 = mvk.clip_section_along_axis(ori_model, scroll_scale=0.4)
    # ori_slice_0.plot()
    # ori_slice_1 = mvk.clip_section_along_axis(ori_model, scroll_scale=0.5)
    # ori_slice_1.plot()
    # ori_slice_2 = mvk.clip_section_along_axis(ori_model, scroll_scale=0.6)
    # ori_slice_2.plot()
    # ori_slice_3 = mvk.clip_section_along_axis(ori_model, scroll_scale=0.8)
    # ori_slice_3.plot()

    noddyData = NoddyModelData(root=r'F:\djc\NoddyDataset', max_model_num=50)
    gme_models = GmeModelList('gme_model', root=root_path, noddy_data=noddyData)  #
    geodata = gme_models.geodata[0]
    drills = mvk.visual_sample_data(geodata, is_show=False, drill_radius=25)
    mvk.visual_multiple_model([drills[0], svm_slice_x, svm_slice_y], [drills[0], rbf_slice_x, rbf_slice_y],
                              [drills[0], idw_slice_x, idw_slice_y], [drills[0], gnn_slice_x, gnn_slice_y],
                              [drills[0], ori_slice_x, ori_slice_y], [drills[0], rf_slice_x, rf_slice_y])
