import copy
import numpy as np
import torch
import pyvista as pv
import model_visual_kit as mvk
from geomodel_analysis import GmeModelList
import os
from retrieve_noddy_files import NoddyModelData
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import metrics


def intersect_model(ori_model, pred_model):
    class_scalar = ori_model.active_scalars
    class_pred = pred_model.active_scalars
    class_intersect = class_scalar - class_pred
    model = copy.deepcopy(ori_model)
    model.cell_data['stratum'] = class_intersect
    model_1 = model.threshold(value=[-5, -0.1])
    model_2 = model.threshold(value=[0.1, 5])
    plotter = pv.Plotter()
    plotter.add_mesh(model_1)
    plotter.add_mesh(model_2)
    plotter.show()
    return model


def output_class_pixel_count(model):
    labels = model.active_scalars
    unique_labels = sorted(np.unique(labels))
    stratum_num = len(unique_labels)
    epsilon = 0.1
    pixel_map = {}
    for sit in np.arange(stratum_num):
        stratum_mesh = model.threshold([unique_labels[sit] - epsilon, unique_labels[sit] + epsilon])
        pixel_num = len(stratum_mesh.active_scalars)
        pixel_map[unique_labels[sit]] = pixel_num
    return pixel_map


def confuse_matrix_model(ori_model, pred_model, labels_name, title=None, save_path=None):
    class_scalar = ori_model.active_scalars
    class_pred = pred_model.active_scalars
    plot_matrix(class_scalar, class_pred, labels_name, title, save_path=save_path)


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None, save_path=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # 绘制图例
    # 图像标题
    if title is not None:
        pl.title(title)
    num_local = np.array(range(len(labels_name)))
    # 绘制坐标
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    if save_path is not None:
        pic_name = title + '.jpg'
        save_path = os.path.join(save_path, pic_name)
        pl.savefig(save_path)


if __name__ == '__main__':
    path_1 = os.path.abspath('..')
    root_path = os.path.join(path_1, 'geomodel_workshop')
    # 三维模型
    svm_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\svm.vtk")
    rbf_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\rbf.vtk")
    idw_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\idw.vtk")
    gnn_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\gnn.vtk")
    ori_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\ori.vtk")
    rf_model = pv.read(
        r"E:\Code\duanjc\PyCode\GeoScience\geomodel_workshop\processed\backup\论文实验\classical method geomodelling\3_14\rf.vtk")

    # confuse_matrix_model(ori_model, svm_model, [0, 1, 2, 3, 4, 5], title='confuse-matrix-svm',
    #                      save_path=os.path.join(root_path, 'output'))
    # confuse_matrix_model(ori_model, rbf_model, [0, 1, 2, 3, 4, 5], title='confuse-matrix-rbf',
    #                      save_path=os.path.join(root_path, 'output'))
    # confuse_matrix_model(ori_model, idw_model, [0, 1, 2, 3, 4, 5], title='confuse-matrix-idw',
    #                      save_path=os.path.join(root_path, 'output'))
    # confuse_matrix_model(ori_model, rf_model, [0, 1, 2, 3, 4, 5], title='confuse-matrix-rf',
    #                      save_path=os.path.join(root_path, 'output'))
    confuse_matrix_model(ori_model, gnn_model, [0, 1, 2, 3, 4, 5], title='confuse-matrix-gnn',
                         save_path=os.path.join(root_path, 'output'))

    # intersect_model(ori_model, svm_model)
    # intersect_model(ori_model, rbf_model)
    # intersect_model(ori_model, idw_model)
    # intersect_model(ori_model, rf_model)
    # intersect_model(ori_model, gnn_model)
    # 二维模型
    # pixel_map_ori = output_class_pixel_count(ori_model)
    # pixel_map_svm = output_class_pixel_count(svm_model)
    # pixel_map_rbf = output_class_pixel_count(rbf_model)
    # pixel_map_idw = output_class_pixel_count(idw_model)
    # pixel_map_rf = output_class_pixel_count(rf_model)
    # pixel_map_gnn = output_class_pixel_count(gnn_model)
    #
    # pixel_map = {}
    # for k in pixel_map_ori.keys():
    #     pixel_map[k] = []
    #     pixel_map[k].append(pixel_map_ori[k])
    #     pixel_map[k].append(pixel_map_svm[k])
    #     pixel_map[k].append(pixel_map_rbf[k])
    #     pixel_map[k].append(pixel_map_idw[k])
    #     pixel_map[k].append(pixel_map_rf[k])
    #     pixel_map[k].append(pixel_map_gnn[k])
    #
    # out_path = os.path.join(root_path, 'tile.txt')
    # pd_list = []
    # for k, v in pixel_map.items():
    #     pd_n = pd.DataFrame(v)
    #     pd_list.append(pd_n)
    # pd_file = pd.concat(pd_list, axis=1)
    # pd_file.dropna(axis=0, how='any')
    # pd_file.to_csv(out_path, index=False, header=False, sep='\t')

    # mvk.export_points_labels_dat_file(file_path=os.path.join(root_path, 'tile.csv'), **pixel_map)
