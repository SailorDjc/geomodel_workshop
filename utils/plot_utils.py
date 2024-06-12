import numpy as np
import matplotlib
import pyvista as pv

from data_structure.geodata import BoreholeSet, Borehole, Grid, Section
import sklearn
import os
import copy
import time
from vtkmodules.all import vtkPolyDataMapper
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from geograph_parse import GeoMeshGraphParse
# from matplotlib import pyplot as plt
import imageio as iio
from threading import Thread
from utils.vtk_utils import CreateLUT

matplotlib.use("TkAgg")


# 根据神经网络的预测值，生成基于网格的地质模型，网格已经在geodata中指定
def visual_predicted_values_model(grid_data, cell_values, is_show=True, save_path=None):
    if isinstance(cell_values, torch.Tensor):
        cell_values = cell_values.cpu().numpy()
    if cell_values.ndim > 1:
        scalars = np.argmax(cell_values, axis=1)
    else:
        scalars = np.float32(cell_values)
    if isinstance(grid_data, (Grid, Section)):
        gen_mesh = copy.deepcopy(grid_data.vtk_data)
    else:
        gen_mesh = copy.deepcopy(grid_data)
    gen_mesh.cell_data['stratum'] = scalars
    gen_mesh.set_active_scalars(name='stratum')
    if is_show:
        visual_multiple_model([grid_data, gen_mesh])
    if save_path is not None:
        if isinstance(grid_data, (Grid, Section)):
            dir_path, file_name = os.path.split(save_path)
            export_grid = Grid(grid_vtk=gen_mesh)
            export_grid.label_dict = grid_data.label_dict
            print('save predicted model {} to {}.'.format(file_name, dir_path))
            export_grid.save(dir_path=dir_path, out_name=file_name)
        else:
            filename, extension = os.path.splitext(save_path)
            if extension != '.vtk':
                save_path += '.vtk'
            gen_mesh.save(filename=save_path)
        print('Geodata has been saved to {}.'.format(save_path))
    return gen_mesh


def build_plot_from_horizon_metrics(scalar_means: np.ndarray, residual_means: np.ndarray,
                                    variance: np.ndarray, filename: str):
    scale_for_variance = 20 / 0.001
    variance_s = variance * scale_for_variance

    n = scalar_means.size
    x = np.arange(n)
    width = 0.3 * np.exp(-(n - 1) ** 2) + 0.3
    legend_txt = str(round(variance.min(), 4)) + r' $min$ $s_{var}$'

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(x, scalar_means, s=variance_s, label=legend_txt, alpha=0.4)
    ax.plot(x, scalar_means, lw=3)
    ax.set_xlabel('horizon i')
    ax.set_ylabel('scalar mean')
    ax.legend(loc="upper left", markerscale=0.3, frameon=False)
    ax2 = ax.twinx()
    ax2.bar(x, residual_means, width=width, color="red", alpha=0.4)
    ax2.set_ylabel("residual")
    ax.set_xticks(x)
    fig.savefig(filename)


def build_plot_from_unit_metrics(class_ids: np.ndarray, residual_means: np.ndarray, filename: str):
    n = residual_means.size
    width = 0.3 * np.exp(-(n - 1) ** 2) + 0.3

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(class_ids, residual_means, width=width, color="red")
    ax.set_xlabel('Class Ids')
    ax.set_ylabel('Residual Means')
    ax.set_xticks(class_ids)
    fig.savefig(filename)


colors_default = ["ff0000", "28e5da", "0000ff", "ffff00", "c8bebe", "f79292", "fffff0", "f18c1d", "23dcaa",
                  "d785ec", "9d5b13", "e4e0b1", "894509", "af45f5", "fff000", ]


# pyvista窗体截图保存事件回调
class SaveGraphicCallBack:
    def __init__(self, plotter, dir_path=None):
        self.plotter = plotter
        self.output = dir_path
        if self.output is None:
            self.output = os.path.dirname(os.path.abspath(__file__))
            self.output = os.path.join(self.output, '../output')
            if not os.path.exists(self.output):
                os.makedirs(self.output)

    def __call__(self, *args, **kwargs):
        file_name = 'pic_' + str(int(time.time())) + '.png'
        self.plotter.screenshot(filename=os.path.join(self.output, file_name))
        print('Picture {} has been saved to {}.'.format(file_name, self.output))


# 控制每个地层的可见性， 可以展示钻孔与地质格网模型：
# geo_object_list:  list
# 输入参数为列表，[BoreholeSet, Grid]， 目前支持BoreholeSet和Grid两种类型的数据
# font_file 字体文件的路径，传入字体文件ttf，支持中文显示
def control_visibility_with_layer_label(geo_object_list: list, lookup_table=None
                                        , is_detach=True, grid_smooth=False, show_edge=False
                                        , labels_info=None, text_info=None, font_file=None):
    if not isinstance(geo_object_list, list):
        raise ValueError('Input must be list.')
    plotter = pv.Plotter()

    plotter.track_click_position()

    start_pos_y = 12.0
    start_pos_x = 5.0
    size = 20
    classes_list = [list(item.get_classes()) for item in geo_object_list]
    classes_list = sum(classes_list, [])
    if -1 in classes_list:
        classes_list.remove(-1)
    max_value = max(classes_list)
    min_value = min(classes_list)
    if lookup_table is None:
        # lookup_table = pv.LookupTable('Set1', n_values=256)  # coolwarm
        # lookup_table.scalar_range = (min_value, max_value)
        lookup_table = CreateLUT(min=min_value, max=max_value)
        lookup_table.below_range_color = 'white'
    _actor_list = []

    class SetVisibilityCallback:
        """Helper callback to keep a reference to the actor being modified."""

        def __init__(self, actor):
            self.actor = actor

        def __call__(self, state):
            self.actor.SetVisibility(state)

    class SetOpacityCallback:

        def __init__(self, actor_list: list):
            self.actor_list = actor_list

        def __call__(self, slider_val):
            for a_i in range(len(self.actor_list)):
                self.actor_list[a_i].GetProperty().SetOpacity(slider_val)

    for geo_object in geo_object_list:
        if not isinstance(geo_object, (BoreholeSet, Grid, Section)):
            raise ValueError('Input data type is not supported.')
        else:
            grid_flag = False
            if isinstance(geo_object, Grid):
                grid_flag = True
            if is_detach:
                vtk_data_dict = geo_object.detach_vtk_component_with_label()
                for label in sorted(vtk_data_dict):
                    if isinstance(vtk_data_dict[label], pv.UnstructuredGrid) and grid_smooth:
                        vtk_data_dict[label] = vtk_data_dict[label].extract_geometry().smooth(boundary_smoothing=False
                                                                                              , n_iter=100
                                                                                              , relaxation_factor=0.1
                                                                                              , edge_angle=120)
                    if label == -1:
                        color = 'black'
                    else:
                        color = lookup_table.map_value(label)

                    actor = plotter.add_mesh(vtk_data_dict[label], color=color
                                             , show_edges=show_edge)
                    if grid_flag:
                        _actor_list.append(actor)
                    # 按钮点击事件-可见性
                    callback = SetVisibilityCallback(actor)
                    plotter.add_checkbox_button_widget(callback, value=True, position=(start_pos_x, start_pos_y),
                                                       size=size,
                                                       border_size=1, color_on=lookup_table.map_value(label)
                                                       , color_off='grey', background_color='grey')
                    text = str(label)
                    if labels_info is not None:
                        if label in labels_info.keys():
                            text = str(labels_info[label])
                            if text_info is not None:
                                text += text_info[labels_info[label]]
                    plotter.add_text(text=text, position=(start_pos_x + size + 1, start_pos_y), font_size=10
                                     , font_file=font_file)
                    start_pos_y = start_pos_y + size + (size // 10)
                if geo_object.name is None:
                    geo_name = geo_object.__class__.__name__
                else:
                    geo_name = geo_object.__class__.__name__ + '_' + geo_object.name
                plotter.add_text(text=geo_name
                                 , position=(start_pos_x, start_pos_y), font_size=12)
                start_pos_y = start_pos_y + size + (size // 10) + 2
            else:
                if isinstance(geo_object, BoreholeSet):
                    if geo_object.vtk_data is None:
                        geo_object.generate_vtk_data_as_tube()
                actor = plotter.add_mesh(geo_object.vtk_data, show_edges=show_edge)
                if grid_flag:
                    _actor_list.append(actor)
                # 按钮点击事件-可见性
                callback = SetVisibilityCallback(actor)
                plotter.add_checkbox_button_widget(callback, value=True, position=(start_pos_x, start_pos_y),
                                                   size=size, border_size=1, color_on='red'
                                                   , color_off='grey', background_color='grey')
                if geo_object.name is None:
                    geo_name = geo_object.__class__.__name__
                else:
                    geo_name = geo_object.__class__.__name__ + '_' + geo_object.name
                plotter.add_text(text=geo_name, position=(start_pos_x + size + 1, start_pos_y), font_size=12)
                start_pos_y = start_pos_y + size + (size // 10)
    if len(_actor_list) > 0:
        callbace_opacity = SetOpacityCallback(actor_list=_actor_list)
        plotter.add_slider_widget(callbace_opacity, value=1, rng=(0, 1), title='Opacity Of Grid'
                                  , pointa=(0.8, 0.1), pointb=(0.95, 0.1))
    # 截屏按钮
    callback = SaveGraphicCallBack(plotter=plotter)
    w_size = plotter.window_size
    b_x = w_size[0] * 0.9
    b_y = w_size[1] * 0.9
    plotter.add_checkbox_button_widget(callback, value=True, position=(b_x, b_y)
                                       , size=50, border_size=1,
                                       color_on='blue', color_off='blue', background_color='blue')
    plotter.add_text(text='save picture'
                     , position=(b_x - 50, b_y - 25), font_size=12)
    return plotter


# 支持Grid类型
def control_threshold_with_scalars(grid: Grid, lookup_table=None):
    plotter = pv.Plotter()
    classes_list = grid.get_classes()
    max_value = max(classes_list)
    min_value = min(classes_list)
    if lookup_table is None:
        lookup_table = pv.LookupTable('coolwarm', n_values=256)
        lookup_table.scalar_range = (min_value, max_value)
        lookup_table.below_range_color = 'white'
    if not isinstance(grid, Grid):
        raise ValueError('Input data type is not supported.')
    mesh = grid.vtk_data
    if not isinstance(grid.vtk_data, (pv.UnstructuredGrid, pv.RectilinearGrid)):
        raise ValueError('Input data is not supported')
    actor = plotter.add_mesh_threshold(mesh)
    actor.mapper.lookup_table = lookup_table
    return plotter


# grid: Grid
# lookup_table: vtkLookupTable, optional
# only_section: boolen, True 只显示剖面
# save_path: str  .vtp
# 读Grid网格模型的平面剖切窗体
def control_clip_with_plane(grid: Grid, lookup_table=None, only_section=False, save_path=None):
    plotter = pv.Plotter()
    classes_list = grid.get_classes()
    max_value = max(classes_list)
    min_value = min(classes_list)
    if lookup_table is None:
        lookup_table = pv.LookupTable('coolwarm', n_values=256)
        lookup_table.scalar_range = (min_value, max_value)
        lookup_table.below_range_color = 'white'
    if not isinstance(grid, Grid):
        raise ValueError('Input data type is not supported.')
    mesh = grid.vtk_data
    if not isinstance(grid.vtk_data, (pv.UnstructuredGrid, pv.RectilinearGrid)):
        raise ValueError('Input data is not supported')
    if only_section:
        actor = plotter.add_mesh_slice(mesh, cmap=lookup_table)
        if save_path is not None:
            mapper = actor.GetMapper()
            polydata = mapper.GetInput()
            polydata.save(filename=save_path)
    else:
        plotter.add_mesh_clip_plane(mesh, cmap=lookup_table)
    return plotter


# 对Grid网格模型的样条曲面剖切窗体
# save_path: str   .vtp
def control_clip_with_spline(grid: Grid, lookup_table=None, spline_points: np.ndarray = None, save_path=None):
    plotter = pv.Plotter()
    classes_list = grid.get_classes()
    max_value = max(classes_list)
    min_value = min(classes_list)
    if lookup_table is None:
        lookup_table = pv.LookupTable('coolwarm', n_values=256)
        lookup_table.scalar_range = (min_value, max_value)
        lookup_table.below_range_color = 'white'
    if not isinstance(grid, Grid):
        raise ValueError('Input data type is not supported.')
    mesh = grid.vtk_data
    if not isinstance(grid.vtk_data, (pv.UnstructuredGrid, pv.RectilinearGrid)):
        raise ValueError('Input data is not supported')
    if spline_points is None:
        bounds = mesh.bounds
        spline_points = np.array([[bounds[0] * 0.1, bounds[1] * 0.1, bounds[2] * 0.1],
                                  [bounds[0] * 0.3, bounds[1] * 0.3, bounds[2] * 0.3],
                                  [bounds[0] * 0.5, bounds[1] * 0.5, bounds[2] * 0.5],
                                  [bounds[0] * 0.7, bounds[1] * 0.7, bounds[2] * 0.7],
                                  [bounds[0] * 0.9, bounds[1] * 0.9, bounds[2] * 0.9]])
    plotter.add_mesh(mesh.outline(), color='black')
    actor = plotter.add_mesh_slice_spline(mesh, initial_points=spline_points, n_handles=len(spline_points))
    if save_path is not None:
        mapper = actor.GetMapper()
        poly_data = mapper.GetInput()
        poly_data.save(filename=save_path)
    return plotter


# 可视化输入的多个pyvista模型，给几个模型就分几个subPlotter显示，若输入是[]列表，则列表中的模型在一个subPlotter中显示
def visual_multiple_model(*visual_mesh_args, camera=None, box=None, is_show=True):
    plot_num = len(visual_mesh_args)
    plotter = pv.Plotter(shape=(1, plot_num))
    for pit, visual_mesh in enumerate(visual_mesh_args):
        if pit == 0:
            if isinstance(visual_mesh, list):
                for model in visual_mesh:
                    plotter.add_mesh(model)
            else:
                plotter.add_mesh(visual_mesh, show_scalar_bar=True)
            plotter.add_axes()
            if box is not None:
                plotter.add_mesh(box, color='k')
            if camera is not None:
                plotter.camera_position = camera
        else:
            plotter.subplot(0, pit)
            if isinstance(visual_mesh, list):
                for model in visual_mesh:
                    plotter.add_mesh(model)
            else:
                plotter.add_mesh(visual_mesh, show_scalar_bar=False)
            plotter.add_axes()
            if box is not None:
                plotter.add_mesh(box, color='k')
            if camera is not None:
                plotter.camera_position = camera
    if is_show:
        plotter.show()


def confuse_matrix_model(ori_model, pred_model, labels_name, title=None, save_path=None):
    class_scalar = ori_model.active_scalars
    class_pred = pred_model.active_scalars
    plot_matrix(class_scalar, class_pred, labels_name, title, save_path=save_path)


# 生成混淆矩阵
def plot_matrix(y_true, y_pred, labels_name, title=None, threshold=0.8, axis_labels=None, save_path=None):
    # 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    if title is None:
        title = "confusion matrix"
    plt.title(title)
    num_local = np.array(range(len(labels_name)))
    # 绘制坐标
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 将百分比打印在相应的格子内，大于threshold的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > threshold else "black")  # 如果要更改颜色风格，需要同时更改此行
    # 显示
    if save_path is not None:
        pic_name = title + '.jpg'
        save_path = os.path.join(save_path, pic_name)
        plt.savefig(save_path)


# TP FN    实际值为正的总数 TP + FN
# FP TN    实际值为负的总数 FP + TN
#          预测值为正的总数 TP + FP
#          预测值为负的总数 FN + TN
# recall = TP / (TP + FN)
# precision = TP / (TP + FP)
# TPR = 灵敏度 = TP / (TP + FN)                            真正率
# FPR = 1 - 特异度 = 1 - TN / (FP + TN) = FP / (FP + TN)   假正率
# 设X为预测值，Y为真实值，从条件概率来看
# precision = P(X=1 | Y=1)
# recall = 灵敏度 = P(X=1 | Y=1)
# 特异度 = P(X=0 | Y=0)
# ROC (Receiver Operating Characteristic)曲线， 接受者操作特征曲线。 ROC曲线的两个指标就是: 横轴 FPR, 纵轴 TPR
# TPR 越高， FPR越低， 即ROC曲线越陡，模型性能就越好
# AUC (Area Under Curve) 指TPR和FPR围成的ROC曲线下的面积，将分类任务的实际值和预测值作为参数输入给 roc_curve()方法可
# 以得到FPR、TPR对应的阈值。
# AUC 一般判断标准： 0.5-0.7 较低； 0.7-0.85 一般； 0.85-0.95 较好； 0.95-1 非常好，一般不可能。
def plot_auc_curve(y_true_list, y_pred_list, labels, title=None, colors=None, legend_items=None, save_path=None
                   , is_show=True):
    fpr_list = []
    tpr_list = []
    auc_list = []
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange']
    if legend_items is None:
        legend_items = ['1st method', '2nd method', '3rd method', '4th method']
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred, pos_label=labels)
        auc = metrics.auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    for i in range(len(auc_list)):
        legend_items[i] += '(AUC = %0.4F)'
    line_width = 1  # 线宽
    plt.figure(figsize=(8, 5))  # 图大小
    for i in range(len(auc_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=line_width, label=legend_items[i] % auc_list[i], color=colors[i])
    if title is None:
        title = "AUC"
    plt.title(title)
    plt.xlim([0.0, 1.0])  # x轴范围
    plt.ylim([0.0, 1.0])  # x轴范围
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()  # 在图中添加网格
    plt.legend(loc="lower right")  # 显示图例并设置位置
    if is_show:
        plt.show()
    if save_path is not None:
        pic_name = title + '.jpg'
        save_path = os.path.join(save_path, pic_name)
        plt.savefig(save_path)


# 'b'蓝色  'g'绿色 'r'红色 'c'青色 'm'品红 'y'黄色 'k'黑色 'w'白色
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


def visual_acc_picture(train_acc, test_acc, title=None, x_label='epoch', y_label='Acc', save_path=None, is_show=True):
    plt.figure(figsize=[14, 5])
    plt.plot(train_acc, "ro-", label="Train Acc")
    plt.plot(test_acc, "bs-", label="Test Acc")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    max_train_acc_value = max(train_acc)  # 求列表最大值
    # max_train_acc_idx = train_acc.index(max_train_acc_value)  # 求最大值对应索引
    max_test_acc_value = max(test_acc)
    # max_test_acc_idx = test_acc.index(max_test_acc_value)
    # print('The {}th epoch, train acc reached the highest value: {}.'.format(max_train_acc_idx, max_train_acc_value))
    # print('The {}th epoch, test acc reached the highest value: {}.'.format(max_test_acc_idx, max_test_acc_value))
    if title is not None:
        plt.title(title)
    if is_show:
        plt.show()
    if save_path is not None:
        pic_name = 'acc_pic.jpg'
        save_path = os.path.join(save_path, pic_name)
        plt.savefig(save_path)


# edge_list: [num_edge, 2]
# 可视化边集，将三角网处理成边集 edge_list，可以用该函数可视化
def visual_edge_list(edge_list, edge_points, is_show=True, e_color='gray', add_points=False, points_idx=None,
                     points_label=None, p_color=None, p_size=2.5):
    # 去除重复边
    new_edge_list = list(set(tuple(sorted(sub)) for sub in edge_list))
    lines = []
    for sub in new_edge_list:
        one_line = [2, sub[0], sub[1]]
        lines.append(np.array(one_line, dtype=int))
    edges = pv.PolyData(edge_points, lines=lines)
    if is_show:
        plotter = pv.Plotter()
        plotter.add_mesh(edges, color=e_color)
        if add_points:
            if points_idx is not None:
                edge_points = copy.deepcopy(edge_points)[points_idx]
            point_data = pv.PolyData(edge_points)
            if points_label is not None:
                point_data.point_data['stratum'] = np.array(points_label)
            if p_color is not None:
                plotter.add_mesh(point_data, render_points_as_spheres=True, point_size=p_size, color=p_color)
            else:
                plotter.add_mesh(point_data, render_points_as_spheres=True, point_size=p_size)
        plotter.add_axes()
        plotter.show()
    return edges


def create_gif(image_list, gif_path, duration=1):
    """
    生成gif文件，原始图片仅支持png格式
    gif_name： 字符串
    duration:  gif图像时间间隔
    """
    if len(image_list) > 0:
        if isinstance(image_list[0], np.ndarray):
            frames = image_list
        elif isinstance(image_list[0], str):
            frames = []
            for image_name in image_list:
                frames.append(iio.imread(image_name))
        else:
            return
        iio.mimsave(gif_path, frames, format='GIF', duration=duration)

# yy = np.linspace(-1, 1, 50)
# rr = np.random.uniform(0.039, 2.938, 50)
# ss = np.random.uniform(0.001, 0.017, 50)
# build_plot_from_horizon_metrics(yy, rr, ss, "test1.png")
#
# y = np.array([0.805, 0.770])
# s = np.array([0.00017, 0.05977])
# r = np.array([0.018, 0.022])
# build_plot_from_horizon_metrics(y, r, s, "test2.png")
#
# t = 5
# pyvista窗体录频事件回调
# class SaveVideoCallBack:
#     def __init__(self, plotter, dir_path=None, vtime=10):
#         self.plotter = plotter
#         self.output = dir_path
#         self.vtime = vtime
#         self.thread = None
#         if self.output is None:
#             self.output = os.path.dirname(os.path.abspath(__file__))
#             dir_name = str(int(time.time()))
#             self.output = os.path.join(self.output, '../output', dir_name)
#             if not os.path.exists(self.output):
#                 os.makedirs(self.output)
#         self.image_list = []
#
#     def __call__(self, state, *args, **kwargs):
#         while state:
#             image = self.plotter.screenshot()
#             self.image_list.append(image)
#         if len(self.image_list) > 0:
#             file_name = 'pic_' + str(int(time.time())) + '.gif'
#             gif_path = os.path.join(self.output, file_name)
#             if isinstance(self.image_list[0], np.ndarray):
#                 frames = self.image_list
#             else:
#                 return
#             iio.mimsave(gif_path, frames, format='GIF', duration=self.vtime)
#             print('Picture {} has been saved to {}.'.format(file_name, self.output))
#             self.image_list = []
