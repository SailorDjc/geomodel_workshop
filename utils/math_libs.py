import numpy as np
from scipy.interpolate import interp1d, interp2d
import scipy.spatial as spt
import math
import torch
from rdp import rdp
import copy


def remove_duplicate_points(points_3d, tolerance=0.000001, is_remove=False):
    ckt = spt.cKDTree(points_3d)
    remove_point_ids = []
    for p_id, one_point in enumerate(points_3d):
        d, pid = ckt.query(one_point, k=3)
        for d_i in range(len(d)):
            if p_id in remove_point_ids:
                continue
            if pid[d_i] != p_id and np.abs(d[d_i]) < tolerance:
                remove_point_ids.append(pid[d_i])
    if is_remove and len(remove_point_ids) > 0:
        points_3d = np.delete(points_3d, remove_point_ids, axis=0)
    return points_3d, remove_point_ids


def simplify_polyline_2d(polyline_points, principal_axis, eps=1):
    axis_labels = ['x', 'y']
    label_to_index = {label: index for index, label in enumerate(axis_labels)}
    axis = principal_axis.lower()
    # 沿轴从左到右排序, principal_axis轴作为主排序序列
    if axis in axis_labels:
        line_points_sort_ind = np.lexsort((polyline_points[:, 1 - label_to_index[axis]]
                                           , polyline_points[:, label_to_index[axis]]))
        line_points_sorted = polyline_points[line_points_sort_ind]
        line_xy_sorted = copy.deepcopy(line_points_sorted[:, 0:2])
        simplify_points = rdp(line_xy_sorted, epsilon=eps)
        return simplify_points


# 用于对线上的点进行加密，输入的 line_points 的 x坐标必须是单调增的
def densify_line_xy_points_with_interp(line_points: np.ndarray, resolution_xy, is_smooth=False
                                       , grid_bounds: np.ndarray = None, is_extent=False):
    control_line_xy = line_points[:, 0:2]
    x = control_line_xy[:, 0].ravel()
    y = control_line_xy[:, 1].ravel()
    if not is_smooth:
        interp_1d = interp1d(x, y, kind='linear', fill_value="extrapolate")
    else:
        interp_1d = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    x_new = []
    for x_i in np.arange(x.shape[0]):
        item = x[x_i]
        x_new.append(item)
        if x_i != len(x) - 1:
            dist = spt.distance.euclidean(control_line_xy[x_i], control_line_xy[x_i + 1], w=None)
            p_num = math.floor(dist / resolution_xy) - 1
            while p_num > 0 and (x[x_i + 1] - item) > 0:
                item = item + (x[x_i + 1] - item) / (p_num + 1)
                x_new.append(item)
                p_num -= 1
    if is_extent and grid_bounds is not None:
        if x[0] > grid_bounds[0]:  # x_min
            x_new.insert(0, grid_bounds[0])
        if x[len(x) - 1] < grid_bounds[1]:  # x_max
            x_new.append(grid_bounds[1])
    x_new = np.array(x_new)
    y_new = interp_1d(x_new)
    line_points_new = None
    if line_points.shape[1] == 2:
        line_points_new = np.array(list(zip(x_new, y_new)))
    elif line_points.shape[1] == 3:
        z = line_points[0, 2]
        z_new = np.full_like(x_new, z)
        line_points_new = np.array(list(zip(x_new, y_new, z_new)))
    return line_points_new


# 计算 熵归一化
def compute_entropy_normalization(prob_data):
    if isinstance(prob_data, torch.Tensor):
        prob_data = prob_data.cpu().numpy()
        # 归一化
        if torch.max(prob_data.view(-1)) > 1 or torch.min(prob_data.view(-1)) < 0:
            prob_data = torch.nn.functional.normalize(prob_data, dim=1)
    if prob_data.ndim > 1:
        ent = torch.sum(torch.mul(prob_data, torch.log(prob_data)), dim=1)
        ent = torch.divide(torch.mul(ent, -1), torch.log(prob_data.ndim))
        return ent


# 列表元素去重，以list_item_1为判断依据，若有list_item_2，则list_item_2与list_item_1等长，删去list_item_1相同位置的元素
def remove_repeated_elements_with_lists(list_item_1, list_item_2=None):
    unique, indexes = np.unique(list_item_1, return_index=True)
    if list_item_2 is not None:
        list_item_2 = list_item_2[indexes]
        return unique, list_item_2
    else:
        return unique




