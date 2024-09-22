import numpy as np
from scipy.interpolate import LinearNDInterpolator
import scipy.spatial as spt
import math
import torch
from rdp import rdp
import copy


# 检查一个三角形与一个长方体是否相交
def check_triangle_box_overlap(tri_points, voxel_points):
    # axis[3]
    # d, p0, p1, p2, rad, fex, fey, fez
    # normal[3], e0[3], e1[3], e2[3]
    # 三角形数
    tri_num = tri_points.shape[0]
    check_flag = np.full((tri_num, ), True)
    box_half_size = caculate_box_length(voxel_points)
    box_center = caculate_box_center(voxel_points)
    tri_offset = np.subtract(tri_points, box_center)
    tri_max_x = np.max(tri_offset[:, :, 0], axis=1)
    tri_min_x = np.min(tri_offset[:, :, 0], axis=1)
    flag_0 = tri_min_x > box_half_size[0]
    flag_1 = tri_max_x < -box_half_size[0]
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(check_flag, flag_2)
    tri_max_y = np.max(tri_offset[:, :, 1], axis=1)
    tri_min_y = np.min(tri_offset[:, :, 1], axis=1)
    flag_0 = tri_min_y > box_half_size[1]
    flag_1 = tri_max_y < -box_half_size[1]
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(check_flag, flag_2)
    tri_max_z = np.max(tri_offset[:, :, 2], axis=1)
    tri_min_z = np.min(tri_offset[:, :, 2], axis=1)
    flag_0 = tri_min_z > box_half_size[2]
    flag_1 = tri_max_z < -box_half_size[2]
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(check_flag, flag_2)

    # 三角形两向量
    v0, v1, v2 = tri_offset[:, 0], tri_offset[:, 1], tri_offset[:, 2]

    e0 = np.subtract(v1, v0)
    e1 = np.subtract(v2, v1)
    # 三角形法向量
    normal = np.cross(e0, e1)
    d = - np.sum((normal * v0), axis=1)
    flag_3 = check_plane_box_overlap(normal=normal, d=d, maxbox=box_half_size)
    check_flag = np.logical_and(check_flag, flag_3)
    e2 = np.subtract(v0, v2)
    fex, fey, fez = np.fabs(e0[:, 0]), np.fabs(e0[:, 1]), np.fabs(e0[:, 2])
    # (e0[Z], e0[Y], fez, fey);
    p0 = e0[:, 2] * v0[:, 1] - e0[:, 1] * v0[:, 2]
    p2 = e0[:, 2] * v2[:, 1] - e0[:, 1] * v2[:, 2]
    min_val = np.minimum(p0, p2)
    max_val = np.maximum(p0, p2)
    rad = fez * box_half_size[1] + fey * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e0[Z], e0[X], fez, fex)
    p0 = -e0[:, 2] * v0[:, 0] + e0[:, 0] * v0[:, 2]
    p2 = -e0[:, 2] * v2[:, 0] + e0[:, 0] * v2[:, 2]
    min_val = np.minimum(p0, p2)
    max_val = np.maximum(p0, p2)
    rad = fez * box_half_size[0] + fex * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e0[Y], e0[X], fey, fex)
    p1 = e0[:, 1] * v1[:, 0] - e0[:, 0] * v1[:, 1]
    p2 = e0[:, 1] * v2[:, 0] - e0[:, 0] * v2[:, 1]
    min_val = np.minimum(p1, p2)
    max_val = np.maximum(p1, p2)
    rad = fey * box_half_size[0] + fex * box_half_size[1]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    fex, fey, fez = np.fabs(e1[:, 0]), np.fabs(e1[:, 1]), np.fabs(e1[:, 2])
    # e1[Z], e1[Y], fez, fey
    p0 = e1[:, 2] * v0[:, 1] - e1[:, 1] * v0[:, 2]
    p2 = e1[:, 2] * v2[:, 1] - e1[:, 1] * v2[:, 2]
    min_val = np.minimum(p0, p2)
    max_val = np.maximum(p0, p2)
    rad = fez * box_half_size[1] + fey * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e1[Z], e1[X], fez, fex)
    p0 = -e1[:, 2] * v0[:, 0] + e1[:, 0] * v0[:, 2]
    p2 = -e1[:, 2] * v2[:, 0] + e1[:, 0] * v2[:, 2]
    min_val = np.minimum(p0, p2)
    max_val = np.maximum(p0, p2)
    rad = fez * box_half_size[0] + fex * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e1[Y], e1[X], fey, fex)
    p0 = e1[:, 1] * v0[:, 0] - e1[:, 0] * v0[:, 1]
    p1 = e1[:, 1] * v1[:, 0] - e1[:, 0] * v1[:, 1]
    min_val = np.minimum(p0, p1)
    max_val = np.maximum(p0, p1)
    rad = fey * box_half_size[0] + fex * box_half_size[1]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e2[Z], e2[Y], fez, fey);
    fex, fey, fez = np.fabs(e2[:, 0]), np.fabs(e2[:, 1]), np.fabs(e2[:, 2])
    p0 = e2[:, 2] * v0[:, 1] - e2[:, 1] * v0[:, 2]
    p1 = e2[:, 2] * v1[:, 1] - e2[:, 1] * v1[:, 2]
    min_val = np.minimum(p0, p1)
    max_val = np.maximum(p0, p1)
    rad = fez * box_half_size[1] + fey * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e2[Z], e2[X], fez, fex)
    p0 = -e2[:, 2] * v0[:, 0] + e2[:, 0] * v0[:, 2]
    p1 = -e2[:, 2] * v1[:, 0] + e2[:, 0] * v1[:, 2]
    min_val = np.minimum(p0, p1)
    max_val = np.maximum(p0, p1)
    rad = fez * box_half_size[0] + fex * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    # (e2[Y], e2[X], fey, fex)
    p1 = e2[:, 1] * v1[:, 0] - e2[:, 0] * v1[:, 1]
    p2 = e2[:, 1] * v2[:, 0] - e2[:, 0] * v2[:, 1]
    min_val = np.minimum(p1, p2)
    max_val = np.maximum(p1, p2)
    rad = fey * box_half_size[0] + fex * box_half_size[2]
    flag_0 = min_val > rad
    flag_1 = max_val < -rad
    flag_2 = np.logical_not(np.logical_or(flag_0, flag_1))
    check_flag = np.logical_and(flag_2, check_flag)
    # if min_val > rad or max_val < -rad:
    #     return False
    flag_result = np.any(check_flag)
    return flag_result


def check_plane_box_overlap(normal, d, maxbox):
    if normal.ndim == 1:
        v_min = [0, 0, 0]
        v_max = [0, 0, 0]
        for q in np.arange(3):
            if normal[q] > 0.0:
                v_min[q] = -maxbox[q]
                v_max[q] = maxbox[q]
            else:
                v_min[q] = maxbox[q]
                v_max[q] = -maxbox[q]
        f1 = np.dot(normal, v_min)
        if f1 + d > 0.0:
            return False
        f2 = np.dot(normal, v_max)
        if f2 + d > 0:
            return True
        return False
    elif normal.ndim == 2:
        v_min = np.full(normal.shape, -1)
        v_max = np.full(normal.shape, -1)
        for q in np.arange(3):
            q_i = np.where(normal[:, q] > 0.0)
            q_j = np.where(normal[:, q] <= 0)
            v_min[q_i, q] = -maxbox[q]
            v_max[q_i, q] = maxbox[q]
            v_min[q_j, q] = maxbox[q]
            v_max[q_j, q] = -maxbox[q]
        # f1 = np.dot(normal, v_min)
        f1 = np.sum((normal * v_min), axis=1)
        flag_0 = np.logical_not(f1 + d > 0.0)

        # if f1 + d > 0.0:
        #     return False
        # f2 = np.dot(normal, v_max)
        f2 = np.sum((normal * v_max), axis=1)
        flag_1 = (f2 + d > 0.0)
        # if f2 + d > 0:
        #     return True
        flag = np.logical_and(flag_0, flag_1)
        return flag
    else:
        raise ValueError('Input Error.')


# np.dot 向量点乘
# 计算长方体长宽高
def caculate_box_length(points):
    x_vec = np.max(points[:, 0]) - np.min(points[:, 0])
    length = np.linalg.norm(x=x_vec)
    y_vec = np.max(points[:, 1]) - np.min(points[:, 1])
    width = np.linalg.norm(x=y_vec)
    z_vec = np.max(points[:, 2]) - np.min(points[:, 2])
    height = np.linalg.norm(x=z_vec)
    return [length/2, width/2, height/2]


def caculate_box_center(points):
    center = np.mean(points, axis=0)
    return center


def points_trans_translate(t_factor, points=None, center=None, only_get_matrix=False):
    if only_get_matrix:
        return get_translate_transform_matrix(t_factor[0], t_factor[1], t_factor[2])
    trans_points_list = []
    if points is None:
        raise ValueError('points is None.')
    for one_point in points:
        trans_point = trans_translate(one_point[0], one_point[0], one_point[0], t_factor[0], t_factor[1], t_factor[2])
        trans_points_list.append(trans_point)
    trans_points = np.vstack(trans_points_list)
    return trans_points


def points_trans_scale(t_factor, center, points=None, only_get_matrix=False):
    if only_get_matrix:
        return get_scale_transform_matrix(center[0], center[1], center[2], t_factor[0], t_factor[1], t_factor[2])
    scale_points_list = []
    if points is None:
        raise ValueError('points is None.')
    for one_point in points:
        scale_point = trans_scale(one_point[0], one_point[1], one_point[2], center[0], center[1], center[2],
                                  t_factor[0], t_factor[1], t_factor[2])
        scale_points_list.append(scale_point)
    scale_points = np.vstack(scale_points_list)
    return scale_points


# 在三维空间中，点(x, y, z) 平移(tx, ty, tz)
def trans_translate(x, y, z, tx, ty, tz):
    T = get_translate_transform_matrix(tx, ty, tz)
    P = np.array([x, y, z, [1] * x.size])
    x_, y_, z_, _ = np.dot(T, P)
    return np.float32([x_, y_, z_])


# 在三维空间中，点(x, y, z)相对于另一点(px,py,pz)进行缩放操作,(sx, sy, sz)是缩放因子。
def trans_scale(x, y, z, px, py, pz, sx, sy, sz):
    T = get_scale_transform_matrix(px, py, pz, sx, sy, sz)
    P = np.array([x, y, z, 1])  # [1] * x.size
    x_, y_, z_, _ = np.dot(T, P)
    return np.float32([x_, y_, z_])


# 获取缩放变换矩阵
def get_scale_transform_matrix(px, py, pz, sx, sy, sz):
    T = [[sx, 0, 0, px * (1 - sx)],
         [0, sy, 0, py * (1 - sy)],
         [0, 0, sz, pz * (1 - sz)],
         [0, 0, 0, 1]]
    T = np.array(T)
    return T


# 获取平移变换矩阵
def get_translate_transform_matrix(tx, ty, tz):
    T = [[1, 0, 0, tx],
         [0, 1, 0, ty],
         [0, 0, 1, tz],
         [0, 0, 0, 1]]
    T = np.array(T)
    return T


# 去除重复坐标点
def remove_duplicate_points(points_3d, tolerance=0.000001, is_remove=False, is_reverse=True, k=3):
    remove_point_ids = []
    if len(points_3d) <= 1:
        return points_3d, remove_point_ids
    ckt = spt.cKDTree(points_3d)
    if not is_reverse:
        for p_id, one_point in enumerate(points_3d):
            d, pid = ckt.query(one_point, k=k)
            for d_i in range(len(d)):
                if p_id in remove_point_ids:
                    continue
                if pid[d_i] != p_id and np.abs(d[d_i]) < tolerance:
                    remove_point_ids.append(pid[d_i])
    else:
        for p_id, one_point in reversed(list(enumerate(points_3d))):
            d, pid = ckt.query(one_point, k=3)
            for d_i in range(len(d)):
                if p_id in remove_point_ids:
                    continue
                if pid[d_i] != p_id and np.abs(d[d_i]) < tolerance:
                    remove_point_ids.append(pid[d_i])
    if is_remove and len(remove_point_ids) > 0:
        points_3d = np.delete(points_3d, remove_point_ids, axis=0)
    return points_3d, remove_point_ids


def add_point_to_point_set_if_no_duplicate(points_set: list, point, tolerance=0.000001, k=3):
    if len(points_set) == 0:
        points_set.append(point)
        return points_set, len(points_set) - 1
    if len(points_set) == 1:
        dist = math.dist(points_set[0], point)
        if dist > tolerance:
            points_set.append(point)
        return points_set, len(points_set) - 1
    ckt = spt.cKDTree(points_set)
    d, pid = ckt.query(point, k=k)
    for d_i in range(len(d)):
        if np.abs(d[d_i]) < tolerance:
            return points_set, pid[d_i]
    points_set.append(point)
    return points_set, len(points_set) - 1


# 简化二维折线，折角小于指定角度的拐点删除
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
                                       , z_value=None):
    control_line_xy = line_points[:, 0:2]
    x = control_line_xy[:, 0].ravel()
    y = control_line_xy[:, 1].ravel()
    x_new = []
    y_new = []
    if len(x) == 0:
        raise ValueError('Empty input.')
    for p_i in np.arange(x.shape[0]):
        if p_i != len(x) - 1:
            dist = spt.distance.euclidean(control_line_xy[p_i], control_line_xy[p_i + 1], w=None)
            seg_num = math.floor(dist / resolution_xy)
            # 存入第一个点的坐标
            x_new.append(x[p_i])
            y_new.append(y[p_i])
            if seg_num <= 1:
                continue
            # 使用等分点公式计算
            for pj in np.arange(seg_num - 1):
                namela = (pj + 1) / (seg_num - pj - 1)
                insert_x = (x[p_i] + namela * x[p_i + 1]) / (1 + namela)
                insert_y = (y[p_i] + namela * y[p_i + 1]) / (1 + namela)
                x_new.append(insert_x)
                y_new.append(insert_y)
    # 存入最后一个点的坐标
    x_new.append(x[-1])
    y_new.append(y[-1])
    line_points_new = None
    if z_value is not None:
        z_new = np.full_like(x_new, z_value)
        line_points_new = np.array(list(zip(x_new, y_new, z_new)))
    else:
        if line_points.shape[1] == 2:
            line_points_new = np.array(list(zip(x_new, y_new)))
        elif line_points.shape[1] == 3:
            z = line_points[:, 2].ravel()
            if is_smooth:
                interp = LinearNDInterpolator(list(zip(x, y)), z)
            else:
                interp = LinearNDInterpolator(list(zip(x, y)), z)
            z_new = interp(x_new, y_new)
            line_points_new = np.array(list(zip(x_new, y_new, z_new)))
    return line_points_new


# 计算softmax值
def softmax(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    # 防止上下溢出
    m = np.max(data, axis=1)
    m_tiled = np.tile(m, (data.shape[1], 1)).T
    data = np.subtract(data, m_tiled)
    e_x = np.exp(data)
    nn = np.sum(np.exp(data), axis=1, keepdims=True)
    x = e_x / nn
    return x


def normalization(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) / _range


def standardization(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# 计算 归一化信息熵，默认返回numpy数组
# H(x) = -sigma p(x)ln(p(x))/Smax， Smax=ln(n), n为属性个数
def compute_entropy_normalization(prob_data):
    if isinstance(prob_data, torch.Tensor):
        prob_data = prob_data.cpu().numpy()
        # 若数据不在(0, 1)范围内，则归一化处理, 默认按行归一化
        if torch.max(prob_data.view(-1)) > 1 or torch.min(prob_data.view(-1)) < 0:
            prob_data = softmax(prob_data)
    if prob_data.ndim > 1:
        sn = prob_data.shape[1]
        entropy = softmax(prob_data)
        a = - np.sum(np.multiply(entropy, np.log(entropy)), axis=1)
        b = np.log(sn)
        entropy = np.divide(a, b)
        return entropy
    else:
        raise ValueError('Input prob data error.')


# 列表元素去重，以list_item_1为判断依据，若有list_item_2，则list_item_2与list_item_1等长，删去list_item_1相同位置的元素
def remove_repeated_elements_with_lists(list_item_1, list_item_2=None):
    unique, indexes = np.unique(list_item_1, return_index=True)
    if list_item_2 is not None:
        list_item_2 = list_item_2[indexes]
        return unique, list_item_2
    else:
        return unique


# 获取点云数据集的包围盒
# Parameters: coords 输入是np.array 的二维数组，且坐标是3D坐标
# xy_buffer z_buffer 为大于0的浮点数，是包围盒的缓冲距离
def get_bounds_from_coords(coords: np.ndarray, xy_buffer=0, z_buffer=0):
    assert coords.ndim == 2, "input coords array is not 2D array"
    # assert coords.shape[1] == 3, "input coords are not 3D"

    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)

    x_min = coord_min[0]
    x_max = coord_max[0]
    y_min = coord_min[1]
    y_max = coord_max[1]
    if coords.shape[1] == 3:
        z_min = coord_min[2]
        z_max = coord_max[2]
    else:
        z_min = 0
        z_max = 0
    bounds = np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    if xy_buffer != 0 or z_buffer != 0:
        if xy_buffer == 0:
            xy_buffer = z_buffer
        if z_buffer == 0:
            z_buffer = xy_buffer
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        bounds[0] = bounds[0] - xy_buffer * dx
        bounds[1] = bounds[1] + xy_buffer * dx
        bounds[2] = bounds[2] - xy_buffer * dy
        bounds[3] = bounds[3] + xy_buffer * dy
        bounds[4] = bounds[4] - z_buffer * dz
        bounds[5] = bounds[5] + z_buffer * dz
    return bounds


# bounds 必须是6元数
def bounds_merge(bounds_a, bounds_b):
    if bounds_a is None and bounds_b is not None:
        return np.array(bounds_b)
    elif bounds_a is not None and bounds_b is None:
        return np.array(bounds_a)
    elif bounds_a is not None and bounds_b is not None:
        x_min = np.min([bounds_a[0], bounds_b[0]])
        x_max = np.max([bounds_a[1], bounds_b[1]])
        y_min = np.min([bounds_a[2], bounds_b[2]])
        y_max = np.max([bounds_a[3], bounds_b[3]])
        z_min = np.min([bounds_a[4], bounds_b[4]])
        z_max = np.max([bounds_a[5], bounds_b[5]])
        return np.array([x_min, x_max, y_min, y_max, z_min, z_max])
    else:
        raise ValueError('Input bounds is None.')


def compute_bounds_center(bounds):
    if len(bounds) == 6:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, (z_min + z_max) * 0.5])
        return center
    elif len(bounds) == 4:
        x_min, x_max, y_min, y_max = bounds
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, 0])
        return center
    else:
        raise ValueError('Input error.')


# 包围盒求交
def bounds_intersect(bounds_a, bounds_b, ignore_z=False):
    min_x_a, max_x_a, min_y_a, max_y_a, min_z_a, max_z_a = bounds_a
    min_x_b, max_x_b, min_y_b, max_y_b, min_z_b, max_z_b = bounds_b
    min_x = max(min_x_a, min_x_b)
    min_y = max(min_y_a, min_y_b)
    min_z = max(min_z_a, min_z_b)
    max_x = min(max_x_a, max_x_b)
    max_y = min(max_y_a, max_y_b)
    max_z = min(max_z_a, max_z_b)
    if min_x > max_x or min_y > max_y:
        return None
    if ignore_z is False:
        if min_z > max_z:
            return None
    return np.array([min_x, max_x, min_y, max_y, min_z, max_z])
