import numpy as np
from scipy.interpolate import interp1d, interp2d
import scipy.spatial as spt
import math


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
