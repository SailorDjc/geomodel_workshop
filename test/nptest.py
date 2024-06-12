import numpy as np
import matplotlib.pyplot as plt

# import torch
# y = torch.empty(512, dtype=torch.long).random_(2, 5)
# y[0:100] = torch.randint(8, 12, size=(100,))

import pyvista
sphere = pyvista.Sphere()
plane = pyvista.Plane()
selected = plane.select_enclosed_points(sphere)
pts = plane.extract_points(
     selected['SelectedPoints'].view(bool),
     adjacent_cells=False,
)
pl = pyvista.Plotter()
_ = pl.add_mesh(sphere, style='wireframe')
_ = pl.add_points(pts, color='r')
pl.show()


# 生成二维正态分布数据
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)

# 定义切割规则
bins = 10
xlim = [-3, 3]
ylim = [-3, 3]

# 进行数据切割，并统计每个子区间内数据的数量
counts, xbins, ybins = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])

# 绘制切割后的结果
fig, ax = plt.subplots()
im = ax.imshow(counts.T, origin='lower', extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], cmap='coolwarm')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Data Binning')
fig.colorbar(im)
plt.show()