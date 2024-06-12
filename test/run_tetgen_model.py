import os
import sys
import time
import math
import numpy as np

nn = 4
dlz = [0 for col in range(nn)]
dlz = [20000, 2000, 10000, 20000]
dlx = 40000
dly = 40000
sumdlz = sum(dlz)

print('The length of the cuboid %0.3f' % dlx)
print('The width of the cuboid is %0.3f' % dly)
print('The height of the cuboid is %0.3f' % sumdlz)

boxname = 'cuboid' + '-' + str(dlx) + '-' + str(dly) + '-' + str(sumdlz) + '.poly'
boxf = open(boxname, 'w')

nx = 100
ny = 50
nodenum_part = 2 * (nx + ny)
nodenum = (nn + 1) * nodenum_part
dlz1 = np.append(0, dlz)

for i in range(0, nn):
    dlz1[i + 1] = dlz1[i] + dlz1[i + 1]

# save the information of nodes  储存所有节点信息
node = [[0 for col in range(4)] for row in range(nodenum)]
for i in range(0, nodenum):
    node[i] = [i + 1, 0, 0, 0]  # 第一个值储存节点编号

for i in range(0, len(dlz1)):
    for j in range(0, int(ny / 2)):  # 底面上节点的三维坐标
        node[i * nodenum_part + j][1:] = [dlx / 2, j * dly / ny, dlz1[i] - sumdlz]
    for j in range(int(ny / 2), int(ny / 2 + nx / 2)):
        node[i * nodenum_part + j][1:] = [dlx / 2 - (j - int(ny / 2)) * dlx / nx, dly / 2, dlz1[i] - sumdlz]
    for j in range(int(ny / 2 + nx / 2), int(ny / 2 + nx)):
        node[i * nodenum_part + j][1:] = [-(j - int(nx / 2 + ny / 2)) * dlx / nx, dly / 2, dlz1[i] - sumdlz]
    for j in range(int(ny / 2 + nx), int(ny + nx)):
        node[i * nodenum_part + j][1:] = [-dlx / 2, dly / 2 - (j - int(nx + ny / 2)) * dly / ny, dlz1[i] - sumdlz]

    for j in range(int(nx + ny), int(nx + ny + ny / 2)):
        node[i * nodenum_part + j][1:] = [-dlx / 2, -(j - int(nx + ny)) * dly / ny, dlz1[i] - sumdlz]
    for j in range(int(nx + ny + ny / 2), int(nx + ny + ny / 2 + nx / 2)):
        node[i * nodenum_part + j][1:] = [-dlx / 2 + (j - int(nx + ny + ny / 2)) * dlx / nx, -dly / 2, dlz1[i] - sumdlz]
    for j in range(int(nx + ny + ny / 2 + nx / 2), int(nx + ny + ny / 2 + nx)):
        node[i * nodenum_part + j][1:] = [(j - int(nx + ny + nx / 2 + ny / 2)) * dlx / nx, -dly / 2, dlz1[i] - sumdlz]
    for j in range(int(nx + ny + ny / 2 + nx), int(nx + ny + ny + nx)):
        node[i * nodenum_part + j][1:] = [dlx / 2, -dly / 2 + (j - int(nx + ny + nx + ny / 2)) * dly / ny,
                                          dlz1[i] - sumdlz]

# save the information of facets   存储面的信息
facenum = nn * nodenum_part + nn + 1  # 侧面的面单元和每个立方体的上底面和下底面的和

faceAll = [[0 for col in range(6)] for row in range(facenum - nn - 1)]  # 初始化面的数据

for j in range(0, nn):
    for i in range(0, nodenum_part - 1):
        faceAll[j * nodenum_part + i] = [4, j * nodenum_part + i + 1, j * nodenum_part + i + 2,
                                         j * nodenum_part + i + 2 + nodenum_part,
                                         j * nodenum_part + i + 1 + nodenum_part, j + nn + 2]
    faceAll[(j + 1) * nodenum_part - 1] = [4, (j + 1) * nodenum_part, j * nodenum_part + 1, (j + 1) * nodenum_part + 1,
                                           (j + 2) * nodenum_part, j + nn + 2]

face = [[0 for col in range(5)] for row in range(facenum - nn - 1)]  # 初始化面
faceMark = [[0 for col in range(3)] for row in range(facenum - nn - 1)]  # 初始化面标记

for i in range(0, facenum - nn - 1):
    face[i] = faceAll[i][0:5]
    faceMark[i] = [1, 0, faceAll[i][-1]]

meshsize = [-1]  # the size of mesh

numVolumMark = nn  # nn regions
SolventMark = [4, 0.1, 1, 10]  # the region mark is 1
region = [[0 for col in range(6)] for row in range(numVolumMark)]
for i in range(0, len(region)):
    region[i] = [i + 1, 0, 0, 0, 0, 0]

for i in range(0, len(region)):
    region[i][1] = (node[i * nodenum_part][1] + node[int(nodenum_part / 2 + (i + 1) * nodenum_part)][1]) / 2
    region[i][2] = (node[i * nodenum_part][2] + node[int(nodenum_part / 2 + (i + 1) * nodenum_part)][2]) / 2
    region[i][3] = (node[i * nodenum_part][3] + node[int(nodenum_part / 2 + (i + 1) * nodenum_part)][3]) / 2
    region[i][4] = SolventMark[i]
    region[i][5] = meshsize[0]

line = '# Part 1 - node list\n'
boxf.write(line)
line = '# <# of points> <dimension 3> <# of attributes> <boundary markers>\n'
boxf.write(line)
line = str(nodenum) + '\t' + '3' + '\t' + '0' + '\t' + '1' + '\n'
boxf.write(line)
for i in range(0, nodenum):
    line = str(node[i][0]) + '\t' + str(node[i][1]) + '\t' + str(node[i][2]) + '\t' + str(node[i][3]) + '\t' + str(
        0) + '\n'
    boxf.write(line)

line = '# Part 2 - face list\n'
boxf.write(line)
line = str(facenum) + '\t' + '1' + '\n'
boxf.write(line)
line = '# facet count, boundary marker\n'
boxf.write(line)
line = '#(Eg.) one polygon, no hole, boundary market is 1\n'
boxf.write(line)

for j in range(0, nn):
    line = str(1) + '\t' + str(0) + '\t' + str(j + 1) + '\n'
    boxf.write(line)
    line = str(nodenum_part) + '\t'
    for i in range(j * nodenum_part, (j + 1) * nodenum_part):
        line += str(i + 1) + '\t'
    line += '\n'
    boxf.write(line)
    for k in range(j * nodenum_part, (j + 1) * nodenum_part):
        line = str(faceMark[k][0]) + '\t' + str(faceMark[k][1]) + '\t' + str(faceMark[k][2]) + '\n'
        boxf.write(line)
        line = str(face[k][0]) + '\t' + str(face[k][1]) + '\t' + str(face[k][2]) + '\t' + str(face[k][3]) + '\t' + str(
            face[k][4]) + '\n'
        boxf.write(line)

# the boundary mark of the top facet is
line = str(1) + '\t' + str(0) + '\t' + str(nn + 1) + '\n'
boxf.write(line)
line = str(nodenum_part) + '\t'
for i in range(nn * nodenum_part, (nn + 1) * nodenum_part):
    line += str(i + 1) + '\t'
line += '\n'
boxf.write(line)

line = '# Part 3 - hole list\n'
boxf.write(line)
line = '# <# of polygons> <# of holes> <boundary marker>\n'
boxf.write(line)
line = '0' + '\n'
boxf.write(line)

line = '# Part 4 - region list\n'
boxf.write(line)
line = '#<region #> <x> <y> <z> <region attribute> <region volume constraint>\n'
boxf.write(line)
line = str(numVolumMark) + '\n'
boxf.write(line)
for i in range(0, len(region)):
    line = str(region[i][0]) + '\t' + str(region[i][1]) + '\t' + str(region[i][2]) + '\t' + \
           str(region[i][3]) + '\t' + str(region[i][4]) + '\t' + str(region[i][5]) + '\n'
    boxf.write(line)
boxf.close()

# boxmeshfile = 'cylinder_' + str(dx) + '_' + str(dy) + '-' + 'refined1'+ '.mesh'
cmd = 'tetgen.exe -pq1.3iaAV -k ' + boxname
os.system(cmd)
