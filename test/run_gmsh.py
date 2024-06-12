import os.path

import gmsh
import sys
import numpy as np


def create_box_section(hf, wf, hr, wr, hs, ws, theta):
    """
    :param hf: height of the flange
    :param wf: width of the flange
    :param hr: height of the rib
    :param wr: width of the rib
    :param hs: height of the bottom slab
    :param ws: width of the bottom slab
    :param theta: angle of the flange to deck
    :return: 12*3 array of the coordinate of the points
    """
    coord = np.zeros((12, 3))
    coord[0, :] = [-0.5 * ws, 0, 0]
    coord[1, :] = [coord[0, 0] - (hr + hs) / np.tan(theta), 0, (hr + hs)]
    coord[2, :] = [coord[1, 0] - wf, 0, coord[1, 2]]
    coord[3, :] = [coord[2, 0], 0, coord[2, 2] + hf]
    for i in range(4, 8, 1):
        coord[i, :] = [-1 * coord[7 - i, 0], coord[7 - i, 1], coord[7 - i, 2]]
    coord[8, :] = [coord[0, 0] + wr - hs / np.tan(theta), 0, hs]
    coord[9, :] = [coord[8, 0] - hr / np.tan(theta), 0, coord[8, 2] + hr]
    for i in range(10, 12, 1):
        coord[i, :] = [-1 * coord[19 - i, 0], coord[19 - i, 1], coord[19 - i, 2]]
    return coord


def create_box_girder_mesh(hf, wf, hr, wr, hs, ws, theta, span_len, model_name="box_girder", lc=1e-2):
    coord = create_box_section(hf, wf, hr, wr, hs, ws, theta)
    gmsh.clear()
    gmsh.model.add(model_name)
    # create points:
    for i in range(12):
        gmsh.model.geo.addPoint(coord[i, 0], coord[i, 1], coord[i, 2], lc, i + 1)
    # create line
    for i in range(1, 8):
        gmsh.model.geo.addLine(i, i + 1, i)
    gmsh.model.geo.addLine(8, 1, 8)
    for i in range(9, 12):
        gmsh.model.geo.addLine(i, i + 1, i)
    gmsh.model.geo.addLine(12, 9, 12)
    # create curve_loop
    # outer surface
    gmsh.model.geo.addCurveLoop(np.arange(1, 9, 1), 1)
    # inner surface
    gmsh.model.geo.addCurveLoop(np.arange(9, 13, 1), 2)
    # create plane
    gmsh.model.geo.addPlaneSurface([1, 2], 1)
    # extrude
    gmsh.model.geo.extrude([(2, 1)], 0, span_len, 0)
    # mesh
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate()
    gmsh.write("{0}.vtk".format(model_name))
    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()
    return 1


from data_structure.reader import ReadExportFile

if __name__ == '__main__':


    gmsh.initialize(sys.argv)
    hf = 1
    wf = 2
    hr = 5
    wr = 1
    hs = 1
    ws = 6
    theta = 0.25 * np.pi
    span_len = 2
    create_box_girder_mesh(hf, wf, hr, wr, hs, ws, theta, span_len, model_name="box_girder", lc=0.5)
    reader = ReadExportFile()
    mesh = reader.read_vtk_data(file_path=os.path.join(os.path.abspath('../examples'), '../examples/box_girder.vtk'))
    mesh.plot()