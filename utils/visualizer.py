import sys
import json
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = str(ROOT) + "/human_pose.json"
with open(path, 'r') as f:
    human_pose = json.load(f)
    skeleton_structure = human_pose["skeleton"]
    keypoints = human_pose["keypoints"]


class Visualizer(object):
    def __init__(self, period, datastep,  HPE, joint_list, axes_3D):
        self.plotLines = dict()
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 4
        self.w.setBackgroundColor((255, 255, 255, 255))
        self.w.setWindowTitle('Human tracker')
        self.w.setGeometry(0, 110, 1080, 1080)
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.setColor((150,150,150))
        gx.setSize(3, 10, 8)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-5, 0, 1.5)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.setColor((150,150,150))
        gy.setSize(10, 3, 0)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -5, 1.5)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.setColor((150,150,150))
        gz.setSize(10, 10, 0)
        self.w.addItem(gz)

        self.period = period
        self.datastep = datastep
        self.plotPoints = None
        self.plotLines = None
        self.HPE = HPE

        self.joint_list = list(range(max(joint_list)+1))
        self.axes_3D = axes_3D #For drawing the skeletons: each tuple represents (coordinate index, axis direction)        

        self.itert = 0

    def process_data(self):
        self.itert += 1

        if self.itert%self.datastep!=0:
            return
        
        results, self.colors_list = self.HPE.estimate3D(self.itert)

        if not results or results[0] is None:
            exit()

        lines = []
        points = []
        points_colors = []
        lines_colors = []

        for skeleton_idx, skeleton in enumerate(results):
            number_of_joints = len(self.joint_list)
            x3D = np.zeros(number_of_joints)
            y3D = np.zeros(number_of_joints)
            z3D = np.zeros(number_of_joints)

            skeleton_color = self.colors_list[skeleton_idx]

            for j in self.joint_list:
                idx = str(j)
                if idx in skeleton:
                    x3D[j] = skeleton[idx][self.axes_3D['X'][0]][0]*self.axes_3D['X'][1] / 100
                    y3D[j] = skeleton[idx][self.axes_3D['Y'][0]][0]*self.axes_3D['Y'][1] / 100
                    z3D[j] = skeleton[idx][self.axes_3D['Z'][0]][0]*self.axes_3D['Z'][1] / 100

            for idx in range(len(skeleton_structure)):
                line_x3D = []
                line_y3D = []
                line_z3D = []
                if str(skeleton_structure[idx][0]-1) in skeleton.keys() and str(skeleton_structure[idx][1]-1) in skeleton.keys():
                    if skeleton_structure[idx][0]-1 in self.joint_list and skeleton_structure[idx][1]-1 in self.joint_list:
                        line_x3D.append(x3D[skeleton_structure[idx][0]-1])
                        line_y3D.append(y3D[skeleton_structure[idx][0]-1])
                        line_z3D.append(z3D[skeleton_structure[idx][0]-1])
                        line_x3D.append(x3D[skeleton_structure[idx][1]-1])
                        line_y3D.append(y3D[skeleton_structure[idx][1]-1])
                        line_z3D.append(z3D[skeleton_structure[idx][1]-1])
                        lines.append((line_x3D, line_y3D, line_z3D))
                        lines_colors.append(skeleton_color)

            #
            # Plot the coordinates in 3D
            #
            for j in skeleton.keys():
                p = int(j)
                if p in self.joint_list:
                    points.append([x3D[p], y3D[p], z3D[p]])
                    points_colors.append(skeleton_color)

        for i, line in enumerate(lines):
            lines[i] = np.array([[line[0][0].item(), line[1][0].item(), line[2][0].item()],
                            [line[0][1].item(), line[1][1].item(), line[2][1].item()]])

        self.update_step(np.array(points), lines, points_colors, lines_colors)



    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec()

    def update_step(self, points, lines, points_colors, lines_colors):
        colors = []
        for color in points_colors:
            colors.append(pg.glColor(color))

        if self.plotPoints is not None:
            width = 5
            self.plotPoints.setData(pos=points, color=np.array(colors), size=width)
        else:
            self.plotPoints = gl.GLScatterPlotItem(pos=points, color=np.array(colors), size=5., pxMode=True)
            self.plotPoints.setGLOptions('opaque')
            
            self.w.addItem(self.plotPoints)

        if self.plotLines is not None:
            for i in range(len(self.plotLines)):
                self.w.removeItem(self.plotLines[i])
            self.plotLines.clear()
        else:
            self.plotLines = dict()
        for i, line in enumerate(lines):
            self.plotLines[i] = gl.GLLinePlotItem(
                pos=line, 
                color=pg.glColor(lines_colors[i]),
                width=3, 
                antialias=True
            )
            self.w.addItem(self.plotLines[i])


    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_data)
        self.timer.start(self.period)
        self.start()
