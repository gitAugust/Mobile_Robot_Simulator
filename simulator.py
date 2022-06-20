import math
import numpy as np
import time

from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QObject, Qt
import pyqtgraph as pg
from threading import Thread

from model_mobile_robot import ClassRobot


# Define signal object, inherit from QObject
class MySignals(QObject):
    # When calling the emit method to send a signal,
    # the incoming parameter must be the parameter type specified here
    data_signal = Signal(np.ndarray, pg.ArrowItem, np.ndarray, pg.PlotItem)
    finished_signal = Signal(str)


class Simulator(object):
    def __init__(self):
        loader = QUiLoader()
        # registe PlotWidget
        loader.registerCustomWidget(pg.PlotWidget)
        self.ui = loader.load("main.ui")
        self.ui.Widget_plot.setYRange(-2, 2)
        self.ui.Widget_plot.setXRange(-2, 2)

        self.ui.runButton.clicked.connect(self.handleRun)
        self.ui.stopButton.clicked.connect(self.handleStop)
        self.ui.resetButton.clicked.connect(self.handleReset)

        self.ms = MySignals()
        self.ms.data_signal.connect(self.Plot)
        self.ms.finished_signal.connect(self.finished)

        self.robot = ClassRobot()
        self.posedata = self.robot.pose
        self.posedata_estimated = self.robot.pose_estimated

        self.Thread_stop = False  # flag to stop the robot run thread

    def Plot(self, posedata, arrow, posedata_estimated, ellipse):
        self.ui.Widget_plot.clear()

        self.ui.Widget_plot.plot(posedata[:, 0], posedata[:, 1])
        self.ui.Widget_plot.addItem(arrow)

        mypen = pg.mkPen(color="g", style=Qt.DashLine)  # Dashed green line
        self.ui.Widget_plot.plot(
            posedata_estimated[:, 0], posedata_estimated[:, 1], pen=mypen
        )
        self.ui.Widget_plot.addItem(ellipse)

        # update location data
        self.ui.label_x.setText(str(posedata[-1, 0])[0:5])
        self.ui.label_y.setText(str(posedata[-1, 1])[0:5])
        self.ui.label_x_estimation.setText(str(posedata_estimated[-1, 0])[0:5])
        self.ui.label_y_estimation.setText(str(posedata_estimated[-1, 1])[0:5])

    def finished(self, inf):
        QMessageBox.about(self.ui, "Information", inf)

    def handleRun(self):
        def Run():
            self.Thread_stop = False
            i = 0
            while True:
                goal = self.ui.GoalEdit.text()
                goal = goal.split(",")
                goal = [float(x) for x in goal]
                if len(goal) != 2:
                    goal = (0, 0)
                else:
                    goal = tuple(goal)

                dt, w_r, w_l = self.robot.controller(goal)
                self.robot.run(w_r, w_l)
                self.robot.estimater(w_r, w_l)
                self.posedata = np.vstack((self.posedata, self.robot.pose))
                self.posedata_estimated = np.vstack(
                    (self.posedata_estimated, self.robot.pose_estimated)
                )
                if i >= 10:
                    arrow_x = (
                        math.cos(self.posedata[-1, 2]) * 0.0002 + self.posedata[-1, 0]
                    )
                    arrow_y = (
                        math.sin(self.posedata[-1, 2]) * 0.0002 + self.posedata[-1, 1]
                    )
                    arrow = pg.ArrowItem(
                        angle=math.degrees(math.pi - self.posedata[-1, 2]),
                        pos=(arrow_x, arrow_y),
                    )

                    ellipse_x = self.posedata_estimated[-1, 0]
                    ellipse_y = self.posedata_estimated[-1, 1]
                    qxx = self.robot.Sigma[1, 1]
                    qyy = self.robot.Sigma[2, 2]
                    qxy = self.robot.Sigma[1, 2]
                    k = math.sqrt((qyy - qxx) ** 2 + 4 * (qxy) ** 2)
                    quu = (qyy + qxx + k) / 2
                    qvv = (qyy + qxx - k) / 2
                    s = 5.991  # 95％ confidence interval corresponding s=5.991
                    a = s * math.sqrt(quu)
                    b = s * math.sqrt(qvv)
                    e = math.atan2(2 * qxy, (qyy - qxx)) / 2
                    # print((qyy-qxx)**2+4*(qxy)**2)
                    # print('qxx=',qxx,'qyy=',qyy,'qxy=',qxy,'\n','k=',k,'quu=',quu,'qvv=',qvv,'\n',a,b,e)
                    ex_data, ey_data = get_ellipse(ellipse_x, ellipse_y, a, b, -e)
                    ErrorEllipse = pg.PlotCurveItem(ex_data, ey_data)
                    self.ms.data_signal.emit(
                        self.posedata, arrow, self.posedata_estimated, ErrorEllipse
                    )
                    i = 0
                time.sleep(0.005)
                i += 1
                if dt <= 0.1:
                    self.ms.finished_signal.emit("Finished")
                    break
                if self.Thread_stop == True:
                    self.ms.finished_signal.emit("Stop")
                    break

        t = Thread(target=Run)
        t.start()

    def handleStop(self):
        self.Thread_stop = True

    def handleReset(self):
        self.Thread_stop = True
        self.ui.Widget_plot.clear()
        self.robot.reset()
        self.posedata = self.robot.pose
        self.posedata_estimated = self.robot.pose_estimated


def get_ellipse(e_x, e_y, a, b, e_angle):
    """
    Get elliptical trajectory
    Args:
            e_x ([type]): [center x]
            e_y ([type]): [center y]
            a ([type]): [ellipse axis along x axis]
            b ([type]): [ellipse axis along y axis]
            e_angle ([type]): [Rotation angle]]

    Returns:
            [list]: [trajectory of x,y]
    """
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * math.cos(angles)
        or_y = b * math.sin(angles)
        length_or = math.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle
        new_x = e_x + length_or * math.cos(new_theta)
        new_y = e_y + length_or * math.sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y


if __name__ == "__main__":
    app = QApplication([])
    Simulator = Simulator()
    Simulator.ui.show()
    app.exec_()
    Simulator.Thread_stop = True
