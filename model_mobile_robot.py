import math
import numpy as np


class ClassRobot(object):
    def __init__(self, dt=0.002):
        self.dt = dt
        self.r = 0.05
        self.l = 0.09
        self.kr = 0.02
        self.kl = 0.02
        self.pose = np.array([0, 0, 0])
        self.pose_estimated = self.pose
        self.controller_pd = 1
        self.controller_pt = 10
        self.w_sat = 13
        self.estimated_Sigma = np.zeros((3, 3))

    def run(self, w_r, w_l):
        """
        Simulate the real robot
        Args:
            w_r ([float]): [control signal angular velocity(rad/s) of right wheels]
            w_l ([float]): [control signal angular velocity(rad/s) of left wheels]
        Returns:
            None
        """
        delta_distance = self.dt * self.r * (w_r + w_l) / 2
        delta_theta = self.dt * self.r * (w_r - w_l) / self.l
        theta = self.pose[2]
        J = (
            self.r
            * self.dt
            / 2
            * np.array(
                [
                    [math.cos(theta), math.cos(theta)],
                    [math.sin(theta), math.sin(theta)],
                    [2 / self.l, -2 / self.l],
                ]
            )
        )
        Sigma = np.array([[self.kr * abs(w_r), 0], [0, self.kl * abs(w_l)]])
        Q = J @ Sigma @ J.T
        Noise = np.random.multivariate_normal(np.array([0, 0, 0]).T, Q, (1))
        Noise = Noise.reshape(3)

        self.pose = (
            self.pose
            + np.array(
                [
                    math.cos(theta) * delta_distance,
                    math.sin(theta) * delta_distance,
                    delta_theta,
                ]
            )
            + Noise
        )

    def controller(self, goal):
        """
        P controller
        Args:
            goal ([tuple]): [desired location (x,y)]
        Returns:
            dd ([float]): [distance error]
            w_r ([float]): [desired control signal of right wheels]
            w_l ([float]): [desired control signal of left wheels]
        """
        dx = goal[0] - self.pose_estimated[0]
        dy = goal[1] - self.pose_estimated[1]
        dd = np.sqrt(dx ** 2 + dy ** 2)
        dtheta = math.atan2(dy, dx) - self.pose_estimated[2]
        dtheta = self._wrap_to_pi(dtheta)
        # if dtheta > math.pi:
        #     dtheta = dtheta - 2 * math.pi
        # elif dtheta < -math.pi:
        #     dtheta = dtheta + 2 * math.pi
        v_d = dd * self.controller_pd
        w_d = dtheta * self.controller_pt
        w_r = (v_d + self.l * w_d / 2) / self.r
        w_l = (v_d - self.l * w_d / 2) / self.r
        if w_r >= self.w_sat:
            rat = self.w_sat / w_r
            w_r = self.w_sat
            w_l = w_l * rat
        if w_l >= self.w_sat:
            rat = self.w_sat / w_l
            w_l = self.w_sat
            w_r = w_r * rat
        return dd, w_r, w_l

    def estimater(self, w_r, w_l):
        """
        Location estimater
        Args:
            w_r ([float]): [control signal angular velocity(rad/s) of right wheels]
            w_l ([float]): [control signal angular velocity(rad/s) of left wheels]
        Returns:
            None
        """
        delta_distance = self.dt * self.r * (w_r + w_l) / 2
        delta_theta = self.dt * self.r * (w_r - w_l) / self.l
        theta = self.pose_estimated[2]
        J = (
            self.r
            * self.dt
            / 2
            * np.array(
                [
                    [math.cos(theta), math.cos(theta)],
                    [math.sin(theta), math.sin(theta)],
                    [2 / self.l, -2 / self.l],
                ]
            )
        )
        Sigma_delta = np.array([[self.kr * abs(w_r), 0], [0, self.kl * abs(w_l)]])
        Q = J @ Sigma_delta @ J.T
        H = np.array(
            [
                [1, 0, -delta_distance * math.sin(theta)],
                [0, 1, delta_distance * math.cos(theta)],
                [0, 0, 1],
            ]
        )

        self.estimated_Sigma = H @ self.estimated_Sigma @ H.T + Q
        self.pose_estimated = self.pose_estimated + np.array(
            [
                math.cos(theta) * delta_distance,
                math.sin(theta) * delta_distance,
                delta_theta,
            ]
        )
        self.pose_estimated[2] = self._wrap_to_pi(self.pose_estimated[2])

    def reset(self, dt=0.002):
        self.dt = dt
        self.r = 0.05
        self.l = 0.09
        self.kr = 0.01
        self.kl = 0.01
        self.pose = np.array([0, 0, 0])
        self.pose_estimated = self.pose
        self.controller_pd = 1
        self.controller_pt = 10
        self.w_sat = 13
        self.estimated_Sigma = np.zeros((3, 3))

    def _wrap_to_pi(self, theta):
        while theta > math.pi:
            theta = theta - 2 * math.pi
        while theta < -math.pi:
            theta = theta + 2 * math.pi
        return theta


if __name__ == "__main__":
    robot = ClassRobot()
    # dt, w_r, w_l = robot.controller((1, 1))
    # robot.run(w_r, w_l)
    # while dt >= 0.1:
    #     dt, w_r, w_l = robot.controller((1, 1))
    #     robot.run(w_r, w_l)
    #     robot.estimater(w_r, w_l)
    #     print(robot.pose)
    #     print(robot.pose_estimated)
