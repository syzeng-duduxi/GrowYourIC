#!/usr/bin/env python3
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis
""" Implement classes for tracers,

to create points along the trajectories of given points.
"""

import numpy as np
import pandas as pd


from . import data
from . import geodyn_analytical_flows
from . import positions


class Tracer():
    """ Data for 1 tracer (including trajectory) """

    def __init__(self, initial_position, model, tau_ic, dt):
        """ initialisation

        initial_position: Point instance
        model: geodynamic model, function model.trajectory_single_point is required
        """
        self.initial_position = initial_position
        self.model = model  # geodynamic model
        try:
            self.model.trajectory_single_point
        except NameError:
            print(
                "model.trajectory_single_point is required, please check the input model: {}".format(model))
        point = [initial_position.x, initial_position.y, initial_position.z]
        self.crystallization_time = model.crystallisation_time(point, tau_ic)
        num_t = min(2, floor((tau_ic - self.crystallization_time) / dt))
        # need to find cristalisation time of the particle
        # then calculate the number of steps, based on the required dt
        # then calculate the trajectory
        self.traj_x, self.traj_y, self.traj_z = self.model.trajectory_single_point(
            self.initial_position, t0, t1, num_t)
        self.time = np.linspace(t0, t1, num_t)
        self.position = np.zeros((num_t, 3))
        self.velocity = np.zeros((num_t, 3))
        self.velocity_gradient = np.zeros((num_t, 9))

    def spherical(self):
        for index, (time, x, y, z) in enumerate(
                zip(self.time, self.traj_x, self.traj_y, self.traj_z)):
            point = positions.CartesianPoint(x, y, z)
            r, theta, phi = point.r, point.theta, point.phi
            self.position[index, :] = [r, theta, phi]
            self.velocity[index, :] = [self.model.u_r(r, theta, time), self.model.u_theta(r, theta, time), self.model.u_phi(r, theta, time)]
            self.velocity_gradient[index, :] = self.velocity_gradient_spherical()

    def velocity_gradient_spherical(self, r, theta, phi, time):
        model = self.model
        return [model.epsilon_rr(r, theta, phi, time),
            model.epsilon_rt(r, theta, phi, time),
            model.epsilon_rp(r, theta, phi, time),
            model.epsilon_rt(r, theta, phi, time),
            model.epsilon_tt(r, theta, phi, time),
            model.epsilon_tp(r, theta, phi, time),
            model.epsilon_rp(r, theta, phi, time),
            model.epsilon_tp(r, theta, phi, time),
            model.epsilon_pp(r, theta, phi, time)]

    def output(self, file="output.csv"):
        data = pd.DataFrame(data=self.position, columns=["r", "theta", "phi"])
        print(data)


class Swarm():
    """ Swarm of tracers.

    Include routines for writing outputs.
    """

    def __init__(self):
        pass
