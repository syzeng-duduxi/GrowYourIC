#!/usr/bin/env python3
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis
""" Implement classes for tracers,

to create points along the trajectories of given points.
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


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
        self.crystallization_time = self.model.crystallisation_time(point, tau_ic)
        num_t = max(2, math.floor((tau_ic - self.crystallization_time) / dt))
        # print(tau_ic, self.crystallization_time, num_t)
        self.num_t = num_t
        if num_t ==0:
            print("oups")
        # need to find cristallisation time of the particle
        # then calculate the number of steps, based on the required dt
        # then calculate the trajectory
        else:
            self.traj_x, self.traj_y, self.traj_z = self.model.trajectory_single_point(
                self.initial_position, tau_ic,  self.crystallization_time, num_t)
            self.time = np.linspace(tau_ic, self.crystallization_time, num_t)
            self.position = np.zeros((num_t, 3))
            self.velocity = np.zeros((num_t, 3))
            self.velocity_gradient = np.zeros((num_t, 9))

    def spherical(self):
        for index, (time, x, y, z) in enumerate(
                zip(self.time, self.traj_x, self.traj_y, self.traj_z)):
            point = positions.CartesianPoint(x, y, z)
            r, theta, phi = point.r, point.theta, point.phi
            grad = self.model.gradient_spherical(r, theta, phi, time)
            self.position[index, :] = [r, theta, phi]
            self.velocity[index, :] = [self.model.u_r(r, theta, time), self.model.u_theta(r, theta, time), self.model.u_phi(r, theta, time)]
            self.velocity_gradient[index, :] = grad.flatten()

    def cartesian(self):
        """ Compute the outputs for cartesian coordinates """
        for index, (time, x, y, z) in enumerate(
                zip(self.time, self.traj_x, self.traj_y, self.traj_z)):
            point = positions.CartesianPoint(x, y, z)
            r, theta, phi = point.r, point.theta, point.phi
            x, y, z = point.x, point.y, point.z
            vel = self.model.velocity(time, [x, y, z]) # self.model.velocity_cartesian(r, theta, phi, time)
            grad = self.model.gradient_cartesian(r, theta, phi, time)
            self.position[index, :] = [x, y, z]
            self.velocity[index, :] = vel[:]
            self.velocity_gradient[index, :] = grad.flatten()

    def output_spher(self, i):
        list_i = i * np.ones_like(self.time)
        data_i = pd.DataFrame(data=list_i, columns=["i"])
        data_time = pd.DataFrame(data=self.time, columns=["time"])
        dt = np.append(np.abs(np.diff(self.time)), [0])
        data_dt = pd.DataFrame(data=dt, columns=["dt"])
        data_pos = pd.DataFrame(data=self.position, columns=["r", "theta", "phi"])
        data_velo = pd.DataFrame(data=self.velocity, columns=["v_r", "v_theta", "v_phi"])
        data_strain = pd.DataFrame(data=self.velocity_gradient, columns=["dvr/dr", "dvr/dtheta", "dvr/dphi", "dvr/dtheta", "dvtheta/dtheta", "dvtheta/dphi","dvphi/dr", "dvphi/dtheta", "dvphi/dphi"])
        data = pd.concat([data_i, data_time, data_dt, data_pos, data_velo, data_strain], axis=1)
        return data
        #data.to_csv("tracer.csv", sep=" ", index=False)

    def output_cart(self, i):
        list_i = i * np.ones_like(self.time)
        data_i = pd.DataFrame(data=list_i, columns=["i"])
        data_time = pd.DataFrame(data=self.time, columns=["time"])
        dt = np.append([0], np.diff(self.time))
        data_dt = pd.DataFrame(data=dt, columns=["dt"])
        data_pos = pd.DataFrame(data=self.position, columns=["x", "y", "z"])
        data_velo = pd.DataFrame(data=self.velocity, columns=["v_x", "v_y", "v_z"])
        data_strain = pd.DataFrame(data=self.velocity_gradient, columns=["dvx/dx", "dvx/dy", "dvx/dz", "dvy/dx", "dvy/dy", "dvy/dz", "dvz/dx", "dvz/dy", "dvz/dz"])
        data = pd.concat([data_i, data_time, data_dt, data_pos, data_velo, data_strain], axis=1)
        return data

class Swarm():
    """ Swarm of tracers.

    Include routines for writing outputs.
    """

    def __init__(self, N, model, dt, output="tracer"):
        self.model = model
        self.output = output
        self.rICB = self.model.rICB
        self.tau_ic = self.model.tau_ic
        self.dt = dt
        N_x, N_y, N_z = N, N, N
        print("Number of tracers: {}".format(N_x*N_y*N_z))
        values_x =   np.linspace(-self.model.rICB, self.model.rICB, N_x)
        values_y = np.linspace(-self.model.rICB, self.model.rICB, N_y)
        values_z = np.linspace(-self.model.rICB, self.model.rICB, N_z)

        i = 0
        for ix, x in enumerate(values_x):
            for iy, y in enumerate(values_y):
                for iz, z in enumerate(values_z):

                    if x**2+y**2+z**2 < self.model.rICB**2:
                        i += 1
                        position = positions.CartesianPoint(x, y, z)
                        self.one_tracer(position, i)
                        if i%100 == 0:
                            print("tracer n. {}".format(i))

        # # self.init_pos = np.zeros((N_t*N_r, 3))
        # for i, theta in enumerate(list_theta):
        #     for j, r in enumerate(list_r):
        #         num =  i*N_r +j
        #         position = positions.SeismoPoint(r, theta, 0.)
        #         # self.init_pos[i*N_r +j, :] = position.x, position.y, position.z
        #         self.one_tracer(position, i)

        self.plot_meridional_cross_section()

    def one_tracer(self, position, i):
        track = Tracer(position, self.model, self.tau_ic, self.dt)
        # print(track.num_t, self.tau_ic)
        track.spherical()
        data = track.output_spher(i)
        if i == 1:
            data.to_csv(self.output+"_spher.csv", sep=" ", index=False)
        else:
            with open(self.output+"_spher.csv", 'a') as f:
                data.to_csv(f, sep=" ", header=False, index=False)
        track.cartesian()
        data = track.output_cart(i+1)
        if i == 1:
            data.to_csv(self.output+"_cart.csv", sep=" ", index=False)
        else:
            with open(self.output+"_cart.csv", 'a') as f:
                data.to_csv(f, sep=" ", header=False, index=False)


    def plot_meridional_cross_section(self):
        " plot only x and y values (no y) "
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # ICB
        theta = np.linspace(0., 2 * np.pi, 1000)
        ax.plot(self.rICB*np.sin(theta), self.rICB*np.cos(theta), 'k', lw=3)
        # data
        data = pd.read_csv(self.output+"_cart.csv", sep=" ")
        sc = ax.scatter(data["x"], data["z"], s=1, c=data["time"])
        # ax.plot(self.init_pos[:, 2], self.init_pos[:, 0], '.r')
        ax.set_xlim([-1.01*self.rICB, 1.01*self.rICB])
        ax.set_ylim([-1.01*self.rICB, 1.01*self.rICB])
        plt.colorbar(sc)
        plt.show()