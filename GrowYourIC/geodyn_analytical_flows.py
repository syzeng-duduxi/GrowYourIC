#!/usr/bin/env python3
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis
""" Define classes for models from analytical solutions (Yoshida and Karato's models) """

from __future__ import division
from __future__ import absolute_import


import numpy as np
import matplotlib.pyplot as plt  # for figures
# from mpl_toolkits.basemap import Basemap  # to render maps
import math
from scipy.integrate import ode
from scipy.optimize import fsolve
from scipy.misc import derivative

# personal routines
from . import positions
from . import intersection
from . import geodyn
from . import mineral_phys


year = 3600*24*365.25

def e_r(r, theta, phi): # in cartesian
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def e_theta(r, theta, phi):
    return np.array([np.cos(theta) * np.cos(phi),
                     np.cos(theta) * np.sin(phi), -np.sin(theta)])


def e_phi(r, theta, phi):
    return np.array([-np.sin(phi), np.cos(phi), 0.])


def e_x(r, theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.cos(theta) * np.cos(phi), -np.sin(phi)])


def e_y(r, theta, phi):
    return np.array([np.sin(theta) * np.sin(phi),
                     np.cos(theta) * np.sin(phi), np.cos(phi)])


def e_z(r, theta, phi):
    return np.array([np.cos(theta), np.sin(theta), 0.])


def A_ij(theta, phi):
    """ Matrix for base change from spherical to cartesien:

    V_X = A_ij * V_S,
    where V_X is cartesian velocity, V_S spherical velocity (ordered as theta, phi, r)
    """
    A = np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)],
                [np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)],
                [-np.sin(theta), 0., np.cos(theta)]])
    return A #np.concatenate((e_t, e_p, e_r), axis=1)

def velocity_from_spher_to_cart(vel_spher, r, theta, phi):
    """ Careful, velocity in spherical as [Vtheta, Vphi, Vr] """
    sum_j = 0.
    Aij = A_ij(theta, phi)
    for j in range(3):
        sum_j += Aij[:, j]* vel_spher[j]
    return sum_j

def inverse_Jacobien(r, theta, phi):
    """ Matrix used for base change from spherical to cartesien. Notation J_mk """
    return np.array([[ np.cos(theta)*np.cos(phi)/r, np.cos(theta)*np.sin(phi)/r, -np.sin(theta)/r],
                     [ -np.sin(phi)/np.sin(theta)/r, np.cos(phi)/np.sin(theta)/r, 0.],
                     [ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]])

def derivatives_A_ij(theta, phi):
    """ Matrix used for base change from spherical to cartesien. Notation D_ijm """
    D1 = np.array([[-np.sin(theta)*np.cos(phi), 0., np.cos(theta)*np.cos(phi)],
                   [-np.sin(theta)*np.sin(phi), 0., np.cos(theta)*np.sin(phi)],
                   [-np.cos(theta), 0., -np.sin(theta)]])
    D2 = np.array([[-np.cos(theta)*np.sin(phi), -np.cos(phi), -np.sin(theta)*np.sin(phi)],
                   [np.cos(theta)*np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)],
                   [0., 0., 0.]])
    D3 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    D = np.stack((D1, D2, D3), axis=2)
    # print(D[0, 0, 0], D[0, 0, 2], D[0, 1, 1], D[0, 1, 2])
    # print("D:  ", D.shape)
    return D

def velocity_gradient_spher_to_cart(V_spher, L_spher, r, theta, phi):
    """  convert velocity gradient from spherical to cartesian coordinate systems.

    L^X = \sum_i^3 (\sum_i^3 (V^S_j * D_{ijm} + A_{ij} * L^S_{jm}) J_{mk})
    ^X: cartesian coordinates
    ^S: spherical coordinates

    V_spher as (np.array) [V_theta, V_phi, V_r]
    L_spher as (np.array) [[dV_t/dt , dV_t/dp, dV_t/dr],
                [dV_p/dt , dV_p/dp, dV_p/dr],
                [dV_r/dt , dV_r/dp, dV_r/dr]
    """
    L_cart = np.zeros((3,3))
    D_ijm = derivatives_A_ij(theta, phi)
    A = A_ij(theta, phi)
    J_mk =  inverse_Jacobien(r, theta, phi)
    for i in range(3):
        for k in range(3):
            sum_m = 0
            for m in range(3):
                sum_j = 0.
                for j in range(3):
                    sum_j += V_spher[j] * D_ijm[i, j, m] + A[i, j] * L_spher[j, m]
                sum_m += sum_j * J_mk[m, k]
            L_cart[i, k] = sum_m
    return L_cart


class Analytical_Model(geodyn.Model):

    def proxy_singlepoint(self, point, proxy_type):
        """ evaluate the proxy on a single positions.Point instance."""
        proxy = {}  # empty dictionnary
        x, y, z = point.x, point.y, point.z
        if proxy_type == "constant":
            proxy["constant"] = 1.
        elif proxy_type == "vMises_tau_ic":
            proxy["vMises_tau_ic"] = self.deformation(self.tau_ic, point)
        elif proxy_type == "vMises_acc":
            time = self.crystallisation_time([x, y, z], self.tau_ic)
            proxy["age"] = (self.tau_ic - time)
            vMises_acc = self.deformation_accumulated(
                point, time, self.tau_ic, 20)
            proxy["vMises_acc"] = vMises_acc
            #proxy["log_vMises_acc"] = np.log10(vMises_acc)
            if proxy["vMises_acc"]  > 10:
                    pass # proxy["vMises_acc"] = 10.
        elif proxy_type == "vMises_cart":
            proxy["vMises_cart"] = self.deformation_from_cart(self.tau_ic, point)
        elif proxy_type == "age":
            time = self.crystallisation_time([x, y, z], self.tau_ic)
            proxy["age"] = (self.tau_ic - time)
        elif proxy_type == "growth rate":
            time = self.crystallisation_time([x, y, z], self.tau_ic)
            position_crys = self.crystallisation_position([x, y, z], time)
            # in m/years. growth rate at the time of the crystallization.
            proxy["growth rate"] = self.effective_growth_rate(
                time, position_crys)
        return proxy

    def radius_ic(self, t):
        """ radius of the inner core with time. """
        return self.rICB * (t / self.tau_ic)**self.alpha

    def u_growth(self, t):
        """ growth rate at a given time t (dimensional) """
        if t < 0.0001*self.tau_ic:
            return 0.
        return (t / self.tau_ic)**(self.alpha - 1) * \
            self.alpha * self.rICB / self.tau_ic

    def crystallisation_time(self, point, tau_ic):
        """ Return the crystallisation time.

        The cristallisation time of a particle in the inner core is defined as the intersection between the trajectory and the radius of the inner core.

        Args:
            point: [x, y, z]
            tau_ic: time
        Return: time
        """
        if np.sqrt(point[0]**2 + point[1]**2 + point[2]**2) < self.rICB:
            tau_2 = tau_ic
        else:
            tau_2 = 1.01 * tau_ic
        return self.find_time_beforex0(point, tau_ic, tau_2)

    def crystallisation_position(self, point, time):
        """ Return the crystallisation position.

        The cristallisation time of a particle in the inner core is defined as
        the intersection between the trajectory and the radius of the inner core.
        This function return the position of the particle at this time.

        Args:
            point: [x, y, z]
            time: calulated from crystallisation_time
        Return: time
        """
        _point = self.integration_trajectory(time, point, self.tau_ic)
        return positions.CartesianPoint(_point[0], _point[1], _point[2])

    def find_time_beforex0(self, point, t0, t1):
        """ find the intersection between the trajectory and the radius of the IC
        if needed, can be re defined in derived class!

        point : [x, y, z]
        """
        return intersection.zero_brentq(
            self.distance_to_radius, point, t0, a=0., b=t1)

    def distance_to_radius(self, t, r0, t0):
        return self.trajectory_r(t, r0, t0) - self.radius_ic(t)

    def trajectory_r(self, t, r0, t0):
        """ for a point at position r0 at time t0, return the radial component of the position of the point at time t.

            """
        trajectory = self.integration_trajectory(t, r0, t0)
        r = trajectory[0]**2 + trajectory[1]**2 + trajectory[2]**2
        return np.sqrt(r)

    def integration_trajectory(self, t1, r0, t0):
        """ integration of the equation dr(t)/dt = v(r,t)

        return the position of the point at the time t1.
        r0: initial position
        t0: initial time
        t1: tmax of the integration
            """
        r = ode(self.velocity).set_integrator('dopri5')
        r.set_initial_value(r0, t0)
        return np.real(r.integrate(r.t + (t1 - t0)))

    def trajectory_single_point(self, cart_point, t0, t1, num_t):
        """ return the trajectory of a point (a positions.Point instance) between the times t0 and t1, 

        knowing that it was at the position.Point at t0, given nt times steps.
        """
        time = np.linspace(t0, t1, num_t)
        x, y, z = np.zeros(num_t), np.zeros(num_t), np.zeros(num_t)
        x[0], y[0], z[0] = cart_point.x, cart_point.y, cart_point.z
        for i, t in enumerate(time):
            point = self.integration_trajectory(t, [cart_point.x, cart_point.y, cart_point.z], t0)
            x[i], y[i], z[i] = point[0], point[1], point[2]
        return x, y, z

    def deformation_accumulated(self, point, t_crys, tau_ic, N):
        """ Accumulation of strain on the parcel of material located at position point at t_ic """
        trajectoire_x, trajectoire_y, trajectoire_z = self.trajectory_single_point(
            point, t_crys, tau_ic, N)
        deformation_acc = 0.
        time = np.linspace(t_crys, tau_ic, N)
        for i, ix in enumerate(trajectoire_x):
            position_point = positions.CartesianPoint(
                trajectoire_x[i], trajectoire_y[i], trajectoire_z[i])
            deformation_acc = deformation_acc + \
                (self.deformation(time[i], position_point))**2
        deformation_acc = np.sqrt(deformation_acc) / N
        return deformation_acc

    def deformation(self, time, point):
        """ Von Mises equivalent strain

        sqrt(sum epsilon**2)
        (given as equivalent strain / eta, as eta not defined)
        !! no phi velocities
        inputs:
            -   time: float
            -   point: positions.Point instance
        output: float
        """
        Point_full_position = point
        r, theta, phi = Point_full_position.r, (
            90. - Point_full_position.theta) * np.pi / 180., Point_full_position.phi * np.pi / 180.
        # coefficients
        # radius of inner core. Has to be set to 1 if r is already
        # non-dimensional.
        a = self.rICB
        epsilon_rr = partial_derivative(self.u_r, 0, [r, theta])
        epsilon_tt = partial_derivative(
            self.u_theta, 1, [r, theta]) / r + self.u_r(r, theta) / r
        epsilon_pp = self.u_r(r,
                              theta) / r + self.u_theta(r,
                                                        theta) * np.cos(theta) / np.sin(theta) / r
        def vt_r(r, theta):
            return self.u_theta(r, theta) / r
        epsilon_rt = 0.5 * (r * partial_derivative(vt_r, 0,
                                                   [r, theta]) + partial_derivative(self.u_r, 1, [r, theta]) / r)
        return np.sqrt(2. / 3. * (epsilon_rr**2 + epsilon_tt **
                                  2 + epsilon_pp**2 + 2 * epsilon_rt**2))

    def effective_growth_rate(self, t, point):
        """ Effective growth rate at the point r.

        v_{g_eff} = || v_growth - v_geodynamic*e_r ||
        v_geodynamic is already in cartesian coordinates.
        v_growth = ||v_growth|| * vec{e}_r (the unit vector for the radial direction)
        point.er() gives the cartesian coordinates of the vector e_r
        point.proj_er(vect) gives the value of the vector projected on the vector e_r
        r is the position, described as x,y,z
        This function is used for points that are at the surface: r(t) is a point at the surface of the inner core at the time t.
        """
        r = np.array([point.x, point.y, point.z])
        vitesse = point.proj_er(self.velocity(t, r))  # projected on e_r
        growth = self.u_growth(t) - vitesse
        return growth

class Yoshida96(Analytical_Model):
    """ Analytical model from Yoshida 1996 with preferential flow at the equator. """

    def __init__(self, vt=0., S=2./5.):
        self.name = "Yoshida model based on Yoshida et al. 1996"
        self.rICB = 1.
        self.alpha = 0.5
        self.S2 = S
        self.tau_ic = 1.
        self.u_t = vt  # 0.5e-3
        if not vt == 0.:
            self.name = "Yoshida model + translation"

    def verification(self):
        pass

    def velocity(self, time, point):
        """ Velocity at the given position and given time (cartesian coord.).

        time: time (float)
        point: [x, y, z]
        Output is velocity in cartesian geometry [v_x, v_y, v_z]
        """
        # start by defining the spherical unit vector in cartesian geometry (so that we can use all equations from Yoshida 1996)
        # theta is colatitude! Angles are in radians to be used in cos and sin
        # functions.
        if len(point) == 3 and isinstance(point, type(np.array([0, 0, 0]))):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                return [0., 0., 0.]
                print("oups")
        Point_full_position = positions.CartesianPoint(
            point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, \
                        (90. - Point_full_position.theta) * np.pi / 180., \
                        Point_full_position.phi * np.pi / 180.
        if theta <0: print("oups")
        norm_u = self.u_growth(time) * self.S2  # growth rate (average)
        # S2 = self.S2  # S2 coefficient, see Yoshida 1996 for definition
        # radius of inner core. Has to be set to 1 if r is already
        # non-dimensional.
        a = self.radius_ic(time)
        u_r = norm_u * self.u_r(r, theta, time)
        u_theta = norm_u * self.u_theta(r, theta, time)
        u_phi = norm_u * self.u_phi(r, theta, time)
        velocity = u_r * e_r(r, theta, phi) + u_theta * \
            e_theta(r, theta, phi) + u_phi * e_phi(r, theta, phi)
        # with artificial translation
        velocity = velocity + self.u_t * np.array([1, 0, 0])
        return velocity

    def u_r(self, r, theta, time):
        a = self.radius_ic(time)
        return (8. * (r / a) - 3. * (r / a)**3) * (3. * np.cos(theta) *
                                                   np.cos(theta) - 1.) / 10.

    def u_theta(self, r, theta, time):
        a = self.radius_ic(time)
        return (-24. * (r / a) + 15. * (r / a)**3) * (np.cos(theta) * np.sin(theta)) / \
            10.

    def u_phi(self, r, theta, time):
        return 0.

    def epsilon_rr(self, r, theta, phi, time):
        a = self.radius_ic(time)
        return (8 - 9 * (r / a)**2) * (3 * np.cos(theta)**2 - 1) / 10

    def epsilon_tt(self, r, theta, phi, time):
        a = self.radius_ic(time)
        return (8 * (2 - 3 * np.cos(theta)**2) + 3 *
                r**2 * (7 * np.cos(theta)**2 - 4)) / 10

    def epsilon_pp(self, r, theta, phi, time):
        a = self.radius_ic(time)
        return (-8 + 3 * (r / a)**2 * (2 * np.cos(theta)**2 + 1)) / 10

    def epsilon_rt(self, r, theta, phi, time):
        a = self.radius_ic(time)
        return 24 / 10 * (-1 + (r / a)**2) * np.cos(theta) * np.sin(theta)

    def epsilon_rp(self, r, theta, phi, time):
        return 0.

    def epsilon_tp(self, r, theta, phi, time):
        return 0.

    def vonMises_eq(self, r, theta, phi, time):
        sum = self.epsilon_pp(r, theta, phi, time)**2\
            + self.epsilon_rr(r, theta, phi, time)**2\
            + self.epsilon_tt(r, theta, phi, time)**2\
            + 2 * self.epsilon_rp(r, theta, phi, time)**2\
            + 2 * self.epsilon_rt(r, theta, phi, time)**2\
            + 2 * self.epsilon_tp(r, theta, phi, time)**2
        return np.sqrt(2 / 3 * sum)

    def deformation_old(self, time, point):
        r, theta, phi = point.r, (90. - point.theta) * \
            np.pi / 180., point.phi * np.pi / 180.
        return  self.vonMises_eq(r, theta, phi, time)

    def gradient_spherical(self, r, theta, phi, time):
        """ gradient of velocity in spherical coordinates

        (np.array) [[dV_t/dt , dV_t/dp, dV_t/dr],
                   [dV_p/dt , dV_p/dp, dV_p/dr],
                   [dV_r/dt , dV_r/dp, dV_r/dr]
        """
        norm_u = self.u_growth(time)*self.S2   # growth rate (average)
        # S2 coefficient, see Yoshida 1996 for definition
        # radius of inner core. Has to be set to 1 if r is already
        # non-dimensional.
        a = self.radius_ic(time)
        L_tt = (-24.*r/a + 15.*(r/a)**3) * (np.cos(theta)**2-np.sin(theta)**2)/10
        L_tr = (-24./a + 45.*r**2/(a)**3) * np.cos(theta) * np.sin(theta)/10
        L_rt = (8.*r/a - 3.*r**3/a**3) * (-6*np.cos(theta)*np.sin(theta))/10
        L_rr = (8./a - 9.*r**2/(a)**3) * (3*np.cos(theta)**2-1.) /10
        return norm_u  * np.array([[L_tt, 0., L_tr], [0., 0., 0.], [L_rt, 0., L_rr]])

    def gradient_cartesian(self, r, theta, phi, time):
        """ gradient of velocity in cartesian coordinates """
        L_S = self.gradient_spherical(r, theta, phi, time)
        V_S = self.S2 * self.u_growth(time)  * np.array([self.u_theta(r, theta, time), self.u_phi(r, theta, time), self.u_r(r, theta, time) ])
        return velocity_gradient_spher_to_cart(V_S, L_S, r, theta, phi)

    #def velocity_cartesian(self, r, theta, phi, time):
    #    velocity_spher = [self.u_theta(r, theta, time), self.u_phi(r, theta, time), self.u_r(r, theta, time)]
    #    return velocity_from_spher_to_cart(velocity_spher, r, theta, phi)

    def deformation(self, time, point):
        r, theta, phi = point.r, (90. - point.theta) * \
            np.pi / 180., point.phi * np.pi / 180.
        L_X = self.gradient_cartesian(r, theta, phi, time)
        epsilon = 0.5*(L_X+L_X.T)
        return np.sqrt(2/3*np.sum(epsilon**2))



class LorentzForce(Analytical_Model):

    def __init__(self):
        self.name = "Lorentz Force based on Karato 1986"
        self.rICB = 1.
        self.u_growth = 1.
        self.tau_ic = 1.
        self.P = 1e4

    def verification(self):
        pass

    def P20(self, r):
        """ Coefficient P_2^0 at the given position and given time.

        point: [x, y, z]
        Output: float
        """
        P = self.P
        return (-r**6 + 14. / 5. * r**4 - 9. / 5. * r**2 + 204. / 5 * r**4 / (19. +
                                                                              5. * P) - 544. / 5. * r**2 / (19. + 5. * P)) / (3.**3 * 7. * np.sqrt(5.))

    def Y20(self, theta):
        """ Spherical Harmonics Y_2^0 at the given position

        point: [x, y, z]
        Output: float
        """
        return np.sqrt(5) / 2. * (3 * np.cos(theta)**2 - 1.)

    def u_r(self, r, theta):
        return 2. * 3. * self.Y20(theta) * self.P20(r) / r

    def u_theta(self, r, theta):
        def p20_r(radius):
            return self.P20(radius) * radius
        return derivative(p20_r, r, dx=1e-6) / r * \
            derivative(self.Y20, theta, dx=1e-6)

    def velocity(self, time, point):
        """ Velocity at the given position and given time.

        time: time (float)
        point: [x, y, z]
        Output is velocity in cartesian geometry [v_x, v_y, v_z]
        """
        Point_full_position = positions.CartesianPoint(
            point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, (
            90. - Point_full_position.theta) * np.pi / 180., Point_full_position.phi * np.pi / 180.
        # def spherical coordinates vectors in cartesian coordinates
        e_r = np.array([np.sin(theta) *
                        np.cos(phi), np.sin(theta) *
                        np.sin(phi), np.cos(theta)])
        e_theta = np.array([np.cos(theta) * np.cos(phi),
                            np.cos(theta) * np.sin(phi), -np.sin(theta)])

        velocity = self.u_r(r, theta) * e_r + self.u_theta(r, theta) * e_theta
        return velocity

    def deformation(self, time, point):
        """ Von Mises equivalent strain

        sqrt(sum epsilon**2)
        (given as equivalent strain / eta, as eta not defined)
        inputs:
            -   time: float
            -   point: positions.Point instance
        output: float
        """
        # spherical coordinates
        # positions.CartesianPoint(point[0], point[1], point[2])
        Point_full_position = point
        r, theta, phi = Point_full_position.r, (
            90. - Point_full_position.theta) * np.pi / 180., Point_full_position.phi * np.pi / 180.
        # coefficients
        # radius of inner core. Has to be set to 1 if r is already
        # non-dimensional.
        a = self.rICB
        epsilon_rr = partial_derivative(self.u_r, 0, [r, theta])
        epsilon_tt = partial_derivative(
            self.u_theta, 1, [r, theta]) / r + self.u_r(r, theta) / r
        epsilon_pp = self.u_r(r,
                              theta) / r + self.u_theta(r,
                                                        theta) * np.cos(theta) / np.sin(theta) / r

        def vt_r(r, theta):
            return self.u_theta(r, theta) / r
        epsilon_rt = 0.5 * (r * partial_derivative(vt_r, 0,
                                                   [r, theta]) + partial_derivative(self.u_r, 1, [r, theta]) / r)
        return np.sqrt(2. / 3. * (epsilon_rr**2 + epsilon_tt **
                                  2 + epsilon_pp**2 + 2 * epsilon_rt**2))


def partial_derivative(func, var=0, point=[]):
    """ Partial derivative of a function fun

    var indicates which derivative to use: 0 for first argument, 1 for second, etc.
    """
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx=1e-5)
