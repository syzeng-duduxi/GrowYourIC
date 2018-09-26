#!/usr/local/bin/python
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis


from __future__ import division
from __future__ import absolute_import


import numpy as np
import matplotlib.pyplot as plt  # for figures
from mpl_toolkits.basemap import Basemap  # to render maps
import math
from scipy.integrate import ode
from scipy.optimize import fsolve
from scipy.misc import derivative

# personal routines
from . import positions
from . import intersection
from . import geodyn
from . import mineral_phys


def e_r(r, theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
def e_theta(r, theta, phi):
    return np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
def e_phi(r, theta, phi):
    return np.array([-np.sin(phi), np.cos(phi), 0.])

def e_x(r, theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)])
def e_y(r, theta, phi):
    return np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)])
def e_z(r, theta, phi):
    return np.array([np.cos(theta), np.sin(theta), 0.])
 





class Analytical_Model(geodyn.Model):

    def integration_trajectory(self, t1, r0, t0):
        """ integration of the equation dr(t)/dt = v(r,t)

        return the position of the point at the time t1.
        r0: initial position
        t0: initial time
        t1: tmax of the integration
            """
        r = ode(self.velocity).set_integrator('dopri5')
        # .set_f_params() if the function has any parameters
        r.set_initial_value(r0, t0)
        return np.real(r.integrate(r.t + (t1 - t0)))

    def trajectory_single_point(self, point, t0, t1, num_t):
        """ return the trajectory of a point (a positions.Point instance) between the times t0 and t1, knowing that it was at the position.Point at t0, given nt times steps. 
        """
        time = np.linspace(t0, t1, num_t)
        x, y, z = np.zeros(num_t), np.zeros(num_t), np.zeros(num_t)
        x[0], y[0], z[0] = point.x, point.y, point.z
        for i, t in enumerate(time):
            point = self.integration_trajectory(t, [x[0], y[0], z[0]], t0)
            x[i], y[i], z[i] = point[0], point[1], point[2]
        return x, y, z

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
            proxy["vMises_acc"] = self.deformation_accumulated(point, time, self.tau_ic, 20) 
        elif proxy_type == "age":
            time = self.crystallisation_time([x, y, z], self.tau_ic)
            proxy["age"] = (self.tau_ic - time)
        return proxy

    def radius_ic(self, t):
        """ radius of the inner core with time. 
        """
        return self.rICB * (t / self.tau_ic)**self.alpha

    def u_growth(self, t):
        """ growth rate at a given time t (dimensional) """
        if t<0.01:
            return 0.
        return (t/self.tau_ic)**(self.alpha-1)*self.alpha*self.rICB/self.tau_ic

    def u_a(self, t):
        """ u_growth/r_icb """
        return 1./self.tau_ic #TODO verify (not true for alpha != 0.5 )
        
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

    def find_time_beforex0(self, point, t0, t1):
        """ find the intersection between the trajectory and the radius of the IC
        if needed, can be re defined in derived class!

        point : [x, y, z]
        """
        return intersection.zero_brentq(self.distance_to_radius, point, t0, a=0., b=t1)

    def distance_to_radius(self, t, r0, t0):
        return self.trajectory_r(t, r0, t0) - self.radius_ic(t)

    def trajectory_r(self, t, r0, t0):
        """ for a point at position r0 at time t0, return the radial component of the position of the point at time t.

            """
        trajectory = self.integration_trajectory(t, r0, t0)
        #r, t, p = positions.from_cartesian_to_seismo(trajectory[0], trajectory[1], trajectory[2])
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
        # .set_f_params() if the function has any parameters
        r.set_initial_value(r0, t0)
        return np.real(r.integrate(r.t + (t1 - t0)))

    def trajectory_single_point(self, point, t0, t1, num_t):
        """ return the trajectory of a point (a positions.Point instance) between the times t0 and t1, knowing that it was at the position.Point at t0, given nt times steps. 
        """
        time = np.linspace(t0, t1, num_t)
        x, y, z = np.zeros(num_t), np.zeros(num_t), np.zeros(num_t)
        x[0], y[0], z[0] = point.x, point.y, point.z
        for i, t in enumerate(time):
            point = self.integration_trajectory(t, [x[0], y[0], z[0]], t0)
            x[i], y[i], z[i] = point[0], point[1], point[2]
        return x, y, z

    def deformation_accumulated(self, point, t_crys, tau_ic, N):
        """ Accumulation of strain on the parcel of material located at position point at t_ic """
        trajectoire_x, trajectoire_y, trajectoire_z = self.trajectory_single_point(point, t_crys, tau_ic, N)
        deformation_acc = 0.
        time = np.linspace(t_crys, tau_ic, N)
        for i, ix in enumerate(trajectoire_x):
            position_point = positions.CartesianPoint(trajectoire_x[i], trajectoire_y[i], trajectoire_z[i])
            deformation_acc = deformation_acc + (self.deformation(time[i], position_point))**2
        deformation_acc = np.sqrt(deformation_acc)/N
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
        # spherical coordinates
        Point_full_position = point # positions.CartesianPoint(point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, (90.-Point_full_position.theta)*np.pi/180., Point_full_position.phi*np.pi/180. 
        # coefficients
        a = self.rICB #radius of inner core. Has to be set to 1 if r is already non-dimensional.
        epsilon_rr = partial_derivative(self.u_r, 0, [r, theta])
        epsilon_tt = partial_derivative(self.u_theta, 1, [r, theta])/r+self.u_r(r, theta)/r
        epsilon_pp = self.u_r(r, theta)/r + self.u_theta(r, theta)*np.cos(theta)/np.sin(theta)/r
        def vt_r(r, theta):
            return self.u_theta(r, theta)/r
        epsilon_rt = 0.5*(r*partial_derivative(vt_r, 0, [r, theta])+partial_derivative(self.u_r, 1, [r, theta])/r)
        return np.sqrt(2./3.*(epsilon_rr**2+epsilon_tt**2+epsilon_pp**2+2*epsilon_rt**2))





class Model_Yoshida96(Analytical_Model):
    """ Analytical model from Yoshida 1996 with preferential flow at the equator. """

    def __init__(self):
        self.name = "Yoshida model based on Yoshida et al. 1996"
        self.rICB = 1.
        self.alpha = 0.5
        self.S2 = 2./5.
        self.tau_ic = 1.
        self.u_t = 0.#0.5e-3

    def verification(self):
        pass

    def velocity(self, time, point):
        """ Velocity at the given position and given time. 
         
        time: time (float)
        point: [x, y, z]
        Output is velocity in cartesian geometry [v_x, v_y, v_z]
        """
        # start by defining the spherical unit vector in cartesian geometry (so that we can use all equations from Yoshida 1996)
        #theta is colatitude! Angles are in radians to be used in cos and sin functions.
        if len(point)==3 and type(point)== type(np.array([0,0,0])):
            if point[0]==0 and point[1]==0 and point[2]==0:
                return [0., 0., 0.]
        Point_full_position = positions.CartesianPoint(point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, (90.-Point_full_position.theta)*np.pi/180., Point_full_position.phi*np.pi/180. 
        norm_u = self.u_growth(time) #growth rate (average)
        S2 = self.S2 #S2 coefficient, see Yoshida 1996 for definition
        a = self.rICB #radius of inner core. Has to be set to 1 if r is already non-dimensional.
        u_r = norm_u*S2* self.u_r(r, theta, time)  #(8.*(r/a)-3.*(r/a)**3) * (3.*np.cos(theta)*np.cos(theta)-1.)/10.
        u_theta = norm_u*S2* self.u_theta(r, theta, time) #(-24.*(r/a) + 15. *(r/a)**3) * (np.cos(theta)*np.sin(theta))/10. 
        u_phi = 0. #self.u_phi(r, theta)
        velocity = u_r * e_r(r, theta, phi) + u_theta * e_theta(r, theta, phi) + u_phi*e_phi(r, theta, phi)
        velocity = velocity + self.u_t*np.array([1, 0, 0])#with artificial translation
        return velocity

    def u_r(self, r, theta, time):
        a = self.radius_ic(time)
        return (8.*(r/a)-3.*(r/a)**3) * (3.*np.cos(theta)*np.cos(theta)-1.)/10. #2.*3.*self.Y20(theta)*self.P20(r)/r
    
    def u_theta(self, r, theta, time):
        a = self.radius_ic(time)
        return (-24.*(r/a) + 15. *(r/a)**3) * (np.cos(theta)*np.sin(theta))/10. # derivative(p20_r, r, dx=1e-6)/r*derivative(self.Y20, theta, dx=1e-6)

    def epsilon_rr(self, r, theta, phi, time):
        a = self.radius_ic(time) 
        return (8-9*(r/a)**2)*(3*np.cos(theta)**2-1)/10
    def epsilon_tt(self, r, theta, phi, time):
        a = self.radius_ic(time) 
        return (  8*(2-3*np.cos(theta)**2) +3*r**2*(7*np.cos(theta)**2-4)  )/10
    def epsilon_pp(self, r, theta, phi, time):
        a = self.radius_ic(time) 
        return (-8 + 3*(r/a)**2*(2*np.cos(theta)**2+1))/10
    def epsilon_rt(self, r, theta, phi, time):
        a = self.radius_ic(time) 
        return 24/10*(-1+(r/a)**2)*np.cos(theta)*np.sin(theta)
    def epsilon_rp(self, r, theta, phi, time):
        return 0.
    def epsilon_tp(self, r, theta, phi, time):
        return 0.

    def vonMises_eq(self, r, theta, phi, time):
        sum = self.epsilon_pp(r, theta, phi, time)**2\
                +self.epsilon_rr(r, theta, phi, time)**2\
                +self.epsilon_tt(r, theta, phi, time)**2\
                +2*self.epsilon_rp(r, theta, phi, time)**2\
                +2*self.epsilon_rt(r, theta, phi, time)**2\
                +2*self.epsilon_tp(r, theta, phi, time)**2
        return np.sqrt(2/3*sum)

    def deformation(self, time, point):
        r, theta, phi = point.r, (90.-point.theta)*np.pi/180., point.phi*np.pi/180. 
        return self.u_a(time)*self.vonMises_eq(r, theta, phi, time)



class Model_LorentzForce(Analytical_Model):

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
        #TODO add value P 
        P = self.P
        return (-r**6+14./5.*r**4-9./5.*r**2  +204./5*r**4/(19.+5.*P) -544./5.*r**2/(19.+5.*P)  )     /(3.**3*7.*np.sqrt(5.)) 

    def Y20(self, theta):
        """ Spherical Harmonics Y_2^0 at the given position
        
        point: [x, y, z]
        Output: float
        """ 
        return np.sqrt(5)/2.*(3*np.cos(theta)**2-1.)

    def u_r(self, r, theta):
        return 2.*3.*self.Y20(theta)*self.P20(r)/r
    
    def u_theta(self, r, theta):
        def p20_r(radius):
            return self.P20(radius)*radius
        return derivative(p20_r, r, dx=1e-6)/r*derivative(self.Y20, theta, dx=1e-6)

    def velocity(self, time, point):
        """ Velocity at the given position and given time. 
         
        time: time (float)
        point: [x, y, z]
        Output is velocity in cartesian geometry [v_x, v_y, v_z]
        """ 
        Point_full_position = positions.CartesianPoint(point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, (90.-Point_full_position.theta)*np.pi/180., Point_full_position.phi*np.pi/180. 
        # def spherical coordinates vectors in cartesian coordinates
        e_r = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        
        velocity = self.u_r(r, theta) * e_r + self.u_theta(r, theta)*e_theta
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
        Point_full_position = point # positions.CartesianPoint(point[0], point[1], point[2])
        r, theta, phi = Point_full_position.r, (90.-Point_full_position.theta)*np.pi/180., Point_full_position.phi*np.pi/180. 
        # coefficients
        a = self.rICB #radius of inner core. Has to be set to 1 if r is already non-dimensional.
        epsilon_rr = partial_derivative(self.u_r, 0, [r, theta])
        epsilon_tt = partial_derivative(self.u_theta, 1, [r, theta])/r+self.u_r(r, theta)/r
        epsilon_pp = self.u_r(r, theta)/r + self.u_theta(r, theta)*np.cos(theta)/np.sin(theta)/r
        def vt_r(r, theta):
            return self.u_theta(r, theta)/r
        epsilon_rt = 0.5*(r*partial_derivative(vt_r, 0, [r, theta])+partial_derivative(self.u_r, 1, [r, theta])/r)
        return np.sqrt(2./3.*(epsilon_rr**2+epsilon_tt**2+epsilon_pp**2+2*epsilon_rt**2))

def partial_derivative(func, var=0, point=[]):
    """ Partial derivative of a function fun
    
    var indicates which derivative to use: 0 for first argument, 1 for second, etc.
    """
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-5)


if __name__ == "__main__":

    Yoshida = Model_Yoshida96()

