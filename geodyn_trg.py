#!/usr/local/bin/python
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis



import numpy as np
import matplotlib.pyplot as plt #for figures
from mpl_toolkits.basemap import Basemap #to render maps
import math
from scipy.integrate import ode
from scipy.optimize import fsolve

#personal routines
import positions
import intersection 
import geodyn 
import mineral_phys_data


# def exact_translation(point, velocity, direction=positions.CartesianPoint(1,0,0)):
#     # TODO : is this function used anywhere??
#     RICB = 1221.
#     x_0, y_0, z_0 = point.x, point.y, point.z
#     mean_direction = np.sqrt(direction.x**2+ direction.y**2+ direction.z**2)
#     a, b, c = direction.x/mean_direction, direction.y/mean_direction, direction.z/mean_direction
# 
#     solution_1 = x_0*a+y_0*b+z_0*c + np.sqrt((x_0*a+y_0*b+z_0*c)**2-(x_0**2+y_0**2+z_0**2-RICB**2))
#     solution_2 = x_0*a+y_0*b+z_0*c - np.sqrt((x_0*a+y_0*b+z_0*c)**2-(x_0**2+y_0**2+z_0**2-RICB**    2))
#     ## TO DO : verify that we can remove solution_2 ? Is solution_1 always max?
#     return max(solution_1, solution_2)


class ModelTRG(geodyn.Model):
    
    def __init__(self):
        pass

    def velocity(self, t, position):
        """ Velocity at the given position and given time. 
            
            Need to be implemented in derived classes. 
            Needed in cartesian coordinates.
            
            Example :
            velocity, in cartesian coordinates
    
            position is a cartesian position [x, y, z]
            vt is the translation velocity (numpy array (1,3))
            rotation velocity is $\omega \vec{e}_z \times \vec{r}$.
            
            return vt + np.array([-omega * r[1], omega * r[0], 0.] )
            """
        raise NotImplementedError("need to implement velocity() in derived class!")

    def radius_ic(self, t):
        """ radius of the inner core with time. 
            
            Need to be implemented in derived classes.
            """
        raise NotImplementedError("need to implement radius_ic() in derived class!")

    def growth_ic(self, t):
        """ growth rate of the inner core with time. 
            
            Need to be implemented in derived classes.
            """
        raise NotImplementedError("need to implement growth_ic() in derived class!")

    def effective_growth_rate(self, t, point):
        """ Effective growth rate at the point r.

        v_{g_eff} = || v_growth + v_geodynamic ||
        v_geodynamic is already in cartesian coordinates.
        v_growth = ||v_growth|| * vec{e}_r (the unit vector for the radial direction)
        r is the position, described as x,y,z
        This function is used for points that are at the surface: r(t) is a point at the surface of the inner core at the time t.
        """
        r = np.array([point.x, point.y, point.z])
        vitesse = self.growth_ic(t)*point.er()-self.velocity(t, r)
        ## TO DO : project on \vec{e}_r to get the radial component only?
        return np.sqrt(vitesse[0]**2+vitesse[1]**2+vitesse[2]**2)

    def find_time_beforex0(self, point, t0, t1):
        """ find the intersection between the trajectory and the radius of the IC
        if needed, can be re defined in derived class!

        point : [x, y, z]
        """
        return intersection.zero_brentq(self.distance_to_radius, point, t0, a=0., b=t1)


    def proxy_singlepoint(self, point):
        """ evaluate the proxy on a single positions.Point instance."""
        proxy = {} #empty dictionnary

        if point.r< self.rICB:
            x, y, z = point.x, point.y, point.z
            time = self.find_time_beforex0([x, y, z], self.tau_ic, self.tau_ic)
        else: #in case a point is choosen artificially outside the IC, time bigger than tau_ic is allowed. This should in general not be used. 
            x, y, z = point.x, point.y, point.z
            time = self.find_time_beforex0([x, y, z], self.tau_ic, self.tau_ic*1.01)
        
        ## calculate the proxy needed (proxy_type value)
        if self.proxy_type == "age": 
            proxy["age"] = (self.tau_ic-time)*1.e3
        elif self.proxy_type == "domain_size":
            proxy["domain_size"] = self.domain_size(self.tau_ic-time)
        elif self.proxy_type == "dV_V":
            proxy["dV_V"] = self.dV_V(self.tau_ic-time)
        elif self.proxy_type == "theta" or self.proxy_type == "phi":
            point = self.integration_trajectory(time, [x,y,z], self.tau_ic)
            proxy["position"] = positions.CartesianPoint(point[0], point[1], point[2])
            proxy["phi"] = proxy["position"].phi
            proxy["theta"] = proxy["position"].theta
        elif self.proxy_type == "growth rate": 
            proxy["growth rate"] = self.effective_growth_rate(time, point)
        else: 
            print "unknown value for proxy_type."
            proxy = 0.
        return proxy

    def distance_to_radius(self, t, r0, t0):
        return self.trajectory_r(t, r0, t0)-self.radius_ic(t)

    def trajectory_r(self, t, r0, t0):
        """ for a point at position r0 at time t0, return the radial component of the position of the point at time t.
            
            """
        trajectory = self.integration_trajectory(t, r0, t0)
        #r, t, p = positions.from_cartesian_to_seismo(trajectory[0], trajectory[1], trajectory[2])
        r = trajectory[0]**2+trajectory[1]**2+trajectory[2]**2
        return np.sqrt(r)

    def integration_trajectory(self, t1, r0, t0):
        """ integration of the equation dr(t)/dt = v(r,t)
        
        return the position of the point at the time t1.
        r0: initial position
        t0: initial time
        t1: tmax of the integration
            """
        r = ode(self.velocity).set_integrator('dopri5')
        r.set_initial_value(r0, t0) # .set_f_params() if the function has any parameters
        return np.real(r.integrate(r.t+(t1-t0)))

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
    
    def domain_size(self, age):
        if self.units == None: #take age in seconds
            age = age * 1.e9 * np.pi* 1e7 #age inner core = 1Byears
        else: age = age # age has to be in seconds!
        return mineral_phys_data.domain_size(age)
    
    def dV_V(self, age):
        adimfrequency = mineral_phys_data.adimensional_frequency(self.domain_size(age))
        polynome = mineral_phys_data.export_matlab_data("Belanoshko_Vp_poly")
        return mineral_phys_data.convert_CM2008_velocity(adimfrequency, polynome)



    def plot_equatorial(self, t0, t1, Nt = 200,  N=40):
        # Plot the inner core boundary
        phi = np.linspace(0., 2*np.pi, N)
        x, y = self.rICB*np.cos(phi), self.rICB*np.sin(phi)
        plt.plot(x, y, 'b')
        for i, phi in enumerate(phi):
            trajx, trajy, trajz = self.trajectory_single_point(positions.CartesianPoint(x[i], y[i], 0.), t0, t1, Nt)
            trajectory_r = np.sqrt(trajx**2. + trajy**2 + trajz**2)
            mx = np.ma.masked_array(trajx, mask = trajectory_r >= self.rICB)
            my = np.ma.masked_array(trajy, mask = trajectory_r >= self.rICB)
            plt.plot(mx, my)
            plt.plot(mx[::10], my[::10], '+b')
            velocity = self.velocity(self.tau_ic, [trajx, trajy, trajz])
            plt.quiver(mx[::10], my[::10], np.ma.masked_array(velocity[0], mask = trajectory_r >= self.rICB)[::10], np.ma.masked_array(velocity[1], mask = trajectory_r >= self.rICB)[::10], units='width')
        plt.axis("equal")
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        #plt.show()

    def translation_velocity(self):
        try:
            self.vt
            assert(len(self.vt)==3)
        except (NameError, AttributeError):
            print "translation velocity has not been given. Please enter the three components of the velocity: (be careful, it has to be dimensionless)"
            value = map(float, raw_input("Translation velocity: ").split())
            self.set_vt(value)
        return self.vt

    def rotation_velocity(self, r):
        try:
            self.omega
        except (NameError, AttributeError):
            print "Rotation velocity has not been defined. Please enter the value for omega: "
            value = float(input("Rotation rate: "))
            self.set_rotation(value)
        return np.array([-self.omega * r[1], self.omega * r[0], 0.] )    


class PureTranslation(ModelTRG):
    
    def __init__(self):
        self.name = "Translation"

    def velocity(self, t, position):
      #  position = np.array(position)
      #  if position.ndim == 1: 
      #      ncols= 1
      #  else: 
      #      nlines, ncols = position.shape
      #  return np.repeat([self.vt], ncols, axis=0)
        return self.translation_velocity() 

    def radius_ic(self, t):
        return self.rICB
    
    def growth_ic(self, t):
        return 0.

    def verification(self):
        """ For pure translation, velocity has to be > 2*ricb/t_ic """
        v = np.sqrt(self.vt[0]**2+self.vt[1]**2+self.vt[2]**2)
        if v<= 2.* self.rICB/self.tau_ic :
            raise ValueError, "For pure translation, velocity has to be > 2*ricb/t_ic."
        try:
            self.rICB
            self.tau_ic
            self.vt
            self.proxy_type
        except NameError:
            raise NameError, "please verify the number of parameters.PureTranslationreqires: rICB, rau_ic, vt and proxy_type."
        self.units()



class TranslationRotation(ModelTRG):

    def __init__(self):
        self.name = "TranslationRotation"
    
    def velocity(self, t, r):
        """ velocity at the point position

        position is a np.array [x, y, z] 
        """
        return self.translation_velocity() + self.rotation_velocity(r)
   
    def radius_ic(self, t):
        return self.rICB

    def verification(self):
        """ For translation, velocity has to be (at least) > 2*ricb/t_ic. Please note that with rotation, it may need an even higher velocity. """
        v = np.sqrt(self.vt[0]**2+self.vt[1]**2+self.vt[2]**2)
        if v<= 2.* self.rICB/self.tau_ic :
            raise ValueError, "For translation, velocity has to be (at least) > 2*ricb/t_ic."
        try:
            self.rICB
            self.tau_ic
            self.vt
            self.proxy_type
            self.omega
        except NameError:
            raise NameError, "please verify the number of parameters. TranslationRotation requires: rICB, tau_ic, vt, omega and proxy_type."
        self.units()


class PureRotation(ModelTRG):

    def __init__(self):
        self.name = "Rotation"
    
    def velocity(self, t, r):
        """ velocity at the point position

        position is a np.array [x, y, z] 
        """
        
        return self.rotation_velocity() 

    def verification(self):
        """ pure rotation cannot give age values because the streamlines do not cross the inner core boundary."""
        if proxy_type == 'age':
            raise ValueError, "pure rotation cannot give age values because the streamlines do not cross the     inner core boundary." 
        try:
            self.rICB
            self.tau_ic
            self.proxy_type
            self.omega
        except NameError:
            raise NameError, "please verify the number of parameters. PureRotation requires: rICB, tau_ic, omega and proxy_type."
        self.units()

class PureGrowth(ModelTRG):

    def __init__(self):
        self.name = "PureGrowth"
    
    def velocity(self, t, r):
        """ velocity at the point position
    
        position is a np.array [x, y, z]
        """
        return np.array([0.,0.,0.]) 
  
    def verification(self):
        try:
            self.rICB
            self.tau_ic
            self.proxy_type
            self.exponent_growth
        except NameError:
            raise NameError, "please verify the number of parameters. PureGrowth requires: rICB, tau_ic, exponent_growth and proxy_type."
        self.units()

    def growth_ic(self, t):
        if self.exponent_growth == 0.:
            return 0.
        else:
            if t <=0. :
                return 0.
            else:
                return self.exponent_growth*self.rICB*(t/self.tau_ic)**(self.exponent_growth-1.)

    def radius_ic(self, t):
        return self.rICB*(t/self.tau_ic)**self.exponent_growth


class TranslationGrowth(ModelTRG):

    def __init__(self):
        self.name = "Translation and Growth"

    def velocity(self, t, r):
        """ velocity at the point position
    
        position is a np.array [x, y, z]
        """
        return self.translation_velocity() 
  
    def radius_ic(self, t):
        return self.growth_ic(t)

    def verification(self):
        try:
            self.rICB
            self.tau_ic
            self.proxy_type
            self.exponent_growth
            self.vt
        except NameError:
            raise NameError, "please verify the number of parameters. TranslationGrowth requires: rICB, tau_ic, exponent_growth, vt and proxy_type."
        self.units()

    def growth_ic(self, t):
        if self.exponent_growth == 0.:
            return 0.
        else:
            if t <=0. :
                return 0.
            else:
                return self.exponent_growth*self.rICB*(t/self.tau_ic)**(self.exponent_growth-1.)

    def radius_ic(self, t):
        return self.rICB*(t/self.tau_ic)**self.exponent_growth


class TranslationGrowthRotation(ModelTRG):

    def __init__(self):
        self.name = "Translation, Rotation and Growth"


    def velocity(self, t, r):
        """ velocity at the point position
    
        position is a np.array [x, y, z]
        """
        return self.translation_velocity()+ self.rotation_velocity(r)

    def verification(self):
        try:
            self.rICB
            self.tau_ic
            self.vt
            self.exponent_growth
            self.omega
            self.proxy_type
        except NameError:
            raise NameError, "At least one parameter is missing, please verify. Translation, Growth and Translation require: rICB, tau_ic, vt, exponent_growth, omega, proxy_type."
        self.units()

    def growth_ic(self, t):
        if self.exponent_growth == 0.:
            return 0.
        else:
            if t <=0. :
                return 0.
            else:
                return self.exponent_growth*self.rICB*(t/self.tau_ic)**(self.exponent_growth-1.)

    def radius_ic(self, t):
        return self.rICB*(t/self.tau_ic)**self.exponent_growth



def translation_velocity(center, amplitude):
    """ Give the value of the translation velocity to be used in geodyn models, in the format [x, y, z]
    
    center: [latitude, longitude]
    """

    direction = positions.SeismoPoint(1., center[0], center[1])
    return amplitude * np.array([direction.x, direction.y, direction.z])


def radial_derivative(fun, pos, dr, *args):
    """ compute the radial derivative of function fun(pos, *args) at the point r with finite difference 
    
    d fun/ dr = (fun(r+dr) - fun(r-dr))/(2*dr)

    Args:
        fun: a function that is defined as fun(pos, *args)
        pos: [x, y, z] list 
        dr: a length <1. 
        *args: additional arguments required by function fun
    Return the value of the radial derivative.
    """
    x, y, z = pos[0], pos[1], pos[2]
    r, theta, phi = positions.from_cartesian_to_seismo(x, y, z)
    r1 = min(1., r+dr)
    r2 = max(0., r-dr)
    x1, y1, z1 = positions.from_seismo_to_cartesian(r1, theta, phi)
    x2, y2, z2 = positions.from_seismo_to_cartesian(r2, theta, phi)

    return (fun([x1, y1, z1], *args)-fun([x2, y2, z2], *args))/(r1-r2)

