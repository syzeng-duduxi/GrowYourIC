# -*- coding: UTF-8 -*-


import numpy as np
import matplotlib.pyplot as plt #for figures
#from mpl_toolkits.basemap import Basemap #to render maps
import math

from GrowYourIC import tracers, positions, geodyn, geodyn_trg, geodyn_static, plot_data, data, geodyn_analytical_flows

#plt.rcParams['figure.figsize'] = (8.0, 3.0) #size of figures
cm = plt.cm.get_cmap('viridis_r')

#V = 0.2 # translation velocity
#S2 = 1/5.

#Yoshida = geodyn_analytical_flows.Yoshida96(V, S=S2)
#file = "Fig/Yoshida_{}_S2_{}".format(V, S2)
#print(file)

V = [0.2, 0.4]
S2 = [1/5., 4/5., 2.]
    
for vitesse in V:
    for value_S in S2: 
        Yoshida = geodyn_analytical_flows.Yoshida96(vitesse, S=value_S)
        file = "Fig/Yoshida_{}_S2_{}".format(vitesse, value_S)
        print(file)

        npoints = 50 #number of points in the x direction for the data set. 
        data_set = data.PerfectSamplingCut(npoints, rICB = 1.)
        data_set.method = "bt_point"

        # Age plot with velocity field
        proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="age", verbose = False)
        data_set.plot_c_vec(Yoshida, proxy=proxy, nameproxy="age")
        plt.savefig(file+"_age.pdf")


# accumulated deformation
#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_acc", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_acc")
#plt.savefig(file+"_vM_acc.pdf")
#data_set.plot_c_vec(Yoshida, proxy=np.log10(proxy), cm=cm, nameproxy="log_vMises_acc")
#plt.savefig(file+"_log_vM_acc.pdf")

# tracers with age
#tracers.Swarm(5, Yoshida, Yoshida.tau_ic/400, "ici", plane="meridional")


#data_set = data.PerfectSamplingCut(20, rICB = 1.)
#data_set.method = "bt_point"
#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="age", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, nameproxy="age")

#plt.show()

#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_tau_ic", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_tau_ic")
#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_acc", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_acc")


#Karato = geodyn_analytical_flows.Model_LorentzForce()
#proxy = geodyn.evaluate_proxy(data_set, Karato, proxy_type="vMises_tau_ic", verbose = False)
#data_set.plot_c_vec(Karato, proxy=proxy, cm=cm, nameproxy="vMises_tau_ic")

#Karato.P = 1e-4
#proxy = geodyn.evaluate_proxy(data_set, Karato, proxy_type="vMises_tau_ic", verbose = False)
#data_set.plot_c_vec(Karato, proxy=proxy, cm=cm, nameproxy="vMises_tau_ic")

#proxy = geodyn.evaluate_proxy(data_set, Karato, proxy_type="age", verbose = False)
#data_set.plot_c_vec(Karato, proxy=proxy, cm=cm, nameproxy="age")




#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="age", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="age")
#plt.savefig(file+"_tage.pdf")


#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_tau_ic", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_tau_ic")
#plt.savefig(file+"_t_vM.pdf")

#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_cart", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_cart")
#plt.savefig("Yoshida_vM.pdf")

#Karato.P = 1e4
#proxy_1 = geodyn.evaluate_proxy(data_set, Karato, proxy_type="vMises_tau_ic", verbose = False)
#data_set.plot_c_vec(Karato, proxy=proxy_1, cm=cm, nameproxy="vMises_tau_ic")



#npoints = 50 #number of points in the x direction for the data set. 
#data_set = data.PerfectSamplingCut(npoints, rICB = 1.)
#data_set.method = "bt_point"
#proxy_2 = geodyn.evaluate_proxy(data_set, Karato, proxy_type="age", verbose = False)
#data_set.plot_c_vec(Karato, proxy=proxy_2, cm=cm, nameproxy="age")

#npoints = 100 #number of points in the x direction for the data set. 
#data_set = data.PerfectSamplingCut(npoints, rICB = 1.)
#data_set.method = "bt_point"
#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_acc", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_acc")


plt.show()