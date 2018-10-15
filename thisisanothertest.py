import numpy as np
import matplotlib.pyplot as plt #for figures
#from mpl_toolkits.basemap import Basemap #to render maps
import math

from GrowYourIC import positions, geodyn, geodyn_trg, geodyn_static, plot_data, data, geodyn_analytical_flows

#plt.rcParams['figure.figsize'] = (8.0, 3.0) #size of figures
cm = plt.cm.get_cmap('viridis_r')


Yoshida = geodyn_analytical_flows.Yoshida96()


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


npoints = 30 #number of points in the x direction for the data set. 
data_set = data.PerfectSamplingCut(npoints, rICB = 1.)
data_set.method = "bt_point"
 

#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="age", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="age")
#plt.savefig("Ýoshida_age.pdf")


proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_tau_ic", verbose = False)
data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_tau_ic")
#plt.savefig("Yoshida_vM.pdf")

proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_cart", verbose = False)
data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_cart")
#plt.savefig("Yoshida_vM.pdf")

#proxy = geodyn.evaluate_proxy(data_set, Yoshida, proxy_type="vMises_acc", verbose = False)
#data_set.plot_c_vec(Yoshida, proxy=proxy, cm=cm, nameproxy="vMises_acc")
#plt.savefig("Yoshida_vM_acc.pdf")


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