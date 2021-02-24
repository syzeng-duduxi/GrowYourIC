# import statements
import numpy as np
import matplotlib.pyplot as plt #for figures
from mpl_toolkits.basemap import Basemap #to render maps
import math
import json #to write dict with parameters

from GrowYourIC import positions, geodyn, geodyn_trg, geodyn_static, plot_data, data

plt.rcParams['figure.figsize'] = (8.0, 3.0) #size of figures
cm = plt.cm.get_cmap('viridis')
cm2 = plt.cm.get_cmap('winter')

## real data set
data_set = data.SeismicFromFile("~/ownCloud/Research/Projets/CIDER_IC/GrowYourIC/GrowYourIC/data/WD11.dat")
residual = data_set.real_residual()

velocity_center = [0., -80]#center of the eastern hemisphere
r, t, p = data_set.extract_rtp("bottom_turning_point")
dist = positions.angular_distance_to_point(t, p, *velocity_center)



fig, ax = plt.subplots(2)
ax[0].hist(1221*(1-r))

zeta = data_set.extract_zeta()

ax[1].hist(zeta)



fig, ax = plt.subplots(sharey=True, figsize=(8, 2))
cm2 = plt.cm.get_cmap('winter')
sc1 = ax.scatter(p, residual, c=zeta, s=10,cmap=cm2, linewidth=0)
ax.set_xlabel("longitude")
ax.set_ylabel("residuals")
ax.set_xlim([-180, 180])
#sc2 = ax[1].scatter(dist, residual, c="k", s=10,cmap=cm2, linewidth=0)
#ax[1].set_xlabel("angular distance to ({}, {})".format(*velocity_center))
#ax[1].set_xlim([0, 180])
#fig.suptitle("Dataset: {},\n geodynamic model: {}".format(data_set_random.name, geodynModel.name))
cbar2 = fig.colorbar(sc1)
cbar2.set_label("zeta")

fig, ax = plt.subplots(figsize=(8, 2))
rICB_dim = 1221. #in km
sc=ax.scatter(p,rICB_dim*(1.-r), c=residual, s=10,cmap=cm, linewidth=0)
ax.set_ylim(-0,120)
fig.gca().invert_yaxis()
ax.set_xlim(-180,180)
cbar = fig.colorbar(sc)
cbar.set_label("Residual")
ax.set_xlabel("longitude")
ax.set_ylabel("depth (km)")
ax.plot([11,11],[10,30], 'k')
ax.plot([21,21],[30,58], 'k')
ax.plot([38,38],[58,110], 'k')
ax.plot([-80,100], [30,30], 'k:')
ax.plot([-80,100], [58,58], 'k:')


points = [13, 234, 456, 1234, 2343, 27, 56, 567, 789]

for point_value in points: 
    point = data_set[point_value]
    print(point)
    point.straight_in_out(30)
    traj_r = np.zeros(30)
    traj_p = np.zeros(30)
    for i, po in enumerate(point.points):
        r, t, p = po.r, po.theta, po.phi-180.
        traj_r[i] =rICB_dim*(1.-r) 
        traj_p[i] = p
    ax.plot(traj_p, traj_r, 'k')

plt.savefig("test.pdf")
print(r.shape, residual.shape)

fig, ax = plt.subplots(1, 4, sharey=True, sharex=True)
sc = ax[0].scatter(residual, zeta, c=dist , cmap="seismic", linewidth=0, s=10)
cbar = fig.colorbar(sc)

masks = [np.squeeze(rICB_dim*(1.-r))<30, np.squeeze(rICB_dim*(1.-r))>58, (np.squeeze(rICB_dim*(1.-r))>30)*np.squeeze(rICB_dim*(1.-r))<58] 
#mask = np.squeeze(rICB_dim*(1.-r))<30
#print(mask.shape, zeta.shape)
zeta = np.squeeze(zeta)
dist = np.squeeze(dist)
for i, mask in enumerate(masks):
    ax[i+1].scatter(np.ma.masked_where(mask, (residual)), np.ma.masked_where(mask, zeta), c= np.ma.masked_where(mask, dist), s=10, cmap="seismic", linewidth=0)


plt.show()