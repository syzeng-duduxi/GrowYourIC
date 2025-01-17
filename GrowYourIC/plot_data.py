#!/usr/bin/env python3
# Project : From geodynamic to Seismic observations in the Earth's inner core
# Author : Marine Lasbleis
""" Define routines for maps. Maybe non-useful. """


import numpy as np
import matplotlib.pyplot as plt  # for figures
# from mpl_toolkits.basemap import Basemap  # to render maps

# personal routines
from . import positions


def setting_map():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(projection='moll', lon_0=0., resolution='c')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    m.drawmeridians(np.arange(0, 360, 30))
    m.drawparallels(np.arange(-90, 90, 30))
    m.drawmapboundary(fill_color='#99ffff')
    return m, fig


def setting_map_ortho():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(projection='ortho', lat_0=45, lon_0=-100, resolution='c')
    m.drawcoastlines(linewidth=0.25)
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')
    m.drawmeridians(np.arange(0, 360, 30))
    m.drawparallels(np.arange(-90, 90, 30))
    m.drawmapboundary(fill_color='#99ffff')
    return m, fig
