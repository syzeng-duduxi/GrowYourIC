#!/usr/bin/env python3

from GrowYourIC import tracers, positions, geodyn, geodyn_trg, geodyn_static, plot_data, data, geodyn_analytical_flows
import numpy as np

if __name__ == "__main__":


    model = geodyn_analytical_flows.Yoshida96(0.)
    tracers.Swarm(30, model, model.tau_ic/200)