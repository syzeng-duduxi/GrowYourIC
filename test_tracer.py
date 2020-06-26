#!/usr/bin/env python3

from GrowYourIC import tracers, positions, geodyn, geodyn_trg, geodyn_static, plot_data, data, geodyn_analytical_flows
import numpy as np

if __name__ == "__main__":


    # model = geodyn_analytical_flows.Yoshida96(0.00)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_000")

    # model = geodyn_analytical_flows.Yoshida96(0.00)
    # tracers.Swarm(50, model, model.tau_ic/1000, "Yoshida/test_tracers_000_1000pts")

    # model = geodyn_analytical_flows.Yoshida96(0.10)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_010")

    # model = geodyn_analytical_flows.Yoshida96(0.20)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_020")


    # model = geodyn_analytical_flows.Yoshida96(0.05)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_005")

    # model = geodyn_analytical_flows.Yoshida96(0.15)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_015")

    # model = geodyn_analytical_flows.Yoshida96(0.25)
    # tracers.Swarm(50, model, model.tau_ic/200, "Yoshida/test_tracers_025")


    model = geodyn_analytical_flows.Yoshida96(0, S=-3)
    tracers.Swarm(7, model, model.tau_ic/400, "Yoshida_S_-3", plane="meridional")

    model = geodyn_analytical_flows.Yoshida96(0, S=-2)
    tracers.Swarm(7, model, model.tau_ic/400, "Yoshida_S_-2", plane="meridional")

    model = geodyn_analytical_flows.Yoshida96(0, S=1)
    tracers.Swarm(7, model, model.tau_ic/400, "Yoshida_S_1", plane="meridional")

    model = geodyn_analytical_flows.Yoshida96(0, S=2)
    tracers.Swarm(7, model, model.tau_ic/400, "Yoshida_S_2", plane="meridional")

    model = geodyn_analytical_flows.Yoshida96(0, S=3)
    tracers.Swarm(7, model, model.tau_ic/400, "Yoshida_S_3", plane="meridional")
