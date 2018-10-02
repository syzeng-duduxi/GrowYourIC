from GrowYourIC import tracers, positions, geodyn, geodyn_trg, geodyn_static, plot_data, data, geodyn_analytical_flows



if __name__ == "__main__":

    model = geodyn_analytical_flows.Model_Yoshida96()
    position = positions.CartesianPoint(20., 2., 0.)
    track = tracers.Tracer(position, model, 1e6, 5)
    print(track.traj_x, track.traj_y, track.traj_z)
    
    track.spherical()
    track.output(1)


    tracers.Swarm(3, model)