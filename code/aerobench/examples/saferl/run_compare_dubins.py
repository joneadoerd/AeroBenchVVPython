import math
import pickle
import os
from aerobench.util import get_script_path, StateIndex

import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim, F16SimState

from aerobench.visualize import plot

from wingman_autopilot import WingmanAutopilot

class WingmanF16State:
    '''object containing simulation state

    With this interface you can run partial simulations, rather than having to simulate for the entire time bound

    if you just want a single run with a fixed time, it may be easier to use the run_f16_sim function
    '''

    def __init__(self, initial_state):

        ### Initial Conditions ###
        power = 9 # engine power level (0-10)

        # Default alpha & beta
        alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
        beta = 0                # Side slip angle (rad)

        # Initial Attitude
        alt = 1000        # altitude (ft)
        #vt = 600          # initial velocity (ft/sec)
        phi = 0           # Roll angle from wings level (rad)
        theta = 0         # Pitch angle from nose level (rad)
        #psi = 0           # Yaw angle from North (rad)

        x0, y0, heading0, v0 = initial_state

        vt = v0
        #psi = heading0 + np.pi
        psi = np.pi / 2 - heading0
        pe = x0
        pn = y0

        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        initial_state = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, pn, pe, alt, power]

        self.ap = WingmanAutopilot(target_heading=psi, target_vel=v0, target_alt=alt, stdout=True)

        step = 1.0
        self.tmax = 0.0 # current time in simulation

        
        self.fss = F16SimState(initial_state, self.ap, step=1.0, extended_states=True)

    def one_step_with_control(self, control):
        '''run one step with the passed in control (rudder, throttle)'''

        self.ap.targets[0] -= control[0] # modify target heading by rudder control input
        self.ap.targets[1] += control[1] # modify target velocity by throttle control input

        self.tmax += 1.0
        self.fss.simulate_to(self.tmax, update_mode_at_start=True) # TODO: check if we need to be increasing the time by 1.0 here

def main():
    'main function'

    dubins_filename = "dubins_data.pkl"
    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    with open(dubins_file_path, 'rb') as file:
        data = pickle.load(file)
        
    print(f"Loaded data from {dubins_filename}")
    states_np_list, actions_np_list = data
    traj_one = states_np_list[0]
    actions_one = actions_np_list[0]
    num_steps = traj_one.shape[1]
    init_state = traj_one[:, 0] # x0, y0, heading0, v0
    
    tmax = 70 # simulation time 

    f16 = WingmanF16State(init_state)

    #ap = WingmanAutopilot(target_heading=math.pi/2, target_vel=400, target_alt=alt, stdout=True)

    #step = 1/30
    #extended_states = False
    #res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')
   
    # print(f"res['states']: {res['states']}")

    AP_VEL_TARGET_INDEX = 1 
    AP_PSI_TARGET_INDEX = 0
    vel_targets = [f16.ap.targets[AP_VEL_TARGET_INDEX]]
    psi_targets = [f16.ap.targets[AP_PSI_TARGET_INDEX]]
    u_refs = None

    for i, action in enumerate(actions_one.T):
        f16.one_step_with_control(action)
        vel_targets.append(f16.ap.targets[AP_VEL_TARGET_INDEX])
        psi_targets.append(f16.ap.targets[AP_PSI_TARGET_INDEX])

        if u_refs is None:
            first_state = f16.fss.states[0]
            u_refs = [f16.ap.get_u_ref(0, first_state)]

        u_ref = f16.ap.get_u_ref(0, f16.fss.states[-1])
        u_refs.append(u_ref)

    res = {'states': np.array(f16.fss.states), 'times': np.array(f16.fss.times),
           'u_list': f16.fss.u_list, 'ps_list': f16.fss.ps_list, 'Nz_list': f16.fss.Nz_list, 'Ny_r_list': f16.fss.Ny_r_list}

    throttle_cmds = [u_tuple[3] for u_tuple in f16.fss.u_list]

    plot.plot_single(res, 'psi', title='Psi (heading angle, radians)')
    plt.plot(res['times'], res['states'][:, StateIndex.PSI], 'k-', label='actual')

    # plot the target psi as a dotted line
    plt.plot(res['times'], psi_targets, 'r--', label='target')
    # add reference line for 3pi/2
    plt.axhline(y=3*np.pi/2, color='g', linestyle='--', label='3pi/2')
    plt.legend()
    filename = 'psi.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # vt
    #plot.plot_single(res, 'vt', title='Velocity (ft/sec)') # clears the plot
    # clear plot
    plt.clf()
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(res['times'], res['states'][:, StateIndex.VT], 'k-', label='actual')
    axs[0].plot(res['times'], vel_targets, 'r--', label='target')
    axs[0].set_title('Velocity (ft/sec)')

    axs[1].plot(res['times'], throttle_cmds, 'b-', label='throttle')
    axs[1].set_title('Throttle Cmds')

    axs[2].plot(res['times'], [u_ref[3] for u_ref in u_refs], 'b-', label='throttle')
    axs[2].set_title('Throttle URefs')

    # add legend entry for blue '-' line called 'actual'
    

    plt.legend()
    filename = 'vel.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    ax = plot.plot_overhead(res)
    plt.plot(res['states'][:, StateIndex.POS_E], res['states'][:, StateIndex.POS_N], 'k-', label='F16 recreation')

    # add overhead plot of first dubins path traj_one rows 0 and 1
    ax.plot(traj_one[0], traj_one[1], 'r', label='Dubins')
    ax.plot(traj_one[0, 0], traj_one[1, 0], 'ro')
    ax.legend()

    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    
    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

if __name__ == '__main__':
    main()
