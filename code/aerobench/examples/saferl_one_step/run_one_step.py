import math

import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import StateIndex, get_state_names

from aerobench.visualize import plot

from wingman_autopilot import WingmanAutopilot

def load_csv():
    'load csv file data'

    filename = 'eval_csvfile.csv'

    # first line is header, load into a dict keyed by the tuple: int(round(float(col0))), str(col1)
    import csv
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader) # skip header
        alldata = {}
        
        for row in reader:
            key = int(round(float(row[0]))), str(row[1])
            # use header[2:] as keys
            rowdata = {}
            for i, col in enumerate(header[2:]):
                rowdata[col] = float(row[i+2])

            alldata[key] = rowdata

    print(f"Loaded {len(alldata)} rows from {filename}")
    return alldata

def get_wingman_autopilot_init(x0=0, y0=0, heading0=np.pi/2, v0=550):
    ### Initial Conditions ###
    power = 9 #9 # engine power level (0-10)

    # Default alpha & beta
    alpha = 0 #deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1000 #3600 #3800        # altitude (ft)
    vt = v0          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = heading0 #math.pi/8   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, y0, x0, alt, power]
    return init

def main():
    'main function'

    csvdata = load_csv()

    init_x = csvdata[(0, 'lead')]['x']
    init_y = csvdata[(0, 'lead')]['y']
    init_heading = csvdata[(0, 'lead')]['heading']
    init_v = csvdata[(0, 'lead')]['v']

    #print(f"init_x: {init_x}, init_y: {init_y}, init_heading: {init_heading}, init_v: {init_v}")
    # print all lead state 0 values
    print("Lead State at step 0 in csv:")
    data0 = csvdata[(0, 'lead')]
    for key, value in data0.items():
        print(f"  {key}: {value}")

    init = get_wingman_autopilot_init(init_x, init_y, np.pi/2 - init_heading, init_v)

    print(f'\nF16 initial state:')
    var_names = get_state_names()
    for i, name in enumerate(var_names):
        print(f'  {name}: {init[i]}')

    ap = WingmanAutopilot(target_heading=np.pi/2 - init_heading, target_vel=init_v, 
        target_alt=1000, stdout=True)

    tmax = 1 # simulation time

    step = 1/30
    extended_states = False
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')
    #print(f"res.keys(): {res.keys()}")
    # print(f"res['states']: {res['states']}")
    # print(f"res['states'].shape: {res['states'].shape}")
    # print(f"res['states'][-1]: {res['states'][-1]}")

    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    plot.plot_single(res, 'alt', title='Altitude (ft)')
    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # vt
    plot.plot_single(res, 'vt', title='Velocity (ft/sec)')
    filename = 'vel.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_overhead(res)
    filename = 'overhead.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # get state at step 1 from csv file
    x1 = csvdata[(1, 'lead')]['x']
    y1 = csvdata[(1, 'lead')]['y']
    heading1 = csvdata[(1, 'lead')]['heading']
    v1 = csvdata[(1, 'lead')]['v']
    print(f"In CSV file at step 1: x: {x1}, y: {y1}, heading: {heading1}, v: {v1}")

    # get state at step 1 from simulation
    
    #print(f"Simulated: x: {res['states'][-1][9]}, y: {res['states'][-1][10]}, heading: {res['states'][-1][5]}, v: {res['states'][-1][0]}")
    # use StateIndex.XXX
    x_sim = res['states'][-1][StateIndex.POS_E]
    y_sim = res['states'][-1][StateIndex.POS_N]
    heading_sim = np.pi/2 - res['states'][-1][StateIndex.PSI]
    v_sim = res['states'][-1][StateIndex.VT]

    print(f"Simulated at time 1.0: x: {x_sim}, y: {y_sim}, heading: {heading_sim}, v: {v_sim}")

    


if __name__ == '__main__':
    main()
