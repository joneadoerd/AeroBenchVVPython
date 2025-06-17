import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import numpy as np
from aerobench.lowlevel.low_level_controller import LowLevelController
from aerobench.highlevel.autopilot import Autopilot
from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import print_state
import csv


def main():
    print("Enter F-16 Target Data:")
    lat = float(input("Latitude: "))
    lon = float(input("Longitude: "))
    alt = float(input("Altitude (meters): "))
    heading = float(input("Heading (degrees): "))
    print("\nF-16 Target Data:")
    print(f"Latitude: {lat}")
    print(f"Longitude: {lon}")
    print(f"Altitude: {alt} meters")
    print(f"Heading: {heading} degrees")

    # Set up initial F-16 state (using default trim, but set position and heading)
    llc = LowLevelController()
    initial_state = llc.xequil.copy()
    initial_state[9] = lat   # pos_n (for demo, not true lat)
    initial_state[10] = lon  # pos_e (for demo, not true lon)
    initial_state[11] = alt  # alt
    initial_state[5] = np.deg2rad(heading)  # psi (yaw)

    class SimpleAutopilot(Autopilot):
        def __init__(self, llc):
            super().__init__('level', llc)
        def get_u_ref(self, t, x_f16):
            # Hold level flight, straight
            return [1.0, 0.0, 0.0, 0.5]  # Nz, ps, Ny_r, throttle

    step = float(input("Simulation step time (seconds): "))
    ap = SimpleAutopilot(llc)
    tmax = 10.0  # seconds
    res = run_f16_sim(initial_state, tmax, ap, step=step)
    print("\nSimulation finished. Final state:")
    print_state(res['states'][-1])

    # Export trajectory to CSV for map view, including 3D velocity components
    csv_filename = "f16_sim_trajectory.csv"
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "lat(pos_n)", "lon(pos_e)", "alt", "vt", "vn", "ve", "vd"])
        for t, state in zip(res['times'], res['states']):
            vt = state[0]  # total speed
            alpha = state[1]  # angle of attack (rad)
            beta = state[2]   # sideslip angle (rad)
            theta = state[4]  # pitch angle (rad)
            psi = state[5]    # yaw angle (rad)
            # Body to NED (North-East-Down) velocity components
            # See Stevens & Lewis, or F-16 model docs
            # v_n = vt * cos(alpha) * cos(beta) * cos(theta) * cos(psi)
            # v_e = vt * cos(alpha) * cos(beta) * cos(theta) * sin(psi)
            # v_d = -vt * sin(theta)
            # But more accurately, use:
            vn = vt * np.cos(alpha) * np.cos(beta) * np.cos(theta) * np.cos(psi)
            ve = vt * np.cos(alpha) * np.cos(beta) * np.cos(theta) * np.sin(psi)
            vd = -vt * np.sin(theta)
            writer.writerow([t, state[9], state[10], state[11], vt, vn, ve, vd])
    print(f"\nTrajectory saved to {csv_filename}")


if __name__ == "__main__":
    main()
