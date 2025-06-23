import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import deg2rad

from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import StateIndex
from aerobench.visualize import plot


def generate_initial_conditions(alt=3800, vt=540, psi=0, power=9):
    '''Create initial condition vector with variation'''
    alpha = deg2rad(2.1215)
    beta = 0
    phi = 0
    theta = 0
    return [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]


def generate_waypoints(offset_e=0, offset_n=0):
    '''Generate a waypoint path with offsets for each aircraft'''
    return [
        [1000 + offset_e, 3000 + offset_n, 4000],
        [3000 + offset_e, 8000 + offset_n, 3900],
        [1000 + offset_e, 18000 + offset_n, 3750],
        [1500 + offset_e, 28000 + offset_n, 300]
    ]


def run_all_simulations(configs, tmax=75, step=1/30):
    '''Run all aircraft simulations and return results'''
    all_results = []
    for config in configs:
        ap = WaypointAutopilot(config["wps"], stdout=True)
        print(f"Starting simulation for Aircraft {config['id']}")
        res = run_f16_sim(config["init"], tmax, ap, step=step,
                          extended_states=True, integrator_str='rk45')
        
        # Inspect available keys
        print(f"Keys in result for Aircraft {config['id']}: {list(res.keys())}")
        
        # Just use the full result for now
        res["id"] = config["id"]
        res["wps"] = config["wps"]
        res["autopilot"] = ap
        all_results.append(res)
    return all_results


def animate_aircrafts(all_results, tmax, step):
    fig, ax = plt.subplots()
    ax.set_title("Aircraft Overhead View")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.grid(True)

    ax.set_xlim(-20000, 10000)
    ax.set_ylim(-20000, 10000)

    colors = ['blue', 'green', 'red']
    markers = ['o', '^', 's']
    lines = []
    labels = []

    state_key = 'states'  # corrected key from your info
    time_key = 'times'

    # Plot waypoint paths and points first (static)
    for i, res in enumerate(all_results):


        plot.plot_overhead(res, waypoints=res["wps"], llc=res["autopilot"].llc)
        filename = 'overhead.png'
        plt.savefig(filename)
        print(f"Made {filename}")
        


        waypoints = res["wps"]
        way_x = [wp[0] for wp in waypoints]
        way_y = [wp[1] for wp in waypoints]
        ax.plot(way_x, way_y, marker='x', linestyle='None', color=colors[i], label=f'WP Aircraft {i+1}')
        ax.plot(way_x, way_y, linestyle='-', color=colors[i], alpha=0.5)

    # Create markers for aircraft positions
    for i in range(len(all_results)):
        line, = ax.plot([], [], marker=markers[i], color=colors[i], label=f'Aircraft {i+1}', markersize=8)
        label = ax.text(0, 0, "", fontsize=10, color=colors[i])
        lines.append(line)
        labels.append(label)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines + labels

    def update(frame):
        t = frame * step
        for i, res in enumerate(all_results):
            state = res[state_key]
            time = res[time_key]
            idx = max(j for j, t_j in enumerate(time) if t_j <= t)
            s = state[idx]
            east = s[StateIndex.POS_E]
            north = s[StateIndex.POS_N]
            lines[i].set_data([east], [north])
            labels[i].set_position((east, north + 400))
            labels[i].set_text(f"A{i+1}")
        return lines + labels

    num_frames = int(tmax / step)
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  init_func=init, blit=True, interval=33)

    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    '''Main function to simulate and animate 3 aircraft'''
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0   
    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    aircraft_configs = [
        {"id": 1, "init": init, "wps":[[-5000, -7500, alt],
                 [-15000, -7500, alt-500],
                 [-15000, 5000, alt-200]]},
        # {"id": 2, "init": generate_initial_conditions(psi=math.pi / 6), "wps": generate_waypoints(500, 1000)},
        # {"id": 3, "init": generate_initial_conditions(psi=math.pi / 4), "wps": generate_waypoints(-500, -1000)},
    ]

    tmax = 20 # simulation time
    step = 1/30

    all_results = run_all_simulations(aircraft_configs, tmax, step)
    print(f"results keys: {[list(res.keys()) for res in all_results]}")
    animate_aircrafts(all_results, tmax, step)


if __name__ == '__main__':
    main()
