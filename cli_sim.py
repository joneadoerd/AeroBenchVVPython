import sys
import json
import argparse
from typing import List, Dict, Any

from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.run_f16_sim import run_f16_sim
from aerobench.util import convert_result_ft_to_meter

# Data structures matching the Rust structs
class F16State:
    def __init__(self, vt, alpha, beta, phi, theta, psi, p, q, r, pn, pe, h, pow ,time = None):
        self.vt = vt
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.p = p
        self.q = q
        self.r = r
        self.pn = pn
        self.pe = pe
        self.h = h
        self.pow = pow
        self.time = time 
    def to_dict(self):
        return self.__dict__

class Position:
    def __init__(self, alt, lat, lon):
        self.alt = alt
        self.lat = lat
        self.lon = lon
    def to_dict(self):
        return self.__dict__

class Target:
    def __init__(self, id, init_state, waypoints):
        self.id = id
        self.init_state = init_state
        self.waypoints = waypoints
    def to_dict(self):
        return {
            'id': self.id,
            'init_state': self.init_state.to_dict(),
            'waypoints': [wp.to_dict() for wp in self.waypoints]
        }

class Simulation:
    def __init__(self, targets, time_step, max_time):
        self.targets = targets
        self.time_step = time_step
        self.max_time = max_time
    def run_all_simulations(self,sim_input, tmax=None, step=None):
        '''Run all aircraft simulations using Simulation struct input and return a list of SimulationResult.'''
        all_results = []
        # Use tmax and step from sim_input if not provided
        tmax = tmax if tmax is not None else sim_input.get('max_time', 75)
        step = step if step is not None else sim_input.get('time_step', 1/30)
        for target in sim_input['targets']:
            ap = WaypointAutopilot([[wp['lat'], wp['lon'], wp['alt']] for wp in target['waypoints']], stdout=True)
            # print(f"Starting simulation for Aircraft {target['id']}")
            # Build initial state from dict if needed
            init = [
                target['init_state']['vt'],
                target['init_state']['alpha'],
                target['init_state']['beta'],
                target['init_state']['phi'],
                target['init_state']['theta'],
                target['init_state']['psi'],
                target['init_state']['p'],
                target['init_state']['q'],
                target['init_state']['r'],
                target['init_state']['pn'],
                target['init_state']['pe'],
                target['init_state']['h'],
                target['init_state']['pow'],
            ]
            res = run_f16_sim(init, tmax, ap, step=step,
                            extended_states=True, integrator_str='rk45')
            # print(f"Keys in result for Aircraft {target['id']}: {list(res.keys())}")
            # Build SimulationResult
            # Convert all states to F16State (only first 13 elements of each state), and attach time
            final_states = [F16State(*s[:13], time=t) for s, t in zip(res['states'], res['times'])] if 'states' in res else []
            sim_result = SimulationResult(
                target_id=target['id'],
                time=res['times'],
                waypoints=[Position(**wp) for wp in target['waypoints']],
                run_time=res.get('runtime', tmax),
                final_state=final_states
            )
            all_results.append(sim_result)
        return SimulationResultList(all_results)
    def to_dict(self):
        return {
            'targets': [target.to_dict() for target in self.targets],
      # In the provided Python code snippet, the `'time_step'` attribute is a key in the dictionary
      # returned by the `to_dict` method of the `Simulation` class. This key represents the time step
      # used in the simulation.
            'time_step': self.time_step,
            'max_time': self.max_time
        }
        

class SimulationResult:
    def __init__(self, target_id, time, waypoints, run_time, final_state):
        self.target_id = target_id
        self.time = time
        self.waypoints = waypoints
        self.run_time = run_time
        self.final_state = final_state
    

    def to_dict(self):
        return {
            'target_id': self.target_id,
            'time': self.time,
            'waypoints': [wp.to_dict() for wp in self.waypoints],
            'run_time': self.run_time,
            'final_state': [fs.to_dict() for fs in self.final_state]
        }
class SimulationResultList:
    def __init__(self, results: List[SimulationResult]):
        self.results = results

    def to_dict(self):
        return {
            'results': [r.to_dict() for r in self.results]
        }
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', choices=['meter', 'feet'], default='feet', help='Output unit for state data')
    args = parser.parse_args()
    # Read JSON input from stdin
    input_json = sys.stdin.read()
    sim_data = json.loads(input_json)
    # Parse input into objects
    sim = Simulation(
        [Target(t['id'], F16State(**t['init_state']), [Position(**wp) for wp in t['waypoints']]) for t in sim_data['targets']],
        sim_data['time_step'],
        sim_data['max_time']
    )
    # Run all simulations and get SimulationResultList
    sim_result_list = sim.run_all_simulations(sim_data)
    # If meter, convert all F16State data to meters
    if args.unit == 'meter':
        for sim_result in sim_result_list.results:
            for fs in sim_result.final_state:
                fs.vt *= 0.3048
                fs.pn *= 0.3048
                fs.pe *= 0.3048
                fs.h *= 0.3048
    # Output results as JSON to stdout (for Rust or other consumers)
    output = json.dumps(sim_result_list.to_dict())
    print(output)
    with open('sim_results.json', 'w') as f:
        f.write(output)

if __name__ == "__main__":
    main()
