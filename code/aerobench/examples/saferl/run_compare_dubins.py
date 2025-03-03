'''load rollouts from safeRL and compare to f-16 recreations with same inputs

Get a predictive model for online RTA
'''


import os
import hashlib
import numpy as np
from scipy.io import savemat

from compare_util import load_data_and_sim_f16, eval_predictor_accuracy, DubinsPredictor, make_linear_predictor, plot_prediction_vs_dubins
from aerobench.util import get_script_path

def hash_file(filename, algorithm="sha256"):
    """Compute the hash of a file using the specified algorithm (default: SHA-256)."""
    hash_func = getattr(hashlib, algorithm)()
    with open(filename, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def main():
    '''main function'''

    np.set_printoptions(suppress=True, precision=3)

    dubins_filename = 'dubins_data.pkl'

    script_dir = get_script_path(__file__)
    dubins_file_path = os.path.join(script_dir, dubins_filename)

    file_hash = hash_file(dubins_file_path)
    print(f"SHA-256 hash of {dubins_filename}: {file_hash}")

    all_f16_res_dicts = load_data_and_sim_f16(dubins_file_path, file_hash, single_index=None)

    MAX_STEPS_TO_PREDICT = 19
    predictor = make_linear_predictor(all_f16_res_dicts, MAX_STEPS_TO_PREDICT)

    # save to linear_predictor.mat
    
    filename = 'linear_predictor.mat'
    data = {'max_steps': MAX_STEPS_TO_PREDICT}

    for step in range(1, MAX_STEPS_TO_PREDICT+1):
        A = predictor.A_dict[step]
        residuals = predictor.residuals_dict[step]
        data[f'A_{step}'] = A
        data[f'residuals_{step}'] = residuals

    savemat(filename, data)
    print(f"Saved predictor A matrices and residuals to {filename}")


    for num_steps_to_predict in [MAX_STEPS_TO_PREDICT]: #range(1, 20):
        

        max_error_vector = eval_predictor_accuracy(all_f16_res_dicts, predictor, num_steps_to_predict=num_steps_to_predict)
        print(f"{num_steps_to_predict=}, {max_error_vector=}")

        plot_prediction_vs_dubins(all_f16_res_dicts, predictor, num_steps_to_predict=num_steps_to_predict)

    #compare_dubins_data() # create data file from f-16 recreations of dubins paths

if __name__ == '__main__':
    main()
