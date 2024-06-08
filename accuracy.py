import numpy as np
import random
import quantisations as qt
import basics as bs

quantization_functions = {
    "scalar_quantisation_max": scalar_quantisation_max,
    "scalar_quantisation_percentile": scalar_quantisation_percentile,
    "scalar_quantisation_tensorrt": scalar_quantisation_tensorrt
}


# Function to compare accuracies
# Function to compare accuracies
def compare_accuracies_euc(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'c_t': 0, 'w_t': 0, 'c_s': 0, 'w_s': 0, 'c': 0, 'w': 0, 'c_t_s': 0, 'w_t_s': 0, 'c_s_s': 0, 'w_s_s': 0, 'c_ss': 0, 'w_ss': 0} for name in quant_funcs.keys()}

    fn_threshold = 10
    sf_threshold = 10.734  

    for i in range(m):
        a, b, imga, imgb, n = pairs[i]

        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        for name, quant_func in quant_funcs.items():
            a_quant = quant_func(a)
            b_quant = quant_func(b)
            a_qn_s = a_quant / np.linalg.norm(a_quant)
            b_qn_s = b_quant / np.linalg.norm(b_quant)

            a_n = a / np.linalg.norm(a)
            b_n = b / np.linalg.norm(b)

            # SFace calculations
            a_s = bs.get_embedding(imga)
            b_s = bs.get_embedding(imgb)

            a_q_tensor_s = qt.quantize_tensor(a_s)
            b_q_tensor_s = qt.quantize_tensor(b_s)
            a_qn_t_s = a_q_tensor_s / np.linalg.norm(a_q_tensor_s)
            b_qn_t_s = b_q_tensor_s / np.linalg.norm(b_q_tensor_s)

            a_quant_s = quant_func(a_s)
            b_quant_s = quant_func(b_s)
            a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
            b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

            a_n_s = a_s / np.linalg.norm(a_s)
            b_n_s = b_s / np.linalg.norm(b_s)

            # Facenet comparison
            if n:
                if bs.euclidean_distance(a_qn_t, b_qn_t) > fn_threshold:
                    counters[name]['w_t'] += 1
                else:
                    counters[name]['c_t'] += 1
                if bs.euclidean_distance(a_qn_s, b_qn_s) > fn_threshold:
                    counters[name]['w_s'] += 1
                else:
                    counters[name]['c_s'] += 1
                if bs.euclidean_distance(a_n, b_n) > fn_threshold:
                    counters[name]['w'] += 1
                else:
                    counters[name]['c'] += 1

            # SFace comparison
            if bs.euclidean_distance(a_qn_t_s, b_qn_t_s) > sf_threshold:
                counters[name]['w_t_s'] += 1
            else:
                counters[name]['c_t_s'] += 1
            if bs.euclidean_distance(a_qn_s_s, b_qn_s_s) > sf_threshold:
                counters[name]['w_s_s'] += 1
            else:
                counters[name]['c_s_s'] += 1
            if bs.euclidean_distance(a_n_s, b_n_s) > sf_threshold:
                counters[name]['w_ss'] += 1
            else:
                counters[name]['c_ss'] += 1

    return counters

# Function to compare accuracies
def compare_accuracies_cos(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'c_t': 0, 'w_t': 0, 'c_s': 0, 'w_s': 0, 'c': 0, 'w': 0, 'c_t_s': 0, 'w_t_s': 0, 'c_s_s': 0, 'w_s_s': 0, 'c_ss': 0, 'w_ss': 0} for name in quant_funcs.keys()}

    fn_threshold_cos = 0.4
    sf_threshold_cos = 0.593

    for i in range(m):
        a, b, imga, imgb, n = pairs[i]
        
        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        for name, quant_func in quant_funcs.items():
            a_quant = quant_func(a)
            b_quant = quant_func(b)
            a_qn_s = a_quant / np.linalg.norm(a_quant)
            b_qn_s = b_quant / np.linalg.norm(b_quant)

            a_n = a / np.linalg.norm(a)
            b_n = b / np.linalg.norm(b)

            # SFace calculations
            a_s = bs.get_embedding(imga)
            b_s = bs.get_embedding(imgb)

            a_q_tensor_s = qt.quantize_tensor(a_s)
            b_q_tensor_s = qt.quantize_tensor(b_s)
            a_qn_t_s = a_q_tensor_s / np.linalg.norm(a_q_tensor_s)
            b_qn_t_s = b_q_tensor_s / np.linalg.norm(b_q_tensor_s)

            a_quant_s = quant_func(a_s)
            b_quant_s = quant_func(b_s)
            a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
            b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

            a_n_s = a_s / np.linalg.norm(a_s)
            b_n_s = b_s / np.linalg.norm(b_s)

            # Facenet comparison
            if n:
                if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                    counters[name]['w_t'] += 1
                else:
                    counters[name]['c_t'] += 1
                if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                    counters[name]['w_s'] += 1
                else:
                    counters[name]['c_s'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['w'] += 1
                else:
                    counters[name]['c'] += 1
            else:
                if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                    counters[name]['c_t'] += 1
                else:
                    counters[name]['w_t'] += 1
                if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                    counters[name]['c_s'] += 1
                else:
                    counters[name]['w_s'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['c'] += 1
                else:
                    counters[name]['w'] += 1

            # SFace comparison
            if n:
                if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                    counters[name]['w_t_s'] += 1
                else:
                    counters[name]['c_t_s'] += 1
                if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                    counters[name]['w_s_s'] += 1
                else:
                    counters[name]['c_s_s'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['w_ss'] += 1
                else:
                    counters[name]['c_ss'] += 1
            else:
                if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                    counters[name]['c_t_s'] += 1
                else:
                    counters[name]['w_t_s'] += 1
                if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                    counters[name]['c_s_s'] += 1
                else:
                    counters[name]['w_s_s'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['c_ss'] += 1
                else:
                    counters[name]['w_ss'] += 1

    return counters