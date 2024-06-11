import numpy as np
import quantisations as qt
import basics as bs

quantization_functions = {
    "scalar_quantisation_max": qt.scalar_quantisation_max,
    "scalar_quantisation_percentile": qt.scalar_quantisation_percentile,
    "scalar_quantisation_tensorrt": qt.scalar_quantisation_tensorrt
}

def compare_accuracies_euc(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'correct_tensor_facenet': 0, 'wrong_tensor_facenet': 0, 'correct_scalar_facenet': 0, 'wrong_scalar_facenet': 0, 'correct_noquant_facenet': 0, 'wrong_noquant_facenet': 0, 'correct_tensor_sface': 0, 'wrong_tensor_sface': 0, 'correct_scalar_sface': 0, 'wrong_scalar_sface': 0, 'correct_noquant_sface': 0, 'wrong_noquant_sface': 0} for name in quant_funcs.keys()}

    fn_threshold = 10
    sf_threshold = 10.734  

    for i in range(m):
        a, b, imga, imgb, n = pairs[i]

        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        a_n = a / np.linalg.norm(a)
        b_n = b / np.linalg.norm(b)

        # SFace calculations
        a_s = bs.get_embedding(imga)
        b_s = bs.get_embedding(imgb)

        a_q_tensor_s = qt.quantize_tensor(a_s)
        b_q_tensor_s = qt.quantize_tensor(b_s)
        a_qn_t_s = a_q_tensor_s / np.linalg.norm(a_q_tensor_s)
        b_qn_t_s = b_q_tensor_s / np.linalg.norm(b_q_tensor_s)

        a_n_s = a_s / np.linalg.norm(a_s)
        b_n_s = b_s / np.linalg.norm(b_s)

        for name, quant_func in quant_funcs.items():
            # Scalar quantization for Facenet
            a_quant = quant_func(a)
            b_quant = quant_func(b)
            a_qn_s = a_quant / np.linalg.norm(a_quant)
            b_qn_s = b_quant / np.linalg.norm(b_quant)

            # Scalar quantization for SFace
            a_quant_s = quant_func(a_s)
            b_quant_s = quant_func(b_s)
            a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
            b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

            # Facenet comparison
            if n:
                if bs.euclidean_distance(a_qn_t, b_qn_t) > fn_threshold:
                    counters[name]['wrong_tensor_facenet'] += 1
                else:
                    counters[name]['correct_tensor_facenet'] += 1
                if bs.euclidean_distance(a_qn_s, b_qn_s) > fn_threshold:
                    counters[name]['wrong_scalar_facenet'] += 1
                else:
                    counters[name]['correct_scalar_facenet'] += 1
                if bs.euclidean_distance(a_n, b_n) > fn_threshold:
                    counters[name]['wrong_noquant_facenet'] += 1
                else:
                    counters[name]['correct_noquant_facenet'] += 1

            # SFace comparison
            if bs.euclidean_distance(a_qn_t_s, b_qn_t_s) > sf_threshold:
                counters[name]['wrong_tensor_sface'] += 1
            else:
                counters[name]['correct_tensor_sface'] += 1
            if bs.euclidean_distance(a_qn_s_s, b_qn_s_s) > sf_threshold:
                counters[name]['wrong_scalar_sface'] += 1
            else:
                counters[name]['correct_scalar_sface'] += 1
            if bs.euclidean_distance(a_n_s, b_n_s) > sf_threshold:
                counters[name]['wrong_noquant_sface'] += 1
            else:
                counters[name]['correct_noquant_sface'] += 1

    return counters


def compare_accuracies_cos(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'correct_tensor_facenet': 0, 'wrong_tensor_facenet': 0, 'correct_scalar_facenet': 0, 'wrong_scalar_facenet': 0, 'correct_noquant_facenet': 0, 'wrong_noquant_facenet': 0, 'correct_tensor_sface': 0, 'wrong_tensor_sface': 0, 'correct_scalar_sface': 0, 'wrong_scalar_sface': 0, 'correct_noquant_sface': 0, 'wrong_noquant_sface': 0} for name in quant_funcs.keys()}

    fn_threshold_cos = 0.4
    sf_threshold_cos = 0.593

    for i in range(m):
        a, b, imga, imgb, n = pairs[i]
        
        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        a_n = a / np.linalg.norm(a)
        b_n = b / np.linalg.norm(b)

        # SFace calculations
        a_s = bs.get_embedding(imga)
        b_s = bs.get_embedding(imgb)

        a_q_tensor_s = qt.quantize_tensor(a_s)
        b_q_tensor_s = qt.quantize_tensor(b_s)
        a_qn_t_s = a_q_tensor_s / np.linalg.norm(a_q_tensor_s)
        b_qn_t_s = b_q_tensor_s / np.linalg.norm(b_q_tensor_s)

        a_n_s = a_s / np.linalg.norm(a_s)
        b_n_s = b_s / np.linalg.norm(b_s)

        for name, quant_func in quant_funcs.items():
            # Scalar quantization for Facenet
            a_quant = quant_func(a)
            b_quant = quant_func(b)
            a_qn_s = a_quant / np.linalg.norm(a_quant)
            b_qn_s = b_quant / np.linalg.norm(b_quant)

            # Scalar quantization for SFace
            a_quant_s = quant_func(a_s)
            b_quant_s = quant_func(b_s)
            a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
            b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

            # Facenet comparison
            if n:
                if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                    counters[name]['wrong_tensor_facenet'] += 1
                else:
                    counters[name]['correct_tensor_facenet'] += 1
                if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                    counters[name]['wrong_scalar_facenet'] += 1
                else:
                    counters[name]['correct_scalar_facenet'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['wrong_noquant_facenet'] += 1
                else:
                    counters[name]['correct_noquant_facenet'] += 1
            else:
                if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                    counters[name]['correct_tensor_facenet'] += 1
                else:
                    counters[name]['wrong_tensor_facenet'] += 1
                if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                    counters[name]['correct_scalar_facenet'] += 1
                else:
                    counters[name]['wrong_scalar_facenet'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['correct_noquant_facenet'] += 1
                else:
                    counters[name]['wrong_noquant_facenet'] += 1

            # SFace comparison
            if n:
                if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                    counters[name]['wrong_tensor_sface'] += 1
                else:
                    counters[name]['correct_tensor_sface'] += 1
                if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                    counters[name]['wrong_scalar_sface'] += 1
                else:
                    counters[name]['correct_scalar_sface'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['wrong_noquant_sface'] += 1
                else:
                    counters[name]['correct_noquant_sface'] += 1
            else:
                if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                    counters[name]['correct_tensor_sface'] += 1
                else:
                    counters[name]['wrong_tensor_sface'] += 1
                if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                    counters[name]['correct_scalar_sface'] += 1
                else:
                    counters[name]['wrong_scalar_sface'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['correct_noquant_sface'] += 1
                else:
                    counters[name]['wrong_noquant_sface'] += 1

    return counters