import numpy as np
import random
import quantisations as qt
import basics as bs


# Function to compare accuracies
def compare_accuracies_euc(pairs, m=1000):
    # Initialize counters
    c_t = w_t = c_s = w_s = 0
    c = w = c_t_s = w_t_s = c_s_s = w_s_s = 0
    c_ss = w_ss = 0

    fn_threshold=10
    sf_threshold=10.734  

    for i in range(m):
        # print(i)
        a, b, imga, imgb = pairs[i]
        
        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        a_quant = qt.scalar_quantisation_percentile(a)
        b_quant = qt.scalar_quantisation_percentile(b)
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

        a_quant_s = qt.scalar_quantisation_percentile(a_s)
        b_quant_s = qt.scalar_quantisation_percentile(b_s)
        a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
        b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

        a_n_s = a_s / np.linalg.norm(a_s)
        b_n_s = b_s / np.linalg.norm(b_s)

        # Facenet comparison
        if n:
            if bs.euclidean_distance(a_qn_t, b_qn_t) > fn_threshold:
                w_t += 1
            else:
                c_t += 1
            if bs.euclidean_distance(a_qn_s, b_qn_s) > fn_threshold:
                w_s += 1
            else:
                c_s += 1
            if bs.euclidean_distance(a_n, b_n) > fn_threshold:
                w += 1
            else:
                c += 1
        else:
            if bs.euclidean_distance(a_qn_t, b_qn_t) > fn_threshold:
                c_t += 1
            else:
                w_t += 1
            if bs.euclidean_distance(a_qn_s, b_qn_s) > fn_threshold:
                c_s += 1
            else:
                w_s += 1
            if bs.euclidean_distance(a_n, b_n) > fn_threshold:
                c += 1
            else:
                w += 1

        # SFace comparison
        if n:
            if bs.euclidean_distance(a_qn_t_s, b_qn_t_s) > sf_threshold:
                w_t_s += 1
            else:
                c_t_s += 1
            if bs.euclidean_distance(a_qn_s_s, b_qn_s_s) > sf_threshold:
                w_s_s += 1
            else:
                c_s_s += 1
            if bs.euclidean_distance(a_n_s, b_n_s) > sf_threshold:
                w_ss += 1
            else:
                c_ss += 1
        else:
            if bs.euclidean_distance(a_qn_t_s, b_qn_t_s) > sf_threshold:
                c_t_s += 1
            else:
                w_t_s += 1
            if bs.euclidean_distance(a_qn_s_s, b_qn_s_s) > sf_threshold:
                c_s_s += 1
            else:
                w_s_s += 1
            if bs.euclidean_distance(a_n_s, b_n_s) > sf_threshold:
                c_ss += 1
            else:
                w_ss += 1

    correct_f_c = [c, c_s, c_t]
    incorrect_f_c = [w, w_s, w_t]
    correct_c = [c_ss, c_s_s, c_t_s]
    incorrect_c = [w_ss, w_s_s, w_t_s]

    return correct_f_c, incorrect_f_c, correct_c, incorrect_c

# Function to compare accuracies
def compare_accuracies_cos(pairs, m=1000):
    # Initialize counters
    c_t = w_t = c_s = w_s = 0
    c = w = c_t_s = w_t_s = c_s_s = w_s_s = 0
    c_ss = w_ss = 0

    fn_threshold_cos = 0.4
    sf_threshold_cos = 0.593

    for i in range(m):
        # print(i)
        a, b, imga, imgb = pairs[i]
        
        # Facenet calculations
        a_q_tensor = qt.quantize_tensor(a)
        b_q_tensor = qt.quantize_tensor(b)
        a_qn_t = a_q_tensor / np.linalg.norm(a_q_tensor)
        b_qn_t = b_q_tensor / np.linalg.norm(b_q_tensor)

        a_quant = qt.scalar_quantisation_percentile(a)
        b_quant = qt.scalar_quantisation_percentile(b)
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

        a_quant_s = qt.scalar_quantisation_percentile(a_s)
        b_quant_s = qt.scalar_quantisation_percentile(b_s)
        a_qn_s_s = a_quant_s / np.linalg.norm(a_quant_s)
        b_qn_s_s = b_quant_s / np.linalg.norm(b_quant_s)

        a_n_s = a_s / np.linalg.norm(a_s)
        b_n_s = b_s / np.linalg.norm(b_s)

        # Facenet comparison
        if n:
            if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                w_t += 1
            else:
                c_t += 1
            if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                w_s += 1
            else:
                c_s += 1
            if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                w += 1
            else:
                c += 1
        else:
            if bs.get_cos_dist_numpy(a_qn_t, b_qn_t) > fn_threshold_cos:
                c_t += 1
            else:
                w_t += 1
            if bs.get_cos_dist_numpy(a_qn_s, b_qn_s) > fn_threshold_cos:
                c_s += 1
            else:
                w_s += 1
            if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                c += 1
            else:
                w += 1

        # SFace comparison
        if n:
            if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                w_t_s += 1
            else:
                c_t_s += 1
            if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                w_s_s += 1
            else:
                c_s_s += 1
            if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                w_ss += 1
            else:
                c_ss += 1
        else:
            if bs.get_cos_dist_numpy(a_qn_t_s, b_qn_t_s) > sf_threshold_cos:
                c_t_s += 1
            else:
                w_t_s += 1
            if bs.get_cos_dist_numpy(a_qn_s_s, b_qn_s_s) > sf_threshold_cos:
                c_s_s += 1
            else:
                w_s_s += 1
            if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                c_ss += 1
            else:
                w_ss += 1

    correct_f_c = [c, c_s, c_t]
    incorrect_f_c = [w, w_s, w_t]
    correct_c = [c_ss, c_s_s, c_t_s]
    incorrect_c = [w_ss, w_s_s, w_t_s]

    return correct_f_c, incorrect_f_c, correct_c, incorrect_c