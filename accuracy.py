import numpy as np
import quantisations as qt
import basics as bs
import pandas as pd
import time

quantization_functions = {
    "scalar_quantisation_max": qt.scalar_quantisation_max,
    "scalar_quantisation_percentile": qt.scalar_quantisation_percentile
}

def gather_timings(func, repetitions=100):
    times = []
    for _ in range(repetitions):
        _, exec_time = time_function(func, ...)
        times.append(exec_time)
    return times


def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def compare_accuracies_euc(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'correct_tensor_facenet': 0, 'wrong_tensor_facenet': 0, 'correct_scalar_facenet': 0, 'wrong_scalar_facenet': 0, 'correct_noquant_facenet': 0, 'wrong_noquant_facenet': 0, 'correct_tensor_sface': 0, 'wrong_tensor_sface': 0, 'correct_scalar_sface': 0, 'wrong_scalar_sface': 0, 'correct_noquant_sface': 0, 'wrong_noquant_sface': 0} for name in quant_funcs.keys()}
    
    # Initialize time counters
    time_counters = {name: {'tensor_time': 0, 'scalar_time': 0, 'noquant_time': 0} for name in quant_funcs.keys()}


    fn_threshold = 10
    sf_threshold = 10.734  
    p = 0
    for _ in range(m):

        a_f = None
        b_f = None

        while a_f == None and b_f == None:
            try:
                imga, imgb, n = pairs[p]
                a_f = bs.get_embedding_facenet(imga)
                b_f = bs.get_embedding_facenet(imgb)
            except Exception as e:
                # print(f"Error getting embeddings (attempt {p+1}/{2000}): {e}")
                a_f = None
                b_f = None
                p = p + 1
        p = p+1

        # Facenet calculations
        a_tensor_facenet, tensor_time_a = time_function(qt.quantize_tensor, a_f)
        b_qtensor_facenet, tensor_time_b = time_function(qt.quantize_tensor, b_f)

        # SFace calculations
        a_s = bs.get_embedding(imga)
        b_s = bs.get_embedding(imgb)

        # print(bs.euclidean_distance(a_tensor_facenet, b_qtensor_facenet))

        a_tensor_sface, tensor_time_s_a = time_function(qt.quantize_tensor, a_s)
        b_tensor_sface, tensor_time_s_b = time_function(qt.quantize_tensor, b_s)


        for name, quant_func in quant_funcs.items():
       
            # Scalar quantization for Facenet
            a_quant_facenet, scalar_time_a = time_function(quant_func, a_f)
            b_quant_facenet, scalar_time_b = time_function(quant_func, b_f)

            # Scalar quantization for SFace
            a_quant_sface, scalar_time_s_a = time_function(quant_func, a_s)
            b_quant_sface, scalar_time_s_b = time_function(quant_func, b_s)

            # Accumulate timing
            time_counters[name]['tensor_time'] += tensor_time_a + tensor_time_b + tensor_time_s_a + tensor_time_s_b
            time_counters[name]['scalar_time'] += scalar_time_a + scalar_time_b + scalar_time_s_a + scalar_time_s_b
            time_counters[name]['noquant_time'] += 0  # No additional time for no quantization

            # Facenet comparison
            if n:
                if bs.euclidean_distance(a_tensor_facenet, b_qtensor_facenet) > fn_threshold:
                    counters[name]['wrong_tensor_facenet'] += 1
                else:
                    counters[name]['correct_tensor_facenet'] += 1
                if bs.euclidean_distance(a_quant_facenet, b_quant_facenet) > fn_threshold:
                    counters[name]['wrong_scalar_facenet'] += 1
                else:
                    counters[name]['correct_scalar_facenet'] += 1
                if bs.euclidean_distance(a_f, b_f) > fn_threshold:
                    counters[name]['wrong_noquant_facenet'] += 1
                else:
                    counters[name]['correct_noquant_facenet'] += 1
            else:
                if bs.euclidean_distance(a_tensor_facenet, b_qtensor_facenet) > fn_threshold:
                    counters[name]['correct_tensor_facenet'] += 1
                else:
                    counters[name]['wrong_tensor_facenet'] += 1
                if bs.euclidean_distance(a_quant_facenet, b_quant_facenet) > fn_threshold:
                    counters[name]['correct_scalar_facenet'] += 1
                else:
                    counters[name]['wrong_scalar_facenet'] += 1
                if bs.euclidean_distance(a_f, b_f) > fn_threshold:
                    counters[name]['correct_noquant_facenet'] += 1
                else:
                    counters[name]['wrong_noquant_facenet'] += 1
            if n:
                # SFace comparison
                if bs.euclidean_distance(a_tensor_sface, b_tensor_sface) > sf_threshold:
                    counters[name]['wrong_tensor_sface'] += 1
                else:
                    counters[name]['correct_tensor_sface'] += 1
                if bs.euclidean_distance(a_quant_sface, b_quant_sface) > sf_threshold:
                    counters[name]['wrong_scalar_sface'] += 1
                else:
                    counters[name]['correct_scalar_sface'] += 1
                if bs.euclidean_distance(a_s, b_s) > sf_threshold:
                    counters[name]['wrong_noquant_sface'] += 1
                else:
                    counters[name]['correct_noquant_sface'] += 1

            else: 
                if bs.euclidean_distance(a_tensor_sface, b_tensor_sface) > sf_threshold:
                    counters[name]['correct_tensor_sface'] += 1
                else:
                    counters[name]['wrong_tensor_sface'] += 1
                if bs.euclidean_distance(a_quant_sface, b_quant_sface) > sf_threshold:
                    counters[name]['correct_scalar_sface'] += 1
                else:
                    counters[name]['wrong_scalar_sface'] += 1
                if bs.euclidean_distance(a_s, b_s) > sf_threshold:
                    counters[name]['correct_noquant_sface'] += 1
                else:
                    counters[name]['wrong_noquant_sface'] += 1

    return counters, time_counters

def compare_accuracies_cos(pairs, m=1000, quant_funcs=None):
    if quant_funcs is None:
        quant_funcs = quantization_functions

    # Initialize counters
    counters = {name: {'correct_tensor_facenet': 0, 'wrong_tensor_facenet': 0, 'correct_scalar_facenet': 0, 'wrong_scalar_facenet': 0, 'correct_noquant_facenet': 0, 'wrong_noquant_facenet': 0, 'correct_tensor_sface': 0, 'wrong_tensor_sface': 0, 'correct_scalar_sface': 0, 'wrong_scalar_sface': 0, 'correct_noquant_sface': 0, 'wrong_noquant_sface': 0} for name in quant_funcs.keys()}
    
    # Initialize time counters
    time_counters = {name: {'tensor_time': 0, 'scalar_time': 0, 'noquant_time': 0} for name in quant_funcs.keys()}


    fn_threshold_cos = 0.4
    sf_threshold_cos = 0.593
    p=0
    for _ in range(m):
        # imga, imgb, n = pairs[i]
        a_f = None
        b_f = None

        while a_f == None and b_f == None:
            try:
                imga, imgb, n = pairs[p]
                a_f = bs.get_embedding_facenet(imga)
                b_f = bs.get_embedding_facenet(imgb)
            except Exception as e:
                # print(f"Error getting embeddings (attempt {p+1}/{2000}): {e}")
                a_f = None
                b_f = None
                p = p + 1
        p = p+1

        a_n = a_f / np.linalg.norm(a_f)
        b_n = b_f / np.linalg.norm(b_f)

        # Facenet calculations
        a_qtensor_facenet, tensor_time_a = time_function(qt.quantize_tensor, a_n)
        b_qtensor_facenet, tensor_time_b = time_function(qt.quantize_tensor, b_n)

        # SFace calculations
        a_s = bs.get_embedding(imga)
        b_s = bs.get_embedding(imgb)

        a_n_s = a_s / np.linalg.norm(a_s)
        b_n_s = b_s / np.linalg.norm(b_s)

        a_tensor_sface, tensor_time_s_a = time_function(qt.quantize_tensor, a_n_s)
        b_tensor_sface, tensor_time_s_b = time_function(qt.quantize_tensor, b_n_s)

        # for name, quant_func in quant_funcs.items():

        for name, quant_func in quant_funcs.items():
        
            # Scalar quantization for Facenet
            a_quant_facenet, scalar_time_a = time_function(quant_func, a_n)
            b_quant_facenet, scalar_time_b = time_function(quant_func, b_n)
        

            # Scalar quantization for SFace
            a_quant_sface, scalar_time_s_a = time_function(quant_func, a_n_s)
            b_quant_sface, scalar_time_s_b = time_function(quant_func, b_n_s)
            

            # Accumulate timing
            time_counters[name]['tensor_time'] += tensor_time_a + tensor_time_b + tensor_time_s_a + tensor_time_s_b
            time_counters[name]['scalar_time'] += scalar_time_a + scalar_time_b + scalar_time_s_a + scalar_time_s_b
            time_counters[name]['noquant_time'] += 0  # No additional time for no quantization

            # Facenet comparison
            if n:
                if bs.get_cos_dist_numpy(a_qtensor_facenet, b_qtensor_facenet) > fn_threshold_cos:
                    counters[name]['wrong_tensor_facenet'] += 1
                else:
                    counters[name]['correct_tensor_facenet'] += 1
                if bs.get_cos_dist_numpy(a_quant_facenet, b_quant_facenet) > fn_threshold_cos:
                    counters[name]['wrong_scalar_facenet'] += 1
                else:
                    counters[name]['correct_scalar_facenet'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['wrong_noquant_facenet'] += 1
                else:
                    counters[name]['correct_noquant_facenet'] += 1
            else:
                if bs.get_cos_dist_numpy(a_qtensor_facenet, b_qtensor_facenet) > fn_threshold_cos:
                    counters[name]['correct_tensor_facenet'] += 1
                else:
                    counters[name]['wrong_tensor_facenet'] += 1
                if bs.get_cos_dist_numpy(a_quant_facenet, b_quant_facenet) > fn_threshold_cos:
                    counters[name]['correct_scalar_facenet'] += 1
                else:
                    counters[name]['wrong_scalar_facenet'] += 1
                if bs.get_cos_dist_numpy(a_n, b_n) > fn_threshold_cos:
                    counters[name]['correct_noquant_facenet'] += 1
                else:
                    counters[name]['wrong_noquant_facenet'] += 1

            # SFace comparison
            if n:
                if bs.get_cos_dist_numpy(a_tensor_sface, b_tensor_sface) > sf_threshold_cos:
                    counters[name]['wrong_tensor_sface'] += 1
                else:
                    counters[name]['correct_tensor_sface'] += 1
                if bs.get_cos_dist_numpy(a_quant_sface, b_quant_sface) > sf_threshold_cos:
                    counters[name]['wrong_scalar_sface'] += 1
                else:
                    counters[name]['correct_scalar_sface'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['wrong_noquant_sface'] += 1
                else:
                    counters[name]['correct_noquant_sface'] += 1
            else:
                if bs.get_cos_dist_numpy(a_tensor_sface, b_tensor_sface) > sf_threshold_cos:
                    counters[name]['correct_tensor_sface'] += 1
                else:
                    counters[name]['wrong_tensor_sface'] += 1
                if bs.get_cos_dist_numpy(a_quant_sface, b_quant_sface) > sf_threshold_cos:
                    counters[name]['correct_scalar_sface'] += 1
                else:
                    counters[name]['wrong_scalar_sface'] += 1
                if bs.get_cos_dist_numpy(a_n_s, b_n_s) > sf_threshold_cos:
                    counters[name]['correct_noquant_sface'] += 1
                else:
                    counters[name]['wrong_noquant_sface'] += 1

    return counters, time_counters