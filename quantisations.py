import numpy as np
import tensorflow as tf
import basics as bs
import accuracy as ac

def calibrate_percentile(embedding, calibration_percentage=99):
    """
    Calibrate the range of values in the embedding based on a specified percentile.
    """
    lower_bound = np.percentile(embedding, (100 - calibration_percentage) / 2)
    upper_bound = np.percentile(embedding, 100 - (100 - calibration_percentage) / 2)
    return lower_bound, upper_bound

def scalar_quantisation_percentile(values, lower_bound=None, upper_bound=None):
    """
    Quantize the values to int8 using the calibrated range.
    """
    # Ensure the input is a numpy array
    values = np.array(values, dtype=np.float32)
    
    # If no calibration bounds are provided, calculate them
    if lower_bound is None or upper_bound is None:
        lower_bound, upper_bound = calibrate_percentile(values)
    
    # Calculate the scale factor and zero point
    qmin = -128
    qmax = 127
    scale = (upper_bound - lower_bound) / (qmax - qmin)
    zero_point = qmin - round(lower_bound / scale)
    
    # Quantize the values
    quantized_values = np.clip(np.round(values / scale + zero_point), qmin, qmax).astype(np.int8)
    
    return quantized_values

def quantize_tensor(input_tensor, T=tf.qint8, mode='MIN_COMBINED', round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None, ensure_minimum_range=0.01):
    quantized_tensor, _, _ = tf.quantization.quantize(
        input=input_tensor,
        min_range= np.min(input_tensor),
        max_range=np.max(input_tensor),
        T=T,
        mode=mode,
        round_mode=round_mode,
        name=name,
        narrow_range=narrow_range,
        axis=axis,
        ensure_minimum_range=ensure_minimum_range
    )
    return quantized_tensor.numpy()
