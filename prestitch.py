import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

observable_depth_ns = 1400

'''
1. remove excess depth
238ns sampling window is way too long to return signal past 800 pixels
'''
def depth_correction(arr):
    cut_pixels = int((6000/15000) * observable_depth_ns) # px/ns * ns
    return observable_depth_ns, arr[:cut_pixels, :]

'''
2. Self detection removal
~ every 65 pixels the reciever conducts a self detection, we can just locate the anomalously low amplitudes
'''
def remove_self_detection(arr):
    sample = arr[:, 2]
    sdindex = np.where(sample == np.max(sample)) # A good choice row for self detection
    remove_indices = []
    array_shape = arr.shape
    for i in range(0, array_shape[1]):
        if arr[sdindex, i] < sample[sdindex]/2:
            remove_indices.append(i)
    arr = np.delete(arr, remove_indices, axis=1)
    
    return arr

def phase_correction(arr, p):
    for i in range(0, p):
        arr = arr[:-1, :]
        arr = np.vstack((np.zeros((1, arr.shape[1])), arr))
    return arr
