import numpy as np
from scipy.fft import fft, ifft, fftfreq

'''
7. Band-pass filtering
'''
def bandpass_filter(trace, dt, low_f, high_f):
    n = len(trace)
    freqs = fftfreq(n, dt)
    fft_trace = fft(trace)

    mask = (np.abs(freqs) >= low_f) & (np.abs(freqs) <= high_f)
    fft_trace[~mask] = 0

    return np.real(ifft(fft_trace))
