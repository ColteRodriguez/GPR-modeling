import numpy as np

'''
6. Apply Exponential Gain for attenuation
adjust param for desired depth and visualization
'''
param = 0.07
def apply_gain(data,  param, gain_type='exponential', agc_window=40):
    n_samples, n_traces = data.shape
    gained_data = np.copy(data)

    if gain_type == 'linear':
        gain = np.linspace(1, 1 + param * n_samples, n_samples).reshape(-1, 1)
        gained_data *= gain

    elif gain_type == 'exponential':
        gain = np.exp(param * np.arange(n_samples)).reshape(-1, 1)
        gained_data *= gain

    elif gain_type == 'agc':
        for i in range(n_traces):
            trace = gained_data[:, i]
            for j in range(0, n_samples, agc_window):
                window = trace[j:j + agc_window]
                rms = np.sqrt(np.mean(window ** 2)) if np.mean(window ** 2) > 0 else 1
                gained_data[j:j + agc_window, i] = window / rms

    return gained_data