import numpy as np

'''
5. Lateral Energy equalization -- Okay this one's written by Chatgpt
'''

def lateral_energy_equalization(gpr_data, window_size):
    # gpr_data: shape (n_samples, n_traces)
    equalized = np.copy(gpr_data)
    half_w = window_size // 2

    for i in range(gpr_data.shape[1]):
        # Determine window boundaries
        start = max(0, i - half_w)
        end = min(gpr_data.shape[1], i + half_w + 1)

        # Extract the window of traces
        window = gpr_data[:, start:end]

        # Compute energy per trace in the window (sum of squares)
        energies = np.sum(window**2, axis=0)

        # Compute mean energy in the window
        mean_energy = np.mean(energies)

        # Energy of current trace
        current_energy = np.sum(gpr_data[:, i]**2)

        # Scale factor to match mean energy
        if current_energy > 0:
            scale = np.sqrt(mean_energy / current_energy)
            equalized[:, i] *= scale

    return equalized
