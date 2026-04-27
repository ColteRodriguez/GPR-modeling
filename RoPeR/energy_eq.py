import numpy as np

'''
5. Lateral Energy equalization
'''
def lateral_energy_equalization(gpr_data, window_size):
    equalized = np.copy(gpr_data)
    half_w = window_size // 2

    for i in range(gpr_data.shape[1]):
        # window boundaries
        start = max(0, i - half_w)
        end = min(gpr_data.shape[1], i + half_w + 1) # Dont exceeded array bounds

        window = gpr_data[:, start:end]
        
        energies = np.sum(window**2, axis=0)
        mean_energy = np.mean(energies)

        current_energy = np.sum(gpr_data[:, i]**2)

        # Scale to match mean energy
        if current_energy > 0:
            scale = np.sqrt(mean_energy / current_energy)
            equalized[:, i] *= scale

    return equalized
