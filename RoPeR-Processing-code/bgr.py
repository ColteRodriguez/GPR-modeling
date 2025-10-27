import numpy as np

'''
Background operations library for GPR processing used in Rodriguez, 2026 senior thesis. 

- Naming convention for methods will be all lowercase with underscore (_) as spacees
- Naming convention for variables will be the same as for methods except in the case where mathematical naming convention takes precedentc (e.g. cases where 2d arrays act as matrices)
- Each method will be preceeded by a breif description of the method, and code should be tracable with minimal in-line comments.
'''


'''
8. Time Lag adjustment
Removes air waves (n rows from the top down), based on time_lag_ns, which is defined within the method scope. Returns the cut input array.
'''
def time_lag(arr):
    time_lag_ns = 212
    time_lag_ns = 100
    cut_pixels = int((6000/15000) * time_lag_ns) # px/ns * ns
    return arr[cut_pixels:][:]

'''
3. Removal of Ringing Noise
Background removal by subtraction of the mean at each trace (row of data)
'''
def remove_bg(arr):
    horizontal_avg = np.mean(arr, axis=1, keepdims=True)
    return arr - horizontal_avg, arr.shape

'''
3. Removal of Ringing Noise
Background removal by subtraction of the mean at each trace (row of data) within a given sliding window, whose width, n, is passed as a parameter. Returns out, 
with out.shape==arr.shape and the array shape, which is extraneous and can be removed.
'''
def subtract_window_mean(arr, n):
    rows, cols = arr.shape
    out = arr.copy()
    
    for start in range(0, cols, n):
        end = min(start + n, cols)
        window = arr[:, start:end]
        mean = window.mean(axis=1, keepdims=True)
        out[:, start:end] = window - mean
    
    return out, out.shape


'''
4. Data deduplication -- work with an acceptable scan sample to cut down on runtime
Utilizes a correlation matrix constructed from A-scans to remove duplicated data (defined as two 1d traces sharing >= 95% data distribution)
'''
def dedupe(arr, array_shape):
    arr_short = arr[25:250] # formerly 100:350
    thresh = 0.95

    def coeff(a, b):
        corr_matrix = np.corrcoef(a, b)
        return corr_matrix[0, 1]
    
    anom = []
    duplicates = []
    for i in range(0, array_shape[1]-1):
        corr = coeff(arr_short[:, i], arr_short[:, i+1])
        anom.append(corr)
        if corr >= thresh:
            avg = (arr[:, i]+arr[:, i+1])/2
            duplicates.append((arr[:, i], arr[:, i+1]))
            arr[:, i] = avg
            arr[:, i+1] = avg
    return arr, duplicates, anom

'''
3. Removal of Ringing Noise
Background removal using Eigenvalues as defined in Al-Nuaimy 2010. Emphasizes dipping and point refelctors while greatly reducing horizontal banding.
'''
def eigen_background_removal(X):
    U, S, V_T = np.linalg.svd(X, full_matrices=False)

    s_max, u_max, v_max = S[0], U[:, 0], V_T[0, :]
    s_min, u_min, v_min = S[-1], U[:, -1], V_T[-1, :]

    # Outer products
    P  = s_max * np.outer(u_max, v_max)
    P2 = s_min * np.outer(u_min, v_min)

    def normalize(M):
        return (M - np.min(M)) / (np.max(M) - np.min(M))

    Out1 = normalize(X) - normalize(P)
    Out2 = normalize(X) - normalize(P2)
    Out = normalize(Out1) * normalize(Out2)

    Output = Out - np.mean(Out)

    return Output