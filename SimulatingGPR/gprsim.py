import numpy as np
import math
import matplotlib.pyplot as plt
from colorama import Fore
import scipy

def colored_noise_ar1(N, rho, scale=0.2):
    """Generate temporally correlated (AR1) noise."""
    noise = np.random.randn(N)
    for i in range(1, N):
        noise[i] = rho * noise[i-1] + np.sqrt(1-rho**2) * noise[i]
    return noise * scale

def spatial_noise(nx, nt, strength):
    """Low-frequency banding across traces."""
    vertical = np.random.randn(nt)
    vertical = scipy.ndimage.gaussian_filter1d(vertical, sigma=20)
    return np.tile(vertical[:,None], (1,nx)) * strength

def norm2d(array):
    min_val = np.min(array)
    max_val = np.max(array)
    
    # Apply the normalization formula
    normalized_array = 2 * ((array - min_val) / (max_val - min_val)) - 1
    
    return normalized_array

    
def gprsim(eps_r, rf, dt, dx, reflectors, region_shape, wavetype, SNR):
    """
    Simulates a Ground Penetrating Radar (GPR) radargram for a given subsurface model.

    This function generates synthetic GPR data by modeling wave propagation 
    from a surface antenna to one or more point reflectors in a homogeneous medium 
    characterized by a specified relative permittivity. The resulting radargram 
    shows the time-domain reflection response as a function of antenna position.

    Parameters
    ----------
    eps_r : float
        Relative permittivity (dielectric) of the medium.
    
    rf : float
        Frequency parameter controlling the Gaussian wavelet width.
        Larger values produce narrower, higher-frequency pulses.
    
    dt : float
        Sampling interval [s].
    
    dx : float
        Spatial sampling interval (antenna step size) [m].
    
    reflectors : list of tuple(float, float)
        List of subsurface reflector coordinates, each given as (x, z).
        - x : horizontal position of reflector [m] (relative to region_shape[0]).
        - z : depth (two-way travel time) [s].
    
    region_shape : tuple(float, float)
        Physical extent of the simulated region, (xlen, tlen):
        - xlen : total horizontal distance [m].
        - tlen : total recording time window [s].
    
    wavetype : {'spike', 'gaussian'}
        Type of source wavelet used to simulate reflections.
        - 'spike'    : ideal impulse response.
        - 'gaussian' : Gaussian-modulated waveform controlled by `rf`.
    
    SNR : float
        Signal-to-noise ratio. Controls the intensity of additive Gaussian noise.
        If `math.inf`, no noise is added.

    Returns
    -------
    data : ndarray of shape (nt, nx)
        Simulated radargram matrix where:
        - Rows represent time samples.
        - Columns represent antenna positions along the surface (traces).
    
    x_positions : ndarray of shape (nx,)
        Horizontal positions of antenna along the surface [m].
    
    t_samples : ndarray of shape (nt,)
        Time samples corresponding to rows in `data` [s].

    """

    if wavetype not in ['spike', 'gaussian']:
        raise Exception(f'{wavetype} not an allowed wavetype.')
        
    global c 
    c = 3e8
    v = c / np.sqrt(eps_r)

    xlen, tlen = region_shape # Physical extent in two dimension, in units of [m] and [t]

    # Antenna positions along surface
    x_positions = np.arange(0, xlen, dx)
    nx = len(x_positions) # The number of scans taken

    # Time axis, max penetration depth in ns
    t_samples = np.arange(0, tlen, dt)
    nt = len(t_samples) # The number of scans taken
        
    data = np.zeros((nx, nt))      
    for j, x_ant in enumerate(x_positions):
        for (xr, zr) in reflectors:
            (xr, zr_m) = (xr, zr*v) # zr in [s], conv. to [m] to get twt
            
            # calculate the twt from the xr to x_ant
            twt = 2 * (np.sqrt((xr - x_ant) ** 2 + zr_m ** 2)/v) # Must convert reflector position to vel est. position [units of s]
            it = int(np.round(twt / dt)) # Convert travel time to nearest sample index
            
            # # Insert a simple spike (or Gaussian wavelet)
            if 0 <= it < nt:
                if wavetype == "spike":
                    data[j, it] = 1
                elif wavetype == "gaussian":
                    wavelet = np.exp(-(((t_samples - twt)) ** 2) * (rf**2))
                    omega0 = 2*np.pi*rf    # central frequency: 30 MHz (choose any)
                    sigma = 22e-9
                    morlet = np.exp(1j * omega0 * (t_samples - twt)) \
                             * np.exp(-(t_samples - twt)**2 / (2*sigma**2))
                    wavelet = morlet.real
                    wavelet /= np.max(np.abs(wavelet))   # important!
                    data[j,:] += wavelet

            # Only add noise once per trace
            if SNR != math.inf:
                # temporally correlated noise
                noise = colored_noise_ar1(nt, rho=0.9)
                
                # scale to match SNR
                noise *= np.std(data[j,:]) / np.sqrt(SNR)
            
                data[j,:] += noise
                
    if SNR != math.inf:
        data = data.T  # convert to [time, trace] first if needed
        spatial = spatial_noise(nx, nt, strength=1.5)
        data += spatial
        data = data.T

    return norm2d(data.T), x_positions, t_samples

def dipping_reflector(p1, p2):
    '''
    Generate a dipping reflector. Assumes dx == 1. Function should be updated to accound for dx, dt this way we don't have a buch of wasted overlapping dipping reflectors. The result should not 
    change with gprsim though bc that checks at each trace for reflectors
    '''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    slope = dy / dx

    # Build reflector line from p1 → p2
    line = []
    for x_offset in range(dx + 1):
        x = p1[0] + x_offset
        y = p1[1] + slope * x_offset
        line.append((x, y))

    return line

    

def NMO_correction(data, eps_r, t_0, x_0, region_shape, dx, dt):
    """
    Performs a Normal Moveout correction on a simulated gpr radargram using thevo output and metrics from gprsim

    Parameters
    ----------
    data : ndarray of shape (nt, nx)
        Simulated radargram matrix where:
        - Rows represent time samples.
        - Columns represent antenna positions along the surface (traces). 
        
    eps_r : float
        Relative permittivity (dielectric) of the medium.

    t_0, x_0 : float, int
        Guess corrdinates of the reflector in the simulated radargram. Can be easily modified to handle multiple refelctors but this is pointless as NMO can only correct one at a time
        
    region_shape : tuple(float, float)
        Physical extent of the simulated region, (xlen, tlen):
        - xlen : total horizontal distance [m].
        - tlen : total recording time window [s].
    
    dt : float
        Sampling interval [s].
    
    dx : float
        Spatial sampling interval (antenna step size) [m].

    Returns
    -------
    NMO_corrected_data : ndarray of shape (nt, nx)
        migrated data matrix in the same x, t space as input data:
        - Rows represent time samples.
        - Columns represent antenna positions along the surface (traces).

    """
    v = (c/np.sqrt(eps_r))/1e9         # [ns]
    
    NMO_corrected_data = np.zeros_like(data)
    nt, nx = data.shape
    for x in range(nx):
        x_offset = np.abs((x*dx) - x_0)
        t_correction = (np.sqrt(t_0**2+(2*x_offset/v)**2)-t_0)/(dt/1e-9)
        trace_original = data[:, x]
        t_range = np.arange(0, nt, 1) # 
        
        shifted = np.interp(t_range, t_range - t_correction, trace_original, left=0, right=0)
        NMO_corrected_data[:, x] = shifted
        
    return NMO_corrected_data


def minimize_tracewise_slope_window(data, n_candidates, dx, window):
    """
    Tracewise slope-constrained data preconditioning in noisy 2D data.
    
    Identifies a continuous path across a noisy 2D array
    by selecting, for each column (trace), the n-maximum intensity point that
    minimizes deviation from a locally consistent slope. The algorithm uses a 
    moving window to estimate the average slope of recentsegments, and choosing
    the candidate in the next column whose slope best matches this local trend.
    
    Parameters
    ----------
    data : np.ndarray of shape (n_rows, n_traces)
        2D array representing signal intensity, where each column corresponds to a
        spatial or temporal "trace" and each row corresponds to a sampled value (e.g.,
        depth, time, or vertical position).
    
    n_candidates : int
        Number of top local maxima (by intensity) to consider per column. A higher
        value increases robustness but also computation time.
    
    dx : float
        Horizontal step size between adjacent traces (used for slope calculation).
    
    window : int
        Number of previous traces to include in the local slope averaging window.
        Controls how strongly the algorithm enforces slope smoothness.
    
    Returns
    -------
    x : np.ndarray of shape (n_traces,)
        The horizontal coordinate values for each trace.
    
    path : np.ndarray of shape (n_traces,)
        The row indices of the selected path through the data, forming a smooth,
        slope-consistent trace across columns.
    
    Notes
    -----
    - *See smooth_path_dp* The algorithm performs a greedy minimization of local slope deviation, not a
      global optimization. For more robust pathfinding in very noisy data, consider
      extending it with a dynamic programming or cost-accumulation approach. 
    - The function assumes that larger values in `data` correspond to more likely
      "signal" locations (i.e., bright ridges or maxima).
    - The algorithm requires the the 1-max (the maximum) of the first two traces to fall on the hyperbola (becasue the alg is locally greedy, this is a major pitfall -- 
      leading to a runaway steep-slope path.
    """
    n_traces = data.shape[1]
    x = np.arange(n_traces) * dx

    # Precompute top-n candidate indices for each column
    sorted_indices = np.argsort(data, axis=0)[::-1, :]
    candidates = sorted_indices[:n_candidates, :].T  # shape (n_traces, n_candidates)

    # Initialize path with top candidate of first two columns
    path = np.zeros(n_traces, dtype=int)
    path[0] = candidates[0, 0]
    path[1] = candidates[1, 0]

    # Iterate through columns
    for i in range(2, n_traces):
        # Adjust window near beginning
        w = min(window, i)

        # Compute mean slope of recent steps
        local_trends = [path[i-j] - path[i-j-1] for j in range(1, w)]
        local_trend_mean = np.mean(local_trends) / dx

        trace_candidates = candidates[i]
        slope_diff = ((trace_candidates - path[i-1]) / dx) - local_trend_mean

        min_slope_cand = np.argmin(slope_diff**2)
        path[i] = trace_candidates[min_slope_cand]

    return x, path
    

def smooth_path_dp(data, dx, lam):
    """
    Dynamic programming path extraction with slope smoothness constraint.

    This function identifies a globally optimal path through a 2D data array
    (e.g., a noisy image or trace stack) by maximizing signal intensity while
    enforcing slope continuity between adjacent columns. It uses dynamic
    programming to accumulate an optimal cost matrix `C`, balancing the
    contribution of local intensity and smoothness between consecutive points.
    The final path is then reconstructed by backtracking from the maximum
    cumulative cost.

    Parameters
    ----------
    data : np.ndarray of shape (nt, nx)
        2D array representing the signal intensity field. Each column corresponds
        to a spatial or temporal "trace", and each row represents a vertical or
        temporal sample (e.g., depth or time). The algorithm finds a continuous
        high-intensity ridge across columns.
    
    lam : float
        Smoothness weighting factor (0 ≤ lam ≤ 1). Higher values increase the
        penalty on large vertical jumps between adjacent columns, promoting smoother
        paths; lower values emphasize following strong intensity values even if the
        path is jagged.

    Returns
    -------
    x : np.ndarray of shape (nx,)
        Column indices corresponding to each trace position.
    
    path : np.ndarray of shape (nx,)
        Row indices of the optimal path through the data, reconstructed from
        backtracking the maximum cumulative cost.
    
    C : np.ndarray of shape (nt, nx)
        Accumulated cost matrix representing the maximum achievable cumulative
        score up to each point (t, x) in the grid.
    
    backtrack : np.ndarray of shape (nt, nx)
        Index matrix storing the best predecessor for each (t, x), used to recover
        the final optimal path during backtracking.

    Notes
    -----
    - The dynamic programming formulation implicitly accounts for multi-column
      (global) slope trends since each column’s optimal cost includes all previous
      transitions. Thus, even though the local slope penalty is evaluated only
      against the immediate previous column, the optimization captures consistent
      longer-term slope behavior.
    - The computational complexity is O(nt² × nx), which can be reduced by
      restricting the allowed vertical transition range between columns.
    """
    nt, nx = data.shape
    C = np.zeros_like(data)
    backtrack = np.zeros_like(data, dtype=int)

    # Initialize first column
    C[:, 0] = data[:, 0]

    # DP forward pass, populate the memoized cost (C) and store the best path so far. This can be improved from n^2 but I thnk it's alright for now
    for x in range(1, nx):
        for t in range(nt):
            # Compute smoothness penalty relative to previous column, slope cost may not be needed
            penalties = ((1-lam)*C[:, x-1]) - (lam * (np.arange(nt) - t)**2)
            best_prev = np.argmax(penalties)
            C[t, x] = data[t, x] + penalties[best_prev]
            backtrack[t, x] = best_prev

    # Backtrack the best path
    path = np.zeros(nx, dtype=int)
    path[-1] = np.argmax(C[:, -1])
    for x in range(nx-2, -1, -1):
        path[x] = backtrack[path[x+1], x+1]

    return np.arange(0, nx), -path, C, backtrack

# I don't really like this method. Is it a unit test? I think it should be trashed personally
def visualize_preconditioning():
    fig, ax = plt.subplots(3, 3, figsize=(18,15))
    
    for SNR, i in zip([0.1, 0.05, 0.01], range(0,3)):
        # Parameters
        eps_r = 3                  # relative dielectric permittivity
        rf = 400e6                 # radar frequency different from pulse frequency
        dt = 1e-9                  # seconds
        dx = 1                     # meters
        region_shape = (70, 1e-6)  # grid (x,z)
        wavetype = 'gaussian'
        
        # Point reflectors at (x,t) where x [m] and t [s]
        reflectors = [(35, 50e-9)]
        
        data, x_positions, t_samples = gprsim(eps_r, rf, dt, dx, reflectors, region_shape, wavetype, SNR)
        
        im = ax[i][0].imshow(data, aspect='auto', cmap='seismic')
        if i == 2:
            ax[i][0].set_xlabel("Antenna position (m)")
        ax[i][0].set_ylabel("Time (ns)")
        ax[i][0].set_title(f"Simulated Radargram: SNR={SNR}")
        cbar = fig.colorbar(im, label="Amplitude")
        
        t, x = data.argsort(axis=0)[::-1,:][0], np.arange(0, data.shape[1]) * dx 
        x_window, t_window = minimize_tracewise_slope_window(data, 20, 1, 5)
        x_greedy, t_greedy = minimize_tracewise_slope_greedy(data, 15, dx)
        x_dynamic, t_dynamic, c, b = smooth_path_dp(data, dx, lam=0.001)
        
        ax[i][1].set_title(f"Data Preconditioning Using n-max Signals SNR={SNR}")
        im1 = ax[i][1].imshow(data, aspect='auto', extent=[0,70,-1000,0], cmap='seismic')
        ax[i][1].scatter(x, -t, label="Max signal at each trace")
        ax[i][1].plot(x_window, -t_window, label="informed greedy", c='black', linestyle='-')
        ax[i][1].plot(x_greedy, -t_greedy, label="simple greedy", c='black', linestyle='--')
        ax[i][1].plot(x_dynamic, t_dynamic, label="memoized cost", c='black', linestyle=':')
        ax[i][1].legend(loc="lower right")
    
    
        im2 = ax[i][2].imshow((c-c.min(axis=0))/(c.max(axis=0)-c.min(axis=0)), aspect='auto')
        ax[i][2].set_title("Normalized Score Matrix (memoized cost method \n -- accounting for slope and intensity)")


def fit_hyperbola(data, num_hyperbolas, method, dx, dt, x, t):
    '''
    "Finding an viable set of points for hyperbola fitting can be done by hand mapping reflectors on a radargram or automatically using algorithms"
    
    arameters
        ----------
        data : np.ndarray of shape (nt, nx)
            2D array representing the signal intensity field. Each column corresponds
            to a spatial or temporal "trace", and each row represents a vertical or
            temporal sample (e.g., depth or time). The algorithm finds a continuous
            high-intensity ridge across columns.
        
        method : string
            use pinv or np.polyfit. When pinv returns an error, np.polyfit becomes the default. The polyfit reduces the problem to a simpler liear form
    
        Returns
        -------
        v : needs comenting
        
        z : needs comenting
        
        x0 : needs comenting
        
    '''
    if method not in ['fit_from_max', 'faster_fit', 'robust_fit']:
        raise Exception(f'{method} not an allowed method')

    # Option one. SImple, but requires handholding (really nice input)
    if method == 'fit_from_max': 
        # Option 1: Lets just go through and get the time index of the highest magnitude frequency then fit a hyperbola
        t_points = data.argmax(axis=0) * dt # [ns] --> [s]
        x_points = np.arange(0, data.shape[1]) * dx # m

        # getting these points is essential for this method -- otherwise it fails
        apex = np.argmin(t_points)
        x0 = x_points[apex] # [m]
        t0 = t_points[apex]/2 # [s]
        
        # linear reg of t^2 vs (x-x0)^2
        x_offset_term = (x_points - x0)**2
        t_term = t_points**2
        # fit y = mx + b, x = x_offset
        slope, int = np.polyfit(x_offset_term, t_term, 1)  # returns slope, intercept

        if slope <0:
            return 0, 0, 0, 0
        v = 2.0 / np.sqrt(slope)      # subsurface velocity (m/s)
        z = v * np.sqrt(int) / 2.0    # depth (m)
        
        print(f"v = {v:.1f} m/s, depth z = {z:.2f} m, apex x0 = {x0:.3f} m, t0 = {t0} s")
        return v, z, x0, t0
        
    # Actually better than np.polyfit for noisy data
    if method=='robust_fit':
        # Our linearized form of the hyperbola equation
        A = np.column_stack([x**2, -2*x, 1.0*np.ones_like(x)])
        b = t**2
        
        # solve with pseudoinverse -- do this twice to reduce outliers
        consts = np.linalg.pinv(A) @ b
        r = A@consts - b  
        sigma = 1.5 * np.median(np.abs(r - np.median(r))) # scale mad
        
        mask = np.abs(r) <= 0.5 * sigma
        consts2 = np.linalg.pinv(A[mask]) @ b[mask]
        alpha, beta, gamma = consts2
        
        if alpha < -100000:
            print(Fore.RED + "Caught invalud value in sqrt: can not calculate hyperbola parameters from pinv. Switching to np.polyfit to avoid distruption. If np.polyfit quits, the input data may be too noisy")
            return fit_hyperbola(c, num_hyperbolas, 'fit_from_max', dx, dt, x, t)
        else:
            v = np.sqrt(1/alpha)
            x0 = beta * v**2
            t0 = np.sqrt(gamma-(x0**2/v**2))
        # print(f"v = {v:.1f} m/s, depth z = {z:.2f} m, apex x0 = {x0:.3f} m, t0 = {t0} s, risiduals={r}")
        return 2*v*1e9, x0, (t0/2)*1e-9

    return None

def minimize_tracewise_slope_greedy(data, n_candidates, dx):
    t = data.argsort(axis=0)[::-1,:][0]
    t_og = data.argsort(axis=0)[::-1,:][0]* -1
    x = np.arange(0, data.shape[1]) * dx # 
    t_nmax = data.argsort(axis=0)[::-1,:] * -1

    candidates = np.array([data.argsort(axis=0)[::-1,:][n] for n in range(0,n_candidates)])
    for i in range(0,len(x)): # go through each trace
        candidate_indices = candidates[:, i]
        min_slope = math.inf
        slopes = []
        for cand in candidate_indices:
            slope = np.abs((cand-t[i-1])/(x[i]-x[i-1])) # calcuate the slope for each candidate index
            slopes.append(slope)
            if slope < min_slope: # If the slope suddenly becomes eggerdious
                t[i]=cand
                min_slope = slope
    return x, t

