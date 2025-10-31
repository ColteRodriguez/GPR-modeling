import numpy as np
import math

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
                    data[j, :] += wavelet
                
                if SNR != math.inf:
                    # Gaussian noise with mean 0 and variance 1
                    noise = np.random.randn(nt)
                    noise = noise * np.std(data[j, :])/np.sqrt(SNR)
                    data[j,:]+=noise
            
    return data.T, x_positions, t_samples


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


def fit_hyperbola(data, num_hyperbolas, method, dx, dt):
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
        t0 = t_points[apex] # [s]
        
        # linear reg of t^2 vs (x-x0)^2
        x_offset_term = (x_points - x0)**2
        t_term = t_points**2
        # fit y = mx + b, x = x_offset
        slope, int = np.polyfit(x_offset_term, t_term, 1)  # returns slope, intercept
        
        v = 2.0 / np.sqrt(slope)      # subsurface velocity (m/s)
        z = v * np.sqrt(int) / 2.0    # depth (m)
        
        print(f"v = {v:.1f} m/s, depth z = {z:.2f} m, apex x0 = {x0:.3f} m, t0 = {t0} s")

    # # I think the c guess and optimization problem is interesting, but I fear it will be slow and innacurate
    # if method=='faster_fit':
        

    # # This is almost identical to what np.polyfit does but I wanted to implement it. Just use the pseudoinverse to obtain noise-resistant least squares solution
    # if method=='robust_fit':


    return None