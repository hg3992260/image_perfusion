import numpy as np
from scipy.stats import gamma

def generate_gamma_variate_aif(t, arrival_time=10, peak_time=20, width=5, amplitude=100):
    """
    Generates a Gamma-variate function to simulate AIF.
    C(t) = K * (t - t0)^alpha * exp(-(t-t0)/beta)
    """
    # Simplified Gamma variate
    # Peak is roughly at t0 + alpha*beta
    # Let's use scipy's gamma pdf for simplicity
    
    # Shift time
    t_shifted = t - arrival_time
    t_shifted[t_shifted < 0] = 0
    
    # Parameters for shape
    a = (peak_time - arrival_time) / width # shape parameter
    scale = width # scale parameter
    
    y = gamma.pdf(t_shifted, a, scale=scale)
    y = y / np.max(y) * amplitude
    return y

def create_phantom_data(n_frames=60, dt=1.0, size=64):
    """
    Creates a synthetic 3D (x, y, t) dataset simulating perfusion.
    Includes an "artery" pixel and "tissue" pixels with different hemodynamics.
    """
    t = np.arange(n_frames) * dt
    data = np.zeros((size, size, n_frames))
    
    # 1. Generate AIF
    aif = generate_gamma_variate_aif(t, arrival_time=5, peak_time=10, width=2, amplitude=100)
    
    # 2. Define Tissue Types
    # Normal Tissue: Delayed and dispersed version of AIF
    # CBF determines amplitude scaling
    # MTT determines dispersion (width)
    
    # Convolve AIF with an exponential residue function R(t) = exp(-t/MTT)
    # C(t) = CBF * (AIF * R)(t)
    
    def get_tissue_curve(cbf, mtt, delay=0):
        # Residue function
        r_t = np.exp(-t / mtt)
        r_t[0] = 1 # boundary
        
        # AIF with delay
        # Simple shift
        aif_delayed = np.interp(t - delay, t, aif, left=0, right=0)
        
        # Convolve
        # Convolution theorem: C = CBF * convolve(AIF, R) * dt
        # We use full convolution and trim
        c_t = cbf * np.convolve(aif_delayed, r_t, mode='full')[:n_frames] * dt
        return c_t

    # 3. Fill Phantom
    # Background
    bg_noise = np.random.normal(0, 1, (size, size, n_frames))
    data += bg_noise
    
    # Artery Region (Top Left)
    data[10:15, 10:15, :] += aif.reshape(1, 1, -1)
    
    # Normal Tissue (Center)
    # CBF=60, MTT=4s
    norm_curve = get_tissue_curve(cbf=60/6000, mtt=4, delay=2) # Scaled down CBF
    data[20:44, 20:44, :] += norm_curve.reshape(1, 1, -1)
    
    # Ischemic Tissue (Bottom Right) - Penumbra
    # Low CBF, High MTT
    # CBF=20, MTT=8s
    penumbra_curve = get_tissue_curve(cbf=20/6000, mtt=8, delay=4)
    data[30:50, 30:50, :] += penumbra_curve.reshape(1, 1, -1)
    
    # Infarct Core (Bottom Left)
    # Very Low CBF, Variable MTT
    infarct_curve = get_tissue_curve(cbf=5/6000, mtt=10, delay=5)
    data[40:55, 10:25, :] += infarct_curve.reshape(1, 1, -1)
    
    # Return data and AIF location
    return data, (12, 12)
