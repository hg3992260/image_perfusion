import numpy as np
from scipy.linalg import toeplitz, svd

def calculate_ct_perfusion(ct_data, aif_coords, mask=None, dt=1.0, threshold=0.2):
    """
    Calculates CT Perfusion maps (CBF, CBV, MTT).
    
    Args:
        ct_data (np.ndarray): 4D CT data (x, y, z, time) or 3D (x, y, time).
        aif_coords (tuple): Coordinates of the AIF voxel (x, y) or (x, y, z).
        mask (np.ndarray): Boolean mask of brain tissue.
        dt (float): Time interval between frames in seconds.
        threshold (float): SVD truncation threshold.
        
    Returns:
        dict: Dictionary containing 'cbf', 'cbv', 'mtt' maps.
    """
    # Ensure data is 3D (flat_pixels, time) for vectorized processing
    # If 4D input (x, y, z, t), flatten spatial dims
    original_shape = ct_data.shape[:-1]
    time_points = ct_data.shape[-1]
    
    # Reshape to (N_pixels, Time)
    flat_data = ct_data.reshape(-1, time_points)
    
    # Extract AIF signal
    if len(aif_coords) == 2:
        # Assuming single slice processing or user provided 2D coords
        # If data is 3D (x,y,t)
        aif_signal = ct_data[aif_coords[0], aif_coords[1], :]
    elif len(aif_coords) == 3:
        # If data is 4D (x,y,z,t)
        aif_signal = ct_data[aif_coords[0], aif_coords[1], aif_coords[2], :]
    else:
        raise ValueError("Invalid AIF coordinates")
        
    # Baseline correction (simple approach: subtract mean of first few frames)
    # Assuming first 5 frames are baseline
    baseline_frames = 5
    aif_baseline = np.mean(aif_signal[:baseline_frames])
    aif_conc = aif_signal - aif_baseline
    aif_conc[aif_conc < 0] = 0 # Concentration cannot be negative
    
    # Pre-calculate SVD inverse matrix for AIF
    # A * R = C -> R = A_inv * C
    # A is Lower Triangular Toeplitz of AIF * dt
    n = time_points
    padding = np.zeros(n - 1)
    first_col = aif_conc
    first_row = np.zeros(n)
    first_row[0] = aif_conc[0]
    A = toeplitz(first_col, first_row) * dt
    
    U, S, Vt = svd(A)
    
    # Truncated SVD Inverse
    max_s = np.max(S)
    S_inv = np.zeros((n, n))
    for i in range(n):
        if S[i] >= threshold * max_s:
            S_inv[i, i] = 1.0 / S[i]
            
    # P = V * S_inv * U.T (Note: Vt is V.T)
    # P = Vt.T @ S_inv @ U.T
    P_inv = Vt.T @ S_inv @ U.T
    
    # Vectorized Deconvolution
    # R_all (N, T) = C_all (N, T) @ P_inv.T (T, T)
    # But wait, C_all needs baseline correction first
    
    # Calculate tissue concentration (baseline subtraction)
    # Vectorized baseline subtraction
    pixel_baselines = np.mean(flat_data[:, :baseline_frames], axis=1, keepdims=True)
    tissue_conc = flat_data - pixel_baselines
    tissue_conc[tissue_conc < 0] = 0
    
    # Apply mask if provided
    if mask is not None:
        flat_mask = mask.flatten()
        # Only process masked pixels to save time (though matrix mult is fast)
        # For simplicity, we process all and zero out later, or masking here would be better
        # Let's process all for code simplicity in this demo, unless huge
        pass

    # Deconvolution
    # shape: (N, T) = (N, T) dot (T, T)
    residue_functions = tissue_conc @ P_inv.T
    
    # Calculate Parameters
    # CBF = max(R(t))
    cbf_map_flat = np.max(residue_functions, axis=1)
    
    # CBV = Integral(C) / Integral(AIF)
    aif_integral = np.trapz(aif_conc, dx=dt) + 1e-6
    tissue_integrals = np.trapz(tissue_conc, axis=1, dx=dt)
    cbv_map_flat = tissue_integrals / aif_integral
    
    # MTT = CBV / CBF
    # Avoid divide by zero
    mtt_map_flat = np.zeros_like(cbf_map_flat)
    valid_mask = cbf_map_flat > 1e-3 # Threshold for validity
    mtt_map_flat[valid_mask] = cbv_map_flat[valid_mask] / cbf_map_flat[valid_mask]
    
    # Calculate Tmax / TTP
    # TTP: Time to Peak of Concentration Curve
    ttp_indices = np.argmax(tissue_conc, axis=1)
    ttp_map_flat = ttp_indices * dt
    
    # Tmax: Time to Peak of Residue Function R(t)
    # Theoretically R(t) peaks at t=0 for bolus injection, but with delay, it peaks at delay time.
    tmax_indices = np.argmax(residue_functions, axis=1)
    tmax_map_flat = tmax_indices * dt
    
    # Reshape back to original dimensions
    cbf_map = cbf_map_flat.reshape(original_shape)
    cbv_map = cbv_map_flat.reshape(original_shape)
    mtt_map = mtt_map_flat.reshape(original_shape)
    ttp_map = ttp_map_flat.reshape(original_shape)
    tmax_map = tmax_map_flat.reshape(original_shape)
    
    # Apply mask to results
    if mask is not None:
        cbf_map[~mask] = 0
        cbv_map[~mask] = 0
        mtt_map[~mask] = 0
        ttp_map[~mask] = 0
        tmax_map[~mask] = 0
        
    # Scale Units
    # CBF: mL/100g/min. 
    # Current units: 
    # C ~ HU. AIF ~ HU. 
    # R = C/A * 1/dt. Unit is 1/time (s^-1).
    # To get mL/100g/min:
    # We need a scaling factor K depending on CT scanner and contrast.
    # Often K = 6000 (to convert seconds to min and 1/g to 100g/mL approx if density assumed)
    # This is highly scanner dependent. We use a generic factor here.
    K = 60 * 100 # Simple conversion: 1/s -> 1/min (*60), ratio -> % (*100)
    
    cbf_map *= K
    cbv_map *= 100 # CBV is dimensionless ratio, often expressed as mL/100g (which is %)
    # MTT is in seconds, no change needed.
    
    return {
        'cbf': cbf_map,
        'cbv': cbv_map,
        'mtt': mtt_map,
        'ttp': ttp_map,
        'tmax': tmax_map
    }
