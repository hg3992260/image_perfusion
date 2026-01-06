import numpy as np
from scipy.linalg import toeplitz, svd

def calculate_mri_perfusion(mri_data, aif_coords, te, mask=None, dt=1.0, threshold=0.2):
    """
    Calculates MRI Perfusion maps (CBF, CBV, MTT) using DSC-MRI principles.
    
    Args:
        mri_data (np.ndarray): 4D MRI data (x, y, z, time) or 3D (x, y, time).
        aif_coords (tuple): Coordinates of the AIF voxel.
        te (float): Echo Time in seconds (e.g., 0.030 for 30ms).
        mask (np.ndarray): Boolean mask of brain tissue.
        dt (float): Time interval between frames in seconds (TR).
        threshold (float): SVD truncation threshold.
        
    Returns:
        dict: Dictionary containing 'cbf', 'cbv', 'mtt' maps.
    """
    # Ensure data is 3D (flat_pixels, time)
    original_shape = mri_data.shape[:-1]
    time_points = mri_data.shape[-1]
    flat_data = mri_data.reshape(-1, time_points)
    
    # 1. Signal to Concentration Conversion
    # S(t) = S0 * exp(-k * TE * C(t))
    # C(t) = - (1 / (k * TE)) * ln(S(t) / S0)
    # We absorb k into the relative concentration units as it's a global scaling factor.
    # C_rel(t) = - (1 / TE) * ln(S(t) / S0)
    
    # Calculate Baseline S0 (average of first few pre-contrast frames)
    baseline_frames = 5
    s0_flat = np.mean(flat_data[:, :baseline_frames], axis=1, keepdims=True)
    
    # Avoid division by zero
    s0_flat[s0_flat == 0] = 1e-9
    
    # Calculate Ratio S(t)/S0
    # Note: Adding small epsilon to avoid log(0)
    signal_ratio = flat_data / s0_flat
    signal_ratio[signal_ratio <= 0] = 1e-9 
    
    # Calculate Concentration
    # C = -1/TE * ln(S/S0)
    tissue_conc = - (1.0 / te) * np.log(signal_ratio)
    
    # Handle noise (where S(t) > S0 -> ln > 0 -> C < 0)
    tissue_conc[tissue_conc < 0] = 0
    
    # 2. Extract AIF Concentration
    # Need to convert AIF signal to concentration too!
    if len(aif_coords) == 2:
        aif_signal = mri_data[aif_coords[0], aif_coords[1], :]
    elif len(aif_coords) == 3:
        aif_signal = mri_data[aif_coords[0], aif_coords[1], aif_coords[2], :]
    else:
        raise ValueError("Invalid AIF coordinates")
        
    aif_s0 = np.mean(aif_signal[:baseline_frames])
    if aif_s0 == 0: aif_s0 = 1e-9
    
    aif_ratio = aif_signal / aif_s0
    aif_ratio[aif_ratio <= 0] = 1e-9
    aif_conc = - (1.0 / te) * np.log(aif_ratio)
    aif_conc[aif_conc < 0] = 0
    
    # 3. Prepare Deconvolution Matrix (Same as CT)
    n = time_points
    padding = np.zeros(n - 1)
    first_col = aif_conc
    first_row = np.zeros(n)
    first_row[0] = aif_conc[0]
    A = toeplitz(first_col, first_row) * dt
    
    U, S, Vt = svd(A)
    
    max_s = np.max(S)
    S_inv = np.zeros((n, n))
    for i in range(n):
        if S[i] >= threshold * max_s:
            S_inv[i, i] = 1.0 / S[i]
            
    P_inv = Vt.T @ S_inv @ U.T
    
    # 4. Deconvolution
    residue_functions = tissue_conc @ P_inv.T
    
    # 5. Parameter Calculation
    cbf_map_flat = np.max(residue_functions, axis=1)
    
    aif_integral = np.trapz(aif_conc, dx=dt) + 1e-6
    tissue_integrals = np.trapz(tissue_conc, axis=1, dx=dt)
    cbv_map_flat = tissue_integrals / aif_integral
    
    mtt_map_flat = np.zeros_like(cbf_map_flat)
    valid_mask = cbf_map_flat > 1e-3
    mtt_map_flat[valid_mask] = cbv_map_flat[valid_mask] / cbf_map_flat[valid_mask]
    
    # Calculate Tmax / TTP
    ttp_indices = np.argmax(tissue_conc, axis=1)
    ttp_map_flat = ttp_indices * dt
    
    tmax_indices = np.argmax(residue_functions, axis=1)
    tmax_map_flat = tmax_indices * dt
    
    # Reshape
    cbf_map = cbf_map_flat.reshape(original_shape)
    cbv_map = cbv_map_flat.reshape(original_shape)
    mtt_map = mtt_map_flat.reshape(original_shape)
    ttp_map = ttp_map_flat.reshape(original_shape)
    tmax_map = tmax_map_flat.reshape(original_shape)
    
    if mask is not None:
        cbf_map[~mask] = 0
        cbv_map[~mask] = 0
        mtt_map[~mask] = 0
        ttp_map[~mask] = 0
        tmax_map[~mask] = 0
        
    # Scale Units
    # For MRI, absolute quantification is difficult without knowledge of AIF partial volume effects.
    # Often relative CBF/CBV are reported.
    # However, to be consistent with CT, we can apply similar scaling if we assume k is calibrated.
    # Usually rCBF (relative CBF) is sufficient.
    # We apply a generic scaling factor for display purposes.
    K = 60 * 100 
    cbf_map *= K
    cbv_map *= 100
    
    return {
        'cbf': cbf_map,
        'cbv': cbv_map,
        'mtt': mtt_map,
        'ttp': ttp_map,
        'tmax': tmax_map
    }
