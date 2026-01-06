import numpy as np
from scipy.linalg import toeplitz, svd

def create_convolution_matrix(aif, dt):
    """
    Creates a convolution matrix from the Arterial Input Function (AIF).
    
    Args:
        aif (np.ndarray): 1D array of AIF concentration values.
        dt (float): Time interval between frames.
        
    Returns:
        np.ndarray: Convolution matrix A.
    """
    n = len(aif)
    # Standard time-domain convolution matrix (lower triangular Toeplitz)
    # A * R = C
    # The matrix should be constructed such that:
    # C(t) = dt * sum(AIF(t-tau) * R(tau))
    # We use a Toeplitz matrix construction.
    # First column is AIF, first row is [AIF[0], 0, 0, ...]
    
    # However, for bSVD (Block-circulant SVD), which is insensitive to delay,
    # we construct a block-circulant matrix. 
    # Here we implement a standard SVD approach with simple Volterra discretization first
    # for simplicity and clarity, but with SVD regularization.
    
    # Using simple lower triangular Toeplitz for standard deconvolution
    padding = np.zeros(n - 1)
    first_col = aif
    first_row = np.zeros(n)
    first_row[0] = aif[0]
    
    # scipy.linalg.toeplitz constructs a Toeplitz matrix
    # We need a lower triangular one.
    A = toeplitz(first_col, first_row)
    return A * dt

def svd_deconvolution(tissue_conc, aif, dt, threshold=0.2):
    """
    Performs SVD deconvolution to estimate the Residue Function R(t) and CBF.
    
    Args:
        tissue_conc (np.ndarray): 1D array of tissue concentration values C_tissue(t).
        aif (np.ndarray): 1D array of Arterial Input Function values C_a(t).
        dt (float): Time interval between frames.
        threshold (float): SVD truncation threshold (relative to max singular value).
                           Default is 0.2 (20%).
                           
    Returns:
        dict: Dictionary containing:
            - 'cbf': Cerebral Blood Flow (mL/100g/min if units are consistent)
            - 'cbv': Cerebral Blood Volume (mL/100g)
            - 'mtt': Mean Transit Time (s)
            - 'residue_function': The calculated R(t) array
    """
    n = len(tissue_conc)
    
    # 1. Construct the convolution matrix A
    A = create_convolution_matrix(aif, dt)
    
    # 2. Perform SVD on A: A = U * S * V.T
    U, S, Vt = svd(A)
    
    # 3. Regularization (Truncated SVD)
    # Filter singular values
    max_s = np.max(S)
    # Create diagonal inverse matrix with truncation
    S_inv = np.zeros((n, n))
    for i in range(n):
        if S[i] >= threshold * max_s:
            S_inv[i, i] = 1.0 / S[i]
        else:
            S_inv[i, i] = 0.0
            
    # 4. Calculate R(t)
    # R = V * S_inv * U.T * C_tissue
    # Note: numpy/scipy svd returns Vt (V transposed)
    # So R = Vt.T * S_inv * U.T * C_tissue
    
    # Operation order: 
    # temp = U.T @ tissue_conc
    # temp = S_inv @ temp
    # R = Vt.T @ temp
    
    R = Vt.T @ (S_inv @ (U.T @ tissue_conc))
    
    # 5. Calculate Parameters
    # CBF is the peak of the residue function (theoretically R(0), but max is safer)
    cbf = np.max(R)
    
    # CBV is the area under the curve of concentration divided by area of AIF
    # According to Central Volume Principle: CBV = CBF * MTT
    # And C_tissue(t) = CBF * (AIF ** R)(t) -> Integral(C) = CBF * Integral(AIF) * Integral(R)
    # So Integral(R) = Integral(C) / (CBF * Integral(AIF)) ?? No.
    # Actually, Integral(C_tissue) = CBV * Integral(AIF) / k ??
    # Simpler: CBV = Integral(C_tissue) / Integral(AIF)
    # This is model-independent.
    
    cbv_integral = np.trapz(tissue_conc, dx=dt) / (np.trapz(aif, dx=dt) + 1e-6)
    
    # Alternatively calculated from R(t): CBV = CBF * Integral(R(t))
    # Let's use the integral method as it's often more robust.
    
    # MTT = CBV / CBF
    if cbf > 0:
        mtt = cbv_integral / cbf
    else:
        mtt = 0
        
    # Scale units if necessary (usually handled outside, here we return raw values)
    # Standard medical units often require scaling (e.g., * 6000 for mL/100g/min if density is 1g/mL)
    
    return {
        'cbf': cbf,
        'cbv': cbv_integral,
        'mtt': mtt,
        'residue_function': R
    }
