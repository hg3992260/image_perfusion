import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.ct_perfusion import calculate_ct_perfusion
from src.core.mri_perfusion import calculate_mri_perfusion
from src.visualization.plotter import plot_perfusion_maps
from src.utils.phantom import create_phantom_data

def main():
    print("Starting Perfusion Analysis Simulation...")
    
    # 1. Simulate Data (CT-like linear concentration)
    print("Generating Phantom Data...")
    phantom_conc, aif_coords = create_phantom_data(n_frames=60, dt=1.0, size=64)
    
    # Create mask (simple threshold)
    mask = np.mean(phantom_conc, axis=-1) > 1.0
    
    # ---------------------------------------------------------
    # CT Perfusion Workflow
    # ---------------------------------------------------------
    print("\n--- Running CT Perfusion Analysis ---")
    # For CT, HU is proportional to concentration.
    # We use the phantom data directly as CT HU enhancement (after baseline subtraction in real life)
    # We add a baseline HU for realism
    ct_data = phantom_conc + 40 # 40 HU baseline
    
    ct_results = calculate_ct_perfusion(
        ct_data=ct_data,
        aif_coords=aif_coords,
        mask=mask,
        dt=1.0,
        threshold=0.15
    )
    
    print("CT Calculation Complete.")
    print(f"Max CBF: {np.max(ct_results['cbf']):.2f} mL/100g/min")
    
    # Save CT Results
    if not os.path.exists('doc'):
        os.makedirs('doc')
    
    plot_perfusion_maps(
        ct_data, 
        ct_results, 
        save_path='doc/ct_perfusion_results.png'
    )

    # ---------------------------------------------------------
    # MRI Perfusion Workflow (DSC)
    # ---------------------------------------------------------
    print("\n--- Running MRI Perfusion Analysis ---")
    # Convert concentration to Signal
    # S = S0 * exp(-TE * C)
    # Let TE = 0.030 s (30ms)
    TE = 0.030
    S0 = 1000
    
    # mri_signal = S0 * exp(-TE * phantom_conc * Scale)
    # Phantom conc is in arbitrary units, let's assume it scales to R2*
    # We use the phantom_conc directly as R2* change roughly
    mri_data = S0 * np.exp(-TE * phantom_conc * 10) # *10 for strong signal drop
    
    # Add Rician noise (simplified as Gaussian here for speed)
    noise = np.random.normal(0, 20, mri_data.shape)
    mri_data += noise
    mri_data = np.abs(mri_data) # Rician is magnitude
    
    mri_results = calculate_mri_perfusion(
        mri_data=mri_data,
        aif_coords=aif_coords,
        te=TE,
        mask=mask,
        dt=1.0,
        threshold=0.15
    )
    
    print("MRI Calculation Complete.")
    print(f"Max CBF: {np.max(mri_results['cbf']):.2f} (relative units)")
    
    plot_perfusion_maps(
        mri_data, 
        mri_results, 
        save_path='doc/mri_perfusion_results.png'
    )
    
    print("\nAnalysis Complete. Results saved to 'doc/' folder.")

if __name__ == "__main__":
    main()
