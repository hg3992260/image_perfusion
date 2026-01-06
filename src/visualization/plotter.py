import matplotlib.pyplot as plt
import numpy as np

def plot_perfusion_maps(original_img, maps, slice_idx=None, save_path=None):
    """
    Plots the original image and the calculated perfusion maps (CBF, CBV, MTT).
    
    Args:
        original_img (np.ndarray): The original CT/MRI image volume.
        maps (dict): Dictionary containing 'cbf', 'cbv', 'mtt' maps.
        slice_idx (int): The slice index to display. If None, uses the middle slice.
        save_path (str): Optional path to save the figure.
    """
    if slice_idx is None:
        if original_img.ndim == 4:
            slice_idx = original_img.shape[2] // 2
        elif original_img.ndim == 3:
            # If 2D+time or 3D volume
            # If 3D volume (x,y,z), original might be 4D (x,y,z,t)
            # Let's assume original_img is the spatial volume at a specific timepoint 
            # or the whole 4D dataset.
            # If 4D: (x, y, z, t)
            if original_img.shape[-1] > 50: # Likely time dim is last
                 # Wait, if 4D, slice_idx applies to Z.
                 slice_idx = original_img.shape[2] // 2
            else:
                 # If 3D (x, y, t) -> Single slice over time
                 slice_idx = 0 # Only one slice
        else:
            slice_idx = 0

    # Extract slice for display
    # Assume input is (x, y, z, t) or (x, y, t)
    # We display the average over time for the anatomical reference
    if original_img.ndim == 4:
        # (x, y, z, t)
        anat_slice = np.mean(original_img[:, :, slice_idx, :], axis=-1)
        cbf_slice = maps['cbf'][:, :, slice_idx]
        cbv_slice = maps['cbv'][:, :, slice_idx]
        mtt_slice = maps['mtt'][:, :, slice_idx]
    elif original_img.ndim == 3:
        # (x, y, t) -> Single slice dynamic
        anat_slice = np.mean(original_img, axis=-1)
        cbf_slice = maps['cbf']
        cbv_slice = maps['cbv']
        mtt_slice = maps['mtt']
    else:
        raise ValueError("Unsupported image dimension")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Anatomical Reference
    axes[0, 0].imshow(anat_slice, cmap='gray')
    axes[0, 0].set_title('Average Time Image')
    axes[0, 0].axis('off')
    
    # CBF
    im1 = axes[0, 1].imshow(cbf_slice, cmap='jet')
    axes[0, 1].set_title('CBF (mL/100g/min)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # CBV
    im2 = axes[1, 0].imshow(cbv_slice, cmap='jet')
    axes[1, 0].set_title('CBV (mL/100g)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # MTT
    im3 = axes[1, 1].imshow(mtt_slice, cmap='jet')
    axes[1, 1].set_title('MTT (s)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    # In a real GUI app, we wouldn't block, but for script:
    # plt.show() 
