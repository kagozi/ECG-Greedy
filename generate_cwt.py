# ============================================================================
# STEP 2: Generate CWT Representations (Scalograms & Phasograms)
# ============================================================================
# Run this after 1_load_and_standardize.py
# Processes standardized signals in batches to avoid memory issues
# Outputs: train/val/test scalograms and phasograms

import os
import pickle
import numpy as np
import pywt
from scipy.ndimage import zoom
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
SAMPLING_RATE = 100
IMAGE_SIZE = 224
BATCH_SIZE = 100  # Process this many samples at a time to manage memory

print("="*80)
print("STEP 2: GENERATE CWT REPRESENTATIONS")
print("="*80)

# ============================================================================
# CWT GENERATOR CLASS
# ============================================================================

class CWTGenerator:
    """
    Generate scalograms and phasograms from standardized ECG signals
    Processes data in batches to avoid memory overflow
    """
    
    def __init__(self, sampling_rate=100, image_size=224, wavelet='cmor1.5-1.0'):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = wavelet
        
        # Generate scales for target frequency range
        freq_min, freq_max = 0.5, 40.0
        n_scales = 128
        
        cf = pywt.central_frequency(wavelet)
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_scales)
        self.scales = (cf * sampling_rate) / freqs
        
        print(f"\nCWT Generator Configuration:")
        print(f"  Wavelet: {wavelet}")
        print(f"  Scales: {len(self.scales)} (freq range: {freq_min}-{freq_max} Hz)")
        print(f"  Output size: {image_size}×{image_size}")
    
    def compute_cwt_single_lead(self, signal_1d):
        """Compute CWT for a single lead"""
        try:
            coefficients, _ = pywt.cwt(
                signal_1d,
                self.scales,
                self.wavelet,
                sampling_period=1.0 / self.sampling_rate
            )
            return coefficients
        except Exception as e:
            print(f"  Warning - CWT error: {e}")
            return None
    
    def generate_scalogram(self, coefficients):
        """Generate scalogram (power spectrum) from CWT coefficients"""
        # Power spectrum
        scalogram = np.abs(coefficients) ** 2
        
        # Log scaling for better visualization
        scalogram = np.log10(scalogram + 1e-10)
        
        # Robust normalization using percentiles
        p5, p95 = np.percentile(scalogram, [5, 95])
        scalogram = np.clip(scalogram, p5, p95)
        
        # Normalize to [0, 1]
        min_val, max_val = scalogram.min(), scalogram.max()
        if max_val - min_val > 1e-10:
            scalogram = (scalogram - min_val) / (max_val - min_val)
        else:
            scalogram = np.zeros_like(scalogram)
        
        return scalogram.astype(np.float32)
    
    def generate_phasogram(self, coefficients):
        """Generate phasogram (phase information) from CWT coefficients"""
        # Extract phase
        phase = np.angle(coefficients)
        
        # Normalize phase from [-π, π] to [0, 1]
        phasogram = (phase + np.pi) / (2 * np.pi)
        
        return phasogram.astype(np.float32)
    
    def resize_to_image(self, cwt_matrix):
        """Resize CWT matrix to target image size"""
        zoom_factors = (
            self.image_size / cwt_matrix.shape[0],
            self.image_size / cwt_matrix.shape[1]
        )
        return zoom(cwt_matrix, zoom_factors, order=1)
    
    def process_12_lead_ecg(self, ecg_12_lead):
        """
        Process 12-lead ECG to generate scalogram and phasogram
        
        Args:
            ecg_12_lead: (time, 12) or (12, time) array
            
        Returns:
            scalogram: (12, H, W) array
            phasogram: (12, H, W) array
        """
        # Ensure shape is (12, time)
        if ecg_12_lead.shape[0] != 12:
            ecg_12_lead = ecg_12_lead.T
        
        scalograms = []
        phasograms = []
        
        for lead_idx in range(12):
            # Compute CWT for this lead
            coeffs = self.compute_cwt_single_lead(ecg_12_lead[lead_idx])
            
            if coeffs is None:
                # If CWT fails, use zeros
                scalograms.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                phasograms.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                continue
            
            # Generate scalogram and phasogram
            scalo = self.generate_scalogram(coeffs)
            phaso = self.generate_phasogram(coeffs)
            
            # Resize to target size
            scalo_resized = self.resize_to_image(scalo)
            phaso_resized = self.resize_to_image(phaso)
            
            scalograms.append(scalo_resized)
            phasograms.append(phaso_resized)
        
        # Stack all leads: (12, H, W)
        scalogram_12ch = np.stack(scalograms, axis=0)
        phasogram_12ch = np.stack(phasograms, axis=0)
        
        return scalogram_12ch, phasogram_12ch
    
    def process_dataset_batched(self, X, output_scalo_path, output_phaso_path, batch_size=100):
        """
        Process entire dataset in batches to avoid memory issues
        Saves directly to disk
        
        Args:
            X: (N, time, 12) array of ECG signals (can be memory-mapped)
            output_scalo_path: Path to save scalograms
            output_phaso_path: Path to save phasograms
            batch_size: Number of samples to process at once
        """
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\nProcessing {n_samples} samples in {n_batches} batches...")
        print(f"Output will be: ({n_samples}, 12, {self.image_size}, {self.image_size})")
        
        # Pre-allocate output arrays using memmap for memory efficiency
        shape = (n_samples, 12, self.image_size, self.image_size)
        scalograms = np.memmap(output_scalo_path, dtype='float32', mode='w+', shape=shape)
        phasograms = np.memmap(output_phaso_path, dtype='float32', mode='w+', shape=shape)
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Process this batch
            for i in range(start_idx, end_idx):
                scalo, phaso = self.process_12_lead_ecg(X[i])
                scalograms[i] = scalo
                phasograms[i] = phaso
            
            # Flush to disk every batch
            scalograms.flush()
            phasograms.flush()
        
        # Clean up memmap references
        del scalograms
        del phasograms
        
        # Now load and save as regular numpy arrays for easier loading
        print(f"Converting to standard numpy format...")
        scalograms_final = np.memmap(output_scalo_path, dtype='float32', mode='r', shape=shape)
        phasograms_final = np.memmap(output_phaso_path, dtype='float32', mode='r', shape=shape)
        
        # Save as standard .npy files (this will allow easy loading with mmap_mode)
        np.save(output_scalo_path.replace('.npy', '_final.npy'), scalograms_final)
        np.save(output_phaso_path.replace('.npy', '_final.npy'), phasograms_final)
        
        # Remove temporary memmap files
        del scalograms_final
        del phasograms_final
        
        # Rename final files
        import shutil
        shutil.move(output_scalo_path.replace('.npy', '_final.npy'), output_scalo_path)
        shutil.move(output_phaso_path.replace('.npy', '_final.npy'), output_phaso_path)
        
        print(f"✓ Saved scalograms to: {output_scalo_path}")
        print(f"✓ Saved phasograms to: {output_phaso_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Load metadata
    print("\n[1/4] Loading metadata...")
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
    print(f"  Train: {metadata['train_size']} samples")
    print(f"  Val:   {metadata['val_size']} samples")
    print(f"  Test:  {metadata['test_size']} samples")
    print(f"  Signal shape: {metadata['signal_shape']}")
    
    # Initialize CWT generator
    print("\n[2/4] Initializing CWT generator...")
    cwt_gen = CWTGenerator(
        sampling_rate=SAMPLING_RATE,
        image_size=IMAGE_SIZE,
        wavelet='cmor1.5-1.0'
    )
    
    # Process training set
    print("\n[3/4] Processing TRAINING set...")
    X_train = np.load(os.path.join(PROCESSED_PATH, 'train_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_train,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_train  # Free memory
    
    # Process validation set
    print("\n[3/4] Processing VALIDATION set...")
    X_val = np.load(os.path.join(PROCESSED_PATH, 'val_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_val,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_val
    
    # Process test set
    print("\n[4/4] Processing TEST set...")
    X_test = np.load(os.path.join(PROCESSED_PATH, 'test_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_test,
        output_scalo_path=os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
        output_phaso_path=os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_test
    
    print("\n" + "="*80)
    print("STEP 2 COMPLETE!")
    print("="*80)
    print(f"\nAll CWT representations saved to: {PROCESSED_PATH}")
    print("\nFiles created:")
    print("  - train_scalograms.npy")
    print("  - train_phasograms.npy")
    print("  - val_scalograms.npy")
    print("  - val_phasograms.npy")
    print("  - test_scalograms.npy")
    print("  - test_phasograms.npy")
    print("\nNext step: Run 3_train_models.py to train CNN models")


if __name__ == '__main__':
    main()