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
from numpy.lib.format import open_memmap
# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
SAMPLING_RATE = 100
IMAGE_SIZE = 224
BATCH_SIZE = 100  # Process this many samples at a time to manage memory

print("="*80)
print("STEP 2: GENERATE CWT REPRESENTATIONS")
print("="*80)
os.makedirs(WAVELETS_PATH, exist_ok=True)
# ============================================================================
# CWT GENERATOR CLASS
# ============================================================================
class CWTGenerator:
    """
    Generate scalograms and phasograms from standardized ECG signals
    Processes data in batches to avoid memory overflow
    """
    
    def __init__(self, sampling_rate=100, image_size=224, wavelet='cmor2.0-1.0'):
        self.sampling_rate = sampling_rate
        self.image_size = image_size
        self.wavelet = wavelet
        
        # Frequency range best for ECG (clinical bands)
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
        """Generate scalogram (power spectrum)"""
        scalogram = np.abs(coefficients) ** 2
        
        # ✅ Better dynamic range handling
        scalogram = np.log1p(scalogram)
        
        # ✅ ENHANCED: Robust outlier handling with percentile clipping
        p1, p99 = np.percentile(scalogram, [1, 99])
        scalogram = np.clip(scalogram, p1, p99)
        
        # ✅ Normalize per-lead for stability
        min_val, max_val = scalogram.min(), scalogram.max()
        if max_val - min_val > 1e-10:
            scalogram = (scalogram - min_val) / (max_val - min_val)
        else:
            scalogram = np.zeros_like(scalogram)
        
        return scalogram.astype(np.float32)
    
    def generate_phasogram(self, coefficients):
        """Generate phasogram (phase information)"""
        phase = np.angle(coefficients)
        
        # Normalize from [-π, π] → [0, 1]
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
        """Convert (12-lead ECG) → scalogram + phasogram"""
        
        if ecg_12_lead.shape[0] != 12:
            ecg_12_lead = ecg_12_lead.T
        
        scalograms, phasograms = [], []
        
        for lead_idx in range(12):
            coeffs = self.compute_cwt_single_lead(ecg_12_lead[lead_idx])
            
            if coeffs is None:
                scalograms.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                phasograms.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                continue
            
            scalo = self.generate_scalogram(coeffs)
            phaso = self.generate_phasogram(coeffs)
            
            scalograms.append(self.resize_to_image(scalo))
            phasograms.append(self.resize_to_image(phaso))
        
        return (
            np.stack(scalograms, axis=0),
            np.stack(phasograms, axis=0)
        )
        
    def process_dataset_batched(self, X, output_scalo_path, output_phaso_path, batch_size=100):
        n_samples = len(X)
        shape = (n_samples, 12, self.image_size, self.image_size)

        # create proper .npy memmaps with headers
        scalograms = open_memmap(output_scalo_path, mode='w+', dtype='float32', shape=shape)
        phasograms = open_memmap(output_phaso_path, mode='w+', dtype='float32', shape=shape)

        n_batches = (n_samples + batch_size - 1) // batch_size
        for b in tqdm(range(n_batches), desc="Processing batches"):
            s, e = b * batch_size, min((b + 1) * batch_size, n_samples)
            for i in range(s, e):
                scalo, phaso = self.process_12_lead_ecg(X[i])
                scalograms[i] = scalo
                phasograms[i] = phaso
            scalograms.flush(); phasograms.flush()

        del scalograms, phasograms  # ensure buffers are written

        # verify
        scalograms_view = np.load(output_scalo_path, mmap_mode='r')       # OK now
        phasograms_view = np.load(output_phaso_path, mmap_mode='r')
        print("✓", scalograms_view.shape, scalograms_view.dtype)
        print("✓", phasograms_view.shape, phasograms_view.dtype)
    
    # def process_dataset_batched(self, X, output_scalo_path, output_phaso_path, batch_size=100):
    #     """Memory-safe processing of the whole dataset"""
        
    #     n_samples = len(X)
    #     n_batches = (n_samples + batch_size - 1) // batch_size
    #     print(f"\nProcessing {n_samples} samples in {n_batches} batches...")
        
    #     shape = (n_samples, 12, self.image_size, self.image_size)
        
    #     # ✅ FIX: Create memmap with proper format
    #     scalograms = np.memmap(
    #         output_scalo_path, 
    #         dtype='float32', 
    #         mode='w+', 
    #         shape=shape
    #     )
    #     phasograms = np.memmap(
    #         output_phaso_path, 
    #         dtype='float32', 
    #         mode='w+', 
    #         shape=shape
    #     )
        
    #     for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
    #         start, end = batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_samples)
            
    #         for i in range(start, end):
    #             scalo, phaso = self.process_12_lead_ecg(X[i])
    #             scalograms[i] = scalo
    #             phasograms[i] = phaso
            
    #         scalograms.flush()
    #         phasograms.flush()
        
    #     # ✅ CRITICAL: Delete memmap objects before they're garbage collected
    #     # This ensures data is written properly
    #     del scalograms
    #     del phasograms
        
    #     print(f"✅ Saved scalograms → {output_scalo_path}")
    #     print(f"✅ Saved phasograms → {output_phaso_path}")
        
    #     # ✅ VERIFY the files can be loaded
    #     print(f"  Verifying files...")
    #     test_scalo = np.load(output_scalo_path, mmap_mode='r')
    #     test_phaso = np.load(output_phaso_path, mmap_mode='r')
    #     print(f"  ✓ Scalograms: {test_scalo.shape}, {test_scalo.dtype}")
    #     print(f"  ✓ Phasograms: {test_phaso.shape}, {test_phaso.dtype}")
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
        wavelet='cmor2.0-1.0'
    )
    
    # Process training set
    print("\n[3/4] Processing TRAINING set...")
    X_train = np.load(os.path.join(PROCESSED_PATH, 'train_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_train,
        output_scalo_path=os.path.join(WAVELETS_PATH, 'train_scalograms.npy'),
        output_phaso_path=os.path.join(WAVELETS_PATH, 'train_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_train  # Free memory
    
    # Process validation set
    print("\n[3/4] Processing VALIDATION set...")
    X_val = np.load(os.path.join(PROCESSED_PATH, 'val_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_val,
        output_scalo_path=os.path.join(WAVELETS_PATH, 'val_scalograms.npy'),
        output_phaso_path=os.path.join(WAVELETS_PATH, 'val_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_val
    
    # Process test set
    print("\n[4/4] Processing TEST set...")
    X_test = np.load(os.path.join(PROCESSED_PATH, 'test_standardized.npy'), mmap_mode='r')
    
    cwt_gen.process_dataset_batched(
        X_test,
        output_scalo_path=os.path.join(WAVELETS_PATH, 'test_scalograms.npy'),
        output_phaso_path=os.path.join(WAVELETS_PATH, 'test_phasograms.npy'),
        batch_size=BATCH_SIZE
    )
    
    del X_test
    
    print("\n" + "="*80)
    print("STEP 2 COMPLETE!")
    print("="*80)
    print(f"\nAll CWT representations saved to: {WAVELETS_PATH}")
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