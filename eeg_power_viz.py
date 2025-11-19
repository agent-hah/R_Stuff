import mne
import matplotlib.pyplot as plt
import numpy as np
import os

def create_eeg_power_viz():
    # 1. Load Sample Data
    print("Loading sample data...")
    data_path = mne.datasets.sample.data_path()
    raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
    
    # Load raw data, pick EEG channels only to keep it clean
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(meg=False, eeg=True, eog=False, exclude='bads')
    
    # Crop to a shorter duration for better visualization (e.g., 60 seconds)
    raw.crop(tmax=60)

    # 2. Preprocessing & PSD
    # Compute PSD on raw data (before filtering) to see the frequency spectrum
    print("Generating PSD plot...")
    fig_psd = raw.compute_psd(fmax=50).plot(show=False)
    fig_psd.savefig('eeg_psd.png')
    print("PSD plot saved to eeg_psd.png")

    # Filter to a specific band (e.g., Alpha 8-12 Hz)
    print("Filtering data (8-12 Hz)...")
    raw_alpha = raw.copy().filter(8, 12, n_jobs=1, verbose=False)

    # 3. Topomap
    # Compute average power in Alpha band for each channel
    print("Generating Topomap...")
    # Get data from the filtered object
    alpha_data = raw_alpha.get_data()
    # RMS amplitude per channel
    alpha_rms = np.sqrt(np.mean(alpha_data**2, axis=1))
    
    fig_topo, ax_topo = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(alpha_rms, raw.info, axes=ax_topo, show=False, cmap='viridis')
    ax_topo.set_title('Alpha Band Topomap (8-12 Hz)')
    fig_topo.savefig('eeg_alpha_topomap.png')
    print("Topomap saved to eeg_alpha_topomap.png")

    # 3b. High Gamma Topomap (70-100 Hz)
    print("Filtering data for High Gamma (70-100 Hz)...")
    # Note: We filter from the original 'raw' object again
    raw_gamma = raw.copy().filter(70, 100, n_jobs=1, verbose=False)
    
    print("Generating High Gamma Topomap...")
    gamma_data = raw_gamma.get_data()
    gamma_rms = np.sqrt(np.mean(gamma_data**2, axis=1))
    
    fig_topo_g, ax_topo_g = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(gamma_rms, raw.info, axes=ax_topo_g, show=False, cmap='viridis')
    ax_topo_g.set_title('High Gamma Band Topomap (70-100 Hz)')
    fig_topo_g.savefig('eeg_high_gamma_topomap.png')
    print("High Gamma Topomap saved to eeg_high_gamma_topomap.png")

    # 4. Compute Power Envelope (Channel vs Time)
    # We use the Hilbert transform to get the analytic signal
    print("Computing Hilbert transform...")
    raw_alpha.apply_hilbert(envelope=True)
    
    # The data is now the envelope (amplitude). Power is amplitude squared.
    data = raw_alpha.get_data()
    power_data = data ** 2
    
    # 5. Visualization (Channel vs Time)
    print("Generating Channel vs Time plot...")
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # We use imshow to create the heatmap
    im = ax.imshow(power_data, aspect='auto', origin='lower', 
                   extent=[raw.times[0], raw.times[-1], 0, len(raw.ch_names)],
                   cmap='viridis', vmin=0, vmax=np.percentile(power_data, 95))
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels')
    ax.set_title('EEG Alpha Band Power (8-12 Hz) - Channels vs Time')
    
    ax.set_yticks(np.arange(len(raw.ch_names)))
    ax.set_yticklabels(raw.ch_names, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power ($\mu V^2$)')
    
    plt.tight_layout()
    
    output_file = 'eeg_power_channels_vs_time.png'
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    create_eeg_power_viz()
