"""Step 1-2: Signal acquisition (loading) and signal processing (noise removal)."""
import mne
import numpy as np

CH_TYPES = {f"EEG {i:03d}": "eeg" for i in range(12)}
CH_TYPES.update({"EEG 012": "eog", "EEG 013": "eog"})
CH_NAMES = dict(zip(
    [f"EEG {i:03d}" for i in range(14)],
    ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz", "VEOG", "HEOG"],
))


def read_data(file_path, show_plots=False):
    """Load one EEGLAB .set file, remove eye artifacts with ICA, epoch it,
    apply regression-based baseline correction at Pz and return a
    time-frequency feature array of shape (1, 801, 301)."""
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    raw.set_channel_types(CH_TYPES, verbose=None)
    raw.rename_channels(CH_NAMES)
    raw.set_montage("standard_1020")
    if show_plots:
        raw.plot()

    # --- ICA artifact removal (EOG components) ---
    raw.filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=12, method="picard", random_state=97, max_iter=800)
    ica.fit(raw)
    if show_plots:
        ica.plot_components()
    eog_exclude = ica.find_bads_eog(raw, ch_name=["HEOG", "VEOG"], threshold=0.9)
    ica.exclude = eog_exclude[0]
    ica.apply(raw)

    # --- Epoching: fixed-length 1.6 s epochs, reject > 150 uV ---
    events = mne.make_fixed_length_events(raw, id=1, start=0, stop=None, duration=1.6,
                                          first_samp=True, overlap=0)
    epochs = mne.Epochs(raw.copy(), events, tmin=0.0, tmax=1.6, baseline=None,
                        reject=dict(eeg=150e-6), flat=None, reject_by_annotation=True)

    # --- Regression-based baseline correction (baseline window 0-1 s, channel Pz) ---
    baseline = (0.0, 1.0)
    predictor = epochs.events[:, 2] == epochs.event_id["1"]
    baseline_predictor = (epochs.load_data().copy().crop(*baseline)
                          .pick_channels(["Pz"]).get_data().mean(axis=-1).squeeze()) * 1e6
    design_matrix = np.vstack([predictor, baseline_predictor]).T
    reg_model = mne.stats.linear_regression(epochs, design_matrix,
                                            names=["predictor", "baseline_predictor"])
    reg_predictor = reg_model["predictor"].beta
    reg_predictor.pick_channels(["Pz"])
    if show_plots:
        reg_predictor.plot(picks="Pz")

    # --- Time-domain segment 1.0-1.6 s (301 samples) ---
    t = reg_predictor.times
    start, end = np.where(t >= 1.0)[0][0], np.where(t >= 1.6)[0][0]
    time_array = np.array(reg_predictor.data[:, start:end + 1])

    # --- Time-frequency (Morlet wavelet), delta-band power ---
    freqs = np.logspace(1, 30, num=801, base=1.1)
    tfr = mne.time_frequency.tfr_morlet(reg_predictor, freqs=freqs, n_cycles=freqs / 2,
                                        return_itc=False, average=True)
    delta = (0.5, 4)
    power = tfr.data[:, :, (tfr.freqs >= delta[0]) & (tfr.freqs <= delta[1])].mean(axis=2)

    # Combine -> (trial, frequency, time) = (1, 801, 301)
    return power.reshape(1, 801, 1) * time_array.reshape(1, 1, 301)
