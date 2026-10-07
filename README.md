<img width="843" height="672" alt="image" src="https://github.com/user-attachments/assets/f5a14f79-98ca-4cb2-8864-df6aea8c0634" /># EEG Lying-Probe Detection: Feature Extraction and Machine Learning

A BCI-style pipeline that classifies EEG responses to **lying probe** vs **innocent probe** stimuli.

![General framework](images/General_framework.png)

## Pipeline
| Stage | File | What it does |
|---|---|---|
| Signal acquisition | `src/signal_processing.py` | Loads EEGLAB `.set` files (12 EEG + 2 EOG channels, 10-20 montage) |
| Signal processing | `src/signal_processing.py` | 1 Hz high-pass, ICA (Picard) to remove EOG artifacts, 1.6 s epochs (reject > 150 µV), regression-based baseline correction at Pz |
| Feature extraction | `src/feature_extraction.py` | Morlet time-frequency (delta band) combined with the 1.0-1.6 s time segment; features: positive area, mean PSD amplitude, maximum |
| Classification | `src/classification.py` | SVM (RBF), LDA, and an RBM-based DBN pipeline |

## Dataset
71 lying-probe and 365 innocent-probe recordings (436 total). **The EEG data is not included** in this repository
(participant privacy / size). Place it locally as:
```
Final Data/Lying probe/*.set
Final Data/Innocent probe/*.set
```
Add the dataset source / citation here.

## Usage
```bash
pip install -r requirements.txt
python main.py --data "Final Data"
```

## Results (as obtained in the original notebook)
| Model | Accuracy |
|---|---|
| SVM, 5-fold CV | 83.7% |
| LDA, hold-out (80/20) | 85.2% |
| LDA, 5-fold CV | 83.7% |
| DBN, hold-out | 85.2% |

**Important:** the dataset is imbalanced (365/436 = 83.7% innocent). These accuracies equal the
majority-class baseline: the SVM and LDA predict *innocent* for nearly every sample (lying-class precision / recall / F1 = 0).
Addressing this (class weighting, resampling, balanced accuracy / F1 / ROC-AUC, better features) is future work.

## Known limitations
- Features are computed on a single channel (Pz) and one frequency band.
- `extract_positive_area_voltage` clips negative values in place, which affects the other features computed afterwards.
- Cross-validation is not grouped by subject, so recordings from the same person may appear in both train and test.
## Feature curve before SVM classification
![Featurescurve]<img width="907" height="676" alt="Features curve" src="https://github.com/user-attachments/assets/767dd7a9-7cb6-4ff7-a80c-27df4570f4a4" />
