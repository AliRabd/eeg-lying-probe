"""Run the full pipeline:  python main.py --data "Final Data" """
import argparse
from glob import glob
import os
import numpy as np

from src.signal_processing import read_data
from src.feature_extraction import con_features
from src.classification import run_svm, run_lda, run_dbn

p = argparse.ArgumentParser()
p.add_argument("--data", default="Final Data", help="folder with 'Lying probe' and 'Innocent probe'")
args = p.parse_args()

lying = sorted(glob(os.path.join(args.data, "Lying probe", "*.set")))
innocent = sorted(glob(os.path.join(args.data, "Innocent probe", "*.set")))
print(len(lying), "lying /", len(innocent), "innocent files")

data = np.vstack([read_data(f) for f in lying + innocent])
y = np.array([1] * len(lying) + [0] * len(innocent))

X = con_features(data).T
print("Feature matrix:", X.shape)

run_svm(X, y)
run_lda(X, y)
run_dbn(X, y)
