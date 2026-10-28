"""
Configuration file for the ADNI AD-vs-HC / Unsupervised-HC project.
Edit DATA_DIR and the filenames below to match what your professor sent you.
"""
from pathlib import Path

# ---- Root data directory (edit this to your local folder) ----
DATA_DIR = Path("data")

# ---- Subfolders holding the .nii / .nii.gz files ----
AD_MRI_DIR = DATA_DIR / "AD_MRI"
HC_MRI_DIR = DATA_DIR / "HC_MRI"
HC_PET_DIR = DATA_DIR / "HC_PET"

# ---- Mask / atlas files (from Prof. Presotto's email) ----
MASK_PATH = DATA_DIR / "mask_ICV.nii"
RMASK_PATH = DATA_DIR / "rmask_ICV.nii"      # resampled ICV mask, matched to MRI space
AAL_PATH = DATA_DIR / "aal.nii"              # AAL brain parcellation (standard space)

# ---- Output directories (created automatically by main.py) ----
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Task B (classification) settings ----
N_CV_FOLDS = 5
TEST_SIZE = 0.2
