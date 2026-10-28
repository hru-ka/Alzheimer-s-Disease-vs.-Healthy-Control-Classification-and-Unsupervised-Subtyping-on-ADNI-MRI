# Alzheimer-s-Disease-vs.-Healthy-Control-Classification-and-Unsupervised-Subtyping-on-ADNI-MRI
AD vs. HC classification on ADNI structural MRI using AAL atlas ROI features. Logistic Regression, SVM and MLP with stratified 5-fold CV and class-imbalance handling (106 AD / 346 HC), plus PCA + KMeans clustering of healthy controls with seed-stability (ARI) checks.
This is using AAL ROI features extracted from spatially-normalized ADNI MRI scans.

project partner : ANY DAS

- **Task A** — Unsupervised learning on HC (PCA + clustering + stability checks)
- **Task B** — AD vs HC classification (LogReg / SVM / MLP, cross-validated)

The two tasks are linked through a shared preprocessing + feature-extraction
pipeline (mask → normalize → AAL ROI extraction), and by checking whether
misclassified/borderline HC subjects in Task B fall into the outlier
cluster(s) found in Task A.

## Project structure

adni_project/
├── config.py                    # paths and settings 
├── main.py                      # runs both tasks end to end
├── requirements.txt
├── data/                        # put your data here 
│   ├── AD_MRI/
│   ├── HC_MRI/
│   ├── HC_PET/
│   ├── mask_ICV.nii
│   ├── rmask_ICV.nii
│   └── aal.nii
├── src/
│   ├── data_loading.py          # file listing, subject ID extraction, consistency checks
│   ├── preprocessing.py         # masking + z-score normalization
│   ├── feature_extraction.py    # AAL atlas resampling + ROI feature extraction
│   ├── task_a_unsupervised.py   # PCA + clustering + stability checks (Task A)
│   ├── task_b_classification.py # cross-validated classifiers (Task B)
│   └── evaluation.py            # ROC curve / silhouette plotting
└── outputs/
    ├── figures/                 # saved plots
    └── results/                 # saved CSVs (features, CV results)

## Run

```bash
python main.py
```

This will:
1. Load and sanity-check AD and HC MRI files
2. Apply the ICV mask and normalize each subject's MRI
3. Resample the AAL atlas and extract per-ROI mean-intensity features
4. Run Task A (clustering on HC) and save silhouette + stability results
5. Run Task B (AD vs HC classification) with 5-fold CV and save ROC curves,
   metrics, and the combined feature table to `outputs/`

## Notes / methodology decisions worth mentioning in your report

- **Nearest-neighbor atlas resampling** (`order=0`) is used when aligning the
  AAL atlas to MRI space, to avoid blending integer region labels.
- **Scaling happens inside each CV fold** in Task B, not before — scaling HC
  and AD separately (or on the full dataset before splitting) would either
  erase the between-group signal or leak test information into training.
- **Class imbalance** (106 AD vs 346 HC) is handled via `class_weight="balanced"`
  and by reporting sensitivity/specificity rather than raw accuracy.
- **Stability checks**: Task A reruns KMeans across multiple seeds and reports
  Adjusted Rand Index between runs; Task B uses stratified 5-fold CV and
  reports mean ± std across folds instead of a single train/test split.
