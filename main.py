"""
Main entry point: runs Task A (unsupervised learning on HC) and
Task B (AD vs HC classification) end to end.

Usage (from the project root, with your data folder set up per README.md):
    python main.py
"""
import numpy as np
import nibabel as nib

import config
from src.data_loading import list_nifti_files, extract_subject_id, check_image_consistency
from src.preprocessing import load_volumes, apply_mask, zscore_normalize
from src.feature_extraction import resample_atlas_to_reference, extract_roi_features
from src.task_a_unsupervised import run_pca, cluster_kmeans, stability_check
from src.task_b_classification import build_combined_dataset, run_cross_validation
from src.evaluation import plot_roc_curves, plot_silhouette_scores


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load file lists ----------
    hc_mri_files = list_nifti_files(config.HC_MRI_DIR)
    ad_mri_files = list_nifti_files(config.AD_MRI_DIR)
    print(f"HC MRI files: {len(hc_mri_files)} | AD MRI files: {len(ad_mri_files)}")

    metadata_df, consistency_summary = check_image_consistency({
        "HC MRI": hc_mri_files, "AD MRI": ad_mri_files,
    })
    print(consistency_summary)

    # ---------- 2. Load mask + resample atlas ----------
    rmask_img = nib.load(config.RMASK_PATH)
    rmask_data = rmask_img.get_fdata()

    reference_img = nib.load(hc_mri_files[0])
    atlas_data = resample_atlas_to_reference(config.AAL_PATH, reference_img)
    roi_labels = [int(l) for l in np.unique(atlas_data) if l != 0]
    print(f"Number of ROI labels: {len(roi_labels)}")

    # ---------- 3. Preprocess HC ----------
    hc_volumes = load_volumes(hc_mri_files)
    hc_masked = apply_mask(hc_volumes, rmask_data)
    hc_normalized, hc_zero_std = zscore_normalize(hc_masked)
    if hc_zero_std:
        print(f"WARNING - HC subjects with zero-std intensity (check these scans): {hc_zero_std}")

    # ---------- 4. Preprocess AD ----------
    ad_volumes = load_volumes(ad_mri_files)
    ad_masked = apply_mask(ad_volumes, rmask_data)
    ad_normalized, ad_zero_std = zscore_normalize(ad_masked)
    if ad_zero_std:
        print(f"WARNING - AD subjects with zero-std intensity (check these scans): {ad_zero_std}")

    # ---------- 5. Extract ROI features ----------
    hc_features, hc_missing = extract_roi_features(hc_normalized, atlas_data, roi_labels)
    ad_features, ad_missing = extract_roi_features(ad_normalized, atlas_data, roi_labels)

    hc_subject_ids = [extract_subject_id(f) for f in hc_mri_files]
    ad_subject_ids = [extract_subject_id(f) for f in ad_mri_files]

    # ================= TASK A: Unsupervised learning on HC =================
    print("\n=== TASK A: Unsupervised learning on HC ===")
    reduced_hc, pca_model, hc_scaler = run_pca(hc_features)

    silhouette_df = cluster_kmeans(reduced_hc)
    print(silhouette_df)
    plot_silhouette_scores(silhouette_df, save_path=config.FIGURES_DIR / "silhouette_scores.png")

    best_k = int(silhouette_df.loc[silhouette_df["silhouette"].idxmax(), "k"])
    print(f"Selected k = {best_k} (highest silhouette score)")

    stability = stability_check(reduced_hc, n_clusters=best_k)
    print(f"Cluster stability - mean ARI: {stability['mean_ari']:.3f} (+/- {stability['std_ari']:.3f})")

    # ================= TASK B: AD vs HC classification =================
    print("\n=== TASK B: AD vs HC classification ===")
    combined_df = build_combined_dataset(
        hc_features, ad_features, hc_subject_ids, ad_subject_ids, roi_labels
    )
    combined_df.to_csv(config.RESULTS_DIR / "combined_roi_features.csv", index=False)

    X = combined_df.drop(columns=["Subject", "Label"]).values
    y = combined_df["Label"].values

    # Simple mean-imputation for any missing ROI values before classification
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    results_df, summary, roc_curves = run_cross_validation(X, y, n_splits=config.N_CV_FOLDS)
    print(summary)

    results_df.to_csv(config.RESULTS_DIR / "classification_cv_results.csv", index=False)
    plot_roc_curves(roc_curves, save_path=config.FIGURES_DIR / "roc_curves.png")

    print("\nDone. Results saved in:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
