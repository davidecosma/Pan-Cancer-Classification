import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.special import softmax


def log_to_file(message, file_path):

    with open(file_path, "a") as f:
        f.write(message + "\n")
    print(message)


def fuse_models(logits_dir, datasets, models, fusion_log_path, charts_dir=None):

    models_str = " + ".join(models)

    log_to_file(f"\nFUSION RESULTS ({models_str})", fusion_log_path)

    for file_path, dataset_name in datasets.items():
        log_to_file(f"\n--- Results for: {dataset_name} ---\n", fusion_log_path)

        # Identify fold files
        fold_files = [f for f in os.listdir(logits_dir) 
                      if f.startswith(f"ytest_{dataset_name}") and f.endswith(".txt")]
        num_folds = len(fold_files)

        evaluation_type = "kfold" if num_folds == 10 else "loo"
        fold_accuracies = []

        # For AUC (out-of-fold concatenation)
        all_y_test = []
        all_y_scores = []

        for fold in range(num_folds):
            logits_all_models = {}

            # Load ground truth labels for this fold
            y_test_path = os.path.join(logits_dir, f"ytest_{dataset_name}_{fold + 1}.txt")
            if not os.path.exists(y_test_path):
                raise ValueError(f"File not found: {y_test_path}")
            y_test = np.loadtxt(y_test_path, dtype=int)

            # Load logits for each model
            for model_name in models:
                logits_path = os.path.join(logits_dir, f"logits_{dataset_name}_{fold + 1}_{model_name}.txt")
                if os.path.exists(logits_path):
                    logits_all_models[model_name] = np.loadtxt(logits_path)
                else:
                    logging.error(f"File not found: {logits_path}")
                    raise ValueError(f"File not found: {logits_path}.")

            if len(logits_all_models) == 0:
                raise ValueError(f"No logits found for dataset: {dataset_name}, fold: {fold + 1}.")

            # Fuse logits (simple sum)
            fused_logits = np.sum(list(logits_all_models.values()), axis=0)
            fused_preds = np.argmax(fused_logits, axis=1)

            # Compute accuracy
            accuracy = accuracy_score(y_test, fused_preds)
            fold_accuracies.append(accuracy)
            log_to_file(f"Fusion accuracy {evaluation_type} ({fold + 1}/{num_folds}): {accuracy * 100:.2f}%", fusion_log_path)

            # Store probabilities for AUC
            probs = softmax(fused_logits, axis=1)   # apply softmax
            y_scores = probs[:, 1]                  # class 1 scores
            all_y_test.append(y_test)
            all_y_scores.append(y_scores)

        # Calculate mean accuracy and SEM
        avg_accuracy = np.mean(fold_accuracies)
        std_dev = np.std(fold_accuracies, ddof=1)
        sem = std_dev / np.sqrt(len(fold_accuracies))

        log_to_file(f"\nFinal fusion accuracy: {avg_accuracy * 100:.2f}% (SEM: {sem:.4f})\n", fusion_log_path)

        # Concatenate OOF predictions for AUC
        all_y_test_concat = np.concatenate(all_y_test)
        all_y_scores_concat = np.concatenate(all_y_scores)
        mean_auc = roc_auc_score(all_y_test_concat, all_y_scores_concat)
        error_under_roc = (1 - mean_auc) * 100

        log_to_file(f"Fusion overall AUC (out-of-fold): {mean_auc:.3f}", fusion_log_path)
        log_to_file(f"Fusion error under ROC (%):       {error_under_roc:.3f}\n", fusion_log_path)

        # Plot ROC curve if charts_dir is provided
        if charts_dir:
            os.makedirs(charts_dir, exist_ok=True)
            fpr, tpr, _ = roc_curve(all_y_test_concat, all_y_scores_concat)
            plt.plot(fpr, tpr, color='b', label=f'ROC (AUC = {mean_auc:.3f})')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title(f'ROC Curve - Fusion ({dataset_name})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid(True)
            output_path = os.path.join(charts_dir, f"roc_curve_fusion_{dataset_name}.png")
            plt.savefig(output_path, dpi=1200)
            plt.close()


if __name__ == "__main__":

    # Define directories
    timestamp = "2025-09-14_12-33-54"
    base_results_dir = os.path.join("results", f"results_{timestamp}")
    logits_dir = os.path.join(base_results_dir, "logits")

    # Create fusions folder if not exists
    fusions_dir = os.path.join(base_results_dir, "fusions")
    os.makedirs(fusions_dir, exist_ok=True)

    # Define models to fuse
    models = ["SVM", "IT-KAN"]

    # Build filename automatically from models list
    models_str = "_".join(models)
    fusion_log_path = os.path.join(fusions_dir, f"fusion_{models_str}.log")

    # Create or clear the fusion log file
    charts_dir = os.path.join(fusions_dir, f"fusion_{models_str}_charts")
    os.makedirs(charts_dir, exist_ok=True)

    file_paths = {
        r"GSE13507_trasp_mod.csv"      : "Bladder Urothelial Carcinoma",
        r"GSE39004_trasp_mod.csv"      : "Breast invasive carcinoma cancer",
        r"breast_matrix_trasp_mod.csv" : "Breast cancer",
        r"GSE41657_trasp_mod.csv"      : "Colon adenocarcinoma",
        r"GSE20347_trasp_mod.csv"      : "Esophageal carcinoma",
        r"GSE6631_trasp_mod.csv"       : "Head and Neck squamous cell carcinoma",
        r"GSE15641_1_trasp_mod.csv"    : "Kidney Chromophobe",
        r"GSE15641_2_trasp_mod.csv"    : "Kidney renal clear cell carcinoma",
        r"GSE15641_3_trasp_mod.csv"    : "Kidney renal papillary cell carcinoma",
        r"GSE45267_trasp_mod.csv"      : "Liver hepatocellular carcinoma",
        r"GSE33479_trasp_mod.csv"      : "Lung squamous cell carcinoma",
        r"GSE10072_trasp_mod.csv"      : "Lung adenocarcinoma",
        r"GSE6919_trasp_mod.csv"       : "Prostate adenocarcinoma",
        r"GSE20842_trasp_mod.csv"      : "Rectum adenocarcinoma",
        r"GSE2685_trasp_mod.csv"       : "Stomach adenocarcinoma",
        r"GSE33630_trasp_mod.csv"      : "Thyroid carcinoma",
        r"GSE17025_trasp_mod.csv"      : "Uterine Corpus Endometrial Carcinoma"
    }

    fuse_models(logits_dir, file_paths, models, fusion_log_path, charts_dir)

