import gc
import os
import sys
import logging
import subprocess
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_predict
from torch.utils.data import DataLoader, TensorDataset

from pyDeepInsight import ImageTransformer
from models import KAN, MLP, CNN_1D, CNN, VisionTransformer
from utils import feature_selection, channel_expansion, neighbor_informed_gene_expression


# Configure the root logger to display INFO messages
logger = logging.getLogger("")
logger.setLevel(logging.INFO)


def log_to_file(msg, file_path):

    # Log a message to both the console and a specified file
    file_handler = logging.FileHandler(file_path, mode="a")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(msg)

    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)

    file_handler.close()


def suppress_output():
    sys.stdout = open(os.devnull, 'w')   # Temporarily suppress standard output   


def restore_output():
    sys.stdout = sys.__stdout__          # Restore standard output


def training_loop(n_epochs, optimizer, model, criterion, train_loader, device):
    
    losses = []       # List of loss values 
    
    model.train()     # Model to training mode

    for epoch in range(n_epochs):
        
        loss_train = 0.0

        for features, labels in train_loader:
            
            # Move the features and labels to the GPU
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Training loss for each minibatch
            loss_train += loss.item()

        losses.append(loss_train / len(train_loader))

    return losses


def testing_loop(model, criterion, test_loader, device):
   
    model.eval()             # Model to evaluation mode

    logits_list = []

    with torch.no_grad():    # Disable gradient computation
        correct = 0
        total = 0
        
        test_loss = 0.0

        for features, labels in test_loader: 
            
            # Move the features and labels to the GPU
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Testing loss and logits for each minibatch 
            test_loss += loss.item()
            logits_list.append(outputs.cpu().numpy())

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total

    return test_loss, accuracy, np.vstack(logits_list)


def sem(accuracy_eval, std_errors):
    std_dev = np.std(accuracy_eval, ddof=1) / 100          # Compute the sample standard deviation of the accuracies (normalized by dividing by 100)
    sem = std_dev / np.sqrt(len(accuracy_eval))            # Compute the standard error of the mean (SEM) using the normalized standard deviation
    std_errors.append(sem)                                 # Append the computed SEM to the list of standard errors


def init_dicts(models, dicts):
    for model_name in models:   # For each model, create an empty entry in each dictionary
        for d in dicts:
            d[model_name] = []


def softmax(logits):
    e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))   # Apply the softmax function row-wise to convert scores to probabilities
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def process_fold(train_index, test_index, features, labels, device):

    # Split the data into training and test sets
    X_train = features[train_index, :]       # Training features
    X_test = features[test_index, :]         # Test features
    y_train = labels[train_index]            # Training labels
    y_test = labels[test_index]              # Test label

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler on the training features and apply the transformation
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same transformation to the test features (without refitting)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    X_train_scaled_chi2, X_test_scaled_chi2, selected_feature_indices_chi2 = feature_selection(X_train_scaled, X_test_scaled, y_train, device, method='chi2', max_features=10000)
    X_train_scaled, X_test_scaled, selected_feature_indices_rff = feature_selection(X_train_scaled_chi2, X_test_scaled_chi2, y_train, device, method='RfF', max_features=30)

    # Combine selected feature indices from both methods
    selected_feature_indices = selected_feature_indices_chi2[selected_feature_indices_rff]

    # Original DeepInsight
    tsne = TSNE(n_components=2, perplexity=8, metric='cosine', random_state=1701)   # t-SNE for dimensionality reduction
    it = ImageTransformer(feature_extractor=tsne, pixels=8)                         # Initialize Improved DeepInsight with t-SNE as the feature extractor and set image size to 64 pixels
    _ = it.fit(X_train_scaled, plot=False)                                          # Fit the ImageTransformer to the training data

    return X_train_scaled, X_test_scaled, y_train, y_test, it, selected_feature_indices 

        
def evaluate_models_cv(models, features, labels, num_classes, dataset_name, evaluation_type, device, logits_active, save_selected_features, abs_path):

    # Initialize dictionaries to store per-model metrics
    loss_eval = {}               
    accuracy_eval = {}        
    avg_loss = {}
    avg_accuracy = {}
    sem = {}

    splits = []

    # Initialize all metric dictionaries for the given models
    dicts = [loss_eval, accuracy_eval, avg_loss, avg_accuracy, sem]
    init_dicts(models, dicts)
    
    # Generate data splits based on evaluation type
    if evaluation_type  == "loo":

        loo = LeaveOneOut()         

        splits_dir = os.path.join("data", "splits", "loo")

        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        splits_filename = os.path.join(splits_dir, f'splits_loo_{dataset_name}.npy')

        # Create and save splits if they do not exist
        if not os.path.exists(splits_filename): 
            for train_index, test_index in loo.split(features):
                splits.append((train_index, test_index))

            np.save(splits_filename, np.array(splits, dtype=object))

        splits_array = np.load(splits_filename, allow_pickle=True)

    elif evaluation_type  == "kfold":

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        splits_dir = os.path.join("data", "splits", "kfold")

        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        splits_filename = os.path.join(splits_dir, f'splits_kfold_{dataset_name}.npy')

        # Create and save splits if they do not exist
        if not os.path.exists(splits_filename): 

            for train_index, test_index in kfold.split(features, labels):

                splits.append((train_index, test_index))

            np.save(splits_filename, np.array(splits, dtype=object))

        splits_array = np.load(splits_filename, allow_pickle=True)

    else:
        # Raise error if evaluation type is not supported
        raise ValueError(f"Type {evaluation_type} is not supported")

    # Create directories to store logits, including a timestamped subfolder
    if logits_active:
        logits_dir = os.path.join(abs_path, "logits")
        os.makedirs(logits_dir, exist_ok=True)
    
    # Create directory to save selected features if feature selection is used
    if save_selected_features:
        features_dir = os.path.join(abs_path, "selected_features")
        os.makedirs(features_dir, exist_ok=True)

    # Process each fold in parallel using joblib
    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(train_idx, test_idx, features, labels, device) 
        for train_idx, test_idx in splits_array
    )

    for model_name in models:

        log_to_file(f"\nModel: {model_name}", acc_path)

        # Parameters
        if model_name in ["KAN", "MLP", "SVM"]:
            batch_size = 8
            num_epochs = 100
            learningRate = 0.0001

        elif model_name in ["CNN-1D"]:
            batch_size = 8
            num_epochs = 100
            learningRate = 0.0001

        elif model_name in ["DI-CNN+", "ViT"]:
            batch_size = 30
            num_epochs = 100
            learningRate = 0.0001

        elif model_name == "IT-KAN":
            alpha = 1.0
            beta = 1.0
            sigma = 1.0
            k = 16

            batch_size = 8
            num_epochs = 100
            learningRate = 0.0001

        else:
            raise ValueError(f"Model '{model_name}' is not supported")

        for fold in range(len(splits_array)):
            
            # Define save paths for logits and test labels
            logits_save_path = os.path.join(logits_dir, f"logits_{dataset_name}_{fold + 1}_{model_name}.txt")
            ytest_save_path = os.path.join(logits_dir, f"ytest_{dataset_name}_{fold + 1}.txt")
            features_save_path = os.path.join(features_dir, f"selected_features_{dataset_name}_{fold + 1}.txt")

            # Unpack current fold data
            X_train_scaled, X_test_scaled, y_train, y_test, it, selected_feature_indices = results[fold]
            num_features = X_train_scaled.shape[1]

            # Save test labels if logits logging is enabled
            if logits_active:
                np.savetxt(ytest_save_path, np.array(y_test), fmt='%d')

            # Save selected feature indices if available
            if selected_feature_indices is not None and save_selected_features:
                np.savetxt(features_save_path, np.array(selected_feature_indices), fmt='%d')

            if model_name in ["DI-CNN+", "ViT", "IT-KAN"]:

                # Improved DeepInsight with channel wise expansion (apply the channel expansion function to both the training and test sets)
                if model_name in ["DI-CNN+", "ViT"]:
                    X_train_scaled = channel_expansion(it, X_train_scaled)       # img_train
                    X_test_scaled = channel_expansion(it, X_test_scaled)         # img_test

                if model_name == "IT-KAN":
                    X_train_informed, y_train_tensor = neighbor_informed_gene_expression(X_train_scaled, it.coords(), y_train, alpha=alpha, beta=beta, k=k, sigma=sigma)
                    X_test_informed, y_test_tensor = neighbor_informed_gene_expression(X_test_scaled, it.coords(), y_test, alpha=alpha, beta=beta, k=k, sigma=sigma)

                    # Normalize features to [0, 1]
                    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
                    X_train_informed_scaled = minmax_scaler.fit_transform(X_train_informed.cpu())
                    X_test_informed_scaled = minmax_scaler.transform(X_test_informed.cpu())
                    
                    # Convert scaled data back to PyTorch tensors
                    X_train_informed_scaled = torch.tensor(X_train_informed_scaled, dtype=torch.float32, device=device)
                    X_test_informed_scaled  = torch.tensor(X_test_informed_scaled, dtype=torch.float32, device=device)

                    # Build TensorDatasets for training and testing
                    train_dataset = TensorDataset(X_train_informed_scaled, y_train_tensor)
                    test_dataset  = TensorDataset(X_test_informed_scaled, y_test_tensor)

            if model_name != "IT-KAN":

                # Converting to PyTorch Tensors
                X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

                if model_name == "ViT":

                    # Rearrange the dimensions of the tensors: (batch_size, height, width, channels) to (batch_size, channels, height, width)
                    X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)  
                    X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)

                # Creating DataSets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            # Creating DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            if model_name == "SVM":

                # Train SVM model with probability estimates
                prob = svm_problem(y_train, X_train_scaled)
                param = svm_parameter('-t 0 -q -b 1')
                svm_model = svm_train(prob, param)

                # Suppress output during prediction
                suppress_output()
                
                # Predict with SVM and get probabilities (logits)
                y_pred_svm, _, logits_svm = svm_predict(y_test, X_test_scaled, svm_model, '-b 1')

                # Restore normal output
                restore_output()

                # Ensure logits are ordered as [0,1]
                logits_svm = np.array(logits_svm)
                labels = np.array(svm_model.get_labels())
                logits_fixed = logits_svm[:, [np.where(labels == 0)[0][0], np.where(labels == 1)[0][0]]]
                
                # Save logits if logging is enabled
                if logits_active:
                    np.savetxt(logits_save_path, logits_fixed, fmt='%.6f')

                # Compute accuracy
                correct = (y_pred_svm == y_test).sum()  
                accuracy = (correct / len(y_test)) * 100  
                accuracy_eval[model_name].append(accuracy)

                # Compute hinge loss
                hinge_loss = sum(max(0, 1 - y * y_pred) for y, y_pred in zip(y_test, y_pred_svm)) / len(y_test)
                loss_eval[model_name].append(hinge_loss)

                # Log test accuracy for current fold
                log_to_file(f"Test accuracy {evaluation_type} ({fold + 1}/{len(splits_array)}): {accuracy:.2f}%", acc_path)

            else:
                
                # Load model
                if model_name == "KAN":
                    model = KAN(layers_hidden=[num_features, 100, num_classes], grid_size=9, spline_order=3)
                
                elif model_name == "MLP":
                    model = MLP(input_dim=num_features, hidden_dim=100, output_dim=num_classes)
                    
                elif model_name == "CNN-1D":
                    model = CNN_1D(input_dim=num_features, num_classes=num_classes)

                elif model_name == "DI-CNN+":
                    model = CNN(in_channels=X_train_scaled.shape[1], num_classes=num_classes)   # DeepInsight-CNN (multi-channel) [DI-CNN+]

                elif model_name == "ViT":
                    model = VisionTransformer(patch_size=8,                                # Size of each patch (patch_size x patch_size) that the image is split into
                                                image_size=64,                             # Size of the input images (image_size x image_size pixels)
                                                C=X_train_tensor.shape[1],                 # Number of input channels
                                                num_layers=4,                              # Number of transformer layers in the model
                                                embedding_dim=2096,                        # Dimensionality of the embedding space for each patch
                                                num_heads=8,                               # Number of attention heads in the multi-head attention mechanism
                                                hidden_dim=2096,                           # Dimension of the hidden layer
                                                dropout_prob=0.1,                          # Probability of dropout to prevent overfitting
                                                num_classes=num_classes)                   # Number of classes
                
                elif model_name == "IT-KAN":

                    model = KAN(layers_hidden=[num_features, 100, num_classes], grid_size=9, spline_order=3)

                else:
                    raise ValueError(f"Model '{model_name}' is not supported")
        
                # Move the model to the GPU if available
                model.to(device)
        
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                    
                if model_name in ["KAN", "MLP", "CNN-1D", "IT-KAN"]:
                    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)

                elif model_name in ["DI-CNN+", "ViT"]:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01)

                else:
                    raise ValueError(f"Model '{model_name}' is not supported")
                
                # Training model
                _ = training_loop(num_epochs, optimizer, model, criterion, train_loader, device)
                
                # Testing model
                test_loss, accuracy, logits_nn = testing_loop(model, criterion, test_loader, device)
                loss_eval[model_name].append(test_loss)
                accuracy_eval[model_name].append(accuracy)

                # Save logits if logging is enabled
                if logits_active:
                    np.savetxt(logits_save_path, np.array(logits_nn), fmt='%.6f')

                # Log test accuracy for the current fold
                log_to_file(f"Test accuracy {evaluation_type} ({fold + 1}/{len(splits_array)}): {accuracy:.2f}%", acc_path)

                # Delete the model, training data loader and optimizer to free up memory
                del model
                del train_loader
                del optimizer

                # Perform garbage collection to clean up any unused memory
                gc.collect()

                # Clear the GPU cache to free up memory on the GPU
                torch.cuda.empty_cache()

        # Compute average accuracy for the model
        avg_accuracy[model_name] = sum(accuracy_eval[model_name]) / len(accuracy_eval[model_name])

        # Compute standard error of the mean (SEM) for accuracy
        std_dev = np.std(accuracy_eval[model_name], ddof=1) / 100                                 
        sem[model_name] = std_dev / np.sqrt(len(accuracy_eval[model_name]))

        # Compute average loss for the model                   
        avg_loss[model_name] = sum(loss_eval[model_name]) / len(loss_eval[model_name])

    return avg_loss, avg_accuracy, sem

  

if __name__ == "__main__":

    folder = os.path.join("data", "processed")
    
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

    # List to store the accuracy values for different datasets
    accuracies = {}

    # List to store the standard errors of the mean of accuracies for each dataset
    std_errors = {}

    # Setup for k-fold, models, and output options
    evaluation_type  = "kfold"
    #models = ["IT-KAN"]
    models = ["KAN", "MLP", "SVM", "CNN-1D", "DI-CNN+", "IT-KAN"]
    logits_active = True
    save_selected_features = True
    metrics = True
    chart = True
    
    # Generate a timestamp to make log filenames unique
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Base directory for all results
    results_dir = os.path.join("results", f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    abs_path = os.path.abspath(results_dir)

    # Directory to store model accuracy logs
    acc_path = os.path.join(abs_path, f"accuracies.log") 

    # Define the path for the metrics log
    metrics_path = os.path.join(abs_path, f"metrics.log")

    # Define the path for saving chart results
    charts_dir = os.path.join(abs_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Define the path for saving logits outputs (model predictions before softmax)
    logits_dir = os.path.join("results", f"results_{timestamp}", "logits")

    # If the processed data folder doesn't exist, run the preprocessing script
    processed_folder = os.path.join("data", "processed")
    if not os.path.exists(processed_folder):
        subprocess.run(["python", os.path.join("scripts", "datasets_trasp_mod.py")])

    # Set up CPU usage: detect all cores and configure parallel processing limits
    num_cores = multiprocessing.cpu_count()
    os.environ["LOKY_MAX_CPU_COUNT"] = str(num_cores)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device, "\n")

    log_to_file(f"\nMODEL RESULTS", acc_path)

    for file_path, cancer_type in file_paths.items():

        # Log dataset name and define dataset path
        log_to_file(f"\n--- Results for: {cancer_type} ---", acc_path)
        dataset_path = os.path.join(folder, file_path)

        # Load dataset
        df = pd.read_csv(dataset_path, header=None)

        # Separate labels and features
        labels = df.iloc[:, 0]
        features = df.drop(columns=0)                           
        features.columns = range(features.shape[1])

        # Count class distribution
        cancer_count = (labels == 1).sum()
        normal_count = (labels == 0).sum()

        # Log class distribution for the dataset
        log_to_file(f"\nCancer: {cancer_count}", acc_path)
        log_to_file(f"Normal: {normal_count}\n", acc_path)
        
        # Convert to NumPy arrays
        labels = np.array(labels, dtype=np.int32)                      
        features = np.array(features, dtype=np.float32)                

        # Log shapes of features and labels
        log_to_file(f"Shape of features: {features.shape}", acc_path) 
        log_to_file(f"Shape of labels: {labels.shape}", acc_path)

        # Get data properties
        num_samples = features.shape[0]                                 # Number of samples (rows)
        num_features = features.shape[1]                                # Number of features (columns)
        num_classes = len(torch.unique(torch.tensor(labels)))           # Number of unique classes in the labels

        # Evaluate Models with Leave-One-Out Cross-Validation
        avg_loss, avg_accuracy, sem = evaluate_models_cv(models, features, labels, num_classes, cancer_type, evaluation_type, device, logits_active, save_selected_features, abs_path)

        # Store results
        for model_name in models:
            if model_name not in accuracies:
                accuracies[model_name] = []
            if model_name not in std_errors:
                std_errors[model_name] = []

            accuracies[model_name].append(avg_accuracy[model_name] / 100) 
            std_errors[model_name].append(sem[model_name])

        # Log final statistics
        log_to_file("\nFinal Test Results", acc_path)
        for model_name in models:
            if model_name == "SVM":
                log_to_file(f'{model_name} - Average Hinge Loss: {avg_loss[model_name]:.3f}, Accuracy: {avg_accuracy[model_name]:.2f}%', acc_path)
            else:
                log_to_file(f'{model_name} - Average Loss: {avg_loss[model_name]:.3f}, Accuracy: {avg_accuracy[model_name]:.2f}%', acc_path)
        log_to_file("", acc_path)
    
    print(f"The accuracy results have been saved as 'accuracies.log' in: {abs_path}")

    
    # ---- Metrics part ----

    if metrics:
        
        for file_path, cancer_type in file_paths.items():

            log_to_file(f"\n--- Results for: {cancer_type} ---", metrics_path)
            
            fold_files = [f for f in os.listdir(logits_dir) if f.startswith(f"ytest_{cancer_type}") and f.endswith(".txt")]
            num_folds = len(fold_files)

            for model_name in models:

                all_y_test = []
                all_y_scores = []

                for fold in range(num_folds):

                    # Load real labels
                    y_test_path = os.path.join(logits_dir, f"ytest_{cancer_type}_{fold + 1}.txt")
                    y_test = np.loadtxt(y_test_path, dtype=int)

                    # Load logits
                    logits_path = os.path.join(logits_dir, f"logits_{cancer_type}_{fold + 1}_{model_name}.txt")
                    if os.path.exists(logits_path):
                        logits = np.loadtxt(logits_path)
                        probs = softmax(logits)
                        y_scores = probs[:, 1]

                        # Save predictions by concatenation
                        all_y_test.append(y_test)
                        all_y_scores.append(y_scores)

                    else:
                        raise ValueError(f"File not found: {logits_path}")

                # Concatenate out-of-fold predictions and calculate overall AUC
                all_y_test_concat = np.concatenate(all_y_test)
                all_y_scores_concat = np.concatenate(all_y_scores)
                mean_auc = roc_auc_score(all_y_test_concat, all_y_scores_concat)

                # Calculate the error under the ROC curve
                error_under_roc = (1 - mean_auc) * 100

                # Log overall AUC (out-of-fold) and corresponding error under ROC
                log_to_file(f"\nModel: {model_name}", metrics_path)
                log_to_file(f"  Overall AUC (out-of-fold): {mean_auc:.3f}", metrics_path)
                log_to_file(f"  Error under ROC (%):       {error_under_roc:.3f}\n", metrics_path)

                # Plot ROC curve using out-of-fold predictions (overall ROC)
                fpr, tpr, _ = roc_curve(all_y_test_concat, all_y_scores_concat)
                plt.plot(fpr, tpr, color='b', label=f'ROC (AUC = {mean_auc:.3f})')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.title(f'ROC Curve - {model_name} ({cancer_type})')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.grid(True)
                output_path = os.path.join(charts_dir, f"roc_curve_{cancer_type}_{model_name}.png")
                plt.savefig(output_path, dpi=1200)
                plt.close()

        print(f"The metrics have been saved as 'metrics.log' in: {abs_path}")
        print(f"ROC curves have been saved in: {charts_dir}")
    

    # ---- Chart part ----

    if chart:

        # Create a bar chart of model accuracies across cancer types
        labels = list(file_paths.values())
        x = np.arange(len(labels))
        width = 0.15
        spacing = 0.05

        #models = ["IT-KAN"]
        models = ["IT-KAN", "DI-CNN+", "KAN", "MLP", "SVM", "CNN-1D"]
        colors = ["#36BFC0", "#40E0D0", "#00BFFF", "#0077FF", "#0047AB", "#003366"]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Add horizontal grid lines for reference
        for y in [0.6, 0.7, 0.8, 0.9, 1.0]:
            ax.axhline(y=y, color='#dfdfdf', linestyle='-', linewidth=1, zorder=0)

        n_models = len(models)
        center_offset = (n_models - 1) / 2  
        rects_list = []

        # Plot bars for each model
        for i, model in enumerate(models):
            xpos = x + (i - center_offset) * width
            rects = ax.bar(
                xpos, 
                accuracies[model], 
                width, 
                yerr=std_errors[model], 
                capsize=3, 
                label=model, 
                color=colors[i], 
                ecolor='#727272', 
                zorder=2
            )
            rects_list.append(rects)
        
        # Customize chart appearance
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.6, 1)
        ax.set_title('Model Performance Across Cancer Types (SEM)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=5)
        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()

        # Save chart to file
        output_path = os.path.join(charts_dir, "bar_chart.png")
        plt.savefig(output_path, dpi=1200)
        plt.close()

        print(f"The chart has been saved as 'bar_chart.png' in: {charts_dir}")

