# Pan-Cancer Classification

This repository contains code and tools for pan-cancer data classification using various deep learning architectures, the classical machine learning algorithm SVM, and feature selection techniques. The project includes preprocessing pipelines, feature selection, channel expansion, neural network models, model comparison, and results analysis.


## Project Structure

- `utils/` - Utility modules for channel expansion, feature selection, and neighbor-informed gene expression.
    - `deepinsight_channel_expansion.py` – Channel expansion for DeepInsight image representations.
    - `feature_selection.py` – Feature selection utilities.
    - `neighbor_informed_gene_expression.py` – Neighbor-informed gene expression processing.
- `main.py` – Main script for the entire workflow: data processing, feature selection, channel expansion, model training, evaluation, and results management.
- `models/` – Model implementations (Autoencoder, CNN-1D, CNN, KAN, MLP, ViT).
- `data/` – Raw, processed, and split datasets.
- `results/` – Experiment results, logs, and charts.
- `scripts/` – Support scripts for preprocessing and dataset management.
- `papers/` – Scientific papers and references.


## Requirements

The project was developed and tested with the following package versions (Python 3.13.1):

- PyTorch (2.6.0+cu118, see note below)
- scikit-learn (1.6.1)
- pandas (2.2.3)
- numpy (1.26.4)
- matplotlib (3.10.0)
- joblib (1.4.2)
- libsvm-official (3.35.0)
- pyDeepInsight (0.1.1)
- shap (0.47.1)
- Cython (3.0.12)
- pymrmr (0.1.11)
- skrebate (0.62)
- xgboost (3.0.0)
- scipy (1.15.1)

**Note on PyTorch**:  
Please install the appropriate PyTorch version for your operating system and CUDA setup from the official website:  https://pytorch.org/get-started/locally/


**Recommended hardware**:  
An NVIDIA GPU with CUDA support is strongly recommended to accelerate training and parallelized computations required by the models.


## Quick Installation

Install the required Python packages (except PyTorch) via pip:

```sh
pip3 install \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    matplotlib==3.10.0 \
    joblib==1.4.2 \
    libsvm-official==3.35.0 \
    pyDeepInsight==0.2.0 \
    shap==0.47.1 \
    Cython==3.0.12 \
    pymrmr==0.1.11 \
    skrebate==0.62 \
    xgboost==3.0.0 \
    scipy==1.15.1
```

**pyDeepInsight**: 
The version used in this project (`0.1.1`) is no longer available on PyPI. The latest available version on PyPI is `0.2.0`. You can also install the latest version directly from GitHub:

```sh
python3 -m pip install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
```

**pymrmr**: On macOS, installing pymrmr may fail due to OpenMP incompatibility.


## Test Environment

All tests in this project were conducted on the following setup:

- **Operating System**: Windows 11 Pro 24H2
- **Processor**: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- **Installed RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 4060 with 8 GB VRAM

- **IDEs used**: 
  - Spyder (for dataset exploration and creating transposed matrices via Variable Explorer)  
  - VS Code (for developing, running the project code, and navigating folders/files)



## Usage

1. **Run the Main Workflow**  

    Execute [`main.py`](main.py) from the same directory to perform all steps: data processing, feature selection, channel expansion, model training, and evaluation. Results will be saved in `results/`.

    ```sh
    python3 main.py
    ```

    It is recommended to use a code editor such as VS Code (with Python extensions). Simply open the main project folder, then open `main.py` and run it. You can navigate between folders and files using the VS Code File Explorer, explore outputs, and adjust parameters as needed.

2. **Model Fusion**

    The [`fusion_models.py`](fusion_models.py) script enables the combination of predictions (logits) from multiple models to enhance classification performance. To run it, use the following command:

    ```sh
    python3 fusion_models.py
    ```

    It performs the following steps for each dataset and fold:

    - Loads test labels and logits for selected models.

    - Fuses the logits (sum of logits across models).

    - Computes accuracy and standard error of the mean (SEM).

    - Calculates out-of-fold AUC and error under the ROC curve.

    - Optionally generates ROC curve plots in a specified directory.


3. **Results Analysis**  
   Check log files (`accuracies.log`, `metrics.log`) and charts/visualizations in the `results` folder.
   Fusion-related files and the folder with ROC curve charts are located in the `fusions` subfolder.



## Feature Selection Approach

In this project, the `Chi-2 test` is employed to select the top 10,000 features for downstream analysis. All subsequent experiments are consistently performed on this reduced feature set. Among the approaches tested, the best results are obtained either by using the `Chi-2 selection` alone or by combining `Chi-2` with `Random Forest`, the latter further reducing the feature space from 10,000 to 30 variables. Alternative feature selection methods, including cascading multiple techniques, were also explored; however, these consistently resulted in inferior classification performance. Therefore, the `Chi-2 test`, either alone or in combination with `Random Forest`, was chosen as the most effective feature selection strategy to ensure optimal model accuracy.



## Experiment Results

- **results_2025-08-24_09-36-37**  
  Experiment with 10,000 features selected using Chi-Square.

- **results_2025-09-14_12-33-54**  
  Experiment starting from 10,000 Chi-Square selected features, then reduced to 30 features using Random Forest.



## Overview of `main.py`

`main.py` is the core of the project and manages the entire workflow: from data loading and preprocessing, to feature selection, channel expansion (which uses DeepInsight's ImageTransformer to generate images with multiple channels), to training, evaluation, and saving model results.


### General Structure

#### 1. Imports and Configuration
- Imports all necessary libraries (PyTorch, scikit-learn, pandas, numpy, matplotlib, etc.).
- Configures the logger to save logs both to a file and to the console.

#### 2. Utility Functions
- `log_to_file(msg, file_path)`: logs a message to both the console and a specified file.

- `suppress_output()` / `restore_output()`: temporarily suppresses or restores standard output, useful when running models like SVM to avoid cluttering logs.

- `training_loop(n_epochs, optimizer, model, criterion, train_loader, device)`: trains a PyTorch model for a specified number of epochs and returns the training loss per epoch.

- `testing_loop(model, criterion, test_loader, device)`: evaluates a trained PyTorch model on test data and returns the test loss, accuracy, and logits.

- `sem(accuracy_eval, std_errors)`: computes the standard error of the mean (SEM) for a list of accuracies and appends it to a provided list.

- `init_dicts(models, dicts)`: initializes empty entries for each model in a list of dictionaries, used to store metrics or results.

- `softmax(logits)`: computes probabilities from logits (used for metrics like AUC).

- `process_fold(train_index, test_index, features, labels, device)`: 
  - Splits data into training and test sets for the current fold.
  - Normalizes features using `MinMaxScaler`.
  - Applies feature selection (Chi-2 + Random Forest by default).
  - Initializes an ImageTransformer (DeepInsight) to generate images with multiple channels (prepares data for CNN-1D+ and IT-KAN).  
  - **Returns**:  
    - `X_train_scaled`, `X_test_scaled`: transformed feature matrices (ready for image conversion)  
    - `y_train`, `y_test`: corresponding labels  
    - `image_transformer`: the fitted ImageTransformer object containing gene coordinates

- `evaluate_models_cv(models, features, labels, num_classes, dataset_name, evaluation_type, device, logits_active, abs_path)`: 
  - Performs cross-validation (k-fold or leave-one-out), trains and evaluates each specified model.
  - Saves results, logits, and metrics (accuracy, loss, SEM).
  - Handles different model types (SVM, KAN, MLP, CNN-1D, DI-CNN+, ViT, IT-KAN) with appropriate preprocessing, channel expansion, and data loading.
  - Supports parallel processing of folds using CPU, and GPU acceleration for model training where applicable, to improve hardware efficiency and reduce execution times.
  - **Returns**:  
    - `avg_loss`: average loss across folds/epochs  
    - `avg_accuracy`: average classification accuracy  
    - `sem`: standard error of the mean (SEM) for accuracy (or other metric) 


#### 3. Main Workflow (`if __name__ == "__main__":`)
- Defines dataset paths and output folders.
- Sets execution parameters (e.g., type of validation, models to use).
- Prepares directories for results and logs.
- Loads and processes data, separating features and labels.
- For each dataset:
  - Performs cross-validation and evaluates all specified models.
  - Saves accuracy and loss metrics for each model.
- Optionally executes:
  - **Metrics**: calculates AUC, errors under the ROC curve, and saves ROC curves.
  - **Chart**: generates and saves a bar chart of model accuracies across all datasets.


### Modularity

Each step (preprocessing, feature selection, channel expansion, training, evaluation, saving results) is encapsulated in dedicated functions, making the code modular and easily extendable.


