# Does Automated Data Selection Hurt Fairness and Utility of Machine Learning?
# ELAPSE 

## 🔍 Overview

**ELAPSE** is a framework for evaluating how data selection methods affect both model utility and fairness across a wide range of settings.

## 📁 Repository Structure

- `Datasets/` : Contains the real-world datasets used in ELAPSE.  
- `Code/` : Main codebase to run experiments.  
  - `configs/` : Example experiment configuration files.  
  - `results/` : Folder to store output results per configuration.  
- `Traces/` :  
  - `DatasetProperties.csv` : Evaluated datasets and their corresponding sensitive attributes.
  - `ExperimentConfigurations.csv` : All evaluated configurations.  
  - `ExperimentMeasurements.csv` : Epoch-wise results for the different runs across all evaluated settings.
  - `ExperimentStatistics.csv` : Aggregated metrics and t-test results.  

## 🚀 Running an Experiment

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Edit configuration

Add the config file to `Code/configs` to set the following parameters:

- **Dataset**  
  Set the dataset name, its sensitive attributes and the path to the data.

- **Model and Hyperparameters**  
  Choose the model(s) to be used and specify their hyperparameters.

- **Selection Method, Ratios, and Selection Frequency**  
  Define the data selection method(s) and the corresponding selection ratios, and selection frequency.

- **Number of Runs**  
  Configure how many times to run the experiment.

- **Output Path**  
  Specify the output directory to save results.

#### 🧾  Configuration Example
```json
{
  // Name of the dataset used
  "dataset_name": "ars",

  // Sensitive attribute(s) used to compute fairness metrics
  "sensituve_attributes": ["gender"],

  // Columns to retain (e.g., the sensitive attribute and labels)
  "columns": ["gender", "labels"],

  // Path to the folder containing the dataset files
  "dataset_path": "../Datasets/ARS/",

  // Dataset file names for training, testing, and validation
  "train_file": "train_ars.csv",
  "test_file": "test_ars.csv",
  "val_file": "val_ars.csv",

  // Name of the dataset loading function implemented in the code
  "data_load": "load-ars",

  // Training hyperparameters
  "lr": 0.001,             // Learning rate
  "batch_size": 512,       // Batch size
  "epoch": 400,            // Number of training epochs
  "label_num": 2,          // Number of output classes
  "log_interval": 50,      // Frequency for logging during training
  "runs": 5,               // Number of experiment repetitions

  // Data selection parameters
  "fraction": 0.05,        // Initial fraction of the dataset used
  "select_every": 20,      // Frequency (in epochs) of selecting new data
  "ratios": [0.05, 0.1, 0.2, 0.3], // Selection ratios to evaluate

  // Data selection method(s) to evaluate:
  // 0 = Full, 2 = GradMatch, 3 = Craig, 4 = Glister, 5 = Random
  "values": [3],

  // Machine learning models to train
  "models": ["MLP", "SVM", "Logreg"],

  // Output directory to save results
  "result_path": "./results/ARS",

  // Metrics to compute: fairness (w.r.t. gender) and performance
  "cols": [
    "SPD_gender", "EOD_gender", "AOD_gender",
    "DI_gender", "DcI_gender", "F1_score",
    "Precision", "Recall"
  ]
}
```

### 3. Prepare result folders
Create the following structure:
```
<result_path>/
  └── <method_name>/
      ├── <dataset_name>_<selection_ratio>/
      └── ...
```

### 4. Run the experiment
```bash
python Code/main.py --config Code/configs/config.json
```

## 📊 View Results
Results are saved in the specified result folder.
