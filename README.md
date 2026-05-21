# ELAPSE

**ELAPSE** is a framework for evaluating the impact of data selection methods on ML model utility and fairness.  
It supports configurable experiments across a variety of datasets and ML models, a wide range of ML data selection methods, and measuring various fairness metrics and utility metrics. As a result, ELAPSE produces the experiment traces and their statistical analysis.

<p align="center">
  <img src="./ELAPSE_pipeline.jpg" alt="Overview of ELAPSE experimentation framework" width="1000">
</p>

In the following, we introduce:

- [ELAPSE Features](#elapse-features)  
- [Repository Structure](#repository-structure)  
- [Starting with ELAPSE](#starting-with-elapse)  
- [Running an Experiment](#running-an-experiment)  
- [Producing Traces and Statistics](#producing-traces-and-statistics)  
- [Reproducibility Testing](#reproducibility-testing)  
- [Contributing](#contributing)  
- [Acknowledgments](#acknowledgments)  
- [Publications](#publications) 

---


## ELAPSE Features

- Evaluation of selection methods on fairness and utility metrics  
- Built-in support for common ML models and real-world datasets  
- Modular design for easy extension  
- Epoch-level and aggregated result tracing  
- Integrated t-test analysis for selection impact and variability  
- Reproducibility support through configuration files, released traces, and table-generation scripts


## Repository Structure
```bash
├── code/
│   ├── configs/                  # Experiment configurations
│   ├── dataselection/            # Core modules for data selection 
│   │   ├── selectionmethods/     # Data selection methods
│   │   └── utils/                # Dataset loading and model utilities
│   └── statistics/               # Trace statistics, t-tests, correction, and table generation
├── datasets/                     # ELAPSE datasets
├── results/                      # Output results per configuration
├── traces/                       # Aggregated metrics and t-test results
└── README.md
```
---

## Starting with ELAPSE

### Software Requirements
- Python 3.10.5  
- CORDS 0.0.4  
- All dependencies listed in `code/requirements.txt`

The experiments reported in the paper were run with Python 3.10.5 and CORDS 0.0.4. The main packages used in the evaluation include:

```bash
aif360==0.6.1
fairlearn==0.10.0
matplotlib==3.5.3
numpy==1.26.4
pandas==2.0.3
scikit-learn==1.3.2
scipy==1.13.1
seaborn==0.13.2
statsmodels
cords==0.0.4
torch==2.5.1
torchvision==0.20.1
```

### Hardware Recommendations
Experiments can be run on CPU or GPU. A CUDA-compatible GPU is recommended for faster training when using deep models or large datasets.

### Installation

To install the latest ELAPSE version from source:

```bash
git clone https://github.com/sara-bouchenak/ELAPSE/
cd ELAPSE
conda create -n elapse python=3.10.5
conda activate elapse
pip install -r code/requirements.txt
pip install cords==0.0.4
```



## Running an Experiment

1. Create or select a JSON configuration file in `code/configs/`.

   Example configuration:

   ```json
   {
     "dataset_name": "ars",
     "sensitive_attributes": ["gender"],
     "columns": ["gender", "labels"],
     "dataset_path": "./datasets/ars/",
     "train_file": "train_ars.csv",
     "test_file": "test_ars.csv",
     "val_file": "val_ars.csv",
     "data_load": "load-ars",

     "models": ["MLP", "SVM", "Logreg"],

     "lr": 0.001,
     "batch_size": 512,
     "epoch": 400,
     "label_num": 2,
     "log_interval": 50,

     "runs": 5,

     "fraction": 0.05,
     "select_every": 20,
     "ratios": [0.05, 0.1, 0.2, 0.3],
     "values": [3],

     "result_path": "./results/ARS",
     "cols": [
       "SPD_gender", "EOD_gender", "AOD_gender",
       "DI_gender", "DcI_gender", "F1_score",
       "Precision", "Recall"
     ],
     "warmup_epochs" : 20
   }
   ```

   The field `values` defines the evaluated training/data-selection system:

   ```bash
   0 = Full
   2 = GradMatch
   3 = Craig
   4 = Glister
   5 = Random
   ```

2. Prepare the result folder structure.

```bash
results/<result_path>/
  └── <method_name>/
      └── <dataset_name>_<selection_ratio>/
```

3. Run the experiment.

```bash
python code/main.py --config code/configs/config.json
```

Replace `code/configs/config.json` with the configuration file corresponding to the dataset, model, selection method, and selection ratios that should be evaluated.



## Producing Traces and Statistics

ELAPSE supports detailed trace logging and statistical analysis to evaluate the impact of data selection on model fairness and utility.

### Output Trace Files

- `ExperimentMeasurements.csv`: Epoch-wise metrics for each run across configurations  
- `ExperimentStatistics.csv`: Aggregated results and impact evaluation  
- `ExperimentConfigurations.csv`: All evaluated configuration details  
- `DatasetProperties.csv`: Metadata about datasets and associated sensitive attributes  

### Running the Analysis

Use the following notebooks to generate trace files and conduct statistical tests:

```bash
# Generate experiment traces
jupyter notebook code/statistics/traces.ipynb

# Applies t-tests to evaluate data selection impact on ML model utility and fairness
jupyter notebook code/statistics/selection-impact-t-test.ipynb

# Applies Holm-Bonferroni correction, post-processing thresholds, and effect-size summaries
jupyter notebook code/statistics/selection-impact-t-test-w-correction.ipynb

# Applies t-tests to assess the variability of ML model utility and fairness
jupyter notebook code/statistics/variability-t-test.ipynb
```

The notebook `selection-impact-t-test-w-correction.ipynb` applies the statistical post-processing used in the paper, including:

- Holm-Bonferroni correction over the collected p-values;
- post-processing thresholds of `0.5%`, `0.75%`, `1%`, `2%`, and `3%`;
- effect-size aggregation;
- exclusion of the `Random` baseline from the main comparison tables;
- exclusion of datasets for which specific fairness metrics are not applicable, when relevant;
- generation of CSV and LaTeX files used to build the paper tables.

---

## Reproducibility Testing

The purpose of this section is to make the reproduction of ELAPSE experiments and paper tables straightforward.  
The first step is a simple run to test that ELAPSE is correctly installed, followed by a representative experiment, and finally the reproduction of the paper tables from the released traces.

The experiments reported in the paper were run using Python 3.10.5 and CORDS 0.0.4. The configuration files used to run the experiments are available in `code/configs/`. Each configuration specifies the dataset, model, sensitive attributes, number of runs, selection method, selection ratios, and output folder.

### 1. Run a simple example

This command runs one ELAPSE configuration and checks that the framework can load the dataset, train the model, apply the selected data-selection method, and save the results.

```bash
python code/main.py --config code/configs/config.json
```

Replace `code/configs/config.json` with one of the available configuration files in `code/configs/`.

After the run completes, the result folder should contain CSV files with epoch-level measurements for the evaluated method, dataset, model, and selection ratio. The exact output path is controlled by the `result_path`, `dataset_name`, `models`, `ratios`, and `values` fields in the selected configuration file.

### 2. Run representative experiments

To reproduce representative experiments from the paper, run the corresponding configuration files from `code/configs/`. For example:

```bash
python code/main.py --config code/configs/<DATASET_CONFIG>.json
```

The following configuration fields are the most important for reproducibility:

```json
{
  "dataset_name": "...",
  "sensitive_attributes": ["..."],
  "models": ["..."],
  "runs": 5,
  "ratios": [0.05, 0.1, 0.2, 0.3],
  "values": [0, 2, 3, 4, 5],
  "select_every": 20,
  "warmup_epochs": ...
}
```

The paper compares the full-data baseline against the considered data-selection methods. Therefore, to reproduce the full evaluation for a given dataset and model, make sure that the configuration includes the full baseline and the data-selection methods:

```bash
0 = Full
2 = GradMatch
3 = Craig
4 = Glister
5 = Random
```

The random seeds and number of runs are controlled by the experiment configuration and the implementation of the training pipeline. The paper uses 5 runs per configuration.

### 3. Generate traces from experiment outputs

Once the raw experiment outputs are available in `results/`, generate the trace files with:

```bash
jupyter notebook code/statistics/traces.ipynb
```

This produces the aggregated trace files used in the statistical analysis:

```bash
traces/ExperimentMeasurements.csv
traces/ExperimentStatistics.csv
traces/ExperimentConfigurations.csv
traces/DatasetProperties.csv
```

### 4. Run statistical tests

To compute the paired t-tests comparing each data-selection method against the full-data baseline, run:

```bash
jupyter notebook code/statistics/selection-impact-t-test.ipynb
```

This step produces the raw t-test results, including p-values and effect sizes, in the corresponding result folders.

To compute variability-related t-tests, run:

```bash
jupyter notebook code/statistics/variability-t-test.ipynb
```

### 5. Apply Holm-Bonferroni correction and post-processing thresholds

To reproduce the corrected statistical results used in the revised paper, run:

```bash
jupyter notebook code/statistics/selection-impact-t-test-w-correction.ipynb
```

This notebook applies Holm-Bonferroni correction and produces post-processed impact labels using several practical thresholds:

```bash
0.005   # 0.5%
0.0075  # 0.75%
0.01    # 1%
0.02    # 2%
0.03    # 3%
```

The corrected outputs are saved in:

```bash
results/test-p-value-005-holm-global-postprocessed/
results/effect_size_tables/
results/effect_size_summary/
```

The main corrected table files include:

```bash
ttest_5_holm_global-5C.csv
ttest_5_holm_global-5C-wo-random.csv
effect_size_wide_holm_global.csv
table_effect_size_summary.csv
```

### 6. Reproduce all paper tables from released traces

The paper tables can be reproduced directly from the released traces without rerunning all experiments. This is the recommended option for checking the reported results.

First, make sure that the released traces and statistical results are available in the expected folders:

```bash
traces/
results/test-p-value-005-raw-effect-size/
```

Then run the canonical table-reproduction script:

```bash
python code/statistics/reproduce_paper_tables.py \
  --results-root results \
  --input-dir results/test-p-value-005-raw-effect-size \
  --output-dir results/paper_tables \
  --alpha 0.05 \
  --correction holm \
  --post-thresholds 0.005 0.0075 0.01 0.02 0.03 \
  --exclude-random \
  --exclude-spd-di-datasets voxceleb fairface
```

This script reproduces the corrected tables used in the paper by:

1. loading the raw t-test and effect-size results;
2. applying Holm-Bonferroni correction globally over the tested metrics;
3. assigning impact labels after correction;
4. applying the practical post-processing thresholds;
5. excluding `Random` from the main paper tables;
6. excluding `voxceleb` and `fairface` from SPD/DI summaries when these metrics are not applicable;
7. exporting both CSV and LaTeX versions of the paper tables.

The generated files are saved in:

```bash
results/paper_tables/
```

Expected outputs include:

```bash
table_impact_holm_1pct.csv
table_impact_holm_1pct.tex
table_impact_holm_0p5pct.csv
table_impact_holm_0p75pct.csv
table_impact_holm_2pct.csv
table_impact_holm_3pct.csv
table_effect_size_summary.csv
table_variability_holm.csv
```

### 7. Optional: inspect and modify the table-generation workflow

The following notebook provides an interactive version of the correction and table-generation workflow:

```bash
jupyter notebook code/statistics/selection-impact-t-test-w-correction.ipynb
```

It is useful for inspecting intermediate CSV files, checking the effect of different post-processing thresholds, and validating the final LaTeX tables before adding them to the paper.

### 8. Notes on reproducibility

Some small numerical differences may occur across machines because of differences in hardware, low-level libraries, or nondeterministic operations in model training. For this reason, the recommended way to reproduce the exact tables reported in the paper is to use the released traces and run the table-generation script described above.

To reproduce the complete experimental campaign from scratch, run the relevant configuration files in `code/configs/`, then generate traces and statistical tables using the notebooks and scripts described in this section.

---

## Contributing

We value and encourage contributions from the research and open-source communities to improve the ELAPSE framework. ELAPSE is designed to be modular and extensible, making it easy to integrate new datasets, selection methods, or models.

### How to Extend ELAPSE

- **Add new datasets**:  
  Update the dataset builder in `dataselection.utils.data.datasets` to integrate a new dataset along with its preprocessing.

- **Implement new selection methods**:  
  Extend `dataselection/selectionmethods/` with a new data selection method, and `dataselection.utils.data.dataloader` to account for it.

- **Add new model architectures**:  
  Define additional models in `dataselection.utils.models` and ensure they are compatible with the training pipeline.

- **Improve metrics and trace handling**:  
  Add new fairness/utility metrics, and enhance the statistical analysis workflow.



### Contribution Guidelines

We welcome all types of contributions. Please follow these guidelines to ensure smooth collaboration:

- **Report issues**:  
  If you encounter bugs or have suggestions for improvement, open an issue on the GitHub repository. Include as much relevant detail as possible (e.g., error messages, dataset/config used, and reproduction steps).

- **Propose new features**:  
  For new features or enhancements, submit a feature request. Clearly describe the motivation, proposed functionality, and how it aligns with ELAPSE’s goals.

- **Code contributions**:  
  To contribute code:
  1. Fork the ELAPSE repository.
  2. Create a new branch from `main` or an appropriate development branch.
  3. Implement your changes and ensure they are well-documented and, where applicable, tested.
  4. Submit a pull request with a clear explanation of the changes and their purpose.

- **Code style**:  
  Follow the existing code structure and conventions used in ELAPSE. Consistency improves readability and facilitates code reviews.

- **Testing**:  
  Ensure your changes pass existing tests and, if introducing new features, provide relevant tests.



For substantial changes, consider opening a discussion or draft pull request first to align with the maintainers on design choices.
For any questions or follow-up, please contact the repository maintainers at [nawel.benarba@insa-lyon.fr](mailto:nawel.benarba@insa-lyon.fr), [zeyang.kong@insa-lyon.fr](mailto:zeyang.kong@insa-lyon.fr), and [sara.bouchenak@insa-lyon.fr](mailto:sara.bouchenak@insa-lyon.fr).


---

## Acknowledgments

ELAPSE extends [CORDS](https://github.com/decile-team/cords) by adding support for new datasets with associated sensitive attributes, integrating additional model architectures, and computing a comprehensive set of utility and fairness metrics. We thank the open-source community for the foundational tools and contributions that supported the development of this framework.

---

## Publications

The following publications are related to the data selection strategies supported in ELAPSE.

[1] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, Rishabh Iyer.  
“GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training”.  
*Proceedings of the 38th International Conference on Machine Learning (ICML), July 2021*, Virtual 
Event, PMLR 139:5464–5474.

[2] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Rishabh Iyer.  
“GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning”.  
*Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI), February 2021*, Virtual Event, 
pp. 8110–8118.

[3] Baharan Mirzasoleiman, Jeff Bilmes, Jure Leskovec.  “Coresets for Data-efficient Training of 
Machine Learning Models”. *International Conference on Machine Learning (ICML), July 2020*, Virtual 
Event.
