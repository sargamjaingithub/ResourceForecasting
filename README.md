<<<<<<< HEAD
# ResourceForecasting
A machine learning framework for cloud resource forecasting using time-series algorithms (Random Forest, SVM, Informer, SCINet). Includes interactive Streamlit dashboard for visualizing CPU/memory predictions on real workload traces (Alibaba 2018, Google 2019). Optimize capacity planning and reduce cloud costs.
=======
# Enhancing the output of time series forecasting algorithms for cloud resource provisioning
_This repository was created as a result of the [research manuscript](https://www.sciencedirect.com/science/article/pii/S0167739X25001281)_

### Introduction
This repository contains an implementation for training, testing and evaluating workload resource forecasting algorithms. It also contains the required code and configurations to run all the experiments of the aforementioned manuscript. 

### Code structure
The code is structured in three main parts: data loading, algorithms and evaluation. The three parts are managed by the script `app/train_test_model.py` that trains an algorithm and tests/evaluates it once it is trained. All the required configuration for every execution of the script is input through a json file. This json file contains the information regarding the dataset to load, the preprocessing steps to performed, the algorithm to use, the parameters of the algorithm and so on.

The following list describes in a lower level the three main parts of the code:
- Data loading: loads the dataset, performs preprocessing steps and prepares it for algorithm processing. It is divided in two classes:
  - source: loads the dataset from disk to numpy arrays, splits into train/val/test splits and performs preprocessing steps. The interface is located at `app/data/data_source_abstract.py` and the different subclasses are available at the directory `app/data/sources/`. The current implementation contains two subclasses for loading the Alibaba and Google datasets.
  - type: prepares the dataset for algorithm processing. In the case of deep and machine learning algorithms, creates the forecasting windows and creates a DataLoader for easy access. If a discretization step is required, this part is also in charge of creating the corresponding intervals. On the other hand, for classic methods, a DataLoader is created with the full time series without windows. The interface is located at `app/data/data_type_abstract.py` and the different subclasses are available at the directory `app/data/types/`.
- Algorithm: implements the training and prediction of the algorithm. There are three types, classical methods (`app/models/classic_methods/`), machine learning methods (`app/models/machine_learning_algorithms/`) and deep learning models (`app/models/networks/`), the three of them with the corresponding interface and multiples subclasses in the specified directories. 
- Evaluation: implements the evaluation of the results. It is fully described in the manuscript and can be found at `app/evaluation/evaluation.py`.

Apart from this, there are two additional scripts `app/experiment_motivation_for_new_evaluation.py` and `app/experiment_explainability.py`, that run the remaining experiments that do not involve training and testing an algorithm. Anyhow, these two scripts also employ the described code structure to load the data and use the corresponding algorithms.

### How to set up
There are two ways to run the code, directly through python or through singularity. All the experiments of the manuscript were run with singularity in the MareNostrumIV accelerated cluster from the Barcelona Supercomputing Center.

Follow these instructions to prepare the environment to run directly with python (use a virtual env if desired):
- Install python3.9
- Install requirements.txt

Follow these instructions to prepare the environment to run directly with singularity:
- Build image: `sudo singularity build IMAGE_NAME.sif singularity_image.def`

### How to run
The bash script used to run the manuscript experiments is available at `launcher_multiple_singularity.sh`. It was adapted to use slurm in the aforementioned cluster. The used configurations are available at `experiment_configs/`.

To run directly with python:
```
python3 app/train_test_model.py --config CONFIG_FILE_PATH --output OUTPUT_DIRECTORY_PATH
```
remember to populate the variables:
- CONFIG_FILE_PATH with all the experiment configuration, check `experiment_configs/` for examples
- OUTPUT_DIRECTORY_PATH with the directory path to store the resulting outcome of the experiment

To run with singularity:
```
SINGULARITYENV_CUDA_VISIBLE_DEVICES=CUDA_DEVICES PYTHONUNBUFFERED=false PYTHONPATH=. singularity exec --nv IMAGE_NAME.sif python3 ./app/train_test_model.py --config CONFIG_FILE_PATH --output OUTPUT_DIRECTORY_PATH
```
remember to populate the variables:
- CUDA_DEVICES with the desired GPU devices for the experiment, remember the --nv parameter accordingly
- IMAGE_NAME with the singularity image name
- CONFIG_FILE_PATH with all the experiment configuration, check `experiment_configs/` for examples
- OUTPUT_DIRECTORY_PATH with the directory path to store the resulting outcome of the experiment

### Related publications
In this code there are slight modifications of other work implementations, here they are appropriately cited:

```
Berral, J. L., Buchaca, D., Herron, C., Wang, C., & Youssef, A. (2021, September). Theta-scan: leveraging behavior-driven forecasting for vertical auto-scaling in container cloud. In 2021 IEEE 14th International Conference on Cloud Computing (CLOUD) (pp. 404-409). IEEE.
```

```
Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023, June). Are transformers effective for time series forecasting?. In Proceedings of the AAAI conference on artificial intelligence (Vol. 37, No. 9, pp. 11121-11128).
```

```
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, May). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, pp. 11106-11115).
```

```
Liu, M., Zeng, A., Chen, M., Xu, Z., Lai, Q., Ma, L., & Xu, Q. (2022). Scinet: Time series modeling and forecasting with sample convolution and interaction. Advances in Neural Information Processing Systems, 35, 5816-5828.
```

### How to cite
If using this code please cite this paper:
```
Agullo, Ferran, et al. "Enhancing the output of time series forecasting algorithms for cloud resource provisioning." Future Generation Computer Systems (2025): 107833.
```
>>>>>>> b5dbe9f (Initial commit: Resource Forecasting project)
