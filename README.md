# Unsupervised Domain Adaptation with non-stochastic missing data
This repository contains the code of the following paper "Unsupervised Domain Adaptation with non-stochastic missing data"

# Dependencies
In order to run, the code requires the following Python modules specified in `requirements.txt`
  * Numpy
  * Matplotlib
  * [POT](https://github.com/rflamary/POT) (Python Optimal Transport library)
  * PyTorch
  * sklearn

# Quickstart
* Install Miniconda then `source ~/.bashrc`
* Create conda environment: `conda create --name adaptation-imputation python=3.6 -y`
* Activate environment `source activate adaptation-imputation`
* Install the requirements in this environment `pip install -r requirements.txt`.
* Install the package `pip install -e .` at the root

# Run an experiment
* `cd adaptation-imputation/experiments/launcher`
* `python ../../orchestration/launcher.py --experiment "dann_mnist_usps" --gpu_id=1`

The experiment argument is defined in `adaptation-imputation/experiments/__init__.py`. All hyperparameters are stored in a .py file.

Both DANN and DeepJDOT extensions to missing data are in this repository.
* with no suffix, it will run the model with full data (for digits) else it will run the model on missing data
* with "ignore" suffix, it will run the model ignoring the missing component
* with "zeroimput" suffix, it will run the model with zero imputation for the missing component
* with "imput" suffix, it will run the model with conditional generation of the missing component 

`gpu_id` specifies which gpu machine to use. Jobs can be run on CPU but training time will be long.

# Notes  
Figures are saved in folder `figures` and logs in a seperate `results` folder created when the job is launched

Utils functions are saved in `utils`

# Datasets
## Digits dataset
Digits datasets will download as part of the training script. Code is taken from existing github repos and credit is given in the .py files.

## Criteo dataset
Preprocessed data is available [here][2]. 
If necessary follow steps below to regenerate the data.

### Regenerate Criteo data
* Download Criteo Kaggle dataset from [here][1] into `data` folder
* Run `python data_preprocessing.py` in `data` folder
* Run following UNIX commands:
    * Define UNIX function for seeding:
    
    `get_seeded_random() { seed="$1"  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null ; }`
    * `shuf -o total_source_shuffled.txt < total_source.txt --random-source=<(get_seeded_random 42)`
    * `sed -n -e '1,1183117p' total_source_shuffled.txt > total_source_data.txt`
    * `rm total_source_shuffled.txt total_source.txt`
* Training script is then ready to be run

[1]: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
[2]: https://www.dropbox.com/sh/ino8kcgr9x5jm0r/AACI3wynfb96wrJNTb39y_gga?dl=0