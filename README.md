## Automated Architecture and Hyperparameter Optimization for Deep Neural Networks Applied in Forecasting the Cross-Section of Stock Returns
### Description
This repository contains the code to replicate research and experiments focused on forecasting stock returns using deep learning techniques. 

A key contribution of this work is the exploration of neural network architecture design and hyperparameter optimization through Automated Machine Learning (AutoML) methods, aiming to enhance predictive performance and model robustness. 

The repository includes implementations, data processing pipelines, and evaluations of models leveraging AutoML strategies such as the Tree-Structured Parzen Estimator (TPE).

This project provides insights into advancing stock return prediction by combining financial theory with cutting-edge machine learning tools.

### Setup

1. Modify `src/data/credentials_template.ini` by typing the WRDS username and password
2. Rename `credentials_template.ini` to `credentials.ini`
3. Run `setup.sh`
4. Run the following command from the root directory of the repository: `python src/main.py -h` and check what CLI argument is relevant for the experiment you want to run (i.e. to test the grid search with expanding window on the Gu et al's network run `python src/main.py --guNetworkTuning`).
3. In the CLI arguments, include `--saveDirName <dirName>` to customise the save directory
4. Use the links displayed in the terminal to view the experiment dashboard
