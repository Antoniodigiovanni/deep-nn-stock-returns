# Thesis
Codebase for Antonio Di Giovanni's master thesis


1. Install dependencies from *requirements.txt*
2. Install PyTorch as it is not included in the requirements file
3. Run the R files in src/data/OpenAssetPricingCode, particularyl master.R in order to download data from CRSP and signals.
4. Run the following folder from the root directory of the repository 'thesis/': **python src/main.py -h**) and check what command line is relevant for the experiment you want to run (i.e. to test the grid search with expanding window on the Gu et al's network run **python src/main.py --guNetworkTuning**)