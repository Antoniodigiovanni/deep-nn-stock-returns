# Thesis
### Codebase for Antonio Di Giovanni's master thesis

1. Modify src/data/credentials_template.ini by typing the WRDS username and password
2. Rename credentials_template.ini to credentials.ini
3. Run setup.sh
4. Run the following command from the root directory of the repository 'thesis/': **python src/main.py -h**) and check what CLI argument is relevant for the experiment you want to run (i.e. to test the grid search with expanding window on the Gu et al's network run **python src/main.py --guNetworkTuning**).
3. In the CLI arguments, include --saveDirName <dirName> to customise the save directory
4. Use the links displayed in the terminal to view the experiment dashboard