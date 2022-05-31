#!/bin/bash
#$ -cwd
#$ -m abe
#$ -M a.di.giovanni@tum.de
echo "Starting Job at `date`"
R CMD BATCH /home/tum/antoniodg/thesis/src/data/OpenAssetPricing_Code/master.R
echo "Ending Job at `date`"
