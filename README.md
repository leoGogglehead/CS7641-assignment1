#CS-7641 Assignment 1: Supervised Learning

## Author
Phuc Minh Nguyen
pnguyen333@gatech.edu


## Build
NOTE: MUST HAVE CONDA INSTALLED. 
Run the following commands from directory devenv_conf to set up a minimal conda environment, activate the environment, then install and configure poetry.
- 'conda env create -f environment.yml'
- 'conda activate cs7641_ass1'


## Datasets
Two datasets are used for this Assignment:

### 1) Two Sigma Connect - Rental Listing Inquiries: Classification of Interest Level in Rental Listings
Link:
https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data
Only training data train.json file are used, as labels are not provided for the test set.

Feature engineering:
https://github.com/happycube/kaggle2017/blob/master/renthop/fin-dprep.ipynb
https://www.kaggle.com/code/luisblanche/price-compared-to-neighborhood-median/notebook

Metric: Multi-class log loss


### 2) Default of credit card clients: Classification of (Binary) default payment
Link: 
https://www.openml.org/search?type=data&status=active&id=42477&sort=runs
