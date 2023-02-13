#CS-7641 Assignment 1: Supervised Learning

## Author
Phuc Minh Nguyen
pnguyen333@gatech.edu

## URL
Link to project code: https://github.com/leoGogglehead/CS7641-assignment1

## Overview

This project explored and analyzed 5 different supervised learning algorithms on two different datasets:

    Decision Tree Classifier
    Histogram-based Gradient Boosted Decision Trees
    Multilayer Perceptron
    k-Nearest neighbors
    C-Support Vector Machines


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


### 2) Default of credit card clients: Classification of (Binary) default payment
Link: 
https://www.openml.org/search?type=data&status=active&id=42477&sort=runs

## Processing Data: 
Run the following notebooks to process data, split and write training and testing datasets:

    credit_card_proc.ipynb
    rental_listing_proc.ipynb

## Runing Experiments:
Run the following notebooks to conduct experiments and produce figures:

    credit_card_exp.ipynb
    rental_listing_exp.ipynb
