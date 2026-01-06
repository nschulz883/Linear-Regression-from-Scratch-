# Linear-Regression-from-Scratch-

## Overview

This project is a portfolio-focused Python application designed to demonstrate foundational machine learning and optimisation skills.
It implements linear regression from scratch, allowing users to train a model, track loss over time, and make predictions while learning how gradient descent and feature scaling work.
The project highlights core Python programming, model evaluation, and professional documentation practices.

## Key Skills Demonstrated

Python programming fundamentals

Machine learning concepts and linear regression

Gradient descent optimisation

Feature scaling (min–max normalisation)

Loss tracking and convergence analysis

Gradient clipping for stable training

Writing clear and professional documentation

## Features

Pure Python implementation (no ML libraries)

Single-variable linear regression model

Mean Squared Error (MSE) loss calculation

Manual gradient computation and parameter updates

Feature normalisation for stable learning

Gradient clipping to prevent unstable updates

Training progress tracking across epochs

Final predictions and residual analysis

## Tech Stack

Python

## Sample Data

A small dataset is included for demonstration purposes:
``` 
x = [1, 2, 4, 8, 16]
y = [2, 6, 8, 16, 30]
``` 

This dataset is intentionally simple to allow:

Transparent step-by-step analysis

Easy debugging and validation

Clear observation of convergence behaviour

## Installation
``` 
git clone https://github.com/nschulz883/linear-regression-from-scratch.git
cd linear-regression-from-scratch


(No external dependencies required)
``` 

## Usage

After cloning the repository, run the script:
``` 
python linear_regression.py
``` 

The program will:

Normalise the input data

Train the model using gradient descent

Print training progress at regular intervals

Output final predictions and residuals

