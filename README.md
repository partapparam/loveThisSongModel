# Overview

Capstone-1 project from the [Machine Learning Zoomcamp course delivered by DataTalks Club](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master) conducted by Alexey Grigorev.

### The Problem

Why do we like the songs that we do? What do our favorite songs have in common? Thanks to Spotifys Track Feature API, we can get a better understanding of what a song needs to be in our 'Liked' playlist, or a song that we skip quickly.

### Data

I found this dataset on [Kaggle](https://www.kaggle.com/datasets/geomack/spotifyclassification/data) which has already been assigned with the creators 'Liked' = 1 and 'Not Liked' = 0.
There are 2017 total rows, and 14 columns describing the track, and our target variable - 'target'

The only columns that I dropped from our data were the 'Unnamed: 0' -- which was a useless column, and the 'song_title' column.

- The 'song_title' had no effect on the accuracy of our model so there was no reason to leave it in.

#### Columns

- acousticness: number [float]
  - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- danceability: number [float]
  - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
-

## Classification

We either like a song or we don't, so I used classification models to test what features a song needs for us to like it.

- Decision Trees
- Random Forest
- XGBoost

## Evaluating the model

I used ROC_AUC_SCORE because this is a binary classification problem. I included results for the following as well:

- `accuracy`
- `precision`
- `recall`
- `f1`

## Project Files

- ReadMe.md
- notebook.ipynb
  - Date prep and cleaning
  - EDA
  - Building the models and tuning parameters
- train.py
  - Train the final model and save it to .bin file
- xgb_model.bin
  - the Model and Dict Vectorizer
- app.py
  - Loading the model and serving to a Flask Service at port 9696
- predict-test.py
  - Testing the flask service and the Model with example songs for us to test
- Dockerfile
- Pipfile and Pipfile.lock

## Running the project

To see the full dataset, EDA, and model selection process, run **notebook.ipynb**

To run the model and save it to xgb_model.bin file:
`python train.py`

To test the model:
`python app.py`
or
`gunicorn --bind 0.0.0.0:9696 app:app`

In second terminal, run:
`python predict_test.py`

### Build the Docker Image

`docker build -t satisfaction .`

`docker run -it -p 9696:9696 app:latest `

Run in another terminal:

`python predict_test.py`
