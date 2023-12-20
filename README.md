# Overview

Capstone-1 project from the [Machine Learning Zoomcamp course delivered by DataTalks Club](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master) conducted by Alexey Grigorev.

### The Problem

Why do we like the songs that we do? What do our favorite songs have in common? Thanks to Spotifys Track Feature API, we can get a better understanding of what a song needs to be in our 'Liked' playlist, or a song that we skip quickly.

### Data

I found this dataset on [Kaggle](https://www.kaggle.com/datasets/geomack/spotifyclassification/data) which has already been assigned with the creators 'Liked' = 1 and 'Not Liked' = 0.
There are 2017 total rows, and 14 columns describing the track, and our target variable - 'target'

The only columns that I dropped from our data were the 'Unnamed: 0' -- which was a useless column, and the 'song_title' column.

- The 'song_title' had no effect on the accuracy of our model so there was no reason to leave it in.

#### Features

- Artist
- acousticness: number [float]
  - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- danceability: number [float]
  - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- duration_ms: integer
  - The duration of the track in milliseconds.
- energy: number [float]
  - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- instrumentalness: number [float]
  - Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- key: integer
  - The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- liveness: number [float]
  - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- loudness: number [float]
  - The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
- mode: integer
  - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- speechiness:number [float]
  - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- time_signature: integer
  - An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4".
- tempo: number [float]
  - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- valence: number [float]
  - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

### EDA

- Using the correlation matrix, we saw slight correlations between certain features (acousticness/energy and energy and loudness)
- This was not as helpful as I'd hoped in helping me to form a hypothesis on what would matter most in liking a song, hence why I did drop any of the features.

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
