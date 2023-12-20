import requests

url = "http://0.0.0.0:9696/predict"

survey_one = {
            "acousticness": 0.0102,
            'danceability': 0.833,
            'duration_ms': 204600,
            'energy': 0.434,
            'instrumentalness': 0.021900,
            'key': 2, 
            'liveness': 0.1659,
            'loudness': -8.795,
            'mode': 1,
            'speechiness': 0.4310,
            'tempo': 150.062,
            'time_signature': 4.0,
            'valence': 0.286,
            'target': 1,
            'artist': 'Future',
            'song_title': 'Mask Off'
            }

survey_two = {
            "acousticness": 0.27,
            'danceability': 0.663,
            'duration_ms': 211160,
            'energy': 0.656,
            'instrumentalness': 0.000279,
            'key': 1, 
            'liveness': 0.117,
            'loudness': -6.548,
            'mode': 1,
            'speechiness': 0.0413,
            'tempo': 120.037,
            'time_signature': 4.0,
            'valence': 0.555,
            'target': 0,
            'artist': 'Dierks Bentley',
            'song_title': 'Black'
            }


response_one = requests.post(url, json=survey_one).json()

response_two = requests.post(url, json=survey_two).json()
print('First response - Future-Mask Off', response_one)
print('Second Response - Dierks Bentley - Black', response_two)