import os
import numpy as np
import pandas as pd


def generate_synthetic_data(num_players=100, num_games=50):
    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Simulate player stats (points, assists, rebounds)
    np.random.seed(42)

    player_ids = np.arange(1, num_players + 1)
    games = np.arange(1, num_games + 1)

    data = []
    for player in player_ids:
        for game in games:
            points = np.random.normal(loc=15, scale=5)  # Mean = 15, SD = 5
            assists = np.random.normal(loc=5, scale=2)  # Mean = 5, SD = 2
            rebounds = np.random.normal(loc=8, scale=3)  # Mean = 8, SD = 3
            data.append([player, game, points, assists, rebounds])

    df = pd.DataFrame(data, columns=['PlayerID', 'Game', 'Points', 'Assists', 'Rebounds'])
    df.to_csv('data/synthetic_data.csv', index=False)
    return df
