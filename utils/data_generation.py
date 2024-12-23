import numpy as np
import pandas as pd

def generate_player_data(num_players=50):
    np.random.seed(42)
    data = {
        "PlayerID": range(1, num_players + 1),
        "Speed": np.random.normal(6, 1.5, num_players),
        "Stamina": np.random.normal(7, 1, num_players),
        "Agility": np.random.normal(5.5, 1.2, num_players),
        "WinRate": np.random.uniform(30, 90, num_players)
    }
    return pd.DataFrame(data)

def generate_match_data(player_data, num_matches=500):
    matches = []
    for _ in range(num_matches):
        player1, player2 = player_data.sample(2).to_dict('records')
        winner = 1 if player1['WinRate'] > player2['WinRate'] else 0
        matches.append({
            "Player1_Speed": player1['Speed'],
            "Player1_Stamina": player1['Stamina'],
            "Player1_Agility": player1['Agility'],
            "Player2_Speed": player2['Speed'],
            "Player2_Stamina": player2['Stamina'],
            "Player2_Agility": player2['Agility'],
            "Outcome": winner
        })
    return pd.DataFrame(matches)
