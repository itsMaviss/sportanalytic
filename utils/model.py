from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
#push
def train_model(player_data, match_data):
    # Features include stats for both players
    X = match_data[[
        "Player1_Speed", "Player1_Stamina", "Player1_Agility",
        "Player2_Speed", "Player2_Stamina", "Player2_Agility"
    ]]
    y = match_data["Outcome"]  # Target column: Winner (1 or 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, precision, recall, conf_matrix
