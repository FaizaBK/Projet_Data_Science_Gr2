import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorFlow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#lire fichier contenant les données
uploaded_file = "cleanVinData.csv"
df = pd.read_csv(uploaded_file, sep =';')

df_x = df.drop(columns=['target'])
df_y = df.target

def regression():
    #séparer le jeu de données en train et test
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=42)

    #Entrainer un modèle de regression logistique
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    print(y_pred)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

def decisionTree():
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=42)

    # Créer et entraîner un modèle d'arbre de décision
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Prédire sur les données de test
    y_pred = model.predict(X_test)

    # Évaluer la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy:.2f}")

    features = df.columns[:-1]  # Toutes les colonnes sauf la dernière (qui contient la cible)
    target_name = ['amer','doux','equilibre']  # Définir manuellement les noms des classes

    # Visualiser l'arbre de décision
    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names= features, class_names= target_name, filled=True)
    plt.show()

def tensorflow():
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3, random_state=42)

    # Définition des couches du modèle
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Couche initiale
        Dropout(0.2),  # Couche de régularisation
        Dense(64, activation='relu'),  # Deuxième couche dense
        Dropout(0.2),
        Dense(len(np.unique(y_train)), activation='softmax')  # Couche de sortie
    ])

    # Compilation du modèle
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Entraînement du modèle
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Évaluation sur l'ensemble de test
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Faire des prédictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Afficher les premières prédictions
    print("Predicted classes:", predicted_classes[:10])
    print("True classes:", y_test[:10])

    # Récupération des données de l'entraînement
    history_dict = history.history

    # Afficher la précision d'entraînement et de validation
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Afficher la perte d'entraînement et de validation
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train Loss', marker='o')
    plt.plot(history_dict['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Montrer les graphiques
    plt.tight_layout()
    plt.show()

    # 1. Affichage de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, predicted_classes)

    # Affichage de la matrice de confusion sous forme de graphique
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    # 2. Comparaison des résultats attendus et prédits sous forme de graphique
    plt.figure(figsize=(12, 6))
    indices = np.arange(20)  # Nombre de points à afficher
    plt.bar(indices - 0.2, y_test[:20], 0.4, label="Expected", color='blue')
    plt.bar(indices + 0.2, predicted_classes[:20], 0.4, label="Predicted", color='orange')
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Comparison of Expected vs Predicted Results")
    plt.xticks(indices)
    plt.legend()
    plt.grid(True)
    plt.show()