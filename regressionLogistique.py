import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#lire fichier contenant les données
uploaded_file = "cleanVinData.csv"
df = pd.read_csv(uploaded_file, sep =';')

df_x = df.drop(columns=['target'])
df_y = df.target

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

