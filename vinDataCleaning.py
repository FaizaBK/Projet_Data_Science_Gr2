import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

#lire fichier contenant les données
uploaded_file = "data/vin.csv"
df = pd.read_csv(uploaded_file)

#extraire les variable explicatives
df_ex = df.drop(columns = ['target', 'Unnamed: 0'])

#standariser les variables explicatives
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ex)
scaled_df = pd.DataFrame(scaled_data, columns = df_ex.columns) 

#print(scaled_df.shape)

#Encoder la target avec labelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df.target).astype(int)
y_encoded = pd.DataFrame(y_encoded, columns = ['target'])

#Concatener scaled_df et y_encoded
clean_df = pd.concat([scaled_df,y_encoded], axis =1)

#pour chaque colonne appliquer isolation forest pour éliminer les valeurs extremes
for e in scaled_df.columns:
    #generer un dataframe avec une colonne de zeros
    y = pd.DataFrame({'zeros': [0.0] * clean_df.shape[0]})
    
    #construir un vecteur de deux colonnes pour l'entrée du modèle isolation forest
    X = pd.concat([y,clean_df[e]], axis =1)
    
    #Entrainer un modèle isolation forest
    model = IsolationForest(contamination=0.02)
    model.fit(X)

    #utiliser le modèle pour prédire les valeurs extremes
    X_pred=model.predict(X)
    
    #filtrer notre clean_df en utilisant la sortie du modèle isolation forest
    clean_df = clean_df[X_pred == 1]
    #redémarer l'index
    clean_df = clean_df.reset_index(drop = True)

#Exporter le résultat dans un fichier csv
clean_df.to_csv('cleanVinData.csv', sep=';', index=False)
