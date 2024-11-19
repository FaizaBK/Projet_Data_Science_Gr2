import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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
y_encoded = pd.DataFrame(y_encoded)

#Concatener X et y
clean_df = pd.concat([scaled_df,y_encoded], axis =1)

clean_df.to_csv('cleanVinData.csv', sep=';', index=False)