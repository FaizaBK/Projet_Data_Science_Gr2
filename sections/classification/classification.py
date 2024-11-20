import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix



# Load the data
def load_data(data_path="data/cleanVinData.csv"):
    """Load data and delete unnecessary columns"""
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, sep=';')
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        st.session_state.data = data
        return data
    else:
        st.error("Le fichier est introuvable.")
        return None

# Display DataFrame information
def display_dataframe_info():
    """Display basic information about the DataFrame"""
    st.header("Page Info DataFrame")
    st.caption("Informations sur le fichier")
    data = load_data()
    if data is not None:
        st.write("Voici les 5 premières lignes du fichier :")
        st.dataframe(data.head())
        st.write(f"Nombre de colonnes : {data.shape[1]}")
        st.write(f"Nombre de lignes : {data.shape[0]}")
        st.write("Résumé des statistiques :")
        st.dataframe(data.describe())
        st.write("Valeurs manquantes ou nulles :")
        st.dataframe(data.isna().sum())
        st.session_state.data = data
    else:
        st.error("Le fichier est introuvable.")
        
# Train logistic regression model
def logistic_regression_model():
    """Train and evaluate a Logistic Regression model"""
    st.header("Régression Logistique")
    data = load_data()
    if data is not None:
        if "target" not in data.columns:
            st.error("La colonne 'target' est manquante dans le fichier.")
            return
        
        # Split the data
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

       # Train the model
        st.write("Entraînement du modèle...")
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display the results
        st.write(f"**Accuracy :** {accuracy:.2f}")
        st.write("**Rapport de classification :**")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        st.write("**Matrice de confusion :**")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Réelles")
        st.pyplot(fig)
        
# Train random forest model
def random_forest_model():
    """Train and evaluate a Random Forest model"""
    st.header("Forêt Aléatoire")
    data = load_data()
    if data is not None:
        if "target" not in data.columns:
            st.error("La colonne 'target' est manquante dans le fichier.")
            return
        
        # Split the datas
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Train the model
        st.write("Entraînement du modèle...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display the results
        st.write(f"**Accuracy :** {accuracy:.2f}")
        st.write("**Rapport de classification :**")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        st.write("**Matrice de confusion :**")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Réelles")
        st.pyplot(fig)

        # Feature importance charcterstics
        st.write("**Importance des caractéristiques :**")
        feature_importances = model.feature_importances_
        features = X.columns
        sorted_idx = feature_importances.argsort()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(features[sorted_idx], feature_importances[sorted_idx], color='green')
        ax.set_xlabel("Importance")
        ax.set_title("Importance des caractéristiques")
        st.pyplot(fig)


        # Precision by class
        st.write("**Précision par classe :**")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        precision = [class_report[str(i)]['precision'] for i in range(len(class_report) - 3)]  # exclude the accuracy, macro avg, and weighted avg
        labels = [f"Classe {i}" for i in range(len(precision))]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, precision, color='blue')
        ax.set_xlabel("Classe")
        ax.set_ylabel("Précision")
        ax.set_title("Précision par Classe")
        st.pyplot(fig)

#Classification page in streamlit   
def classification_page():
    st.header("Bienvenue")
    st.caption("Playground pour les modèles de classification")
    
    if st.sidebar.button("Afficher les informations sur le DataFrame"):
        display_dataframe_info()

    options = st.sidebar.selectbox(
        "Choisissez un modèle",
        ["Régression Logistique","Forêt Aléatoire", "Arbre de Décision", "Réseau de Neurones"], 
        format_func=lambda x: "Sélectionnez un modèle" if x == "" else x,
    )

    if options == "Régression Logistique":
        logistic_regression_model()
    elif  options == "Forêt Aléatoire":
        random_forest_model()
    
    