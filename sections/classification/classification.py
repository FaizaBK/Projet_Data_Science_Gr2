import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, classification_report, confusion_matrix



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
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display the results
        st.metric(label="**Précision (Accuracy)**", value=f"{accuracy:.2%}")
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Prédictions", fontsize=10)
        ax.set_ylabel("Valeurs réelles", fontsize=10)
        ax.set_title("Matrice de confusion", fontsize=14)
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
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display the results
        st.metric(label="**Précision (Accuracy)**", value=f"{accuracy:.2%}")

        row1Col1,row1Col2 = st.columns(2)
        with row1Col1:
            # Rapport de classification
            st.write("### Rapport de classification")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

        with row1Col2:
            # Matrice de confusion
            st.write("### Matrice de confusion")
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
            st.pyplot(fig)

        
        row2Col1,row2Col2 = st.columns(2)
        with row2Col1:
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

        with row2Col2:
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


def cross_validation_random_forest():
    """Validation croisée pour une Forêt Aléatoire"""
    st.header("Validation Croisée : Forêt Aléatoire")
    data = load_data()
    if data is not None:
        if "target" not in data.columns:
            st.error("La colonne 'target' est manquante dans le fichier.")
            return

        # Split the data into features (X) and target (y)
        X = data.drop(columns=['target'])
        y = data['target']

        # Model RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Applicate  5-fold cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        row1col1, row1col2 = st.columns(2)
        # Display results
        results_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(scores))],
            "Accuracy": scores
        })
        average_accuracy = scores.mean()
        std_dev_accuracy = scores.std()
        
        with row1col1:
            st.dataframe(results_df)
        with row1col2:
            st.write(f"**Average Accuracy:** {average_accuracy:.2%}")
            st.write(f"**Std Accurancy:** {std_dev_accuracy:.2%}")
            
         # Plot of the scores
        st.write("### Accuracy per Fold")
        fig, ax = plt.subplots()
        sns.barplot(x=results_df["Fold"], y=results_df["Accuracy"], palette="viridis", ax=ax)
        ax.axhline(y=average_accuracy, color="red", linestyle="--", label="Mean Accuracy")
        ax.set_title("Cross-Validation Accuracy")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)
 
def decision_tree_model():
    """Train and visualize a Decision Tree model"""
    st.header("Arbre de Décision")
    data = load_data()
    if data is not None:
        if "target" not in data.columns:
            st.error("La colonne 'target' est manquante dans le fichier.")
            return

        # Split the data
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the Decision Tree
        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Display results
        st.metric(label="**Précision (Accuracy)**", value=f"{accuracy:.2%}")

        # Rapport de classification
        st.write("### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

        # Matrice de confusion
        st.write("### Matrice de confusion")
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
        st.pyplot(fig)

        # Visualisation de l'arbre
        st.write("### Visualisation de l'Arbre de Décision")
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(model, feature_names=X.columns, class_names=model.classes_.astype(str), filled=True, ax=ax)
        st.pyplot(fig)   
            
#Classification page in streamlit   
def classification_page():
    st.header("Classification")
    
    if st.sidebar.button("Afficher les informations sur le DataFrame"):
        display_dataframe_info()

    options = st.sidebar.selectbox(
        "Choisissez un modèle",
        ["","Régression Logistique","Forêt Aléatoire","Validation Croisée_Forêt Aléatoire" ,"Arbre de Décision", "Tensorflow"], 
        format_func=lambda x: "Sélectionnez un modèle" if x == "" else x,
    )

    if options == "Régression Logistique":
        logistic_regression_model()
    elif  options == "Forêt Aléatoire":
        random_forest_model()
    elif options == "Validation Croisée_Forêt Aléatoire":
        cross_validation_random_forest()
    elif options == "Arbre de Décision":
        decision_tree_model()
   