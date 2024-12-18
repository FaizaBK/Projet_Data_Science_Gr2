# Project Data Science  [Site](https://projetdatasciencegr2-machinelearning.streamlit.app/)



## Collaborateurs
  - Faiza MNASRI
  - Virginie GIACOMETTI
  - AlaeEN NASYRY
  - Alexis GUIZARD 

### Description :


#### Cloner le dépôt :

```bash
git clone https://github.com/FaizaBK/Projet_Data_Science_Gr2.git
```

#### Créer et activer un environnement virtuel :

```python

python -m venv .venv

```

Activation :

Sur Windows :

```bash

.\.venv\Scripts\activate

```

Sur macOS/Linux :
```bash

source .venv/bin/activate

```

#### Désactivation :
```bash

desactivate

```

#### Installer les dépendances :

Activer l environnement
```bash
pip install -r ./requirements.txt

```



### Accès à streamlit

```bash
streamlit run app.py 

```




### Structure du projet 

```

Projet_Data_Science_Gr2/
│
├── .venv/                  # Environnement virtuel
│
├── .env                    # Variables d'environnement
│   ├── ROBOFLOW_API_URL="https://detect.roboflow.com"
│   ├── ROBOFLOW_API_KEY="votre_clés_api"
│
├── data/                   # Données en csv
│   ├── diabete.csv         # Fichier initial
│   ├── vin.csv             # Fichier initial
│      
├── section/                   
│   ├── regression
│         ├── regression.py       
|                   
│   ├── classification      
│         ├── classification.py      │      
|
├── nailsdetection/                    
│         ├── nail.py      
|        

```






