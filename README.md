# 🏃‍♂️ Application IA - État Physique des Joueurs (Multijoueur)

Cette application **Streamlit** permet de simuler les données physiques de plusieurs joueurs et de prédire leur **état physique** grâce à un modèle **Random Forest** entraîné sur des données synthétiques.

## 🚀 Fonctionnalités

- Sélection d’un joueur à partir d’un fichier `joueurs.csv`
- Génération aléatoire de données physiques simulées :
  - Distance parcourue (km)
  - Nombre de sprints
  - Temps de jeu (min)
  - Fréquence cardiaque (BPM)
- Prédiction de l'état physique :
  - 🟥 **Fatigué**
  - 🟨 **Correct**
  - 🟩 **En forme**
- Visualisation **radar** de la performance
- Historique par joueur enregistré en session
- Export de l’historique en fichier **CSV**

## 🔧 Prérequis

Installe les dépendances requises avec :

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
