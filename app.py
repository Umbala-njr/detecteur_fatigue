import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# === Génération IA ===
def generer_donnees(n=300):
    np.random.seed(42)
    distance = np.random.normal(8, 2, n)
    sprints = np.random.randint(0, 30, n)
    temps = np.random.randint(30, 95, n)
    bpm = np.random.randint(80, 190, n)

    def etat(d, s, t, b):
        if d < 6 and s < 5 and b > 160:
            return "Fatigué"
        elif d > 10 and s > 20 and b < 150:
            return "En forme"
        else:
            return "Correct"

    etats = [etat(d, s, t, b) for d, s, t, b in zip(distance, sprints, temps, bpm)]
    return pd.DataFrame({
        "distance_km": distance,
        "sprints": sprints,
        "temps_jeu_min": temps,
        "bpm": bpm,
        "etat_physique": etats
    })

df_train = generer_donnees()
X = df_train[["distance_km", "sprints", "temps_jeu_min", "bpm"]]
y = df_train["etat_physique"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# === Initialisation interface ===
st.set_page_config("Multijoueur IA - État Physique")
st.title(" Mode Multijoueur - Test État Physique")

# === Charger joueurs ===
try:
    joueurs_df = pd.read_csv("joueurs.csv")
except:
    st.error("⚠️ Le fichier 'joueurs.csv' est manquant.")
    st.stop()

joueur_select = st.selectbox("👤 Sélectionnez un joueur :", joueurs_df["nom"].tolist())

# === Simuler capteurs ===
st.subheader("📡 Données capteur simulées")
if st.button("📥 Obtenir les données du capteur"):
    distance = round(np.random.uniform(4, 13), 1)
    sprints = np.random.randint(0, 30)
    temps = np.random.randint(30, 95)
    bpm = np.random.randint(85, 190)

    st.write(f"📏 Distance : {distance} km")
    st.write(f"⚡ Sprints : {sprints}")
    st.write(f"⏱️ Temps de jeu : {temps} min")
    st.write(f"❤️ BPM : {bpm}")

    # Analyse IA
    prediction = model.predict([[distance, sprints, temps, bpm]])
    etat = prediction[0]

    # Couleur dynamique
    if etat == "Fatigué":
        st.error(f"🟥 {joueur_select} est **{etat}**")
    elif etat == "En forme":
        st.success(f"🟩 {joueur_select} est **{etat}**")
    else:
        st.warning(f"🟨 {joueur_select} est **{etat}**")

    # Historique joueur (session par nom)
    if f"hist_{joueur_select}" not in st.session_state:
        st.session_state[f"hist_{joueur_select}"] = []

    st.session_state[f"hist_{joueur_select}"].append({
        "Distance (km)": distance,
        "Sprints": sprints,
        "Temps (min)": temps,
        "BPM": bpm,
        "État": etat
    })

    # Radar
    st.subheader(" Radar Performance")
    categories = ["Distance", "Sprints", "Temps", "BPM"]
    valeurs = [distance / 13.0, sprints / 30.0, temps / 95.0, bpm / 190.0]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    valeurs += valeurs[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, valeurs, "o-", linewidth=2)
    ax.fill(angles, valeurs, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title(joueur_select)
    st.pyplot(fig)

# === Historique par joueur
if f"hist_{joueur_select}" in st.session_state:
    st.subheader(" Historique")
    st.dataframe(pd.DataFrame(st.session_state[f"hist_{joueur_select}"]))
    
    # Export CSV
    csv_df = pd.DataFrame(st.session_state[f"hist_{joueur_select}"])
    csv = csv_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📤 Télécharger l'historique (CSV)",
        data=csv,
        file_name=f"historique_{joueur_select}.csv",
        mime='text/csv'
    )

