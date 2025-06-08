import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Nom des classes CIFAR-10
classes = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Chargement des données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisation des pixels entre 0 et 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding des étiquettes
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Création du modèle (réseau dense simple)
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),  # Aplatir les images 32x32x3
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes de sortie
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement du modèle
model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)

# Évaluation sur les données de test
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
print("Précision sur les données de test :", test_accuracy)

# Fonction de prédiction interactive
def predire_image():
    while True:
        entree = input("Entrez un numéro d'image entre 0 et 9999 (ou 'exit' pour quitter) : ")
        if entree.lower() == 'exit':
            break
        try:
            index = int(entree)
            if 0 <= index < len(x_test):
                image = x_test[index]
                prediction = model.predict(np.expand_dims(image, axis=0))
                classe_predite = np.argmax(prediction)
                nom_classe = classes[classe_predite]

                print(f"Classe prédite : {nom_classe} ({classe_predite})")

                plt.imshow(image)
                plt.title(f"Classe prédite : {nom_classe}")
                plt.axis('off')
                plt.show()
            else:
                print("Veuillez entrer un nombre entre 0 et 9999.")
        except ValueError:
            print("Entrée invalide. Entrez un nombre ou 'exit'.")

# Lancer la prédiction
predire_image()
