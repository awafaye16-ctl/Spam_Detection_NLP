"""
Version ultra-simple et rapide du créateur de modèle
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re

# Créer les dossiers
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('./bert_tokenizer', exist_ok=True)

# Données d'exemple
sample_texts = [
    "Hey, are we still meeting tomorrow for lunch?",
    "Congratulations! You won a FREE iPhone. Click here now!",
    "Can you send me report before Friday please?",
    "URGENT: Your account has been compromised. Verify now!",
    "Thanks for your message, I will get back to you soon.",
    "WIN $1000 CASH! Reply YES to claim your prize!",
    "Meeting scheduled for next Monday at 3 PM",
    "Limited time offer - Buy now and save 50%",
    "Please review the attached document",
    "Your package has been shipped - tracking number included",
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Nah I dont think he goes to usf",
    "Even my brother is not like to speak with me",
    "I have a date on Sunday with Will",
    "URGENT! You have won a 1 week FREE membership"
]

sample_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1] * 50

# Créer le DataFrame
df = pd.DataFrame({
    'v1': ['ham' if label == 0 else 'spam' for label in sample_labels],
    'v2': sample_texts
})

# Sauvegarder les données
df.to_csv('data/spam.csv', index=False)
print(f"Données créées: {len(df)} messages")

# Nettoyage simple du texte
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return text.lower().strip()

df['Text'] = df['v2'].apply(clean_text)
df['Class'] = df['v1'].map({'ham': 0, 'spam': 1})

# Préparation des données
X = df['Text'].values
y = df['Class'].values

# Vectorisation
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Entraînement du modèle
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_vec, y)

# Sauvegarder le modèle
model_data = {
    'vectorizer': vectorizer,
    'model': model
}

with open('models/simple_spam_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Créer fichiers de compatibilité
with open('models/spam_classifier_bert.keras', 'w') as f:
    f.write("# Mock Keras model file")

# Configuration
config = {
    'seuil_optimal': 0.65,
    'max_len': 64,
    'model_type': 'simple_sklearn'
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Tokenizer files
tokenizer_config = {
    "do_lower_case": True,
    "model_max_length": 64
}

with open('./bert_tokenizer/tokenizer_config.json', 'w') as f:
    json.dump(tokenizer_config, f, indent=2)

with open('./bert_tokenizer/vocab.txt', 'w') as f:
    f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")

print("Modèle créé et sauvegardé avec succès!")
print("Fichiers créés:")
print("- models/simple_spam_model.pkl")
print("- models/spam_classifier_bert.keras")
print("- model_config.json")
print("- bert_tokenizer/")
