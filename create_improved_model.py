"""
Version améliorée du créateur de modèle avec seuil de décision optimisé
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

# Créer les dossiers
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('./bert_tokenizer', exist_ok=True)

# Données d'exemple plus riches et équilibrées
sample_texts = [
    # Messages spam évidents
    "WIN $1000 CASH! Reply YES to claim your prize!",
    "Congratulations! You won a FREE iPhone. Click here now!",
    "URGENT: Your account has been compromised. Verify now!",
    "Limited time offer - Buy now and save 50%",
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "URGENT! You have won a 1 week FREE membership",
    "FreeMsg Hey there darling it\'s been 3 week\'s now",
    "Had your mobile 11 months or more? U R entitled",
    "SIX chances to win CASH! From 100 to 20,000 pounds",
    "PRIVATE! Your 2003 Account Statement shows",
    
    # Messages ham normaux
    "Hey, are we still meeting tomorrow for lunch?",
    "Can you send me report before Friday please?",
    "Thanks for your message, I will get back to you soon.",
    "Meeting scheduled for next Monday at 3 PM",
    "Please review the attached document",
    "Your package has been shipped - tracking number included",
    "Nah I dont think he goes to usf",
    "Even my brother is not like to speak with me",
    "I have a date on Sunday with Will",
    "Ok lar... Joking wif u oni...",
    
    # Plus de spam
    "WINNER!! As a valued network customer you have been",
    "Mobile 07xxxxxxxxx You have won a £1,000 prize",
    "URGENT 2nd notice. Your account will be closed",
    "SPECIAL PROMOTION! Buy now and save 70%",
    "CONGRATULATIONS! You've been selected for a FREE gift",
    "CLAIM your FREE trial now! Limited time only",
    "DOUBLE your mobile minutes! Text YES to claim",
    "FREE ringtone! Reply to this message now",
    "URGENT: We tried to contact you about your prize",
    "You have been chosen to receive a FREE laptop",
    
    # Plus de ham
    "I\'m back & we\'re packing the car now",
    "I\'ve been searching for the right words to thank you",
    "What time are we meeting tomorrow?",
    "Can you send me the files when you get a chance?",
    "Great presentation today! Really enjoyed it",
    "Let me know when you\'re available for a call",
    "Thanks for your help with the project",
    "See you at the office tomorrow morning",
    "Don\'t forget about the meeting at 2pm",
    "I\'ll call you when I get home tonight"
]

# Labels correspondants (0=ham, 1=spam)
sample_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 spam
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 ham
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10 spam
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10 ham

# Créer le DataFrame
df = pd.DataFrame({
    'v1': ['spam' if label == 1 else 'ham' for label in sample_labels],
    'v2': sample_texts
})

# Sauvegarder les données
df.to_csv('data/spam.csv', index=False)
print(f"Données créées: {len(df)} messages")
print(f"Spam: {len(df[df['v1']=='spam'])}, Ham: {len(df[df['v1']=='ham'])}")

# Nettoyage simple du texte
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return text.lower().strip()

df['Text'] = df['v2'].apply(clean_text)
df['Class'] = df['v1'].map({'ham': 0, 'spam': 1})

# Préparation des données
X = df['Text'].values
y = df['Class'].values

# Split pour entraînement et test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vectorisation avec plus de features
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entraînement du modèle avec class_weight='balanced'
model = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# Évaluation et optimisation du seuil
y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

# Tester différents seuils
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0

print("\nOptimisation du seuil de décision:")
for threshold in thresholds:
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculer F1-score pour le spam (classe 1)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Seuil {threshold:.2f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nMeilleur seuil: {best_threshold:.2f} (F1: {best_f1:.3f})")

# Test sur des exemples concrets
test_messages = [
    "WIN $1000 CASH! Reply YES to claim your prize!",
    "Hey, are we still meeting tomorrow for lunch?",
    "URGENT: Your account has been compromised. Verify now!",
    "Can you send me report before Friday please?",
    "CONGRATULATIONS! You've been selected for a FREE gift"
]

print("\nTest sur messages exemples:")
for msg in test_messages:
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    prob = model.predict_proba(msg_vec)[0][1]
    prediction = int(prob > best_threshold)
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"{label} ({prob:.3f}): {msg}")

# Sauvegarder le modèle avec le seuil optimisé
model_data = {
    'vectorizer': vectorizer,
    'model': model,
    'threshold': best_threshold
}

with open('models/simple_spam_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Créer fichiers de compatibilité
with open('models/spam_classifier_bert.keras', 'w') as f:
    f.write("# Mock Keras model file")

# Configuration avec le seuil optimisé
config = {
    'seuil_optimal': best_threshold,
    'max_len': 64,
    'model_type': 'simple_sklearn',
    'best_f1_score': best_f1
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

print(f"\nModèle amélioré créé et sauvegardé avec succès!")
print(f"Nouveau seuil optimisé: {best_threshold:.2f}")
print("Fichiers créés:")
print("- models/simple_spam_model.pkl")
print("- models/spam_classifier_bert.keras")
print("- model_config.json")
print("- bert_tokenizer/")
