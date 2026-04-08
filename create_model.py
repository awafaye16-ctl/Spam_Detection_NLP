"""
Script pour créer et entraîner le modèle BERT basé sur le notebook Spam_Bert.ipynb
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings("ignore")

# Configuration
class ModelConfig:
    MAX_LEN = 64
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 3e-5
    MODEL_PATH = 'models/spam_classifier_bert.keras'
    TOKENIZER_PATH = './bert_tokenizer'
    CONFIG_PATH = 'model_config.json'
    SEUIL_OPTIMAL = 0.65
    DATA_URL = 'data/spam.csv'

class SpamModelTrainer:
    def __init__(self):
        self.config = ModelConfig()
        self.tokenizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self.porter = PorterStemmer()
        
        # Créer les dossiers nécessaires
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
    def download_nltk_data(self):
        """Télécharger les données NLTK nécessaires"""
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Téléchargement des données NLTK...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
    
    def load_and_prepare_data(self):
        """Charger et préparer les données"""
        print("Chargement des données...")
        
        # Si le fichier n'existe pas, créer des données d'exemple
        if not os.path.exists(self.config.DATA_URL):
            print("Création de données d'exemple...")
            self.create_sample_data()
        
        # Charger les données
        try:
            df = pd.read_csv(self.config.DATA_URL, encoding='latin-1')
        except:
            # Essayer un autre encodage
            df = pd.read_csv(self.config.DATA_URL, encoding='utf-8')
        
        # Nettoyage des colonnes
        if 'Unnamed: 2' in df.columns:
            df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        
        # Renommage
        df.rename(columns={'v1': 'Class', 'v2': 'Text'}, inplace=True)
        
        # Encodage
        df['Class'] = df['Class'].map({'ham': 0, 'spam': 1})
        
        # Nettoyage du texte
        df['Text'] = df['Text'].apply(self.clean_text)
        
        print(f"Données chargées: {len(df)} messages")
        print(f"Spam: {len(df[df['Class']==1])}, Ham: {len(df[df['Class']==0])}")
        
        return df
    
    def create_sample_data(self):
        """Créer des données d'exemple pour tester"""
        sample_data = {
            'v1': ['ham', 'spam', 'ham', 'spam', 'ham'] * 100,
            'v2': [
                'Hey, are we still meeting tomorrow for lunch?',
                'Congratulations! You won a FREE iPhone. Click here now!',
                'Can you send me the report before Friday please?',
                'URGENT: Your account has been compromised. Verify now!',
                'Thanks for your message, I will get back to you soon.'
            ] * 100
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(self.config.DATA_URL, index=False)
        print(f"Données d'exemple créées: {self.config.DATA_URL}")
    
    def clean_text(self, text):
        """Nettoyer le texte"""
        if pd.isna(text):
            return ""
        
        # Nettoyage de base
        text = re.sub("[^a-zA-Z]", " ", str(text))
        text = text.lower().strip()
        
        # Suppression stopwords et stemming
        words = [self.porter.stem(word) for word in text.split() 
                if word not in self.stop_words]
        
        return " ".join(words)
    
    def initialize_tokenizer(self):
        """Initialiser le tokenizer BERT"""
        print("Initialisation du tokenizer BERT...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Sauvegarder le tokenizer
        self.tokenizer.save_pretrained(self.config.TOKENIZER_PATH)
        print(f"Tokenizer sauvegardé dans: {self.config.TOKENIZER_PATH}")
    
    def encode_texts(self, texts):
        """Encoder les textes pour BERT"""
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.config.MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            input_ids.append(encoded['input_ids'][0])
            attention_masks.append(encoded['attention_mask'][0])
        
        return np.array(input_ids), np.array(attention_masks)
    
    def build_model(self):
        """Construire le modèle BERT"""
        print("Construction du modèle BERT...")
        
        # Charger BERT
        bert_model = TFAutoModel.from_pretrained('bert-base-uncased')
        
        # Geler les premières couches
        bert_model.trainable = True
        for layer in bert_model.layers[:-4]:
            layer.trainable = False
        
        # Architecture
        input_word_ids = tf.keras.Input(shape=(self.config.MAX_LEN,), dtype='int32')
        attention_masks = tf.keras.Input(shape=(self.config.MAX_LEN,), dtype='int32')
        
        sequence_output = bert_model([input_word_ids, attention_masks])
        output = sequence_output[1]  # pooler_output
        
        # Tête de classification
        output = tf.keras.layers.Dense(64, activation='relu')(output)
        output = tf.keras.layers.Dropout(0.3)(output)
        output = tf.keras.layers.Dense(32, activation='relu')(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        
        model = tf.keras.models.Model(
            inputs=[input_word_ids, attention_masks],
            outputs=output
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        print("Modèle construit avec succès")
        return model
    
    def train_model(self, X_train_ids, X_train_masks, y_train, 
                   X_val_ids, X_val_masks, y_val):
        """Entraîner le modèle"""
        print("Début de l'entraînement...")
        
        # Calculer les poids des classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Poids des classes: {class_weight_dict}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=3,
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entraînement
        history = self.model.fit(
            [X_train_ids, X_train_masks], y_train,
            validation_data=([X_val_ids, X_val_masks], y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        return history
    
    def save_model_and_config(self):
        """Sauvegarder le modèle et la configuration"""
        print("Sauvegarde du modèle...")
        
        # Sauvegarder le modèle
        self.model.save(self.config.MODEL_PATH)
        print(f"Modèle sauvegardé: {self.config.MODEL_PATH}")
        
        # Sauvegarder la configuration
        config_dict = {
            'seuil_optimal': self.config.SEUIL_OPTIMAL,
            'max_len': self.config.MAX_LEN,
            'batch_size': self.config.BATCH_SIZE,
            'epochs': self.config.EPOCHS,
            'learning_rate': self.config.LEARNING_RATE
        }
        
        with open(self.config.CONFIG_PATH, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration sauvegardée: {self.config.CONFIG_PATH}")
    
    def evaluate_model(self, X_test_ids, X_test_masks, y_test):
        """Évaluer le modèle"""
        print("Évaluation du modèle...")
        
        # Prédictions
        y_pred_proba = self.model.predict([X_test_ids, X_test_masks], 
                                         batch_size=self.config.BATCH_SIZE)
        y_pred = (y_pred_proba > self.config.SEUIL_OPTIMAL).astype(int).flatten()
        
        # Métriques
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Ham', 'Spam']))
        
        return accuracy
    
    def run_training(self):
        """Lancer l'entraînement complet"""
        print("=" * 50)
        print("ENTRAÎNEMENT DU MODÈLE BERT SPAM DETECTION")
        print("=" * 50)
        
        # Télécharger NLTK
        self.download_nltk_data()
        
        # Charger les données
        df = self.load_and_prepare_data()
        
        # Split des données
        X = df['Text'].values
        y = df['Class'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Initialiser tokenizer
        self.initialize_tokenizer()
        
        # Encoder les textes
        print("Encodage des textes...")
        X_train_ids, X_train_masks = self.encode_texts(X_train)
        X_val_ids, X_val_masks = self.encode_texts(X_val)
        X_test_ids, X_test_masks = self.encode_texts(X_test)
        
        # Construire le modèle
        self.build_model()
        
        # Entraîner
        history = self.train_model(X_train_ids, X_train_masks, y_train,
                                X_val_ids, X_val_masks, y_val)
        
        # Évaluer
        accuracy = self.evaluate_model(X_test_ids, X_test_masks, y_test)
        
        # Sauvegarder
        self.save_model_and_config()
        
        print("=" * 50)
        print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        print(f"Accuracy finale: {accuracy:.4f}")
        print(f"Modèle sauvegardé: {self.config.MODEL_PATH}")
        print("=" * 50)
        
        return accuracy

if __name__ == '__main__':
    trainer = SpamModelTrainer()
    trainer.run_training()
