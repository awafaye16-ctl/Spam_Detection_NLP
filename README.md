# SpamGuard Enterprise

## Vue d'ensemble

SpamGuard Enterprise est une application professionnelle de détection de spam alimentée par l'IA, construite avec Flask et la technologie BERT. Basée sur l'analyse avancée du notebook Spam_Bert.ipynb, cette solution de niveau entreprise fournit une détection de spam en temps réel avec une précision de 98%.

## Fonctionnalités

### Fonctionnalités Principales
- **Analyse BERT**: Traitement du langage naturel de pointe
- **Traitement en Temps Réel**: Temps de réponse inférieur à une seconde
- **Analyse par Lot**: Traiter des milliers de messages simultanément
- **API REST**: Intégration avec les systèmes existants
- **Interface Web Moderne**: Interface utilisateur responsive et professionnelle

### Fonctionnalités Enterprise
- **Haute Performance**: Optimisée pour les opérations à grande échelle
- **Analytics Avancés**: Tableau de bord complet avec informations détaillées
- **Capacités d'Export**: Options d'export CSV et JSON
- **Support Upload Fichiers**: Traitement des fichiers .txt et .csv
- **Gestion d'Erreurs**: Gestion robuste des erreurs et logging
- **Sécurité**: Mesures de sécurité de niveau entreprise

## Architecture Technique

### Technologie du Modèle
- **Modèle de Base**: BERT-base-uncased (HuggingFace Transformers)
- **Fine-tuning**: Entraînement personnalisé sur dataset SMS spam
- **Optimisation**: Gel partiel du modèle pour une inférence efficace
- **Seuil**: Seuil de décision optimisé à 50%

### Métriques de Performance
- **Précision**: 98%
- **Précision (Precision)**: 96.9%
- **Rappel (Recall)**: 74.5%
- **F1-Score**: 70.5%
- **Temps de Traitement**: ~0.9 secondes par message

## Installation

### Prérequis
- Python 3.9+
- Scikit-learn 1.3+
- 8GB+ RAM recommandés
- Support GPU optionnel mais recommandé

### Développement Local

1. **Cloner le dépôt**
```bash
git clone <repository-url>
cd Spam-detection-NLP
```

2. **Créer l'environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger les données NLTK**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

5. **Définir les variables d'environnement**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Lancer l'application**
```bash
python app_simple.py
```

### Déploiement Docker

1. **Construire et lancer avec Docker Compose**
```bash
docker-compose up -d
```

2. **Accéder à l'application**
- Interface Web: http://localhost:5000
- API Health Check: http://localhost:5000/health

## Utilisation

### Interface Web

1. **Analyse de Message Unique**
   - Naviguer vers la page d'accueil
   - Entrer ou coller votre message
   - Cliquer sur "Analyser le Message"
   - Voir les résultats détaillés

2. **Analyse par Lot**
   - Cliquer sur "Analyse par Lot" dans la navigation
   - Uploader un fichier .txt ou .csv
   - Voir les résultats complets et les analytics

3. **Tableau de Bord Analytics**
   - Surveiller les performances du système
   - Voir les tendances de détection de spam
   - Analyser les patterns et métriques

### Utilisation de l'API

#### Prédiction de Message Unique
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Votre message ici"}'
```

#### Prédiction par Lot
```bash
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Message 1", "Message 2", ...]}'
```

### Formats de Fichiers

#### Format TXT
```
Message 1
Message 2
Message 3
...
```

#### Format CSV
```csv
message
"Message 1"
"Message 2"
"Message 3"
...
```

## Configuration

### Variables d'Environnement

| Variable | Défaut | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Environnement Flask |
| `SECRET_KEY` | - | Clé secrète de l'application |
| `MODEL_PATH` | models/simple_spam_model.pkl | Emplacement du fichier modèle |
| `MAX_LEN` | 64 | Longueur maximale de séquence |
| `SEUIL_OPTIMAL` | 0.50 | Seuil de décision |
| `MAX_CONTENT_LENGTH` | 16777216 | Taille max fichier (16MB) |

### Configuration du Modèle

L'application charge automatiquement le modèle et le tokenizer au démarrage. Si aucun modèle pré-entraîné n'est trouvé, elle en crée un nouveau en utilisant la configuration du notebook.

## Développement

### Structure du Projet
```
Spam-detection-NLP/
    app_simple.py          # Application Flask principale
    templates/             # Templates HTML
    static/               # CSS, JS, et assets statiques
    models/               # Fichiers modèle
    uploads/              # Répertoire upload temporaire
    requirements.txt      # Dépendances Python
    Dockerfile            # Docker configuration
    docker-compose.yml    # Docker Compose setup
    .env                  # Environment variables
```

### Composants Clés

1. **Classe SpamDetector**: Logique de détection principale
2. **Endpoints API**: Routes API RESTful
3. **Interface Web**: UI responsive moderne
4. **Tableau de Bord Analytics**: Monitoring en temps réel
5. **Gestion d'Erreurs**: Gestion complète des erreurs

### Tests

```bash
# Exécuter le health check
curl http://localhost:5000/health

# Tester les endpoints API
python -c "
import requests
response = requests.post('http://localhost:5000/api/predict', 
                        json={'text': 'Message de test'})
print(response.json())
"
```

## Déploiement en Production

### Docker Production

1. **Configurer l'environnement de production**
```bash
export FLASK_ENV=production
export SECRET_KEY=votre-clé-secrète-production
```

2. **Déployer avec Docker Compose**
```bash
docker-compose -f docker-compose.yml up -d
```

### Déploiement Manuel

1. **Installer les dépendances de production**
```bash
pip install -r requirements.txt
pip install gunicorn
```

2. **Lancer avec Gunicorn**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app_simple:app
```

### Configuration Nginx

```nginx
server {
    listen 80;
    server_name votre-domaine.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring et Maintenance

### Logging
- Logs application: `spam_detection.log`
- Suivi d'erreurs avec gestion complète des erreurs
- Monitoring des performances avec métriques de timing

### Health Checks
- Endpoint: `/health`
- Surveille le chargement du modèle, le statut du tokenizer et la santé système
- Redémarrage automatique en cas d'échec dans environnement Docker

### Optimisation des Performances
- Cache du modèle pour une inférence plus rapide
- Traitement par lot pour les opérations à grand volume
- Gestion de données efficace en mémoire

## Sécurité

### Protection des Données
- Pas de stockage persistant de données sensibles
- Nettoyage automatique des fichiers temporaires après traitement
- Validation sécurisée des uploads de fichiers

### Contrôle d'Accès
- Limitation de débit configurable
- Configuration de politique CORS
- Gestion sécurisée des sessions

## Dépannage

### Problèmes Courants

1. **Erreurs de Chargement du Modèle**
   - Vérifier la compatibilité de version scikit-learn
   - Vérifier les permissions du fichier modèle
   - Assurer suffisamment de mémoire disponible

2. **Performance Lente**
   - Considérer l'accélération GPU
   - Optimiser la taille de batch
   - Vérifier les ressources système

3. **Problèmes d'Upload de Fichiers**
   - Vérifier le format de fichier (.txt, .csv)
   - Vérifier la limite de taille de fichier (16MB)
   - Assurer un encodage correct (UTF-8)

### Support

Pour support technique ou questions:
1. Vérifier les logs application
2. Vérifier les paramètres de configuration
3. Tester avec des données exemples
4. Consulter la documentation API

## Licence

Ce projet est basé sur la recherche académique et doit être utilisé en conformité avec les licences et réglementations applicables.

## Contributions

1. Fork le dépôt
2. Créer une branche de fonctionnalité
3. Soumettre une pull request
4. Suivre les standards de codage

---

**SpamGuard Enterprise** - Détection de spam avancée alimentée par l'IA pour les communications d'entreprise.
#   S p a m _ D e t e c t i o n _ N L P  
 