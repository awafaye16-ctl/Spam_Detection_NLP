# SpamGuard Enterprise

## Vue d'ensemble

SpamGuard Enterprise est une application professionnelle de detection de spam alimentee par l'IA, construite avec Flask et la technologie BERT. Basee sur l'analyse avancee du notebook Spam_Bert.ipynb, cette solution de niveau entreprise fournit une detection de spam en temps reel avec une precision de 98%.

## Fonctionnalites

### Fonctionnalites Principales
- **Analyse BERT**: Traitement du langage naturel de pointe
- **Traitement en Temps Reel**: Temps de reponse inferieur a une seconde
- **Analyse par Lot**: Traiter des milliers de messages simultanement
- **API REST**: Integration avec les systemes existants
- **Interface Web Moderne**: Interface utilisateur responsive et professionnelle

### Fonctionnalites Enterprise
- **Haute Performance**: Optimisee pour les operations a grande echelle
- **Analytics Avances**: Tableau de bord complet avec informations detaillees
- **Capacites d'Export**: Options d'export CSV et JSON
- **Support Upload Fichiers**: Traitement des fichiers .txt et .csv
- **Gestion d'Erreurs**: Gestion robuste des erreurs et logging
- **Securite**: Mesures de securite de niveau entreprise

## Architecture Technique

### Technologie du Modele
- **Modele de Base**: BERT-base-uncased (HuggingFace Transformers)
- **Fine-tuning**: Entrainement personnalise sur dataset SMS spam
- **Optimisation**: Gel partiel du modele pour une inference efficace
- **Seuil**: Seuil de decision optimise a 50%

### Metriques de Performance
- **Precision**: 98%
- **Precision (Precision)**: 96.9%
- **Rappel (Recall)**: 74.5%
- **F1-Score**: 70.5%
- **Temps de Traitement**: ~0.9 secondes par message

## Installation

### Prerequis
- Python 3.9+
- Scikit-learn 1.3+
- 8GB+ RAM recommandes
- Support GPU optionnel mais recommande

### Developpement Local

1. **Cloner le depot**
```bash
git clone <repository-url>
cd Spam-detection-NLP
```

2. **Creer l'environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dependances**
```bash
pip install -r requirements.txt
```

4. **Telecharger les donnees NLTK**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

5. **Definir les variables d'environnement**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Lancer l'application**
```bash
python app_simple.py
```

### Deploiement Docker

1. **Construire et lancer avec Docker Compose**
```bash
docker-compose up -d
```

2. **Acceder a l'application**
- Interface Web: http://localhost:5000
- API Health Check: http://localhost:5000/health

## Utilisation

### Interface Web

1. **Analyse de Message Unique**
   - Naviguer vers la page d'accueil
   - Entrer ou coller votre message
   - Cliquer sur "Analyser le Message"
   - Voir les resultats detailles

2. **Analyse par Lot**
   - Cliquer sur "Analyse par Lot" dans la navigation
   - Uploader un fichier .txt ou .csv
   - Voir les resultats complets et les analytics

3. **Tableau de Bord Analytics**
   - Surveiller les performances du systeme
   - Voir les tendances de detection de spam
   - Analyser les patterns et metriques

### Utilisation de l'API

#### Prediction de Message Unique
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Votre message ici"}'
```

#### Prediction par Lot
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

| Variable | Defaut | Description |
|----------|---------|-------------|
| `FLASK_ENV` | development | Environnement Flask |
| `SECRET_KEY` | - | Cle secrete de l'application |
| `MODEL_PATH` | models/simple_spam_model.pkl | Emplacement du fichier modele |
| `MAX_LEN` | 64 | Longueur maximale de sequence |
| `SEUIL_OPTIMAL` | 0.50 | Seuil de decision |
| `MAX_CONTENT_LENGTH` | 16777216 | Taille max fichier (16MB) |

### Configuration du Modele

L'application charge automatiquement le modele et le tokenizer au demarrage. Si aucun modele pre-entraine n'est trouve, elle en cree un nouveau en utilisant la configuration du notebook.

## Developpement

### Structure du Projet
```
Spam-detection-NLP/
    app_simple.py          # Application Flask principale
    templates/             # Templates HTML
    static/               # CSS, JS, et assets statiques
    models/               # Fichiers modele
    uploads/              # Repertoire upload temporaire
    requirements.txt      # Dependances Python
    Dockerfile            # Docker configuration
    docker-compose.yml    # Docker Compose setup
    .env                  # Environment variables
```

### Key Components

1. **Classe SpamDetector**: Logique de detection principale
2. **Endpoints API**: Routes API RESTful
3. **Interface Web**: UI responsive moderne
4. **Tableau de Bord Analytics**: Monitoring en temps reel
5. **Gestion d'Erreurs**: Gestion complete des erreurs

### Testing

```bash
# Executer le health check
curl http://localhost:5000/health

# Tester les endpoints API
python -c "
import requests
response = requests.post('http://localhost:5000/api/predict', 
                        json={'text': 'Message de test'})
print(response.json())
"
```

## Production Deployment

### Docker Production

1. **Configurer l'environnement de production**
```bash
export FLASK_ENV=production
export SECRET_KEY=votre-cle-secrete-production
```

2. **Deployer avec Docker Compose**
```bash
docker-compose -f docker-compose.yml up -d
```

### Manual Deployment

1. **Installer les dependances de production**
```bash
pip install -r requirements.txt
pip install gunicorn
```

2. **Lancer avec Gunicorn**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app_simple:app
```

### Nginx Configuration

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

## Monitoring and Maintenance

### Logging
- Application logs: `spam_detection.log`
- Suivi d'erreurs avec gestion complete des erreurs
- Monitoring des performances avec metriques de timing

### Health Checks
- Endpoint: `/health`
- Surveille le chargement du modele, le statut du tokenizer et la sante systeme
- Redemarrage automatique en cas d'echec dans environnement Docker

### Performance Optimization
- Cache du modele pour une inference plus rapide
- Traitement par lot pour les operations a grand volume
- Gestion de donnees efficace en memoire

## Security

### Data Protection
- Pas de stockage persistant de donnees sensibles
- Nettoyage automatique des fichiers temporaires apres traitement
- Validation securisee des uploads de fichiers

### Access Control
- Limitation de debit configurable
- Configuration de politique CORS
- Gestion securisee des sessions

## Troubleshooting

### Common Issues

1. **Erreurs de Chargement du Modele**
   - Verifier la compatibilite de version scikit-learn
   - Verifier les permissions du fichier modele
   - Assurer suffisamment de memoire disponible

2. **Performance Lente**
   - Considerer l'acceleration GPU
   - Optimiser la taille de batch
   - Verifier les ressources systeme

3. **Problemes d'Upload de Fichiers**
   - Verifier le format de fichier (.txt, .csv)
   - Verifier la limite de taille de fichier (16MB)
   - Assurer un encodage correct (UTF-8)

### Support

Pour support technique ou questions:
1. Verifier les logs application
2. Verifier les parametres de configuration
3. Tester avec des donnees exemples
4. Consulter la documentation API

## License

Ce projet est base sur la recherche academique et doit etre utilise en conformite avec les licences et reglementations applicables.

## Contributing

1. Fork le depot
2. Creer une branche de fonctionnalite
3. Soumettre une pull request
4. Suivre les standards de codage

---

**SpamGuard Enterprise** - Detection de spam avancee alimentee par l'IA pour les communications d'entreprise.
