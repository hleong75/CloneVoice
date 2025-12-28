# CloneVoice 🎙️

**CloneVoice** est une application de clonage vocal alimentée par l'IA, simple à utiliser mais puissante sous le capot. Elle utilise des modèles d'IA de pointe pour cloner des voix à partir d'échantillons audio.

## ✨ Fonctionnalités

- **Version 1** : Clonage vocal avec fichier CSV manuel
- **Version 2** : Génération automatique du CSV (transcription automatique avec Whisper)
- **Interface simple** : CLI facile à utiliser et API Python
- **IA performante** : Utilise XTTS v2 pour un clonage vocal de haute qualité
- **Multi-langue** : Support de 17+ langues

## 🚀 Installation

### Prérequis

- Python 3.8+
- (Recommandé) GPU NVIDIA avec CUDA pour de meilleures performances

### Installation des dépendances

```bash
pip install -r requirements.txt
```

> **Note** : La première exécution téléchargera automatiquement les modèles d'IA (~2 Go).

## 📖 Utilisation

### Version 1 : Avec fichier CSV manuel

Créez un fichier CSV avec deux colonnes :
- Colonne 1 : Identifiant de l'audio (nom du fichier sans extension)
- Colonne 2 : Transcription du texte

**Exemple de fichier CSV (`data.csv`)** :
```csv
audio_id,transcription
001,Bonjour, comment allez-vous ?
002,Je suis très content de vous rencontrer.
003,À bientôt !
```

**Commande** :
```bash
python clone_voice.py --csv data.csv --audio-dir ./audios --text "Le texte à générer" --output sortie.wav
```

### Version 2 : Mode automatique (sans CSV)

Le programme transcrit automatiquement vos fichiers audio avec Whisper.

```bash
python clone_voice.py --auto --audio-dir ./audios --text "Le texte à générer" --output sortie.wav
```

### Options supplémentaires

```bash
# Spécifier la langue (défaut: fr)
python clone_voice.py --auto --audio-dir ./audios --text "Hello world" --output output.wav --language en

# Utiliser un modèle Whisper plus précis
python clone_voice.py --auto --audio-dir ./audios --text "Bonjour" --output sortie.wav --whisper-model medium

# Désactiver le GPU
python clone_voice.py --auto --audio-dir ./audios --text "Bonjour" --output sortie.wav --no-gpu

# Mode batch avec fichier de textes
python clone_voice.py --csv data.csv --audio-dir ./audios --text-file textes.txt --output-dir ./sorties

# Ajuster les paramètres de qualité (avancé)
python clone_voice.py --auto --audio-dir ./audios --text "Bonjour" --output sortie.wav \
    --temperature 0.5 --speed 1.0 --repetition-penalty 6.0
```

### Paramètres de qualité (avancé)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--temperature` | 0.65 | Contrôle la variation (0.1-1.0). Plus bas = plus déterministe |
| `--speed` | 1.0 | Vitesse de parole (0.5-2.0) |
| `--repetition-penalty` | 5.0 | Pénalité de répétition (1.0-10.0). Plus haut = moins de répétitions |
| `--no-preprocess` | False | Désactiver le prétraitement audio (non recommandé) |

## 🐍 API Python

```python
from src.api import clone_voice, clone_voice_auto, generate_transcriptions

# Version 1 : Avec CSV
result = clone_voice(
    csv_path="data.csv",
    audio_dir="./audios",
    text="Bonjour le monde",
    output_path="sortie.wav"
)

# Version 2 : Mode automatique
result = clone_voice_auto(
    audio_dir="./audios",
    text="Bonjour le monde",
    output_path="sortie.wav"
)

# Avec paramètres personnalisés
result = clone_voice_auto(
    audio_dir="./audios",
    text="Bonjour le monde",
    output_path="sortie.wav",
    temperature=0.5,
    speed=1.0,
    repetition_penalty=6.0
)

# Générer un CSV de transcriptions
csv_path = generate_transcriptions(
    audio_dir="./audios",
    output_csv="transcriptions.csv",
    language="fr"
)
```

## 📁 Structure des fichiers

```
CloneVoice/
├── clone_voice.py       # Point d'entrée principal (CLI)
├── requirements.txt     # Dépendances Python
├── README.md            # Documentation
├── src/
│   ├── __init__.py
│   ├── api.py           # API Python simplifiée
│   ├── csv_parser.py    # Parseur de fichiers CSV
│   ├── voice_cloner.py  # Module de clonage vocal (XTTS)
│   ├── audio_processing.py  # Traitement audio
│   └── auto_transcriber.py  # Transcription automatique (Whisper)
├── tests/
│   └── test_csv_parser.py   # Tests unitaires
└── samples/             # Répertoire pour les échantillons
```

## 🎯 Format des fichiers audio

### Exigences

- **Formats supportés** : WAV, MP3, FLAC, OGG, M4A
- **Durée recommandée** : 6-30 secondes par fichier (optimal: 10s)
- **Qualité recommandée** : Audio clair, sans bruit de fond

### Conseils pour une meilleure qualité

Pour obtenir les meilleurs résultats de clonage vocal :

1. **Qualité audio** :
   - Utilisez des enregistrements de haute qualité (minimum 16kHz)
   - Évitez les bruits de fond (ventilateurs, musique, etc.)
   - Préférez les enregistrements mono

2. **Durée optimale** :
   - Minimum : 3 secondes
   - Optimal : 6-15 secondes
   - Maximum : 30 secondes (au-delà, l'audio sera automatiquement tronqué)

3. **Contenu vocal** :
   - Parole claire et distincte
   - Évitez les chuchotements ou les cris
   - Plusieurs phrases variées sont préférables à une seule répétition

4. **Prétraitement automatique** :
   - Le programme normalise automatiquement le volume
   - Les silences au début/fin sont supprimés
   - Une légère réduction de bruit est appliquée

> **Note** : Si la sortie audio n'est pas claire, essayez d'améliorer la qualité de vos fichiers audio de référence.

## 🌍 Langues supportées

Le modèle XTTS v2 supporte les langues suivantes :
- Français (fr), Anglais (en), Espagnol (es), Allemand (de)
- Italien (it), Portugais (pt), Polonais (pl), Turc (tr)
- Russe (ru), Néerlandais (nl), Tchèque (cs), Arabe (ar)
- Chinois (zh-cn), Japonais (ja), Hongrois (hu), Coréen (ko), Hindi (hi)

## 🔧 Configuration système recommandée

- **CPU** : 4+ cœurs
- **RAM** : 8+ Go
- **GPU** (recommandé) : NVIDIA avec 4+ Go VRAM
- **Stockage** : 5 Go pour les modèles

## 📝 Exemple complet

1. **Préparez vos fichiers audio** dans un répertoire (ex: `./mes_audios/`)

2. **Créez un fichier CSV** (Version 1) ou utilisez le mode automatique (Version 2)

3. **Exécutez le clonage** :
   ```bash
   python clone_voice.py --auto --audio-dir ./mes_audios --text "Ceci est un test de clonage vocal." --output test.wav
   ```

4. **Écoutez le résultat** : Le fichier `test.wav` contient votre texte dit avec la voix clonée !

## 🧪 Tests

```bash
python -m unittest discover tests -v
```

## 📄 Licence

MIT License

## 🙏 Crédits

- [Coqui TTS](https://github.com/coqui-ai/TTS) - Moteur de synthèse vocale et clonage
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription automatique
