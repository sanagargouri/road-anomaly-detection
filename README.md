# Détection d'obstacles inattendus sur la route (RoadAnomaly)

Projet du module Deep Learning : segmenter, pixel par pixel, les obstacles inattendus sur la route
(animaux, cônes, pneus, rochers, charrettes…) à partir du dataset **RoadAnomaly** (60 images 1280 × 720 avec masques),
en suivant la démarche **CRISP-DM**, puis les présenter dans une interface **Streamlit**.

## Objectifs

| Objectif métier | Objectif data science | Évaluation |
| --- | --- | --- |
| BO1 – Détecter et localiser les obstacles | Segmentation binaire obstacle / fond | Precision, recall, F1, IoU, Dice sur le test |
| BO2 – Détecter les obstacles lointains | Détection selon la taille de l'obstacle | Recall par groupe de taille |
| Option – Type d'obstacle | Classification animal / autre obstacle | Accuracy, F1, matrice de confusion |

## Modèles comparés (Keras / TensorFlow)

| N° | Modèle | Entraînement |
| --- | --- | --- |
| 1 | U-Net simplifié | De zéro, avec data augmentation |
| 2 | U-Net simplifié | De zéro, sans augmentation (ablation) |
| 3 | Encodeur-décodeur sans raccourcis (type FCN) | De zéro (ablation) |
| 4 | U-Net avec encodeur ResNet50 | Transfer learning (ImageNet) |
| 5 | DeepLabV3+ avec encodeur ResNet50 | Transfer learning (ImageNet) |

Pleine résolution conservée (1280 × 720), perte BCE + Dice, Adam, early stopping, petite grid search.

## Résultats sur le test

| Modèle | Precision | Recall | F1 | IoU | Dice |
| --- | --- | --- | --- | --- | --- |
| **5 – DeepLabV3+ (retenu)** | 0,83 | **0,76** | **0,80** | **0,66** | **0,78** |
| 4 – U-Net + ResNet50 | **0,87** | 0,62 | 0,72 | 0,56 | 0,71 |
| 1 – U-Net | 0,40 | 0,53 | 0,45 | 0,29 | 0,42 |
| 2 – U-Net sans augmentation | 0,39 | 0,30 | 0,34 | 0,21 | 0,37 |
| 3 – Type FCN | 0,35 | 0,27 | 0,31 | 0,18 | 0,32 |

- Le transfer learning est décisif : F1 de 0,45 à 0,80.
- Obstacles de 0,1 à 1 % de l'image : bien détectés par les modèles pré-entraînés ; les tout petits (< 0,1 %) restent la principale limite.
- Classification animal / autre obstacle : 90 % d'accuracy sur le test.

## Contenu du dépôt

```
├── RoadAnomaly.ipynb        # notebook complet (phases CRISP-DM 1 à 7)
├── app.py                   # interface Streamlit
├── .streamlit/config.toml   # thème de l'interface
├── figures/                 # figures des prédictions
├── RoadAnomaly_modeles/     # historiques d'entraînement et tableaux de résultats (.json, .csv)
└── requirements.txt
```

## Lancer le projet

1. Installer les dépendances : `pip install -r requirements.txt`
2. Télécharger le dataset RoadAnomaly et le placer dans `data/RoadAnomaly/`.
3. Télécharger les poids des modèles (trop lourds pour GitHub) : [lien Google Drive](https://drive.google.com/drive/folders/1jlH-Vu_w94kdC4-vUur0XgfjKwHDq02p?usp=drive_link)
   et les placer dans `RoadAnomaly_modeles/`.
4. Notebook : ouvrir `RoadAnomaly.ipynb` (les cellules d'entraînement rechargent les poids si `REENTRAINER = False`).
5. Interface : `streamlit run app.py`

## Dataset

RoadAnomaly (Lis et al., EPFL) : le dataset n'est pas inclus dans ce dépôt ; voir sa page officielle et le fichier `credits.txt` fourni avec le dataset pour les sources des images.
