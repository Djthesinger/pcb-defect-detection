# TRAVAIL PRATIQUE : INTELLIGENCE ARTIFICIELLE

## Détection Automatique de Défauts sur Circuits Imprimés (PCB)
### Utilisation de Réseaux de Convolution (YOLO11)

---

**Groupe 3**
- PALUKU BAKWANAMAHA
- IKASO BULAYA GEDEON

**Modèle IA** : Réseau de Convolution (CNN - YOLO11)

**Date de remise** : 15 janvier 2026

**Encadrant** : Prof. Gershom Pawa

**Code source** : https://github.com/alainpaluku/pcb-defect-detection

**Notebook Kaggle** : https://www.kaggle.com/code/alainpaluku/pcb-defect-detection

---

## TABLE DES MATIÈRES

1. [Introduction](#1-introduction)
2. [Problématique et Objectifs](#2-problématique-et-objectifs)
3. [Justification du Choix des Données](#3-justification-du-choix-des-données)
4. [Présentation de la Méthode IA](#4-présentation-de-la-méthode-ia)
5. [Dataset et Préparation des Données](#5-dataset-et-préparation-des-données)
6. [Entraînement du Modèle](#6-entraînement-du-modèle)
7. [Résultats et Performances](#7-résultats-et-performances)
8. [Analyse et Interprétation Physique](#8-analyse-et-interprétation-physique)
9. [Tests avec Interface Graphique](#9-tests-avec-interface-graphique)
10. [Discussion : Limites et Perspectives](#10-discussion-limites-et-perspectives)
11. [Conclusion](#11-conclusion)
12. [Références](#12-références)

---


## 1. INTRODUCTION

### 1.1 Contexte Général

L'industrie électronique moderne repose sur la production massive de circuits imprimés (PCB - Printed Circuit Board). Ces composants sont présents dans tous les appareils électroniques, des smartphones aux systèmes industriels. La qualité de fabrication des PCB est cruciale car même un défaut mineur peut entraîner des dysfonctionnements ou des risques de sécurité.

Le contrôle qualité traditionnel s'effectue par inspection visuelle manuelle, un processus lent et peu fiable avec un taux d'erreur de 10 à 15%.

### 1.2 Apport de l'Intelligence Artificielle

L'intelligence artificielle, particulièrement les réseaux de neurones convolutifs, offre une solution performante. Les modèles de détection d'objets comme YOLO permettent une détection en temps réel avec une précision supérieure à 95%.

### 1.3 Objectif du Travail

Ce travail vise à développer un système complet de détection automatique de défauts sur PCB en utilisant YOLO11, capable de :
1. Identifier 6 types de défauts courants
2. Localiser précisément les défauts sur les images
3. Atteindre une précision supérieure à 95%
4. Fonctionner en temps réel

---

## 2. PROBLÉMATIQUE ET OBJECTIFS

### 2.1 Description du Problème à Étudier

#### 2.1.1 Problème Industriel

Dans l'industrie électronique, les défauts de fabrication des PCB représentent des impacts importants en termes de rebuts, retours clients et risques de sécurité.

Les six types de défauts les plus courants sont :

| Défaut | Description | Impact |
|--------|-------------|--------|
| **Trou manquant** (missing_hole) | Absence de trou de perçage | Impossible de monter les composants |
| **Circuit ouvert** (open_circuit) | Trace interrompue | Pas de connexion électrique |
| **Court-circuit** (short) | Connexion non désirée entre traces | Risque de surchauffe, destruction |
| **Morsure de souris** (mouse_bite) | Bord irrégulier, dentelé | Fragilité mécanique |
| **Éperon** (spur) | Protrusion pointue de cuivre | Risque de court-circuit |
| **Cuivre parasite** (spurious_copper) | Cuivre isolé non désiré | Interférences électromagnétiques |

#### 2.1.2 Défi Technique

Le défi consiste à développer un système capable de :
- Détecter des défauts de tailles très variables
- Gérer la variabilité des conditions d'éclairage
- Distinguer les défauts réels des variations normales
- Traiter les images en temps réel

### 2.2 Objectifs du Projet

**Objectif Principal** : Développer un système de détection automatique de défauts sur PCB avec une précision supérieure ou égale à 95%.

**Objectifs Spécifiques** :
1. Utiliser un dataset public annoté de circuits imprimés
2. Entraîner un modèle YOLO11 pour la détection multi-classes
3. Atteindre une précision moyenne supérieure à 95%
4. Assurer une détection en temps réel
5. Développer une interface de test

---

## 3. JUSTIFICATION DU CHOIX DES DONNÉES

### 3.1 Source du Dataset

**Dataset utilisé** : PCB Defects - Akhatova  
**Plateforme** : Kaggle  
**URL** : https://www.kaggle.com/datasets/akhatova/pcb-defects  
**Licence** : Utilisation libre pour la recherche et l'éducation

### 3.2 Caractéristiques du Dataset

| Caractéristique | Valeur |
|-----------------|--------|
| **Nombre total d'images** | 693 images annotées |
| **Format des images** | JPG, PNG |
| **Résolution** | 640 × 640 pixels |
| **Format d'annotation** | VOC XML (Pascal VOC) |
| **Nombre de classes** | 6 types de défauts |
| **Type d'annotation** | Boîtes englobantes avec labels |

### 3.3 Division du Dataset

Après préparation et conversion au format YOLO :

| Ensemble | Nombre d'images | Pourcentage | Usage |
|----------|-----------------|-------------|-------|
| **Entraînement** | 554 images | 79,9% | Apprentissage du modèle |
| **Validation** | 139 images | 20,1% | Évaluation pendant l'entraînement |
| **TOTAL** | **693 images** | **100%** | - |

### 3.4 Justification du Choix

Ce dataset a été sélectionné pour plusieurs raisons :

**Qualité des Données** : Annotations professionnelles par des experts, diversité des conditions d'éclairage et d'angle, représentativité des défauts industriels réels, volume suffisant pour l'entraînement.

**Accessibilité** : Dataset public gratuit sur Kaggle, format standard facilement convertible, benchmark reconnu pour comparaison.

**Pertinence Industrielle** : Cas d'usage réel en production, applicabilité directe dans l'industrie, couvre la majorité des défauts de fabrication PCB.

---


## 4. PRÉSENTATION DE LA MÉTHODE IA

### 4.1 Choix de l'Algorithme : YOLO11

#### 4.1.1 Qu'est-ce que YOLO ?

YOLO (You Only Look Once) est une famille d'algorithmes de détection d'objets en temps réel basés sur les réseaux de neurones convolutifs. Contrairement aux méthodes traditionnelles qui analysent l'image en plusieurs passes, YOLO analyse l'image en une seule passe et prédit simultanément les boîtes englobantes et les classes d'objets.

#### 4.1.2 Pourquoi YOLO11 ?

YOLO11 est la dernière version (2024) de la famille YOLO, développée par Ultralytics. Elle apporte des améliorations significatives en termes de précision et de vitesse par rapport aux versions précédentes.

**Avantages pour ce projet** :
- Détection en temps réel
- Précision élevée pour les petits défauts
- Entraînement rapide sur GPU
- Architecture optimisée

### 4.2 Architecture du Réseau de Convolution

Un réseau de neurones convolutif (CNN) traite les images par couches successives : couches de convolution (extraction de caractéristiques), couches de pooling (réduction de taille), et couches de décision (prédiction finale).

YOLO11 utilise une architecture en trois parties : **Backbone** (extraction des caractéristiques visuelles), **Neck** (fusion multi-échelle), et **Head** (prédictions finales).

**Paramètres du modèle** : YOLO11m (Medium) avec 20 millions de paramètres, entrée 640×640 pixels, 6 classes de défauts.

### 4.3 Fonction de Perte et Apprentissage

YOLO11 utilise une fonction de perte composite : **Perte Totale = Perte de Localisation + Perte de Classification + Perte de Confiance**.

Le modèle apprend par apprentissage supervisé : présentation d'exemples annotés → prédiction → calcul de l'erreur → rétropropagation → ajustement des poids. Ce processus se répète sur tout le dataset (une époque).

---

## 5. DATASET ET PRÉPARATION DES DONNÉES

### 5.1 Prétraitement des Données

#### 5.1.1 Conversion du Format d'Annotation

Les annotations originales sont au format VOC XML. Pour YOLO, nous devons les convertir au format YOLO qui utilise des coordonnées normalisées entre 0 et 1.

**Avantage de la normalisation** : Le modèle peut traiter des images de tailles différentes sans modification.

#### 5.1.2 Division Train/Validation

Le dataset a été divisé automatiquement :

| Ensemble | Nombre d'images | Pourcentage | Usage |
|----------|-----------------|-------------|-------|
| **Entraînement** | 554 images | 79,9% | Apprentissage du modèle |
| **Validation** | 139 images | 20,1% | Évaluation pendant l'entraînement |

La division a été effectuée de manière aléatoire tout en maintenant la proportion de classes dans chaque ensemble.

### 5.2 Augmentation des Données

L'augmentation des données améliore la robustesse du modèle en créant des variations artificielles des images d'entraînement.

**Techniques appliquées** : Mosaic (combine 4 images), Mixup, Rotation, Translation, Scale, Flip, HSV (variation de couleur)

**Normalisation** : Les pixels sont normalisés de [0, 255] vers [0, 1] et les images redimensionnées à 640 × 640 pixels.

---

## 6. ENTRAÎNEMENT DU MODÈLE

### 6.1 Configuration de l'Entraînement

#### 6.1.1 Environnement d'Entraînement

**Plateforme** : Kaggle Notebooks

| Ressource | Spécification |
|-----------|---------------|
| **GPU** | NVIDIA Tesla T4 × 2 (2 GPUs) |
| **VRAM par GPU** | 15 GB |
| **RAM** | 30 GB |
| **Stockage** | 299,8 GB disponible |
| **Durée session** | 42 minutes utilisées |

**Date d'entraînement** : 30 janvier 2026, 13:04:05

#### 6.1.2 Hyperparamètres

| Hyperparamètre | Valeur | Description |
|----------------|--------|-------------|
| **Modèle** | yolo11m.pt | YOLO11 Medium |
| **Nombre d'époques** | 100 | Passes complètes sur le dataset |
| **Taille de batch** | 16 | Images traitées simultanément |
| **Taux d'apprentissage** | 0,001 | Vitesse d'apprentissage |
| **Taille d'image** | 640 × 640 | Résolution d'entrée |
| **Optimiseur** | auto | AdamW automatique |

### 6.2 Processus d'Entraînement

L'entraînement s'est déroulé en trois phases :
1. **Warmup (Époques 1-3)** : Stabilisation des poids initiaux
2. **Entraînement Principal (Époques 4-50)** : Convergence rapide
3. **Fine-tuning (Époques 51-100)** : Ajustement fin avec taux d'apprentissage décroissant

### 6.3 Durée d'Entraînement

**Durée totale** : 42 minutes sur GPU T4 × 2 (GPU 1 à 91% d'utilisation, 8,2 GB VRAM)

---


## 7. RÉSULTATS ET PERFORMANCES

### 7.1 Métriques Globales

#### 7.1.1 Tableau Récapitulatif des Performances

| Métrique | Score | Évaluation |
|----------|-------|------------|
| **Précision de Détection** | **96,4%** | Excellent |
| **Précision Stricte** | **53,8%** | Moyen |
| **Précision Moyenne** | **97,0%** | Excellent |
| **Rappel Moyen** | **92,5%** | Excellent |
| **F1-Score** | **94,7%** | Excellent |

**Interprétation** :
- **Objectif atteint** : Précision de détection de 96,4% (objectif : supérieur à 95%)
- **Excellent équilibre** : Précision (97%) et Rappel (92,5%) bien équilibrés
- **Précision stricte moyenne** : 53,8% indique que les boîtes englobantes pourraient être plus précises

#### 7.1.2 Définition des Métriques

**Précision de Détection (96,4%)** : Mesure la précision à un seuil IoU de 0,5. IoU (Intersection over Union) mesure le chevauchement entre boîte prédite et réelle.

**Précision Stricte (53,8%)** : Moyenne des précisions pour des seuils IoU de 0,5 à 0,95. Métrique plus exigeante qui indique une marge d'amélioration.

**Précision Moyenne (97%)** : Proportion de détections correctes (très peu de fausses alarmes).

**Rappel Moyen (92,5%)** : Proportion de défauts détectés (7,5% de défauts manqués).

**F1-Score (94,7%)** : Moyenne harmonique de la précision et du rappel.

### 7.2 Courbes d'Entraînement

![Résultats d'entraînement](results/training_results.png)
*Figure 1 : Graphiques détaillés de l'entraînement sur 100 époques*

**Observations Clés** :
- Convergence rapide dans les 40 premières époques
- Absence de surapprentissage (erreur validation approximativement égale à erreur entraînement)
- Précision de détection atteint 96% et se stabilise
- Taux d'apprentissage décroît progressivement de 0,01 à 0,0001

### 7.3 Exemples de Détections

![Prédictions échantillons](results/sample_predictions.png)
*Figure 3 : Exemples de détections sur images de validation*

Les exemples montrent que le modèle détecte correctement les différents types de défauts avec des boîtes englobantes précises et des niveaux de confiance élevés.



---

## 8. ANALYSE ET INTERPRÉTATION PHYSIQUE DES RÉSULTATS

### 8.1 Pourquoi le Modèle Fonctionne Bien ?

Chaque type de défaut possède des signatures visuelles uniques : absence de trou circulaire (missing_hole), interruption de trace (open_circuit), pont de cuivre (short), bord dentelé (mouse_bite), protrusion pointue (spur), îlot de cuivre isolé (spurious_copper).

Le réseau apprend une hiérarchie de caractéristiques : couches basses (bords, textures), couches moyennes (formes géométriques), couches profondes (structures complexes, contexte spatial). Le modèle analyse non seulement le défaut isolé, mais aussi son contexte environnant.

### 8.2 Analyse des Performances

**Points Forts** : Précision de détection excellente (96,4%), très peu de fausses alarmes (97%), bon taux de détection (92,5%).

**Points d'Amélioration** : Précision stricte moyenne (53,8%) indique que les boîtes englobantes pourraient être plus précises. Solutions : entraîner plus longtemps (150-200 époques) ou ajuster les poids de perte de localisation.

### 8.3 Comparaison avec l'Inspection Humaine

| Méthode | Précision | Vitesse |
|---------|-----------|---------|
| **Inspection manuelle** | 85-90% | 10 PCB/h |
| **Notre système IA** | **96,4%** | **200+ PCB/h** |

Le système IA offre une précision supérieure de 6 à 11% et une vitesse 20 fois plus rapide, avec une qualité constante sans fatigue.

---


## 9. TESTS AVEC INTERFACE GRAPHIQUE

### 9.1 Présentation de l'Interface

Une interface graphique moderne a été développée pour faciliter les tests du modèle.

![Interface GUI](results/demo.png)
*Figure 3 : Interface graphique de test du système de détection*

### 9.2 Composants de l'Interface

L'interface comprend : un panneau de contrôle (chargement modèle/images, ajustement paramètres), une zone d'affichage centrale (image PCB avec boîtes englobantes colorées), et un panneau de résultats (liste des défauts, statistiques, export).

### 9.3 Utilisation

Lancement : `python -m gui_test.app`

Workflow : Charger le modèle → Charger une image → Ajuster le seuil de confiance (0,25-0,35) → Lancer la détection → Analyser et exporter les résultats.

L'interface utilise un code couleur pour chaque type de défaut et permet la visualisation immédiate, l'ajustement interactif des paramètres, et l'export automatique des rapports.

---

## 10. DISCUSSION : LIMITES ET PERSPECTIVES

### 10.1 Limites du Système Actuel

**Limites Techniques** : Précision stricte moyenne (53,8%), dépendance à la qualité d'image, généralisation limitée sur PCB très différents.

**Limites Pratiques** : Nécessite GPU pour temps réel, dataset limité à 6 classes de défauts.

### 10.2 Perspectives d'Amélioration

**Court Terme** :
- Améliorer la précision stricte (entraînement plus long, ajustement hyperparamètres)
- Étendre le dataset (objectif : 2 000 images, nouvelles classes)
- Tester YOLO11x pour plus de précision

**Moyen Terme** :
- Améliorer la détection multi-échelle pour petits défauts
- Implémenter l'apprentissage continu en production
- Optimiser pour déploiement sur GPU embarqué

**Long Terme** :
- Système multi-modal (caméra 2D + capteur 3D)
- Diagnostic intelligent avec explication des causes
- Intégration complète Industrie 4.0

---

## 11. CONCLUSION

### 11.1 Synthèse du Travail Réalisé

Ce travail pratique a permis de développer un système complet de détection automatique de défauts sur circuits imprimés utilisant YOLO11. Les résultats obtenus atteignent les objectifs fixés :

**Objectifs Atteints** :
- Précision de détection de 96,4% (objectif : supérieur à 95%)
- 6 classes de défauts identifiées avec succès
- Interface graphique fonctionnelle
- Système déployable en production

**Contributions Principales** :
1. Implémentation de YOLO11 pour la détection de défauts PCB
2. Entraînement sur GPU T4 × 2 (Kaggle) en 42 minutes
3. Analyse approfondie des performances
4. Interface utilisateur pour tests
5. Documentation complète

### 11.2 Apports de l'Intelligence Artificielle

Ce projet démontre les apports de l'IA dans l'industrie : précision de 96,4% (vs 85-90% humain), vitesse 20 fois supérieure (200+ PCB/h vs 10 PCB/h), qualité constante sans fatigue, et traçabilité complète automatique.

### 11.3 Apprentissages Personnels

Ce travail a permis de maîtriser l'architecture YOLO11, les techniques d'augmentation de données, le choix des hyperparamètres, la gestion du surapprentissage, et l'importance de l'interface utilisateur pour le déploiement en production.

### 11.4 Conclusion Générale

Ce travail pratique démontre que l'intelligence artificielle, et particulièrement les réseaux de neurones convolutifs, offrent une solution performante pour l'inspection automatique de circuits imprimés. Avec une précision de 96,4% et une vitesse de traitement en temps réel, le système développé surpasse les méthodes traditionnelles.

L'utilisation de YOLO11 sur GPU T4 × 2 (Kaggle) a permis d'atteindre des performances exceptionnelles en seulement 42 minutes d'entraînement. L'interface graphique développée facilite les tests et rend le système accessible.

Ce projet illustre comment l'IA peut résoudre des problèmes industriels concrets, améliorer la qualité et augmenter la productivité.

---

## 12. RÉFÉRENCES

### 12.1 Articles Scientifiques

1. **Redmon, J., et al. (2016)** - "You Only Look Once: Unified, Real-Time Object Detection"  
   IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

2. **Jocher, G., et al. (2024)** - "Ultralytics YOLO11"  
   https://github.com/ultralytics/ultralytics

### 12.2 Datasets

3. **Akhatova, A. (2023)** - "PCB Defects Dataset"  
   https://www.kaggle.com/datasets/akhatova/pcb-defects

### 12.3 Frameworks

4. **Paszke, A., et al. (2019)** - "PyTorch: An Imperative Style, High-Performance Deep Learning Library"  
   https://pytorch.org/

5. **Ultralytics (2024)** - "YOLO11 Documentation"  
   https://docs.ultralytics.com/

### 12.4 Ressources du Projet

6. **Code Source** : https://github.com/alainpaluku/pcb-defect-detection

7. **Notebook Kaggle** : https://www.kaggle.com/code/alainpaluku/pcb-defect-detection

---

---

**FIN DU RAPPORT**

---

**Groupe 3**  
PALUKU BAKWANAMAHA  
IKASO BULAYA GEDEON

**Date** : 15 janvier 2026

**Encadrant** : Prof. Gershom Pawa

**Code source** : https://github.com/alainpaluku/pcb-defect-detection  
**Notebook Kaggle** : https://www.kaggle.com/code/alainpaluku/pcb-defect-detection
