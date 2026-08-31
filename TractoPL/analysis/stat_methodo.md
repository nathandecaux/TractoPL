# Méthodes Statistiques - Analyse Tractométrique v3

- **N sujets** (par exemple, N = 50 patients)
- **100 points** le long d'un faisceau de matière blanche (bundle)
- À chaque point : une mesure de **FA** (Fractional Anisotropy)
- Une variable d'intérêt : **AES** (score d'apathie, variable continue)
- Des confondants : **âge**, éventuellement d'autres variables

**Question scientifique** : Y a-t-il une relation entre l'AES et la FA le long du faisceau ?

---

## Les 3 modèles testés

### 1. Modèle Linéaire Simple

**Équation à chaque point *i* du faisceau :**

```
FA_i = \beta₀ + \beta_1·AES + \beta_2·age + ε
```

**Ce qu'on teste :**
- H₀ : \beta_1 = 0 (pas de relation linéaire entre AES et FA)
- H_1 : \beta_1 ≠ 0 (il existe une relation linéaire)

**Test statistique :**
- **Test F partiel** : compare le modèle complet (avec AES) au modèle réduit (sans AES, uniquement confondants)
- Le test F compare : `Modèle complet` vs `FA_i = \beta₀ + \beta_2·age + ε`
- On obtient une p-value à chaque point du faisceau (100 p-values au total)

**Interprétation :**
- Si p < seuil : la FA à ce point est linéairement associée à l'AES
- \beta_1 positif → AES plus élevé = FA plus élevée à ce point
- \beta_1 négatif → AES plus élevé = FA plus faible à ce point

---

### 2. Modèle Polynomial (degré 2)

**Équation à chaque point *i* :**

```
FA_i = \beta₀ + \beta_1·AES + \beta_2·AES² + \beta_3·age + ε
```

**Ce qu'on teste :**
- H₀ : \beta_1 = \beta_2 = 0 (pas de relation avec AES, ni linéaire ni quadratique)
- H_1 : au moins un des \beta_1 ou \beta_2 ≠ 0

**Test statistique :**
- **Test F partiel** : compare `Modèle complet` vs `FA_i = \beta₀ + \beta_3·age + ε`
- Test si les **deux** termes AES et AES² sont conjointement significatifs
- Utilise `statsmodels.OLS` avec `.compare_lr_test()` (test de rapport de vraisemblance)

**Pourquoi ce modèle ?**
- Détecte les relations **non-linéaires** (courbes en U ou en cloche inversée)
- Exemple : FA pourrait d'abord augmenter avec AES, puis diminuer (relation quadratique)
- Plus flexible que le modèle linéaire

**Métrique supplémentaire :**
- **R² partiel** : proportion de variance de FA expliquée par AES et AES² au-delà des confondants
- `R²_partiel = R²_complet - R²_réduit`

---

### 3. Modèle d'Interaction (slopes par groupe)

**Contexte :** Vous avez aussi un groupe d'appartenance (par exemple : apathique vs non-apathique)

**Équation à chaque point *i* :**

```
FA_i = \beta₀ + \beta_1·AES + \beta_2·groupe + \beta_3·(AES × groupe) + \beta_4·age + ε
```

**Ce qu'on teste :**
- H₀ : \beta_1 = \beta_3 = 0 (pas de relation avec AES, identique dans les deux groupes)
- H_1 : au moins un des termes liés à AES est ≠ 0

**Test statistique :**
- **Test F partiel** : compare `Modèle complet` vs `FA_i = \beta₀ + \beta_2·groupe + \beta_4·age + ε`
- Test si AES a un effet, et si cet effet diffère selon le groupe

**Interprétation :**
- \beta_1 : pente de la relation AES-FA dans le **groupe de référence**
- \beta_3 : **différence de pente** entre les groupes
  - Si \beta_3 ≠ 0 : la relation AES-FA diffère entre les groupes
  - Exemple : AES corrélé négativement avec FA chez les apathiques, mais pas chez les non-apathiques
- On obtient une pente effective par groupe :
  - Groupe référence : `slope = \beta_1`
  - Autre groupe : `slope = \beta_1 + \beta_3`

**Pourquoi ce modèle ?**
- Teste si la **nature** de la relation entre AES et FA dépend du groupe
- Plus informatif qu'une simple comparaison de groupes

---

## Note : Calcul de la F-statistique (test F partiel)

Le test F partiel mesure si les termes ajoutés dans le modèle complet apportent un pouvoir explicatif significatif par rapport au modèle réduit (confondants seuls).

**Formule :**

```
F = [(RSS_réduit - RSS_complet) / q] / [RSS_complet / (n - p)]
```

où :
- **RSS_réduit** : somme des carrés des résidus du modèle réduit (confondants seuls)
- **RSS_complet** : somme des carrés des résidus du modèle complet (confondants + variable d'intérêt)
- **q** : nombre de paramètres ajoutés dans le modèle complet (= nombre de degrés de liberté testés)
  - Modèle linéaire : q = 1 (\beta_1)
  - Modèle polynomial : q = 2 (\beta_1, \beta_2)
  - Modèle interaction : q = 2 (\beta_1, \beta_3)
- **p** : nombre total de paramètres du modèle complet (intercept inclus)
- **n** : nombre de sujets

**Distribution sous H₀ :**

Sous l'hypothèse nulle (les termes ajoutés n'apportent rien), F suit une **loi de Fisher** F(q, n − p).

**Intuition :**
- Le **numérateur** mesure la réduction de l'erreur résiduelle apportée par les termes d'intérêt, normalisée par le nombre de paramètres ajoutés.
- Le **dénominateur** est la variance résiduelle du modèle complet (erreur de base).
- Si F est grand → les termes ajoutés expliquent significativement plus de variance que ce qui serait attendu par hasard.

> **En pratique**, dans le code, c'est `statsmodels.OLS` qui calcule cette statistique via `.compare_lr_test()` (test du rapport de vraisemblance, asymptotiquement équivalent au test F pour les modèles linéaires gaussiens).

---

## Correction pour tests multiples (FWE)

**Problème :** On fait 100 tests (un par point du faisceau) → risque de faux positifs élevé

**Solution : Permutations Freedman-Lane**

### Procédure (pour chaque modèle)

1. **Étape initiale :**
   - Ajuster le modèle réduit (confondants seuls) sur la variable AES
   - Extraire : `AES_fitted` (prédictions) et `AES_résidus` (erreurs)

2. **Pour chaque permutation (par exemple, 1000 permutations) :**
   - Permuter aléatoirement les `AES_résidus` entre les sujets
   - Reconstruire : `AES_permutée = AES_fitted + AES_résidus_permutés`
   - Recalculer les termes du modèle (AES², interactions, etc.) avec `AES_permutée`
   - Ajuster le modèle à chaque point et calculer les 100 p-values
   - Collecter :
     - `min_p` : la plus petite p-value parmi les 100 points
     - `max_cluster` : la longueur du plus grand cluster de points significatifs (p < 0.05)

3. **Seuils FWE calculés :**
   - **alphaFWE** : 5ème percentile des `min_p` → seuil très strict pour contrôler l'erreur de type I
   - **clusterFWE** : 95ème percentile des `max_cluster` → taille minimale d'un cluster pour être considéré significatif

## Comparaison des modèles : Critère d'Akaike (AIC)

**À chaque point du faisceau :**

1. On ajuste les 3 modèles (linéaire, polynomial, interaction)
2. On calcule l'**AIC** (Akaike Information Criterion) pour chaque modèle
   - `AIC = n·log(RSS/n) + 2k`
   - RSS = somme des résidus au carré
   - k = nombre de paramètres du modèle
   - **Plus l'AIC est petit, meilleur est le modèle**

