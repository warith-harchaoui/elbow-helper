# elbow-helper

[🇫🇷](LISEZ-MOI.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](README.md)

**Détection de coude robuste au bruit : elle indique un coude avec son incertitude, ou elle s'abstient franchement.**

Un algorithme peut répondre à la question « où pourrait se trouver un coude ? », et il renverra toujours quelque chose, même sur une droite ou sur du pur bruit. `elbow-helper` enveloppe un repéreur de coude réécrit de zéro dans une procédure de décision prudente, qui répond à une question plus exigeante :

> Le coude candidat est-il fort, unique, persistant, reproductible et improbable sous un modèle sans coude ? Sinon, on le dit.

La priorité de conception est de minimiser les faux coudes, quitte à s'abstenir plus souvent.

## Pourquoi ce paquet existe

Un coude, sur une courbe à rendements décroissants, marque le point au-delà duquel une entrée supplémentaire achète peu de sortie en plus : davantage de groupes en k-means, davantage d'itérations dans un optimiseur, davantage de budget sur un canal publicitaire. Le paquet `kneed`, largement utilisé, excelle à proposer où se situe ce point, mais il ne porte aucune notion de confiance. Une estimation ponctuelle unique sur une courbe bruitée s'avère facile à surestimer en pratique. Ce paquet transforme cette estimation ponctuelle en une décision appuyée sur des preuves, et refuse de répondre quand les preuves sont faibles.

## Dépendances

Le cœur du paquet ne dépend que de `numpy` et de [`os-helper`](https://github.com/warith-harchaoui/os-helper). L'algorithme de repérage du coude est réécrit de zéro (un portage fidèle, en NumPy pur, de la méthode derrière [`arvkevi/kneed`](https://github.com/arvkevi/kneed)) : il n'y a donc aucune dépendance d'exécution à `scipy`, `scikit-learn`, `statsmodels`, `joblib` ou `kneed`. Le tracé de figures reste une option (`matplotlib`, chargé paresseusement, seulement quand on l'appelle).

## Installation

```bash
pip install -e .            # cœur : numpy + os-helper
pip install -e ".[plot]"    # + matplotlib pour la figure de diagnostic
pip install -e ".[dev]"     # + pytest
```

## Prise en main

```python
import numpy as np
from elbow_helper import robust_knee, RobustKneeConfig

x = np.linspace(0, 1, 80)
y = np.where(x <= 0.3, 3*x, 0.9 + 0.2*(x - 0.3))
y = y / y.max() + np.random.default_rng(0).normal(0, 0.02, x.size)

result = robust_knee(x, y, curve="concave", direction="increasing",
                     config=RobustKneeConfig(random_seed=0))

if result.is_clear:
    print(result.knee_x, result.ci90, result.detection_rate, result.null_p_value)
else:
    print("no clear knee:", result.reason)
```

Pour le classique « coude » de k-means ou de scree (convexe, décroissant) :

```python
from elbow_helper import robust_elbow
result = robust_elbow(k_values, inertia)   # curve="convex", direction="decreasing"
```

## Le contrat : une union étiquetée

`robust_knee` renvoie toujours l'un de deux types, tous deux sous-classes de `KneeResult` et distingués par `.is_clear` :

- `ClearKnee` : `knee_x`, `knee_x_norm`, `knee_index`, `ci90` (un intervalle bootstrap à 90 %, en unités des données), `detection_rate`, `smoothing_window`, `sensitivity`, `prominence`, `slope_contrast`, `bic_improvement`, `null_p_value`, et l'ensemble des `diagnostics`.
- `NoClearKnee` : un code `reason` lisible par une machine, plus les `diagnostics`.

On est contraint de traiter l'abstention explicitement : il n'existe aucun repli silencieux vers une estimation hasardeuse.

## Comment la décision se prend : le pipeline

1. **Prétraitement.** On nettoie les données, on les trie, on retire les doublons, on les normalise de façon robuste sur le carré unité, puis on examine leur forme globale avec une corrélation de Spearman (une mesure, fondée sur les rangs, du caractère monotone de la courbe) et une vérification de monotonie pondérée par l'amplitude.
2. **Recherche en espace d'échelles.** On fait tourner le repéreur réécrit de zéro sur une grille de fenêtres de lissage gaussien et de sensibilités, et l'on recueille chaque candidat proposé.
3. **Filtres de base.** On rejette les coudes situés en bordure, ceux à faible proéminence, et ceux dont le rapport proéminence sur bruit est trop bas.
4. **Regroupement par persistance.** On ne garde que les coudes qui réapparaissent à un emplacement stable sur des échelles de lissage consécutives et sur la plupart des sensibilités. On s'abstient si deux coudes semblent également plausibles (`MULTIPLE_PLAUSIBLE_KNEES`).
5. **Confirmation par un modèle.** On exige un changement de pente robuste, obtenu via un estimateur de Theil-Sen (une méthode de régression qui reste fiable même en présence de valeurs aberrantes), plus un ajustement continu en ligne brisée qui surpasse une droite unique en validation croisée par blocs et selon le critère d'information bayésien (_Bayesian Information Criterion_, ou BIC, un score qui récompense l'ajustement tout en pénalisant les paramètres superflus).
6. **Bootstrap.** On relance toute la recherche sur un bootstrap des résidus supposés i.i.d. (indépendants et identiquement distribués, une hypothèse de rééchantillonnage aléatoire du bruit restant) ; le coude doit être redétecté au moins 90 % des fois, avec un intervalle resserré et unimodal.
7. **Test nul « sans coude ».** On effectue un test de Monte-Carlo contre un modèle nul en ligne droite qui reprend l'échelle de bruit du modèle accepté ; le coude observé doit être significatif au seuil p ≤ 0,01.

Seul un candidat qui franchit chaque étape devient un `ClearKnee`.

## Codes d'abstention

`INSUFFICIENT_DATA`, `INVALID_INPUT`, `ZERO_RANGE`, `INCOMPATIBLE_GLOBAL_SHAPE`,
`NO_KNEED_CANDIDATES`, `ALL_CANDIDATES_WEAK`, `NO_PERSISTENT_CLUSTER`,
`MULTIPLE_PLAUSIBLE_KNEES`, `BOUNDARY_KNEE`, `WEAK_SLOPE_CHANGE`,
`SEGMENTED_MODEL_NOT_BETTER`, `BOOTSTRAP_UNSTABLE`, `BOOTSTRAP_MULTIMODAL`,
`NULL_NOT_REJECTED`, `INTERNAL_NUMERICAL_FAILURE`.

## Configuration

Chaque seuil vit dans `RobustKneeConfig`, une dataclass figée (_frozen_) ; on appelle `config.with_(...)` pour en changer un. Tous les seuils positionnels s'expriment en unités d'intervalle x normalisé. Les valeurs par défaut livrées correspondent à un premier prototype fonctionnel : des nombres de réplicats modestes, pour qu'une exécution se termine en quelques secondes.

```python
RobustKneeConfig(bootstrap_replicates=100, null_replicates=200)
# qualité validation :
config.with_(bootstrap_replicates=500, null_replicates=1000)
```

Une remarque sur ces seuils : ce sont des valeurs par défaut calibrées et prudentes, non des constantes universelles. `cluster_tolerance` et `max_neighbor_shift` se situent légèrement au-dessus des 0,05 du plan de référence, pour absorber la gigue de discrétisation d'un à deux échantillons propre au repéreur, aux tailles d'échantillon modestes (n de l'ordre de 60 à 100). Recalibrez-les sur votre propre famille de courbes et de bruit si besoin.

## Diagnostics

```python
from elbow_helper.plotting import plot_diagnostics
plot_diagnostics(x, y, curve="concave", direction="increasing", out="diag.png")
```

Quatre panneaux : la courbe avec le coude repéré et son intervalle, les coudes candidats à travers l'espace d'échelles, la courbe de différence qui a servi à les repérer, et la distribution du bootstrap. L'estimation ponctuelle n'est jamais montrée sans son incertitude.

## Limites

- Le lisseur actuel suppose un espacement régulier, ou quasi régulier, de `x`.
- `curve` et `direction` sont des entrées obligatoires ; la forme n'est jamais devinée.
- La détection est discrétisée aux emplacements des échantillons. À `n` modeste, le coude repéré peut se situer à quelques échantillons de la vraie valeur (erreur médiane autour de, ou en dessous de, 5 % de l'intervalle x, sur la famille synthétique prise en charge).
- Le modèle nul en ligne droite et le bootstrap des résidus i.i.d. conviennent à un bruit à peu près homoscédastique (de variance constante) et non corrélé. Les variantes en bootstrap sauvage (_wild_) ou par blocs mobiles restent à faire.
- Aucune méthode à données finies n'est infaillible. Les objectifs ci-dessus tiennent pour la famille de simulation documentée, non pour toute courbe possible.

## Repéreur autonome

Le repéreur réécrit de zéro s'utilise aussi seul :

```python
from elbow_helper import KneeLocator
kl = KneeLocator(x, y, S=1.0, curve="concave", direction="increasing", online=True)
kl.knee, kl.all_knees
```

## Licence

BSD-3-Clause.
