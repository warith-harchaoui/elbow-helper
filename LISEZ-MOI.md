# elbow-helper

[🇫🇷](https://github.com/warith-harchaoui/elbow-helper/blob/main/LISEZ-MOI.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](https://github.com/warith-harchaoui/elbow-helper/blob/main/README.md)

**Détection de coude robuste au bruit : un coude assorti de son incertitude ou une abstention franche.**

![Logo Elbow Helper](https://raw.githubusercontent.com/warith-harchaoui/elbow-helper/main/assets/logo.png)

Face à la question « où se trouve le coude ? », un algorithme classique répond toujours quelque chose, même sur une droite ou sur du bruit pur. `elbow-helper` enveloppe un repéreur de coude réécrit de zéro dans une procédure de décision prudente, qui pose une question plus exigeante :

> Le coude repéré est-il net : assez marqué, unique, stable d'un rééchantillonnage à l'autre et trop prononcé pour n'être qu'un effet du hasard ? Sinon, le paquet le dit clairement plutôt que de deviner.

La priorité, dès la conception, a été de limiter les faux coudes, quitte à s'abstenir plus souvent qu'un outil moins prudent.

## Documentation

[💻 Documentation](https://harchaoui.org/warith/ai-helpers/docs/elbow-helper-doc/)

[🗺️ Paysage](https://github.com/warith-harchaoui/elbow-helper/blob/main/PAYSAGE.md)

[📋 Exemples](https://github.com/warith-harchaoui/elbow-helper/blob/main/EXEMPLES.md)

## Pourquoi ce paquet existe

Sur une courbe à rendements décroissants, le coude marque l'endroit où chaque unité supplémentaire investie rapporte de moins en moins : des groupes en plus pour un k-means, des itérations en plus pour un optimiseur, du budget en plus sur un canal publicitaire. Les outils de détection existants savent très bien proposer un point ; ce qui leur manque, c'est une notion de confiance. Sur une courbe bruitée, une estimation ponctuelle isolée s'avère trop facile à surinterpréter. `elbow-helper` transforme ce point en une décision appuyée sur des preuves et refuse de trancher quand ces preuves sont trop minces.

## Dépendances

Le paquet entier ne dépend que de `numpy` et de [`os-helper`](https://github.com/warith-harchaoui/os-helper), y compris pour la figure de diagnostic. Le repérage du coude est codé de zéro, en NumPy pur : aucune dépendance à `scipy`, `scikit-learn`, `statsmodels` ni `joblib`. `elbow_helper.plotting` dessine du SVG à la main (voir Diagnostics plus bas) au lieu de passer par matplotlib, donc rien à installer en plus pour tracer une figure. La section Remerciements, en bas de page, détaille l'implémentation dont s'inspire l'algorithme.

## Installation

```bash
pip install -e .            # tout : numpy + os-helper, diagnostics compris
pip install -e ".[dev]"     # + pytest
```

Vous préférez conda ? `conda env create -f environment.yaml && conda activate elbow-helper` installe Python et pip, puis `requirements.txt`, la même liste de dépendances que déclare `pyproject.toml`.

## Prise en main

Voir [`EXEMPLES.md`](https://github.com/warith-harchaoui/elbow-helper/blob/main/EXEMPLES.md) pour d'autres recettes : le coude de k-means, une abstention explicite, une courbe de saturation exponentielle, la figure de diagnostic, le repéreur autonome et le réglage de la configuration.

```python
import numpy as np
from elbow_helper import robust_knee, RobustKneeConfig

x = np.linspace(0, 1, 80)
y = np.where(x <= 0.3, 3*x, 0.9 + 0.2*(x - 0.3))
y = y / y.max() + np.random.default_rng(0).normal(0, 0.02, x.size)

result = robust_knee(x, y, config=RobustKneeConfig(random_seed=0))

if result.is_clear:
    print(result.knee_x, result.ci90, result.detection_rate, result.null_p_value)
else:
    print("no clear knee:", result.reason)
```

`curve` et `direction` sont facultatifs : si on les laisse vides, ils sont déduits des données (voir plus bas), mais rien n'empêche de les fixer soi-même, par exemple `robust_knee(x, y, curve="concave", direction="increasing", ...)`. `y` peut aussi être omis : un simple `robust_knee(y)` prend alors `0, 1, ..., n-1` comme axe `x` implicite.

Pour le coude classique d'un k-means ou d'un scree plot, une courbe convexe et décroissante :

```python
from elbow_helper import robust_elbow
result = robust_elbow(k_values, inertia)   # curve/direction fixés à convex/decreasing
```

## Forme et sens : la déduction automatique

Laisser `curve` ou `direction` vide revient à les déduire des données une fois nettoyées et normalisées. `direction` vient du signe de la tendance entre `x` et `y`. `curve` dépend de la position de la courbe (légèrement lissée) par rapport à la corde reliant son premier et son dernier point : au-dessus, elle est concave ; en dessous, convexe. C'est la définition mathématique du terme et elle s'applique aussi bien à une courbe croissante que décroissante. Les quatre combinaisons concave/convexe et croissant/décroissant sont ainsi couvertes sans jamais demander à l'appelant de nommer la forme à l'avance. Fixer `curve` ou `direction` explicitement l'emporte toujours sur la valeur déduite.

## Le contrat : une union étiquetée

`robust_knee` renvoie toujours l'un de deux types, tous deux sous-classes de `KneeResult` et qu'on distingue par `.is_clear` :

- `ClearKnee` : `knee_x`, `knee_x_norm`, `knee_index`, `ci90` (un intervalle bootstrap à 90 %, en unités des données), `detection_rate`, `smoothing_window`, `sensitivity`, `prominence`, `slope_contrast`, `bic_improvement`, `null_p_value`, plus l'ensemble des `diagnostics`.
- `NoClearKnee` : un code `reason` lisible par une machine, plus les `diagnostics`.

L'abstention doit être traitée explicitement : il n'existe aucun repli silencieux vers une estimation approximative.

## Comment la décision se prend : le pipeline

1. **Prétraitement.** Les données sont nettoyées, triées, dédoublonnées, puis normalisées de façon robuste sur le carré unité. Leur forme globale est ensuite examinée par une corrélation de Spearman (une mesure, fondée sur les rangs, du caractère monotone de la courbe) et par une vérification de monotonie pondérée par l'amplitude.
2. **Recherche en espace d'échelles.** Le repéreur balaie une grille de fenêtres de lissage gaussien et de sensibilités et retient chaque candidat proposé.
3. **Filtres de base.** Les coudes en bordure, ceux de faible proéminence et ceux dont le rapport proéminence sur bruit est trop bas sont écartés.
4. **Regroupement par persistance.** Seuls survivent les coudes qui réapparaissent à un emplacement stable sur plusieurs échelles de lissage consécutives et sur la plupart des sensibilités testées. Le pipeline s'abstient si deux coudes paraissent également plausibles (`MULTIPLE_PLAUSIBLE_KNEES`).
5. **Confirmation par un modèle.** Un changement de pente robuste est exigé, mesuré par un estimateur de Theil-Sen (une régression qui reste fiable même en présence de valeurs aberrantes) et confirmé par un ajustement en ligne brisée continue qui doit surpasser une simple droite, à la fois en validation croisée par blocs et selon le critère d'information bayésien (BIC), un score qui récompense la qualité de l'ajustement tout en pénalisant les paramètres superflus.
6. **Bootstrap.** Toute la recherche est rejouée sur un bootstrap des résidus supposés i.i.d. (indépendants et identiquement distribués). Le coude doit être redétecté dans au moins 90 % des tirages, avec un intervalle resserré et unimodal.
7. **Test nul « sans coude ».** Un test de Monte-Carlo confronte le résultat à un modèle nul en ligne droite reprenant l'échelle de bruit du modèle accepté. Le coude observé doit rester significatif au seuil p ≤ 0,01.

Seul un candidat qui franchit chacune de ces étapes devient un `ClearKnee`.

## Codes d'abstention

`INSUFFICIENT_DATA`, `INVALID_INPUT`, `ZERO_RANGE`, `INCOMPATIBLE_GLOBAL_SHAPE`,
`NO_KNEE_CANDIDATES`, `ALL_CANDIDATES_WEAK`, `NO_PERSISTENT_CLUSTER`,
`MULTIPLE_PLAUSIBLE_KNEES`, `BOUNDARY_KNEE`, `WEAK_SLOPE_CHANGE`,
`SEGMENTED_MODEL_NOT_BETTER`, `BOOTSTRAP_UNSTABLE`, `BOOTSTRAP_MULTIMODAL`,
`NULL_NOT_REJECTED`, `INTERNAL_NUMERICAL_FAILURE`.

## Configuration

Chaque seuil vit dans `RobustKneeConfig`, une dataclass figée (*frozen*) ; pour en changer un, on appelle `config.with_(...)`. Tous les seuils positionnels s'expriment en unités d'intervalle x normalisé. Les valeurs par défaut correspondent à un premier prototype fonctionnel : des nombres de réplicats volontairement modestes, pour qu'une exécution se termine en quelques secondes.

```python
RobustKneeConfig(bootstrap_replicates=100, null_replicates=200)
# qualité validation :
config.with_(bootstrap_replicates=500, null_replicates=1000)
```

Une précision sur ces seuils : ce sont des valeurs par défaut calibrées et prudentes, pas des constantes universelles. `cluster_tolerance` et `max_neighbor_shift` dépassent légèrement les 0,05 du plan de référence, pour absorber la gigue de discrétisation d'un ou deux échantillons propre au repéreur, aux tailles d'échantillon modestes (n de l'ordre de 60 à 100). N'hésitez pas à les recalibrer sur votre propre famille de courbes et de bruit.

## Diagnostics

```python
from elbow_helper.plotting import plot_diagnostics
plot_diagnostics(x, y, curve="concave", direction="increasing", out="diag.svg", language="fr")
```

![Figure de diagnostic : une courbe qui monte puis s'aplatit, avec le coude repéré et sa bande de confiance à 90 % en surbrillance, à côté d'une légende compacte avec la probabilité de détection, la p-value du modèle nul, le contraste de pente, une probabilité a posteriori dérivée du BIC et un score de qualité d'ajustement normalisé contre un pire cas.](https://raw.githubusercontent.com/warith-harchaoui/elbow-helper/main/assets/diagnostics.svg)

Le SVG est écrit à la main, sans matplotlib et sans rien à installer en plus : le diagnostic fait partie du cœur du paquet. La courbe et son coude repéré voisinent avec une légende compacte qui étaye l'estimation : la probabilité de détection, la p-value du modèle nul, le contraste de pente, une probabilité a posteriori dérivée du BIC (les chances, sous l'approximation du facteur de Bayes de Kass et Raftery, que le modèle à coude soit le bon) et un score de qualité d'ajustement normalisé contre un pire cas délibérément pessimiste plutôt que contre la moyenne d'échantillon, trop facile à battre (voir `doc/ELBOW-fr.tex` pour la dérivation des deux). Quand les preuves sont trop faibles, la figure bascule dans un état d'abstention honnête plutôt que d'afficher un résultat trompeur : une courbe grisée en pointillés et la raison, jamais un marqueur qui laisserait croire à plus de confiance que les données n'en autorisent.

## Limites

- Le lisseur actuel suppose un espacement régulier ou presque de `x`.
- La déduction automatique de `curve` et `direction` suppose une tendance réelle et sans ambiguïté. Sur des données trop faibles ou trop bruitées pour trancher avec confiance, le pipeline s'abstient avec `INCOMPATIBLE_GLOBAL_SHAPE` plutôt que de deviner, la même barrière qui filtre déjà les arguments `curve`/`direction` fournis explicitement.
- La détection reste discrétisée aux emplacements des échantillons. À `n` modeste, le coude repéré peut se situer à quelques échantillons de la vraie valeur (erreur médiane proche de 5 % de l'intervalle x, voire en dessous, sur la famille synthétique testée).
- Le modèle nul en ligne droite et le bootstrap des résidus i.i.d. conviennent à un bruit à peu près homoscédastique (de variance constante) et non corrélé. Les variantes en bootstrap sauvage (*wild*) ou par blocs mobiles restent à construire.
- Aucune méthode à données finies n'est infaillible. Les objectifs ci-dessus valent pour la famille de simulation documentée, pas pour n'importe quelle courbe.

## Repéreur autonome

Le repéreur réécrit de zéro s'utilise aussi seul, sans passer par le pipeline complet :

```python
from elbow_helper import KneeLocator
kl = KneeLocator(x, y, S=1.0, curve="concave", direction="increasing", online=True)
kl.knee, kl.all_knees
```

## Plusieurs coudes : `robust_knees`

`robust_knee` répond à « y a-t-il un coude ? ». Une courbe avec plusieurs vrais
changements de régime, trois paliers de prix sur une courbe de demande par
exemple, appelle une autre question : combien de ruptures cette courbe
comporte-t-elle réellement et où ? C'est à cela que répond `robust_knees`
(au pluriel). La recherche balaie tous les découpages possibles par
programmation dynamique, note chaque candidat avec un BIC modifié (un score
de qualité d'ajustement qui pénalise chaque rupture supplémentaire, donc
en ajouter une doit se justifier), puis confirme le nombre gagnant par un
test de permutation à seuil ajusté (Bonferroni) pour maîtriser le taux de
faux positifs à mesure que l'espace de recherche grandit.

```python
import numpy as np
from elbow_helper import RobustKneesConfig, robust_knees

rng = np.random.default_rng(3)
x = np.linspace(0, 1, 100)
y = np.piecewise(
    x,
    [x < 0.3, (x >= 0.3) & (x < 0.65), x >= 0.65],
    [lambda t: 3 * t, lambda t: 0.9 + 0.2 * (t - 0.3), lambda t: 0.97 + 2.2 * (t - 0.65)],
) + rng.normal(0, 0.02, x.size)

result = robust_knees(x, y, config=RobustKneesConfig(random_seed=0, fwer_permutations=200))
print(result)
```

```text
Knees(k=2, x=[0.2929, 0.6465])
```

Les deux ruptures repérées tombent près des vrais changements de régime de
la courbe, à 0,3 et 0,65. Contrairement à `robust_knee`, un résultat vide
ici n'est pas une abstention : c'est la conclusion assurée du pipeline que
la courbe n'a pas de vraie rupture, après avoir survécu aux mêmes filtres
de recherche et de faux positifs qu'un résultat non vide aurait dû
franchir. Seul un échec du prétraitement (entrée invalide, trop peu de
données, intervalle nul) renvoie `InvalidKnees` plutôt que `Knees`. Voir
`research/multiknee/RESULTS.md` et `doc/ELBOW-fr.tex` (sections 5 à 20)
pour la validation derrière ce choix.

## Mathématiques

`doc/ELBOW-fr.tex` ([🇬🇧 doc/ELBOW-en.tex](https://github.com/warith-harchaoui/elbow-helper/blob/main/doc/ELBOW-en.tex)) démontre, depuis les premiers principes, chaque formule que ce paquet met en œuvre : la normalisation du pipeline à coude unique, le filtre de Spearman, la recherche de coude par courbe de différence, le regroupement par persistance, la pente de Theil-Sen, le BIC, la validation croisée par blocs, le bootstrap et le test nul, jusqu'à la recherche multi-coudes derrière `robust_knees` (voir aussi `research/multiknee/RESULTS.md`). Son socle de vraisemblance gaussienne, la théorie générale de pourquoi `L := exp(E[log p])` plutôt qu'un produit brut et comment cette même construction se lit sur un modèle de classification, est isolé dans une note compagne, `doc/LIKELIHOOD-fr.tex` ([🇬🇧 doc/LIKELIHOOD-en.tex](https://github.com/warith-harchaoui/elbow-helper/blob/main/doc/LIKELIHOOD-en.tex)), puisque ce socle ne dépend en rien de l'ajustement de courbes. Le texte privilégie l'intuition, avec un exemple travaillé avant chaque formule, pour des lecteurs allant de la fin du lycée jusqu'à un doctorat en mathématiques appliquées. Les références se trouvent dans `doc/references.bib`, avec quelques renvois vers mes [livres IA préférés](https://deraison.ai/ai-books) là où une technique méritait un traitement plus long. Le document est en LaTeX natif, pas en Markdown, vu le public visé : compilez-le avec `latexmk -pdf ELBOW-fr.tex` (ou `ELBOW-en.tex`, `LIKELIHOOD-fr.tex`, `LIKELIHOOD-en.tex`) depuis `doc/` ou lisez directement les copies déjà compilées, `doc/ELBOW-fr.pdf` / `doc/ELBOW-en.pdf` / `doc/LIKELIHOOD-fr.pdf` / `doc/LIKELIHOOD-en.pdf`.

## Paysage

[🗺️ Paysage](https://github.com/warith-harchaoui/elbow-helper/blob/main/PAYSAGE.md) ([🇬🇧 LANDSCAPE.md](https://github.com/warith-harchaoui/elbow-helper/blob/main/LANDSCAPE.md)) : comment `elbow-helper` se positionne face à `kneed`, `ruptures`, `kneebow`, au `KElbowVisualizer` de Yellowbrick, au paquet R `segmented`, à l'estimation à l'œil et au réflexe de demander à un LLM, noté sur 11 critères et placé sur une carte ACP.

## CLI / API / MCP

Au-delà de la bibliothèque Python, `elbow-helper` ouvre trois autres portes sur le même pipeline : une CLI argparse (toujours installée), une CLI click jumelle et une API HTTP avec un serveur MCP monté dessus. Les quatre restent de simples adaptateurs autour du même cœur partagé (`elbow_helper._core_cli`), donc aucune ne peut dériver de ce que renvoie la bibliothèque elle-même.

```bash
pip install -e .                 # bibliothèque + CLI argparse
pip install -e ".[cli]"          # + la CLI click
pip install -e ".[api]"          # + la surface HTTP FastAPI
pip install -e ".[mcp]"          # + le serveur MCP (inclut [api])
```

```bash
# argparse (toujours disponible)
elbow-helper knee --y-values 0,0.1,0.3,0.6,0.85,0.9,0.92,0.93,0.94,0.95

# CLI click
elbow-helper-click knee --y-values 0,0.1,0.3,0.6,0.85,0.9,0.92,0.93,0.94,0.95

# API HTTP
uvicorn elbow_helper.api:app --reload
curl -X POST localhost:8000/knee -d '{"x": [0,0.1,0.3,0.6,0.85,0.9]}'

# MCP (fastapi-mcp monté sur la même app, sur /mcp)
uvicorn elbow_helper.mcp_server:app --port 8021
```

Chaque surface expose les quatre mêmes opérations, `knee`, `elbow`, `diagnostics` et `locator`, qui correspondent une à une à `robust_knee`, `robust_elbow`, `plot_diagnostics` et au `KneeLocator` autonome. Les données entrent en valeurs inline séparées par des virgules, en fichier `.npy` ou en colonne CSV (CLI) ou en corps JSON (`x`/`y` en listes, HTTP). Les surcharges de `RobustKneeConfig` passent en `--config-json '{"bootstrap_replicates": 500}'` (CLI) ou en objet `config_overrides` (HTTP). L'opération `diagnostics` renvoie le SVG lui-même, pas un objet JSON qui l'enveloppe.

## Auteur

[Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

## Remerciements

L'implémentation Kneedle réécrite de zéro dans `elbow_helper/locator.py` suit l'algorithme décrit par Satopää, Albrecht, Irwin et Raghavan (ICDCSW 2011). Sa logique de parcours, sa table d'orientation et son seuil de sensibilité reprennent de près les choix de [`kneed`](https://github.com/arvkevi/kneed), l'implémentation de Kevin Arvai, publiée sous licence BSD à 3 clauses :

> Copyright (c) 2017, Kevin Arvai
> Tous droits réservés.
>
> La redistribution et l'utilisation sous forme source ou binaire, avec ou sans modification, sont autorisées sous réserve des conditions suivantes : (1) les redistributions du code source doivent conserver l'avis de droit d'auteur ci-dessus, cette liste de conditions et la clause de non-responsabilité qui suit ; (2) les redistributions sous forme binaire doivent reproduire l'avis de droit d'auteur ci-dessus, cette liste de conditions et la clause de non-responsabilité qui suit, dans la documentation ou les autres éléments fournis avec la distribution.

`elbow-helper` n'a aucune dépendance d'exécution à `kneed` : l'algorithme est réécrit en NumPy pur, les appels à scipy étant remplacés comme documenté dans `locator.py`.

## Licence

BSD-3-Clause.
