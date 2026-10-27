# Paysage

[🇫🇷](PAYSAGE.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](LANDSCAPE.md)

Comment `elbow-helper` se compare à *toutes les autres façons de trouver un coude*. Chaque approche est notée sur **le travail de ce projet : rapporter un coude seulement quand les preuves le soutiennent et le dire explicitement sinon**, sans être pénalisée pour exceller à un autre travail (détection brute de points de rupture, visualisation exploratoire, inférence statistique en R).

## En un coup d'œil

<!-- TABLE:START -->
| Outil de détection de coude | Robustesse au bruit | Inférence automatique de la forme | Abstention explicite | Points de rupture multiples | Test de significativité statistique | Quantification de l'incertitude | Rigueur de sélection de modèle | Empreinte de dépendances | Facilité d'utilisation en un appel | Reproductibilité | Mathématiques publiées |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **elbow-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| kneed | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| ruptures | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| kneebow | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| KElbowVisualizer | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Paquet R segmented | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Estimation visuelle manuelle | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| Demander à un LLM | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
<!-- TABLE:END -->

## Carte de positionnement

<!-- FIGURE:START -->
Représentation 2D du tableau ci-dessus.

![Carte de positionnement](https://raw.githubusercontent.com/warith-harchaoui/elbow-helper/main/assets/paysage.png)

La carte est un résumé en 2D des 11 critères : à lire comme une forme, pas comme un classement. « elbow-helper » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal : Simplicité ↔ Précision** et **Vertical : Clarté ↔ Adaptabilité**.
<!-- FIGURE:END -->

## Positionnement

`elbow-helper` ne concurrence pas `ruptures` sur la recherche multi-points de rupture, ni le jugement d'un statisticien sur un jeu de données déjà bien compris. Il répond à une question plus étroite et plus difficile : étant donné une seule courbe et aucun autre contexte, un coude candidat est-il assez solide pour qu'on lui fasse confiance ? La plupart des outils du domaine renvoient toujours une estimation ponctuelle (`kneed`, `kneebow`, le `KElbowVisualizer` de Yellowbrick) ou demandent à un humain de fournir le jugement qu'`elbow-helper` automatise (estimation visuelle, invite adressée à un LLM). Son analogue le plus proche n'est pas un autre localisateur de coude mais le paquet `segmented` de R, qui partage le même réflexe : une affirmation de coude devrait s'accompagner d'une erreur type, pas seulement d'une coordonnée.

## Fiche par outil

### kneed
L'implémentation de référence dont le propre localisateur d'`elbow-helper` est un portage, créditée dans la section Remerciements du README. Excellent dans son unique tâche : étant donné une courbe et un couple `curve`/`direction` explicite, il renvoie un point unique, de façon déterministe, en une ligne de code. Il ne porte aucune notion de confiance et n'offre aucun chemin vers « il n'y a pas de coude ici » au-delà d'un `None` peu explicite. `elbow-helper` enveloppe cette même idée géométrique dans la machinerie de confirmation que kneed laisse entièrement à la charge de l'appelant.

### ruptures
Le bon outil quand la question posée est réellement « combien de points de rupture et où » sur un signal, avec PELT, la segmentation binaire et la recherche par fenêtre déjà intégrés. Il est indifférent à ce que « coude » veut même dire : une courbe à rendements décroissants et un changement de moyenne dans du bruit sont, pour lui, le même genre d'objet. La propre recherche multi-coudes d'`elbow-helper` (`research/multiknee/`) teste ce même programme dynamique proche de PELT qu'utilise `ruptures`, en y ajoutant la couche de sélection de modèle (mBIC, FWER) que `ruptures` laisse à l'utilisateur le soin de configurer.

### kneebow
Un petit paquet construit autour de la même idée de rotation que le localisateur d'`elbow-helper` : faire pivoter la courbe jusqu'à ce qu'elle pointe vers le bas, puis prendre l'extremum. Géométrique, rapide, assez court pour se lire d'une traite. Comme kneed, il s'engage sur une réponse à chaque appel, sans recherche sur une échelle de lissage, sans bootstrap et sans test d'hypothèse nulle derrière le nombre qu'il renvoie. Son installation pèse aussi plus lourd que sa taille ne le laisse croire : le paquet publié déclare numpy, matplotlib et scikit-learn comme dépendances d'exécution, donc il tire la pile graphique et la pile d'apprentissage automatique dont `elbow-helper` se passe délibérément.

### Yellowbrick KElbowVisualizer
La façon la plus répandue, en pratique, de trouver le coude du k-means : il ajuste lui-même l'estimateur sur une plage de k, note la courbe obtenue, la trace, puis — avec `locate_elbow=True`, sa valeur par défaut — marque le pli selon la même construction Kneedle qu'implémente `kneed`. Le réduire à un tracé lui fait donc du tort : c'est un localisateur entouré d'un graphique, qui hérite du comportement de kneed (répondre à chaque appel) en même temps que de sa géométrie. Ce qu'il n'est *pas*, c'est un détecteur général pour une courbe qu'on tient déjà : il possède le balayage sur k, qu'il note avec `distortion`, `silhouette` ou `calinski_harabasz` plutôt que d'accepter un couple `(x, y)` quelconque. Il hérite en outre de scikit-learn et de matplotlib comme dépendances obligatoires. Le propre exemple k-means d'`elbow-helper` (`doc/ELBOW-en.tex`, partie I) vise exactement cette courbe, avec la chaîne de confirmation qu'une lecture Kneedle ne fait jamais tourner.

### Paquet R segmented
L'alternative de loin la plus rigoureuse statistiquement : une véritable régression en ligne brisée, avec erreurs types, test de Davies et prise en charge de points de rupture multiples (`psi`), dans un paquet R mature et validé par les pairs, que Vito M. R. Muggeo maintient et sur lequel il publie depuis 2003. Ce qu'il exige de l'utilisateur est une réelle aisance statistique : une formule de modèle, des valeurs de départ pour les points de rupture et R lui-même, plutôt qu'un simple appel Python en une ligne. `elbow-helper` automatise les parties de ce flux de travail, recherche sur l'échelle de lissage, bootstrap, test nul, que `segmented` laisse au jugement d'un analyste humain.

### Estimation visuelle manuelle
La référence universelle : regarder le tracé, décider où il plie. Un analyste attentif peut honnêtement dire « je ne vois pas de coude net ici », ce qui est déjà plus que ce que la plupart des outils automatisés savent faire, mais le jugement ne se reproduit pas d'une personne à l'autre, ni même d'un jour à l'autre chez la même personne. Il ne passe pas non plus à l'échelle au-delà d'une poignée de courbes.

### Demander à un LLM
Une variante moderne de l'estimation visuelle : coller les données ou une capture d'écran dans un modèle conversationnel et demander où se situe le coude. Les grands modèles de langage décrivent une forme en mots avec aisance et peuvent nuancer leur réponse si on le leur demande, mais la réponse n'est pas déterministe d'une exécution à l'autre, ne porte aucune incertitude calibrée et ne repose sur aucune dérivation reproductible qu'un lecteur pourrait vérifier.

## La thèse

Chaque alternative ici excelle à quelque chose qu'`elbow-helper` ne cherche pas à être : `ruptures` pour compter les points de rupture, `segmented` pour l'inférence statistique complète, Yellowbrick pour prendre en charge le balayage sur k de bout en bout, un LLM pour décrire une forme en mots simples. Ce qu'aucune d'elles ne fait par défaut, c'est refuser de répondre. `elbow-helper` existe pour le cas plus étroit où une mauvaise réponse coûte plus cher qu'une absence de réponse et traite « il n'y a pas de coude net ici » comme un résultat à part entière, aussi bien étayé que les autres.

## Quand choisir quoi

Un paysage qui ne recommande jamais que soi-même est une publicité. Voici donc l'orientation complète, y compris les cas qui relèvent d'ailleurs.

- **Choisir `elbow-helper`** quand un mauvais coude coûte plus cher que pas de coude du tout : un seuil d'autoscaling, un nombre de groupes qui part en production, une décision « nous avons assez de données maintenant ». Ce qui revient est soit une estimation ponctuelle avec son intervalle à 90 %, son taux de détection par bootstrap et sa p-valeur sous modèle nul, soit l'un des quinze codes qui nomment la porte qui s'est fermée.
- **Choisir `kneed`** quand on sait déjà que la courbe plie et qu'on veut seulement la coordonnée. Un appel, deux dépendances, rien à régler. Le localisateur d'`elbow-helper` en est un portage : ce qu'on abandonne ici, c'est la couche de confirmation, pas la géométrie.
- **Choisir `ruptures`** quand la question est « combien de points de rupture et où » sur un signal long. `robust_knees` répond à cette question pour les courbes à rendements décroissants, mais `ruptures` est la bibliothèque de segmentation mature et abondamment citée, avec les fonctions de coût et les stratégies de recherche qui vont avec.
- **Choisir `segmented` de R** quand le point de rupture doit survivre à une relecture par les pairs : une erreur type sur le point de rupture lui-même, le test de Davies, des effets aléatoires. Rien en Python ne le remplace ; `elbow-helper` n'essaie pas.
- **Choisir le `KElbowVisualizer` de Yellowbrick** quand on est déjà dans un pipeline scikit-learn et qu'on veut le balayage sur k ajusté, noté puis tracé par un seul objet. C'est un travail qu'`elbow-helper` ne fait pas du tout.
- **Choisir ses propres yeux** quand il y a une seule courbe, qu'on va la tracer de toute façon et que personne n'aura à reproduire la décision plus tard.

**Quand ne pas choisir `elbow-helper` :**

- **Les courbes applicatives courtes, avec les réglages livrés.** `min_samples` vaut 20 et `min_side_points` vaut 5 : un graphe d'éboulis k-means poussé jusqu'à k = 10 est donc rejeté d'emblée avec `INSUFFICIENT_DATA`. Les valeurs par défaut sont calibrées pour des courbes de mesure plus longues et plus bruitées. Un profil réajusté pour les courbes courtes vit dans `tests/test_real_world_examples.py` et fonctionne, mais choisir ces seuils est précisément le jugement que ce paquet prend à votre place le reste du temps.
- **Quand un nombre est exigé à chaque appel.** Toute la valeur de ce paquet tient à ce qu'il refuse parfois. Si un système en aval ne sait pas traiter une abstention, `kneed` ou `kneebow` répondront toujours et vous laisseront la question de la confiance.
- **La segmentation de signal.** Changements de moyenne dans du bruit, plusieurs régimes le long d'une série temporelle, détection de rupture en ligne : c'est le problème de `ruptures`, pas celui-ci.

## Sources et date de relevé

Les affirmations ci-dessus ont été vérifiées le **2026-10-27**. Les versions et les dépendances déclarées proviennent des métadonnées publiées par chaque projet à cette date. Les notes en étoiles relèvent du jugement éditorial de ce projet face au travail énoncé en tête de page, pas d'un banc d'essai ; la ligne d'`elbow-helper` est une auto-évaluation qu'il vaut mieux lire comme telle.

Deux lignes ne nomment aucun logiciel et ne portent donc aucune version : « Estimation visuelle manuelle » et « Demander à un LLM » sont des pratiques dont les notes sont un argument plutôt qu'une mesure.

- [kneed](https://pypi.org/project/kneed/) — 0.9.0 ; déclare `numpy` et `scipy` ; [documentation](https://kneed.readthedocs.io/en/latest/). Implémente l'algorithme Kneedle de Satopää, Albrecht, Irwin et Raghavan, *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*, 31st International Conference on Distributed Computing Systems Workshops (2011), p. 166-171.
- [ruptures](https://pypi.org/project/ruptures/) — 1.1.10 ; déclare `numpy` et `scipy` ; [documentation](https://centre-borelli.github.io/ruptures-docs/). Cite Truong, Oudre et Vayatis, *Selective review of offline change point detection methods*, Signal Processing 167 (2020) 107299, [doi:10.1016/j.sigpro.2019.107299](https://doi.org/10.1016/j.sigpro.2019.107299).
- [kneebow](https://pypi.org/project/kneebow/) — 1.0.4, publié le 2025-08-29 ; déclare `numpy`, `matplotlib` et `scikit-learn`.
- [Yellowbrick](https://pypi.org/project/yellowbrick/) — 1.5 ; déclare `matplotlib`, `scipy`, `scikit-learn`, `numpy` et `cycler`. La valeur par défaut de `locate_elbow` et les trois métriques prises en charge sont documentées sur la [page du KElbowVisualizer](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html).
- [segmented](https://cran.r-project.org/package=segmented) sur le CRAN — 2.2-2, publié le 2026-09-11 ; auteur et mainteneur Vito M. R. Muggeo.
- `elbow-helper` lui-même — ce dépôt en 0.4.0. Les seuils cités plus haut sont les valeurs par défaut de `src/elbow_helper/config.py` ; le profil réajusté pour courbes courtes se trouve dans `tests/test_real_world_examples.py`.
