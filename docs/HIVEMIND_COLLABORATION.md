# Accord de collaboration CrowdGPT × HiveMind — proposition

**Statut : proposition, à confirmer par les deux parties** (un commentaire « d'accord » de chacun sur
[HiveMind#204](https://github.com/xaviercallens/HiveMind/issues/204) suffit). Rédigée le 27 octobre 2026 à partir de l'échange du
26 octobre sur #204 (demande de @Vxtzq) et de la réponse de @xaviercallens. Rien ici n'engage d'argent : c'est un accord entre deux
projets open source.

Parties : **CrowdGPT** (@Vxtzq — `Vxtzq/CrowdGPT`, modèle `Vxtzq/Crowd-v1`, coordinateur `server.crowdgpt.net`) et **HiveMind / La
Ruche** (@xaviercallens — `xaviercallens/HiveMind`).

## 1. Ce que chacun a demandé ou proposé (source : #204)

| Qui | Demande ou proposition |
|---|---|
| CrowdGPT | Le serveur du coordinateur est CPU uniquement (2 cœurs, 15 Go de RAM, suffisant pour environ 10 contributeurs ; plus de ressources possibles si la demande augmente). Il ne peut pas faire le *loss check* (passe avant du modèle). Demande : le faire sur un GPU communautaire, puis l'envoyer à son backend. |
| CrowdGPT | Compter les contributeurs HiveMind comme contributeurs CrowdGPT, avec un signe distinctif (*flair*) sur le leaderboard. |
| HiveMind | Distribuer `Crowd-v1`, et éventuellement l'entraîner, sur HiveMind. |
| HiveMind | Fournir du calcul GPU (ancien GPU personnel ou VM GCP en mode *spot* pour réduire les coûts). |
| HiveMind | Aller plus loin avec « bring your own bee » (chacun apporte son modèle spécialisé). |

## 2. Engagements proposés

**HiveMind fournit**
1. Le *loss check* sur GPU : `hivemind_loss_check.py` (calcul et vérification) et `hivemind_loss_check_service.py` (coordinateur
   pour le backend : `GET /fl/loss_check/job`, `POST /fl/loss_check/receipt`), proposés dans une PR sur `Vxtzq/CrowdGPT` ; l'agent
   évaluateur `scripts/crowdgpt_loss_check_agent.py` côté HiveMind.
2. Au moins un évaluateur GPU (identifiant `hivemind-t4`) qui prend les jobs **uniquement quand le GPU est libre** selon la politique
   déjà en place sur le T4 : inférence d'abord ; calcul seulement après 30 min sans requête la nuit (22 h – 7 h, heure de Paris) et
   2 h le jour.
3. Les mesures publiées avec leurs limites (voir `docs/CROWDGPT_GPU_LOSS_CHECK.md`) ; aucun chiffre non mesuré.

**CrowdGPT fournit**
1. L'intégration du coordinateur dans son backend, l'ouverture d'un job par candidat, le choix et l'hébergement du jeu held-out.
2. Une clé d'évaluateur par GPU évaluateur (transmise hors de GitHub), et le secret serveur qui reste sur le backend.
3. Le crédit des contributeurs HiveMind sur le leaderboard CrowdGPT avec un *flair* HiveMind.

**Ensemble**
1. Servir `Crowd-v1` sur le routeur HiveMind (déjà au catalogue sous `hivemind/crowd-v1`) avec un lien vers CrowdGPT.
2. Un rapport commun lorsque le premier round vérifié a tourné (chiffres réels uniquement).

## 3. Règles de confiance (non négociables des deux côtés)

- Un candidat n'est promu que si le reçu est signé par un évaluateur connu, porte sur **exactement** les poids et les données du job
  (SHA-256) et passe les contrôles par sondage refaits sur le CPU du backend.
- Pour un GPU communautaire inconnu : **deux évaluateurs indépendants** dont les chiffres concordent (`required_receipts=2`), car le
  sondage seul ne détecte qu'une partie des falsifications (par exemple 3 % pour un seul nombre faux sur 64 avec k = 2).
- La tolérance GPU/CPU est **mesurée** (`calibrate`) avant la mise en service, pas supposée.
- Aucun prompt ni donnée personnelle d'utilisateur n'est échangé : seulement des poids publics, des tokens held-out et des pertes.

## 4. Crédit, points et argent

- Les contributions sont créditées en **points** (HiveMind Reflexion Points, HMT) et sur le leaderboard CrowdGPT. Les points n'ont
  **aucune valeur monétaire** et ne sont ni achetés ni vendus.
- Les coûts de calcul restent à la charge de celui qui fournit la machine ; aucune partie ne facture l'autre.

## 5. Licences

- `Vxtzq/CrowdGPT` est sous GPL-3.0 ; `Vxtzq/Crowd-v1` est sous MIT (carte du modèle). Les fichiers versés dans `Vxtzq/CrowdGPT`
  y sont distribués sous GPL-3.0.
- **À décider par @xaviercallens** : le dépôt HiveMind n'a pas de fichier LICENSE à la racine ; il en faut un pour que la
  réutilisation dans l'autre sens soit claire.

## 6. Prochaines étapes, à discuter (non engagées)

- **Bring your own bee** : une page web où chacun enregistre son modèle spécialisé. @xaviercallens a un prototype sur GCP et propose un
  accès ; tout accès à une infrastructure se décide à part, explicitement, par son propriétaire.
- Promotion auprès d'autres développeurs, au retour de @xaviercallens.

## 7. Durée et sortie

Chaque partie peut arrêter à tout moment en le disant sur #204. Le code déjà publié reste sous sa licence ; les clés d'évaluateur sont
alors révoquées par CrowdGPT.
