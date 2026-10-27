# Loss check CrowdGPT sur GPU communautaire, vérifié par un backend CPU

Demande (relayée par le propriétaire, 27 oct. 2026) : le serveur du coordinateur CrowdGPT est CPU uniquement (2 cœurs, 15 Go de RAM)
et ne peut pas faire le *loss check* (passe avant de `Crowd-v1` sur le jeu held-out) à chaque round. Ce module déplace le calcul sur
un GPU de la communauté HiveMind et ne laisse au backend qu'une vérification par sondage, faisable sur CPU.

Code (ce dépôt) : `hivemind_loss_check.py` (calcul et vérification ; torch + numpy + bibliothèque standard, safetensors pour lire
les poids) et `hivemind_loss_check_service.py` (coordinateur du backend). Ce sont des copies de `src/hivemind/swarm/crowdgpt_loss_check.py`
et `crowdgpt_loss_check_service.py` de [HiveMind](https://github.com/xaviercallens/HiveMind) (PR #234 et #236), où se trouvent aussi
l'outil en ligne de commande (`scripts/crowdgpt_loss_check.py evaluate | verify | calibrate`), l'agent évaluateur GPU
(`scripts/crowdgpt_loss_check_agent.py`) et les tests (`tests/test_crowdgpt_loss_check*.py`).

## Le protocole

1. **Backend** : publie pour le round la base (révision HF), le candidat (révision HF ou fichier), le fichier held-out (tokens `uint32`)
   et une graine `seed`. Il garde pour lui un secret `server_secret`.
2. **GPU (évaluateur)** : `evaluate` calcule la loss moyenne (cross-entropy du token suivant, même découpe que
   `StreamingShardDataset.get_batch`) de la base et du candidat sur N fenêtres choisies par `seed`, puis signe un reçu (HMAC-SHA256,
   une clé par évaluateur) qui lie ces 2N nombres aux SHA-256 exacts des poids (octets bf16) et des données.
3. **Backend (CPU)** : `verify` refuse sans aucune passe avant si l'évaluateur est inconnu, la signature fausse, les poids ou les
   données ne sont pas ceux attendus, les fenêtres ne viennent pas de la graine, ou une loss est hors de `[0, ln(vocab) + 5]`.
   Sinon il recalcule **k** des 2N nombres, choisis par `HMAC(server_secret, empreinte du reçu)` : l'évaluateur ne peut pas savoir
   d'avance lesquels. Si chaque valeur recalculée est à moins de `tol`, la règle du `LossRatchetGate` s'applique :
   `moyenne(après) - moyenne(avant) <= max_regression_tolerance` → ACCEPTED, sinon HELD_PREVIOUS_HEAD.

```python
from hivemind_loss_check import LossCheckReceipt, CrowdV1, cpu_recompute_fn, verify_receipt, weights_sha256
v = verify_receipt(LossCheckReceipt.from_json(body), evaluator_keys={"hivemind-t4": KEY}, server_secret=SECRET,
                   expected_base_sha256=weights_sha256(base), expected_candidate_sha256=weights_sha256(candidate), heldout=heldout,
                   recompute=cpu_recompute_fn(CrowdV1, base, candidate, heldout, window=2048), k=2, tol=2e-3)
if v.accepted: promote(candidate)   # sinon on garde la tête précédente ; v.reason dit pourquoi
```

## Ce que ça garantit, et ce que ça ne garantit pas

- Un évaluateur qui falsifie m des 2N nombres est pris avec la probabilité `1 - C(2N-m, k) / C(2N, k)` (`detection_probability`).
  Avec N = 32 et k = 2 : 1 nombre faux → 3 %, 8 faux → 23 %, 32 faux → 75 %. **Le sondage seul ne suffit donc pas contre un
  évaluateur malhonnête qui ment sur peu de fenêtres.** Pour un évaluateur de confiance (le T4 du projet) c'est un contrôle de
  cohérence ; pour un GPU communautaire inconnu, demander deux reçus indépendants (`receipts_agree`) et/ou augmenter k.
- `tol` doit être **mesuré** entre le GPU de l'évaluateur et le CPU du backend (`calibrate`). Mesuré ici seulement CPU contre CPU :
  fp32 contre bf16 sur les poids réels, écart ≤ 6e-4 nat sur deux fenêtres de 2048 tokens. GPU contre CPU : **pas encore mesuré**.
- Le calcul se fait en fp32 des deux côtés par défaut : sur CPU, le bf16 est 5,5 fois plus lent que le fp32 (mesuré).

## Mesures sur les vrais poids (`Vxtzq/Crowd-v1` révision `e4fba8b`, CPU de cette machine)

| | valeur |
|---|---|
| Notre modèle contre les classes de `client.py` (copiées par AST), 256 tokens | écart max des logits **0,0** ; même loss |
| Paramètres chargés | 657 787 392 (= `modelSize` de `config.json`) |
| Loss held-out, chunk 5482 (code Rust/Python), 2 fenêtres de 2048 | 11,10 et 8,66 nats |
| Loss sur 512 tokens de `README.md` (anglais/markdown) | 7,80 nats ; les mêmes tokens mélangés : 11,40 |
| Coût CPU d'une fenêtre de 2048 tokens, fp32 | 31 s (4 threads), 52 s (2 threads) |
| Bout en bout, vrais poids, 8 fenêtres de 512, candidat = base + bruit N(0, 1e-3) | `evaluate` 187 s (CPU 4 threads, à la place du GPU) ; `verify` k = 2 en 60 s (2 threads) |
| Reçu honnête | écart recalculé 9,5e-7 ; **HELD_PREVIOUS_HEAD** (delta +0,206 : le bruit dégrade bien le modèle) |
| Reçu falsifié (toutes les losses « après » baissées de 1,0, re-signé) | **REJECTED_SPOT_CHECK** (fenêtre 4 : annoncé 9,830, recalculé 10,830) |

Pour mémoire, `status.json` annonce `lastLoss` 0,079 ; la loss held-out mesurée ci-dessus est de 8 à 11 nats (ln(151669) = 11,93 pour
un tirage uniforme). C'est précisément l'écart qu'un loss check held-out doit voir. Ce n'est mesuré que sur deux sources de texte.

Budget backend : avec k = 2 et des fenêtres de 2048 tokens, environ 2 minutes de CPU (2 cœurs) par round, plus le chargement des poids
(2,6 Go en fp32 par modèle chargé). Des fenêtres plus courtes (`--window 512`) divisent ce coût par environ 4.

## Côté HiveMind

L'évaluateur tourne sur n'importe quel GPU de producteur (`scripts/producer_train_kit.py` pour la mise en place du gouverneur). Sur le
T4 du projet, il doit passer par le gouverneur (inférence d'abord) ; ce câblage n'est **pas** fait ici.

## Brancher le coordinateur dans le backend

```python
from hivemind_loss_check import CrowdV1, tokens_sha256, weights_sha256
from hivemind_loss_check_service import LossCheckCoordinator, fastapi_router

coord = LossCheckCoordinator(
    evaluator_keys={"hivemind-t4": open("keys/hivemind-t4").read().strip().encode()},  # une clé par GPU évaluateur
    server_secret=open("keys/loss_check_secret").read().strip().encode(),          # ne quitte jamais le serveur
    state_path="data/loss_check_state.json",
    loader=lambda job: (base_flat, candidate_flat, heldout_tokens),                 # ce que le backend a déjà en mémoire
    k=2, tol=2e-3, max_regression_tolerance=0.0, required_receipts=1)              # 2 pour des GPU communautaires inconnus
app.include_router(fastapi_router(coord))       # ou exposer coord.handle_get_job / coord.handle_post_receipt dans un autre framework

# à chaque round, une fois le candidat agrégé et publié (par ex. une révision HF) :
job = coord.open_job(base_url=..., base_sha256=weights_sha256(base_flat), candidate_url=..., candidate_sha256=weights_sha256(candidate_flat),
                     heldout_url=..., heldout_range="0-10485759", heldout_sha256=tokens_sha256(heldout_tokens), window=2048, n=32)
# puis : coord.decision(job.job_id) -> {"accepted": True/False, "reason": ...} ; promouvoir le candidat seulement si accepted
```

Côté HiveMind, l'évaluateur tourne avec :
`python scripts/crowdgpt_loss_check_agent.py --server https://server.crowdgpt.net --evaluator-id hivemind-t4 --key-file <fichier> --governor-state <état du gouverneur>`.
