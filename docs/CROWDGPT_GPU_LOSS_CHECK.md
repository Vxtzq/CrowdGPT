# Loss check CrowdGPT sur GPU communautaire, vérifié par un backend CPU

Demande (27 oct. 2026) : le serveur du coordinateur CrowdGPT est CPU uniquement (2 cœurs, 15 Go de RAM)
et ne peut pas faire le *loss check* (passe avant de `Crowd-v1` sur le jeu held-out) à chaque round. Ce module déplace le calcul sur
un GPU de la communauté HiveMind et ne laisse au backend qu'une vérification par sondage, faisable sur CPU.

Code dans ce dépôt : `hivemind_loss_check.py` (protocole stdlib + modèle `CrowdV1` avec support optionnel `CrowdEngram` `engramBuckets=227865`),
service coordinateur `hivemind_loss_check_service.py` (`GET /fl/loss_check/job`, `POST /fl/loss_check/receipt`), et tests CI `tests/test_hivemind_integration.py`.

## Le protocole

1. **Backend** : publie pour le round la base (révision HF), le candidat (révision HF ou fichier), le fichier held-out (tokens `uint32`)
   et une graine `seed`. Il garde pour lui un secret `server_secret`.
2. **GPU (évaluateur)** : `make_receipt` calcule la loss moyenne (cross-entropy du token suivant, même découpe que
   `StreamingShardDataset.get_batch`) de la base et du candidat sur N fenêtres choisies par `seed`, puis signe un reçu (HMAC-SHA256,
   une clé par évaluateur) qui lie ces 2N nombres aux SHA-256 exacts des poids (octets bf16) et des données.
3. **Backend (CPU)** : `verify_receipt` refuse sans aucune passe avant si l'évaluateur est inconnu, la signature fausse, les poids ou les
   données ne sont pas ceux attendus, les fenêtres ne viennent pas de la graine, ou une loss est hors de `[0, ln(vocab) + 5]`.
   Sinon il recalcule **k** des 2N nombres, choisis par `HMAC(server_secret, empreinte du reçu)` : l'évaluateur ne peut pas savoir
   d'avance lesquels. Si chaque valeur recalculée est à moins de `tol`, la règle du `LossRatchetGate` s'applique :
   `moyenne(après) - moyenne(avant) <= max_regression_tolerance` → ACCEPTED, sinon HELD_PREVIOUS_HEAD.

```python
from hivemind_loss_check import LossCheckReceipt, CrowdV1, CROWD_V1_ENGRAM_CONFIG, cpu_recompute_fn, verify_receipt, weights_sha256

v = verify_receipt(
    LossCheckReceipt.from_json(body),
    evaluator_keys={"hivemind-t4": KEY, "hivemind-rtx2070": KEY2},
    server_secret=SECRET,
    expected_base_sha256=weights_sha256(base),
    expected_candidate_sha256=weights_sha256(candidate),
    heldout=heldout,
    recompute=cpu_recompute_fn(lambda: CrowdV1(CROWD_V1_ENGRAM_CONFIG), base, candidate, heldout, window=2048),
    k=2,
    tol=2e-3,
)
if v.accepted:
    promote(candidate)  # sinon on garde la tête précédente ; v.reason dit pourquoi
```

## Support `CrowdEngram` (`Step 348+`)

Depuis l'activation de la table bigramme `CrowdEngram` (`227 865 × 1 536` paramètres en `bf16`) sur Hugging Face (`Vxtzq/Crowd-v1`), `hivemind_loss_check.py` fournit :
- `CROWD_V1_CONFIG` (`657 787 392` paramètres tronc seul, rétro-compatible avec `model_bf16.safetensors` plat) ;
- `CROWD_V1_ENGRAM_CONFIG` (`engramBuckets=227865`) et `load_named_crowd_v1_safetensors(path)` pour charger directement `model.safetensors` de Hugging Face (`wte`, `blocks.0..23`, `ln`, `engram.table.weight`).
