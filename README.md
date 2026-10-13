# InfiniteTalk sur RunPod Serverless

Worker RunPod pour [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
(MeiGen-AI, Apache 2.0) : generation de video parlante a partir d'une image
ou d'une video de depart, plus une piste audio.

## Ce que ce modele fait, et ce qu'il coute

InfiniteTalk n'est pas comparable a MuseTalk. MuseTalk incruste une bouche
dans une video existante en une seule passe. InfiniteTalk **regenere chaque
image** par diffusion video (Wan2.1-I2V-14B, 40 pas d'echantillonnage par
morceau de 81 images). La tete, le regard et le buste bougent, et la
synchronisation labiale est supervisee.

Le prix de ce saut de qualite est le temps de calcul, et il est lourd :

| Moteur | Secondes de GPU par seconde de video |
|---|---|
| MuseTalk 1.5 | **3,1** (mesure) |
| MuseTalk + GFPGAN pur | **10,8** (mesure) |
| LatentSync 1.6 | **5,2** (mesure) |
| InfiniteTalk | **a mesurer** — attendre un ordre de grandeur de plus |

Ce dernier chiffre n'est pas une mesure et ne doit pas etre presente comme
telle. Il faut lancer un job en `benchmark: true` et lire `s_par_seconde`
dans la reponse avant d'engager du volume. Les leviers connus pour le
reduire sont `sample_steps`, `use_teacache`, et la LoRA FusionX qui
descend a 8 pas au lieu de 40.

> **Un successeur existe.** Le README amont annonce, le 21 mai 2026,
> [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) :
> meme equipe, Whisper-Large au lieu de wav2vec2 pour une synchronisation
> labiale plus precise, et **8 pas d'inference par distillation** au lieu
> de 40. Sur les deux criteres qui comptent ici — qualite du lip-sync et
> vitesse — il remplace InfiniteTalk. Ce depot construit bien InfiniteTalk,
> comme demande ; la structure (Dockerfile, handler, journal de build) se
> transpose presque telle quelle si vous voulez changer de modele.

## 1. Les poids : 93 Go sur un volume reseau

Ils ne sont **pas** dans l'image Docker : une image de cette taille ne se
telecharge pas a chaque demarrage de worker.

| Jeu | Taille |
|---|---|
| `Wan2.1-I2V-14B-480P` | 82,3 Go |
| `InfiniteTalk/single/infinitetalk.safetensors` | 9,9 Go |
| `chinese-wav2vec2-base` | 0,4 Go |
| **total** | **~92,6 Go** |

Creer un volume reseau de **120 Go** (le telechargement passe par un cache
temporaire), l'attacher a un pod, puis :

```bash
bash download_weights.sh /runpod-volume/infinitetalk/weights
```

Options utiles :

```bash
QUANT=fp8 bash download_weights.sh ...   # +19,5 Go, pour moins de 80 Go de VRAM
LORA=1    bash download_weights.sh ...   # +1,2 Go, permet 8 pas au lieu de 40
```

Le script est reprenable : s'il s'interrompt, le relancer continue ou il
s'etait arrete. Il verifie les fichiers attendus a la fin.

### Les references

Images ou videos de depart, adressables par nom court dans les requetes :

```
/runpod-volume/infinitetalk/refs/inna.png
/runpod-volume/infinitetalk/refs/inna.mp4
```

## 2. Construire l'image

Pousser sur `main` declenche le build. L'image sort en
`ghcr.io/marwanefo/infinitetalk-serverless:latest`.

**En cas d'echec, ne pas ouvrir l'interface GitHub.** La fin du journal,
avec les lignes d'erreur isolees en tete, est lisible directement :

```
https://raw.githubusercontent.com/MarwaneFo/infinitetalk/build-logs/infinitetalk.log
```

## 3. Configurer l'endpoint

### Le GPU

Wan2.1-14B en bf16 occupe environ 28 Go pour le DiT seul, plus 11 Go de T5
et 5 Go de CLIP.

| VRAM | Verdict |
|---|---|
| 80 Go (A100, H100) | confortable, le plus rapide |
| 48 Go (A40, A6000, L40S) | il faut `IT_LOW_VRAM=1` ou `IT_QUANT=fp8`, et c'est plus lent |
| 24 Go | ne pas essayer |

> **Piege deja paye.** La categorie generique **24 Go** de RunPod contient
> des tranches MIG de Blackwell PRO 6000. Decocher `PRO` ne protege pas.
> torch 2.4.1 n'a aucun noyau pour sm_120 : le worker meurt sans laisser
> de trace et les jobs s'empilent en file sans explication. Le handler
> refuse maintenant ces cartes avec un message explicite, mais mieux vaut
> prendre **48 ou 80 Go** d'emblee.

### Les delais

- **Execution timeout** : genereux. Le premier job inclut le chargement
  des 93 Go depuis le volume, qui prend plusieurs minutes, puis la
  diffusion. Compter large et mesurer.
- **Idle timeout** : eleve pendant un lot (300 s ou plus), sinon chaque
  job repaie le chargement complet.
- **Max workers** : chaque worker charge sa propre copie des poids. La
  parallelisation divise l'attente, pas le cout.

### Les variables d'environnement

| Variable | Defaut | Role |
|---|---|---|
| `IT_WEIGHTS` | auto | racine des poids, si elle n'est pas trouvee seule |
| `IT_REFS` | auto | dossier des images et videos de reference |
| `IT_VARIANTE` | `single` | `single` (une personne) ou `multi` (deux) |
| `IT_QUANT` | vide | `fp8` ou `int8`, pour tenir en moins de VRAM |
| `IT_LOW_VRAM` | `0` | `1` deleste le DiT : tient en 48 Go, mais plus lent |
| `IT_T5_CPU` | `0` | `1` garde T5 sur le CPU, libere 11 Go de VRAM |
| `IT_LORA` | vide | chemin d'une LoRA (FusionX : 8 pas au lieu de 40) |
| `IT_LORA_SCALE` | `1.2` | intensite de la LoRA |
| `IT_PROMPT` | neutre | invite par defaut si la requete n'en donne pas |

## 4. L'API

### `POST /v2/<endpoint>/run`

Audio obligatoire, reference obligatoire. Chacun des deux accepte trois
formes : base64, URL, ou nom de fichier sur le volume.

| Champ | Type | Defaut | Role |
|---|---|---|---|
| `audio_base64` | chaine | — | WAV encode en base64 |
| `audio_url` | chaine | — | ou une URL a telecharger |
| `audio` | chaine | — | ou un nom de fichier dans `refs/` |
| `image_base64` / `video_base64` | chaine | — | reference de depart |
| `image_url` / `video_url` | chaine | — | ou une URL |
| `reference` | chaine | — | ou un nom de fichier dans `refs/` |
| `prompt` | chaine | `IT_PROMPT` | description de la scene, en anglais |
| `size` | chaine | `infinitetalk-480` | ou `infinitetalk-720` |
| `mode` | chaine | `streaming` | `streaming` (long) ou `clip` (un morceau) |
| `sample_steps` | entier | `40` | 1 a 60. **Le levier principal de vitesse.** 8 avec la LoRA FusionX |
| `text_guide_scale` | decimal | `5.0` | 1 avec la LoRA |
| `audio_guide_scale` | decimal | `4.0` | 3 a 5 ; monter pour un meilleur lip-sync. 2 avec la LoRA |
| `motion_frame` | entier | `9` | images de recouvrement entre morceaux |
| `max_frame_num` | entier | `1000` | plafond en images (1000 ≈ 40 s) |
| `seed` | entier | `42` | rend le rendu reproductible |
| `color_correction_strength` | decimal | `1.0` | corrige la derive de couleur sur les longues videos |
| `use_teacache` | booleen | `false` | accelere, degrade un peu |
| `teacache_thresh` | decimal | `0.2` | agressivite de TeaCache |
| `use_apg` | booleen | `false` | guidage projete adaptatif |
| `benchmark` | booleen | `false` | renvoie les temps **sans** la video |

Le champ `webhook` se place **a cote de** `input`, pas dedans.

### La reponse

```json
{
  "video_base64": "...",
  "duration_s": 11.7,
  "audio_s": 11.7,
  "size": "infinitetalk-480",
  "mode": "streaming",
  "sample_steps": 40,
  "text_guide_scale": 5.0,
  "audio_guide_scale": 4.0,
  "seed": 42,
  "model": "InfiniteTalk + Wan2.1-I2V-14B-480P",
  "variante": "single",
  "quant": null,
  "gpu": "NVIDIA A40",
  "etapes": { "chargement": 412.0, "entrees": 0.3, "audio": 2.1,
              "diffusion": 1840.5, "encodage": 6.2 },
  "total_s": 2261.1,
  "s_par_seconde": 193.26
}
```

`s_par_seconde` est le chiffre a surveiller : c'est le cout reel du
modele, dans la meme unite que le tableau du haut.

### Exemple

```python
import base64, requests, time

CLE = "..."          # jamais dans un navigateur
ENDPOINT = "..."

with open("voix.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

r = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT}/run",
    headers={"Authorization": f"Bearer {CLE}"},
    json={"input": {
        "audio_base64": audio,
        "reference": "inna.png",
        "prompt": "A woman speaks to the camera in a bright office, "
                  "calm and professional, steady framing.",
        "size": "infinitetalk-480",
        "sample_steps": 40,
        "audio_guide_scale": 4.0,
    }},
    timeout=60,
).json()

job = r["id"]
while True:
    s = requests.get(f"https://api.runpod.ai/v2/{ENDPOINT}/status/{job}",
                     headers={"Authorization": f"Bearer {CLE}"},
                     timeout=30).json()
    if s["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
        break
    time.sleep(10)

sortie = s.get("output") or {}
# Un job COMPLETED peut porter une erreur applicative : toujours verifier.
if "error" in sortie:
    raise RuntimeError(sortie["error"])
open("sortie.mp4", "wb").write(base64.b64decode(sortie["video_base64"]))
```

## 5. Pieges a connaitre

**Un `COMPLETED` peut porter une erreur.** Le handler attrape ses
exceptions et renvoie `{"error": ..., "trace": ...}` plutot que de mourir
en silence. Cote client, tester `output.error` avant de lire
`video_base64`.

**La reponse base64 a un plafond.** Au-dela d'environ une minute de video,
la reponse devient trop lourde. Decouper par diapositive, ce qui tombe
bien pour de l'e-learning.

**Ne reessayer que les echecs de transport survenus avant toute reponse.**
Un `POST /run` qui a abouti a cree un job facture ; le rejouer le facture
deux fois.

**Le commit amont est fige** dans le Dockerfile
(`INFINITETALK_REF=50aa0a9`). C'est volontaire : suivre `main` ferait
changer le comportement au prochain rebuild sans que rien ne l'annonce.

**Le prompt compte.** Wan2.1 est un modele texte-vers-video : une
description vague donne une scene instable. Decrire le cadrage, la
lumiere et l'attitude, en anglais.

**La derive de couleur** apparait au-dela d'une minute en mode image.
`color_correction_strength` l'atenue. La LoRA FusionX l'aggrave.

## Licences

InfiniteTalk est sous Apache 2.0. Wan2.1 a sa propre licence, a verifier
avant un usage commercial. Et cloner la voix ou le visage d'une personne
reelle demande son accord.
