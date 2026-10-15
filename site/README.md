# Studio InfiniteTalk : le site

Une page web protégée par mot de passe. On y dépose un **visage**, un **texte**
ou un **fichier audio**, on choisit un **style**, et on récupère une **vidéo
parlante**.

```
navigateur  ──►  ce site (clés RunPod, travaux, vidéos)
                   │
                   ├─► Chatterbox (RunPod)     texte → voix
                   └─► InfiniteTalk (RunPod)   visage + voix → vidéo
```

Le navigateur ne voit jamais une clé RunPod. Chaque demande devient un travail
enregistré sur le disque du site : on peut fermer la page, la vidéo apparaît
dans « Mes vidéos » quand elle est prête. Si le site redémarre, les travaux en
cours reprennent leur suivi sans être relancés, donc sans être facturés deux fois.

## Ce que coûte une vidéo

Mesuré sur L40S, en mode rapide (bf16 + LoRA FusionX, 8 pas) :

| | Durée |
|---|---|
| Démarrage d'un GPU neuf (poids + chargement) | ~5 min, une fois par worker |
| Un morceau de 3,24 s de vidéo | ~7 min |
| Ordre de grandeur | **~2 min de GPU par seconde de vidéo** |

Les audios de 3,3 s ou moins passent en mode `clip`, soit un seul morceau.
Au-delà, le mode `streaming` enchaîne les morceaux. Il en génère au moins deux
dès que l'audio dépasse 81 images : 3,4 s coûtent donc deux morceaux.

`AUDIO_MAX_S` (60 s par défaut) borne la dépense par demande.

## Les styles

Chaque style agit à deux endroits : la description de scène envoyée à
InfiniteTalk et l'expressivité de Chatterbox quand la voix est synthétisée.
Les deux sont dans `STYLES`, en haut de `app.py`.

| Style | Expressivité de la voix |
|---|---|
| Sérieux / professionnel | 0,40 |
| Content / souriant | 0,60 |
| Enthousiaste / dynamique | 0,85 |
| Bienveillant / pédagogue | 0,50 |

## Variables d'environnement

| Variable | Obligatoire | Rôle |
|---|---|---|
| `SITE_PASSWORD` | oui | Mot de passe d'accès. Sans lui, le site refuse tout. |
| `SITE_USER` | non | Identifiant, `studio` par défaut |
| `RUNPOD_KEY` | oui | Clé API RunPod |
| `INFINITETALK_ENDPOINT` | oui | ID de l'endpoint InfiniteTalk |
| `CHATTERBOX_ENDPOINT` | pour le texte | ID de l'endpoint Chatterbox. Sans lui, seul l'audio est accepté. |
| `CHATTERBOX_KEY` | non | Si différente de `RUNPOD_KEY` |
| `DATA_DIR` | non | Dossier des travaux et des vidéos, `/data` dans le conteneur |
| `AUDIO_MAX_S` | non | Durée maximale d'audio, 60 s par défaut |
| `TEXTE_MAX` | non | Longueur maximale du texte, 900 caractères par défaut |

## Lancer en local

```bash
pip install -r requirements.txt
SITE_PASSWORD=... RUNPOD_KEY=... INFINITETALK_ENDPOINT=... \
CHATTERBOX_ENDPOINT=... uvicorn app:app --port 8000
```

Puis ouvrir http://localhost:8000.

## Mettre en ligne

Le dossier contient un `Dockerfile`. Tout hébergeur qui construit une image
Docker depuis GitHub convient : Railway, Render, Fly.io…

1. Créer le service à partir de ce dépôt, avec **`site/`** comme dossier racine.
2. Renseigner les variables ci-dessus dans l'interface de l'hébergeur, jamais
   dans le code.
3. **Monter un disque persistant sur `/data`.** Sans lui, les travaux et les
   vidéos disparaissent à chaque redéploiement.
4. Le site n'a pas besoin de GPU : la plus petite instance suffit.

## Côté RunPod

- **Endpoint InfiniteTalk** : image `ghcr.io/marwanefo/infinitetalk-serverless`,
  GPU 48 Go PRO, Container Disk 130 Go, variables `IT_MODE_POIDS=rapide` et
  `IT_T5_CPU=1`.
- **Max workers** règle combien de vidéos sont calculées en parallèle. Chaque
  nouveau worker paie ses ~5 min de démarrage.
- Le site porte lui-même le délai d'exécution de chaque job : il le calcule
  d'après la durée de l'audio.
