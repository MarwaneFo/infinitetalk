"""Handler RunPod Serverless pour InfiniteTalk (MeiGen-AI).

InfiniteTalk n'est pas un modele d'incrustation comme MuseTalk. C'est une
diffusion video complete -- Wan2.1-I2V-14B, 40 pas d'echantillonnage par
morceau -- qui regenere chaque image en entier. La tete, le buste et le
regard bougent, pas seulement la bouche. Le calcul s'en ressent lourdement :
compter un ordre de grandeur de plus que MuseTalk, a mesurer avant
d'engager du volume.

Trois choix structurants, chacun paye sur les workers precedents :

1. Les poids (environ 93 Go) vivent sur le volume reseau, jamais dans
   l'image. Une image Docker de cette taille ne se telecharge pas a
   chaque demarrage de worker.

2. Le modele se charge au PREMIER JOB, pas a l'import du module. Un echec
   de chargement revient alors dans la reponse du job, lisible dans la
   console. Charge a l'import, le worker meurt en silence : les travaux
   s'empilent en file et rien n'explique pourquoi.

3. Le pipeline reste resident entre deux jobs. Lire 93 Go depuis le volume
   prend plusieurs minutes ; le refaire a chaque appel rendrait le service
   inutilisable. C'est pour cela qu'on importe InfiniteTalkPipeline au lieu
   d'appeler generate_infinitetalk.py en sous-processus.
"""
import base64
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import runpod

# ---------------------------------------------------------------- 1. chemins

INFINITETALK_ROOT = os.environ.get("INFINITETALK_ROOT", "/opt/InfiniteTalk")

# Racines ou chercher les poids, dans l'ordre. Le volume reseau RunPod est
# monte sur /runpod-volume en serverless et sur /workspace sur un pod.
_RACINES = [
    os.environ.get("IT_WEIGHTS", ""),
    "/runpod-volume/infinitetalk/weights",
    "/runpod-volume/weights",
    "/workspace/infinitetalk/weights",
    os.path.join(INFINITETALK_ROOT, "weights"),
]

# Dossier des references (image ou video de depart), adressables par nom court.
_RACINES_REFS = [
    os.environ.get("IT_REFS", ""),
    "/runpod-volume/infinitetalk/refs",
    "/workspace/infinitetalk/refs",
    os.path.join(INFINITETALK_ROOT, "refs"),
]

PROMPT_DEFAUT = os.environ.get(
    "IT_PROMPT",
    "A person is speaking to the camera, calm and professional, "
    "in a well-lit indoor setting, steady framing, natural expression.")


def _premiere_racine(racines, *attendus):
    """Premiere racine contenant tous les sous-chemins attendus."""
    for r in racines:
        if not r:
            continue
        if all(os.path.exists(os.path.join(r, a)) for a in attendus):
            return r
    return None


def _resoudre_poids():
    """Localise les trois jeux de poids, ou explique ce qui manque.

    On verifie avant de charger : un chemin absent donne ici un message
    clair, alors que plus bas il ressort en trace torch illisible.
    """
    racine = _premiere_racine(_RACINES, "Wan2.1-I2V-14B-480P",
                              "chinese-wav2vec2-base")
    if racine is None:
        raise RuntimeError(
            "Poids introuvables. Racines testees : %s. Attendu dans l'une "
            "d'elles : Wan2.1-I2V-14B-480P/, chinese-wav2vec2-base/, "
            "InfiniteTalk/. Lancer download_weights.sh sur le volume reseau."
            % ", ".join(r for r in _RACINES if r))

    quant = os.environ.get("IT_QUANT") or None          # fp8 | int8 | None
    variante = os.environ.get("IT_VARIANTE", "single")  # single | multi

    poids = {
        "ckpt_dir": os.path.join(racine, "Wan2.1-I2V-14B-480P"),
        "wav2vec_dir": os.path.join(racine, "chinese-wav2vec2-base"),
        "infinitetalk_dir": os.path.join(
            racine, "InfiniteTalk", variante, "infinitetalk.safetensors"),
        "quant_dir": None,
        "quant": quant,
        "lora_dir": os.environ.get("IT_LORA") or None,
    }
    if quant:
        poids["quant_dir"] = os.path.join(
            racine, "InfiniteTalk", "quant_models",
            "infinitetalk_%s_%s.safetensors" % (variante, quant))

    for cle in ("ckpt_dir", "wav2vec_dir", "infinitetalk_dir", "quant_dir"):
        chemin = poids[cle]
        if chemin and not os.path.exists(chemin):
            raise RuntimeError("%s manquant : %s" % (cle, chemin))
    return racine, poids


def _garde_gpu():
    """Refuse explicitement un GPU trop recent pour le torch embarque.

    La categorie generique 24 Go de RunPod contient des tranches MIG de
    Blackwell PRO 6000. torch n'a pas de noyaux compiles pour sm_120 : le
    worker meurt sans laisser de trace et les jobs restent en file. Mieux
    vaut un message.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Aucun GPU visible dans ce worker.")
    majeure, mineure = torch.cuda.get_device_capability(0)
    nom = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("[init] torch=%s gpu=%s sm_%d%d vram=%.0f Go"
          % (torch.__version__, nom, majeure, mineure, vram), flush=True)
    if majeure >= 12:
        raise RuntimeError(
            "GPU %s (sm_%d%d) trop recent pour torch %s : aucun noyau "
            "compile pour cette architecture. Choisir une categorie 48 ou "
            "80 Go sans pastille PRO (A40, A6000, L40S, A100). Attention : "
            "la categorie generique 24 Go contient des tranches MIG de "
            "Blackwell." % (nom, majeure, mineure, torch.__version__))
    if vram < 40:
        print("[init] ATTENTION : %.0f Go de VRAM seulement. Wan2.1-14B "
              "demande du delestage (IT_LOW_VRAM=1) ou un modele quantifie "
              "(IT_QUANT=fp8), et ce sera plus lent." % vram, flush=True)
    return nom, vram


# ------------------------------------------------------------- 2. chargement

_ETAT = {"pret": False, "pipeline": None, "wav2vec": None,
         "poids": None, "gpu": None, "chargement_s": None}


def _args_amont(**surcharges):
    """Namespace d'arguments amont, construit par leur propre parseur.

    On passe par argparse plutot que par un objet maison : le pipeline lit
    des champs qu'on ne connait pas forcement (teacache, apg, fsdp...) et
    _validate_args en derive d'autres, comme sample_shift qui depend de la
    taille. Reconstruire cela a la main se perimerait au premier commit
    amont. Les champs que la validation exploite passent donc par argv,
    pas par setattr apres coup.
    """
    import generate_infinitetalk as gi
    p = _ETAT["poids"]
    argv = ["generate_infinitetalk.py",
            "--ckpt_dir", p["ckpt_dir"],
            "--wav2vec_dir", p["wav2vec_dir"],
            "--infinitetalk_dir", p["infinitetalk_dir"]]
    for drapeau, cle in (("--size", "size"), ("--mode", "mode")):
        if cle in surcharges:
            argv += [drapeau, str(surcharges.pop(cle))]
    if "sample_steps" in surcharges:
        argv += ["--sample_steps", str(int(surcharges.pop("sample_steps")))]

    ancien = sys.argv
    try:
        sys.argv = argv
        args = gi._parse_args()
    finally:
        sys.argv = ancien
    for cle, valeur in surcharges.items():
        setattr(args, cle, valeur)
    return args


def _charger():
    """Monte le pipeline une fois pour toutes. Appele au premier job."""
    if _ETAT["pret"]:
        return
    depart = time.time()

    racine, poids = _resoudre_poids()
    _ETAT["poids"] = poids
    print("[init] poids lus depuis %s" % racine, flush=True)

    # Le code amont construit des chemins relatifs au repertoire courant.
    os.chdir(INFINITETALK_ROOT)
    if INFINITETALK_ROOT not in sys.path:
        sys.path.insert(0, INFINITETALK_ROOT)

    _ETAT["gpu"] = _garde_gpu()

    import wan
    from wan.configs import WAN_CONFIGS
    import generate_infinitetalk as gi

    cfg = WAN_CONFIGS["infinitetalk-14B"]
    pipeline = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=poids["ckpt_dir"],
        quant_dir=poids["quant_dir"],
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=_bool(os.environ.get("IT_T5_CPU"), False),
        lora_dir=[poids["lora_dir"]] if poids["lora_dir"] else None,
        lora_scales=[float(os.environ.get("IT_LORA_SCALE", "1.2"))],
        quant=poids["quant"],
        dit_path=None,
        infinitetalk_dir=poids["infinitetalk_dir"],
    )

    # Delestage : les parametres du DiT ne restent pas tous en VRAM.
    # Indispensable sous 80 Go, au prix de la vitesse.
    if _bool(os.environ.get("IT_LOW_VRAM"), False):
        pipeline.vram_management = True
        pipeline.enable_vram_management(num_persistent_param_in_dit=0)
        print("[init] delestage DiT actif (IT_LOW_VRAM)", flush=True)

    # wav2vec reste sur le CPU : c'est ce que fait le script amont, et ces
    # 380 Mo n'ont rien a gagner a occuper de la VRAM.
    _ETAT["wav2vec"] = gi.custom_init("cpu", poids["wav2vec_dir"])
    _ETAT["pipeline"] = pipeline
    _ETAT["pret"] = True
    _ETAT["chargement_s"] = round(time.time() - depart, 1)
    print("[init] pret en %.1f s" % _ETAT["chargement_s"], flush=True)


# ------------------------------------------------------------ 3. utilitaires

def _bool(valeur, defaut=False):
    if valeur is None:
        return defaut
    if isinstance(valeur, bool):
        return valeur
    return str(valeur).strip().lower() in ("1", "true", "yes", "oui", "on")


def _borne(valeur, mini, maxi, defaut):
    try:
        return max(mini, min(maxi, type(defaut)(valeur)))
    except (TypeError, ValueError):
        return defaut


def _media(entree, dossier, cles_b64, cles_url, cle_nom, defaut_ext):
    """Recupere un media depuis base64, une URL, ou un nom sur le volume."""
    for cle in cles_b64:
        if entree.get(cle):
            prefixe = cle.rsplit("_", 1)[0]
            ext = str(entree.get("%s_ext" % prefixe, defaut_ext)).lstrip(".")
            chemin = os.path.join(dossier, "%s.%s" % (prefixe, ext))
            with open(chemin, "wb") as f:
                f.write(base64.b64decode(entree[cle]))
            return chemin
    for cle in cles_url:
        if entree.get(cle):
            import requests
            url = entree[cle]
            ext = os.path.splitext(url.split("?")[0])[1] or "." + defaut_ext
            chemin = os.path.join(dossier, "%s%s" % (cle.rsplit("_", 1)[0], ext))
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            with open(chemin, "wb") as f:
                f.write(r.content)
            return chemin
    nom = entree.get(cle_nom)
    if nom:
        # Un nom simple, jamais un chemin : sinon n'importe quel fichier du
        # volume devient lisible depuis une requete.
        if "/" in nom or "\\" in nom or ".." in nom:
            raise ValueError("%s doit etre un nom de fichier simple" % cle_nom)
        for r in _RACINES_REFS:
            if not r:
                continue
            chemin = os.path.join(r, nom)
            if os.path.exists(chemin):
                return chemin
        raise RuntimeError(
            "'%s' introuvable. Racines testees : %s"
            % (nom, ", ".join(r for r in _RACINES_REFS if r)))
    return None


def _duree(chemin):
    """Duree d'un media, via ffprobe s'il est present."""
    outil = shutil.which("ffprobe")
    if not outil:
        return None
    r = subprocess.run([outil, "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1",
                        chemin], capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()), 2)
    except ValueError:
        return None


# ---------------------------------------------------------------- 4. handler

def handler(job):
    total = time.time()
    etapes = {}
    entree = job.get("input") or {}

    try:
        depart = time.time()
        premier = not _ETAT["pret"]
        _charger()
        if premier:
            etapes["chargement"] = round(time.time() - depart, 1)

        travail = tempfile.mkdtemp(prefix="it_")
        try:
            # --- audio : obligatoire
            depart = time.time()
            audio = _media(entree, travail, ("audio_base64",),
                           ("audio_url",), "audio", "wav")
            if not audio:
                return {"error": "Fournir audio_base64, audio_url ou audio "
                                 "(nom de fichier sur le volume)."}

            # --- reference : image fixe ou video de depart
            reference = _media(entree, travail,
                               ("image_base64", "video_base64"),
                               ("image_url", "video_url"),
                               "reference", "png")
            if not reference:
                return {"error": "Fournir une reference : image_base64, "
                                 "video_base64, image_url, video_url ou "
                                 "reference (nom de fichier sur le volume)."}
            etapes["entrees"] = round(time.time() - depart, 1)

            # --- reglages
            taille = entree.get("size", "infinitetalk-480")
            if taille not in ("infinitetalk-480", "infinitetalk-720"):
                return {"error": "size doit valoir infinitetalk-480 ou "
                                 "infinitetalk-720."}
            mode = entree.get("mode", "streaming")
            if mode not in ("streaming", "clip"):
                return {"error": "mode doit valoir streaming ou clip."}

            args = _args_amont(
                size=taille,
                mode=mode,
                sample_steps=_borne(entree.get("sample_steps", 40), 1, 60, 40),
                sample_text_guide_scale=_borne(
                    entree.get("text_guide_scale", 5.0), 1.0, 10.0, 5.0),
                sample_audio_guide_scale=_borne(
                    entree.get("audio_guide_scale", 4.0), 1.0, 10.0, 4.0),
                motion_frame=_borne(entree.get("motion_frame", 9), 1, 25, 9),
                max_frame_num=_borne(
                    entree.get("max_frame_num", 1000), 81, 5000, 1000),
                base_seed=int(entree.get("seed", 42)),
                color_correction_strength=_borne(
                    entree.get("color_correction_strength", 1.0),
                    0.0, 1.0, 1.0),
                use_teacache=_bool(entree.get("use_teacache"), False),
                teacache_thresh=_borne(
                    entree.get("teacache_thresh", 0.2), 0.0, 1.0, 0.2),
                use_apg=_bool(entree.get("use_apg"), False),
                audio_save_dir=os.path.join(travail, "audio"),
            )
            os.makedirs(args.audio_save_dir, exist_ok=True)

            # --- plongement audio par wav2vec
            depart = time.time()
            import soundfile as sf
            import torch
            import generate_infinitetalk as gi

            extracteur, encodeur = _ETAT["wav2vec"]
            parole = gi.audio_prepare_single(audio)
            plongement = gi.get_embedding(parole, extracteur, encodeur)
            chemin_pt = os.path.join(args.audio_save_dir, "1.pt")
            chemin_somme = os.path.join(args.audio_save_dir, "sum.wav")
            torch.save(plongement, chemin_pt)
            sf.write(chemin_somme, parole, 16000)
            etapes["audio"] = round(time.time() - depart, 1)

            clip = {
                "prompt": entree.get("prompt") or PROMPT_DEFAUT,
                "cond_video": reference,
                "cond_audio": {"person1": chemin_pt},
                "video_audio": chemin_somme,
            }
            if entree.get("bbox"):
                clip["bbox"] = entree["bbox"]

            # --- diffusion : l'etape longue
            depart = time.time()
            video = _ETAT["pipeline"].generate_infinitetalk(
                clip,
                size_buckget=args.size,
                motion_frame=args.motion_frame,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sampling_steps=args.sample_steps,
                text_guide_scale=args.sample_text_guide_scale,
                audio_guide_scale=args.sample_audio_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                max_frames_num=(args.frame_num if mode == "clip"
                                else args.max_frame_num),
                color_correction_strength=args.color_correction_strength,
                extra_args=args,
            )
            etapes["diffusion"] = round(time.time() - depart, 1)

            # --- ecriture du MP4 et retour
            depart = time.time()
            from wan.utils.multitalk_utils import save_video_ffmpeg
            base = os.path.join(travail, "sortie")
            save_video_ffmpeg(video, base, [chemin_somme],
                              high_quality_save=False)
            sortie = base + ".mp4"
            if not os.path.exists(sortie):
                return {"error": "La video n'a pas ete ecrite : %s" % sortie,
                        "etapes": etapes}
            with open(sortie, "rb") as f:
                donnees = f.read()
            etapes["encodage"] = round(time.time() - depart, 1)

            duree = _duree(sortie)
            reponse = {
                "video_base64": base64.b64encode(donnees).decode(),
                "duration_s": duree,
                "audio_s": _duree(chemin_somme),
                "size": taille,
                "mode": mode,
                "sample_steps": args.sample_steps,
                "text_guide_scale": args.sample_text_guide_scale,
                "audio_guide_scale": args.sample_audio_guide_scale,
                "seed": args.base_seed,
                "model": "InfiniteTalk + Wan2.1-I2V-14B-480P",
                "variante": os.environ.get("IT_VARIANTE", "single"),
                "quant": _ETAT["poids"]["quant"],
                "gpu": _ETAT["gpu"][0] if _ETAT["gpu"] else None,
                "etapes": etapes,
                "total_s": round(time.time() - total, 1),
            }
            # Le cout du modele ramene a l'unite comparable aux autres
            # moteurs : secondes de GPU par seconde de video rendue.
            if duree:
                reponse["s_par_seconde"] = round(reponse["total_s"] / duree, 2)
            if _bool(entree.get("benchmark"), False):
                reponse.pop("video_base64")
                reponse["taille_mp4_octets"] = len(donnees)
            return reponse
        finally:
            shutil.rmtree(travail, ignore_errors=True)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    except Exception as exc:
        # La trace repart dans la reponse : en serverless, une exception
        # avalee laisse un job COMPLETED vide et un worker muet.
        return {"error": "%s: %s" % (type(exc).__name__, exc),
                "trace": traceback.format_exc()[-3000:],
                "etapes": etapes,
                "total_s": round(time.time() - total, 1)}


runpod.serverless.start({"handler": handler})
