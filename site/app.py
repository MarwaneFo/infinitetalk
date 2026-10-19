"""
Studio InfiniteTalk : un visage, un texte ou un audio, un style -> une video.

    SITE_PASSWORD=... RUNPOD_KEY=... INFINITETALK_ENDPOINT=... \
    CHATTERBOX_ENDPOINT=... uvicorn app:app --host 0.0.0.0 --port 8000

Le site est un intermediaire : il garde les cles RunPod, depose les travaux,
les suit, et range les videos. Le navigateur ne voit jamais une cle.

Un rendu dure longtemps -- mesure sur L40S : ~2 min de GPU par seconde de
video. Chaque demande devient donc un travail persistant (un fichier JSON),
suivi par un fil en arriere-plan. L'utilisateur peut fermer la page ; au
redemarrage du serveur, les travaux en cours sont repris la ou ils etaient.
"""
import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave

import requests
from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

ICI = os.path.dirname(os.path.abspath(__file__))
DONNEES = os.environ.get("DATA_DIR", os.path.join(ICI, "donnees"))
TRAVAUX = os.path.join(DONNEES, "travaux")
VIDEOS = os.path.join(DONNEES, "videos")
os.makedirs(TRAVAUX, exist_ok=True)
os.makedirs(VIDEOS, exist_ok=True)

CLE = os.environ.get("RUNPOD_KEY", "").strip()
IT_ID = os.environ.get("INFINITETALK_ENDPOINT", "").strip()
CB_ID = os.environ.get("CHATTERBOX_ENDPOINT", "").strip()
CB_CLE = os.environ.get("CHATTERBOX_KEY", CLE).strip()
MOT_DE_PASSE = os.environ.get("SITE_PASSWORD", "")
UTILISATEUR = os.environ.get("SITE_USER", "studio")

# Garde-fous de cout. Un morceau de 81 images (3,24 s) coute ~7 min de L40S :
# une minute d'audio, c'est deja deux heures de GPU.
AUDIO_MAX_S = float(os.environ.get("AUDIO_MAX_S", "60"))
TEXTE_MAX = int(os.environ.get("TEXTE_MAX", "900"))
# InfiniteTalk refuse un audio qui ne depasse pas un morceau (81 images a
# 25 i/s, soit 3,24 s) : l'assertion amont exige strictement plus.
AUDIO_MIN_S = 3.3
# Jusqu'a cette duree, un seul morceau suffit : mode "clip". Au-dela, le mode
# "streaming" enchaine les morceaux -- et en genere au moins DEUX des que
# l'audio depasse 81 images (lu dans wan/multitalk.py). 3,4 s en streaming
# couterait deux morceaux, ~14 min de GPU au lieu de 7.
CLIP_MAX_S = 3.3

API = "https://api.runpod.ai/v2"

# ------------------------------------------------------------------ styles
# Le style agit a deux endroits : la description de scene donnee a
# InfiniteTalk (Wan2.1 est un modele texte-vers-video, l'invite pilote
# l'attitude), et l'expressivite de Chatterbox quand la voix est synthetisee.
STYLES = {
    "serieux": {
        "nom": "Serieux / professionnel",
        "invite": "A person speaks directly to the camera in a calm, "
                  "professional and composed manner, neutral expression, "
                  "minimal head movement, steady framing, soft even studio "
                  "lighting.",
        "expressivite": 0.40,
    },
    "souriant": {
        "nom": "Content / souriant",
        "invite": "A person speaks warmly to the camera with a gentle, "
                  "genuine smile, cheerful and friendly expression, relaxed "
                  "natural head movement, bright and soft lighting.",
        "expressivite": 0.60,
    },
    "enthousiaste": {
        "nom": "Enthousiaste / dynamique",
        "invite": "A person speaks to the camera with energy and enthusiasm, "
                  "lively facial expressions, expressive eyebrows, animated "
                  "head movements and nods, bright lighting, steady framing.",
        "expressivite": 0.85,
    },
    "pedagogue": {
        "nom": "Bienveillant / pedagogue",
        "invite": "A friendly teacher speaks kindly to the camera, calm and "
                  "reassuring, warm eyes, slow natural nods, patient and "
                  "attentive expression, soft classroom lighting.",
        "expressivite": 0.50,
    },
    "expert": {
        "nom": "Expert / confiant",
        "invite": "A confident expert speaks directly to the camera with "
                  "authority and clarity, steady eye contact, composed posture, "
                  "measured and deliberate head movements, subtle assured "
                  "expression, clean professional studio lighting.",
        "expressivite": 0.45,
    },
    "commercial": {
        "nom": "Commercial / persuasif",
        "invite": "A persuasive presenter speaks warmly and convincingly to "
                  "the camera, engaging confident smile, lively but controlled "
                  "expressions, inviting nods, open friendly attitude, bright "
                  "polished lighting.",
        "expressivite": 0.70,
    },
}

VOIX = {"inna": "Inna (feminine)", "mehdi": "Mehdi (masculine)",
        "voix3": "Voix 3 (clonee)"}
LANGUES = {"fr": "Francais", "en": "Anglais", "es": "Espagnol",
           "de": "Allemand", "it": "Italien", "pt": "Portugais",
           "ar": "Arabe", "nl": "Neerlandais"}

# ------------------------------------------------------------ authentification
# Une page de connexion et un cookie signe, plutot que l'authentification
# HTTP Basic : la fenetre native du navigateur n'apparait pas partout (le
# navigateur integre de VS Code renvoie un sec "Not authenticated"), et un
# site destine a d'autres personnes merite une vraie porte d'entree.
COOKIE = "studio_session"
DUREE_SESSION = 30 * 24 * 3600


def _cle_session():
    # Derivee du mot de passe : en changer deconnecte tout le monde.
    return hashlib.sha256(("session:" + MOT_DE_PASSE).encode()).digest()


def _signer(expiration):
    return hmac.new(_cle_session(), str(expiration).encode(),
                    hashlib.sha256).hexdigest()


def _session_valide(jeton):
    try:
        expiration, signature = jeton.split(".", 1)
        return (int(expiration) > time.time()
                and hmac.compare_digest(signature, _signer(int(expiration))))
    except (AttributeError, ValueError):
        return False


def _authentifie(request):
    if not MOT_DE_PASSE:
        raise HTTPException(503, "SITE_PASSWORD non defini sur le serveur.")
    if _session_valide(request.cookies.get(COOKIE, "")):
        return True
    # Acces par script (curl -u) : HTTP Basic reste accepte, sans jamais
    # etre reclame au navigateur.
    entete = request.headers.get("authorization", "")
    if entete.lower().startswith("basic "):
        try:
            nom, mdp = base64.b64decode(entete[6:]).decode().split(":", 1)
        except (ValueError, UnicodeDecodeError):
            return False
        return (secrets.compare_digest(nom, UTILISATEUR)
                & secrets.compare_digest(mdp, MOT_DE_PASSE))
    return False


def verifier(request: Request):
    """Garde de toutes les routes. Sans SITE_PASSWORD, le site refuse tout :
    il pilote des GPU factures, il ne doit jamais etre ouvert par oubli."""
    if not _authentifie(request):
        raise HTTPException(401, "Connexion requise.")
    return UTILISATEUR


PAGE_CONNEXION = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Studio InfiniteTalk</title><style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1f23;--dim:#6b7079;--line:#e3e1dc;
--accent:#2f5bea;--accent-ink:#fff;--err:#c4372c}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--panel:#1a1d22;
--ink:#e8eaed;--dim:#9aa1ab;--line:#2c3139;--accent:#6b8cff;--accent-ink:#0d1020;
--err:#ff8a80}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;
place-items:center;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,"Segoe UI",Roboto,sans-serif;padding:16px}
form{width:100%;max-width:340px;background:var(--panel);border:1px solid var(--line);
border-radius:12px;padding:24px}h1{font-size:19px;margin:0 0 4px}
p{color:var(--dim);margin:0 0 18px;font-size:14px}
input{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:8px;
background:var(--bg);color:var(--ink);font:inherit;margin-bottom:12px}
button{width:100%;padding:11px;border:0;border-radius:8px;background:var(--accent);
color:var(--accent-ink);font:600 15px/1 inherit;cursor:pointer}
.err{color:var(--err);font-size:13.5px;margin:0 0 12px}</style></head><body>
<form method="post" action="/connexion"><h1>Studio InfiniteTalk</h1>
<p>Entrez le mot de passe pour continuer.</p>__ERREUR__
<input type="password" name="mot_de_passe" placeholder="Mot de passe"
 autocomplete="current-password" autofocus required>
<button type="submit">Se connecter</button></form></body></html>"""


app = FastAPI(title="Studio InfiniteTalk")

# ------------------------------------------------------------------ travaux
_verrou = threading.Lock()


def _chemin(tid):
    return os.path.join(TRAVAUX, "%s.json" % tid)


def lire(tid):
    with _verrou:
        with open(_chemin(tid), encoding="utf-8") as f:
            return json.load(f)


def ecrire(t):
    """Ecriture atomique : un arret brutal ne laisse pas un JSON coupe."""
    with _verrou:
        tmp = _chemin(t["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _chemin(t["id"]))


def maj(tid, **champs):
    t = lire(tid)
    t.update(champs)
    t["maj"] = time.time()
    ecrire(t)
    return t


def journal(tid, message):
    t = lire(tid)
    t.setdefault("journal", []).append([round(time.time()), message])
    t["etape"] = message
    t["maj"] = time.time()
    ecrire(t)


# ------------------------------------------------------------------ RunPod
def _entetes(cle):
    return {"Authorization": "Bearer %s" % cle,
            "Content-Type": "application/json"}


def deposer(endpoint, cle, charge, quoi="job"):
    """POST /run. Ne rejoue que les echecs survenus AVANT toute reponse :
    un POST abouti a cree un job facture, le rejouer le facturerait deux fois.
    """
    derniere = None
    for n in range(4):
        try:
            r = requests.post("%s/%s/run" % (API, endpoint),
                              headers=_entetes(cle), json=charge, timeout=120)
        except requests.RequestException as exc:
            derniere = exc
            time.sleep(2 ** n)
            continue
        if r.status_code == 409 and "max_workers" in r.text:
            raise RuntimeError("%s : l'endpoint RunPod %s est en pause "
                               "(Max workers = 0). Le remettre a 1 au moins."
                               % (quoi, endpoint))
        if r.status_code >= 400:
            raise RuntimeError("%s : RunPod %s : %s"
                               % (quoi, r.status_code, r.text[:300]))
        return r.json()["id"]
    raise RuntimeError("RunPod injoignable : %s" % derniere)


def etat(endpoint, cle, job):
    """GET /status, en tolerant les 404 et 5xx passagers de RunPod."""
    for n in range(8):
        try:
            r = requests.get("%s/%s/status/%s" % (API, endpoint, job),
                             headers=_entetes(cle), timeout=60)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(10)
    raise RuntimeError("etat du job %s illisible" % job)


def attendre(tid, endpoint, cle, job, quoi):
    """Suit un job RunPod jusqu'a son terme ; relaie son avancement."""
    vu = None
    while True:
        s = etat(endpoint, cle, job)
        statut = s.get("status")
        if statut == "COMPLETED":
            sortie = s.get("output") or {}
            if isinstance(sortie, dict) and sortie.get("error"):
                raise RuntimeError("%s : %s" % (quoi, sortie["error"]))
            return sortie
        if statut in ("FAILED", "CANCELLED", "TIMED_OUT"):
            detail = s.get("error") or (s.get("output") or {})
            if isinstance(detail, dict):
                detail = detail.get("error") or json.dumps(detail)[:300]
            raise RuntimeError("%s : %s (%s)" % (quoi, statut, detail))
        # Pendant IN_PROGRESS, output porte le dernier message du worker
        # (progress_update) : telechargement, pas de diffusion, etc.
        message = s.get("output") if statut == "IN_PROGRESS" else None
        if isinstance(message, str) and message != vu:
            vu = message
            journal(tid, "%s : %s" % (quoi, message))
        elif statut == "IN_QUEUE" and vu != "file":
            vu = "file"
            journal(tid, "%s : en file d'attente (demarrage d'un GPU)" % quoi)
        time.sleep(10)


# ------------------------------------------------------------------ audio
def _ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def preparer_audio(donnees):
    """Tout audio -> WAV 48 kHz mono, complete par du silence si trop court.

    Le worker reechantillonne lui-meme a 16 kHz pour analyser la voix, mais
    remet la voix recue dans la video : la reduire ici a 16 kHz donnait un
    son de telephone au telechargement. InfiniteTalk exige au moins un
    morceau entier (3,24 s) : un audio plus court echoue, constate.
    Renvoie (octets_wav, duree_s).
    """
    d = tempfile.mkdtemp(prefix="audio_")
    try:
        entree = os.path.join(d, "entree")
        sortie = os.path.join(d, "sortie.wav")
        with open(entree, "wb") as f:
            f.write(donnees)
        r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", entree,
                            "-ac", "1", "-ar", "48000",
                            "-af", "apad=whole_dur=%.2f" % AUDIO_MIN_S,
                            sortie], capture_output=True)
        if r.returncode or not os.path.exists(sortie):
            raise ValueError("audio illisible : %s"
                             % r.stderr.decode(errors="replace")[:200])
        with wave.open(sortie) as w:
            duree = w.getnframes() / float(w.getframerate())
        with open(sortie, "rb") as f:
            return f.read(), duree
    finally:
        shutil.rmtree(d, ignore_errors=True)


def delai_job_s(duree_audio):
    """Delai d'execution a porter par le job InfiniteTalk.

    Premier job d'un worker neuf : ~5 min de telechargement et chargement.
    Puis ~7 min par morceau de 3,24 s (72 images nouvelles par morceau
    suivant). Marge du simple au double : un job coupe est perdu et paye.
    """
    if duree_audio <= CLIP_MAX_S:
        morceaux = 1
    else:
        morceaux = 1 + max(1, -(-(int(duree_audio * 25) - 81) // 72))
    return int(min(6 * 3600, 2 * (600 + morceaux * 450)))


# ------------------------------------------------------------------ pipeline
def executer(tid):
    """Texte -> voix (Chatterbox) -> video (InfiniteTalk) -> fichier MP4."""
    try:
        t = lire(tid)
        style = STYLES[t["style"]]

        # --- 1. la voix
        if t.get("audio_pret"):
            with open(os.path.join(TRAVAUX, "%s.wav" % tid), "rb") as f:
                wav = f.read()
        elif t["source"] == "texte":
            if not CB_ID:
                raise RuntimeError("CHATTERBOX_ENDPOINT non configure.")
            job = t.get("job_voix")
            if not job:
                entree = {"text": t["texte"], "language": t["langue"],
                          "voice": t["voix"],
                          "exaggeration": style["expressivite"],
                          "temperature": 0.8, "cfg_weight": 0.5}
                job = deposer(CB_ID, CB_CLE, {"input": entree}, "voix")
                maj(tid, job_voix=job, statut="voix")
                journal(tid, "Synthese de la voix (Chatterbox)")
            sortie = attendre(tid, CB_ID, CB_CLE, job, "voix")
            wav, duree = preparer_audio(base64.b64decode(sortie["audio_base64"]))
            if duree > AUDIO_MAX_S + 0.5:
                raise RuntimeError("La voix generee dure %.0f s, au-dela de la "
                                   "limite de %.0f s. Raccourcir le texte."
                                   % (duree, AUDIO_MAX_S))
            with open(os.path.join(TRAVAUX, "%s.wav" % tid), "wb") as f:
                f.write(wav)
            maj(tid, audio_pret=True, duree_audio=round(duree, 2))
            journal(tid, "Voix prete : %.1f s" % duree)
        else:
            raise RuntimeError("audio manquant")

        # --- 2. la video, decoupee en segments traites en parallele
        t = lire(tid)
        if t.get("job_video"):
            # Travail cree avant le decoupage : on reprend son suivi tel quel.
            _ancien_suivi(tid)
            return
        _video_en_segments(tid, style)
    except Exception as exc:
        try:
            maj(tid, statut="erreur", erreur=str(exc)[:600])
            journal(tid, "Erreur : %s" % str(exc)[:300])
        except Exception:
            pass


# ------------------------------------------------------------------ segments
# Une video longue est coupee aux silences en segments d'environ 30 s, deposes
# tous ensemble : RunPod les repartit sur autant de workers que « Max workers »
# l'autorise. Chaque segment repart de la photo, ce qui evite aussi la derive
# de couleur des videos longues. La voix entiere est remise par-dessus a la
# fin : aucune coupure audible aux raccords.
SEG_CIBLE_S = float(os.environ.get("SEG_CIBLE_S", "30"))
SEG_MAX_S = float(os.environ.get("SEG_MAX_S", "36"))
SEG_MIN_S = 6.0
IPS = 25

QUALITES = {
    "standard": {"nom": "Standard", "reglages": {}, "facteur": 1.0},
    # 10 pas au lieu de 8 : details plus fins. Pas encore compare a l'oeil.
    "haute": {"nom": "Haute (10 pas)",
              "reglages": {"sample_steps": 10}, "facteur": 1.3},
}

# Force du guidage audio : a quel point la bouche obeit a la voix. 2 est la
# valeur de reference avec la LoRA FusionX ; au-dela, articulation plus
# marquee, avec un risque de grimace. Sans cout de calcul.
LEVRES = {
    "normale": {"nom": "Normale", "audio_guide_scale": 2.0},
    "marquee": {"nom": "Marquée", "audio_guide_scale": 2.5},
    "forte": {"nom": "Très marquée", "audio_guide_scale": 3.0},
}


def _silences(chemin):
    """Milieux des silences de plus de 0,3 s, en secondes."""
    r = subprocess.run([_ffmpeg(), "-v", "info", "-i", chemin, "-af",
                        "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    milieux, debut = [], None
    for ligne in r.stderr.splitlines():
        if "silence_start:" in ligne:
            debut = float(ligne.split("silence_start:")[1].split()[0])
        elif "silence_end:" in ligne and debut is not None:
            fin = float(ligne.split("silence_end:")[1].split()[0])
            milieux.append((debut + fin) / 2)
            debut = None
    return milieux


def decouper(chemin, duree):
    """Bornes des segments, en secondes, alignees sur les images.

    Le nombre de segments est le plus petit qui respecte SEG_MAX_S, et les
    coupes visent des segments de meme duree : avec 2 workers, deux segments
    de 32 s finissent ensemble, la ou 24 + 22 + 17 s en laisseraient un
    attendre seul. Chaque coupe tombe au milieu du silence le plus proche
    (6 s de tolerance), a defaut pile sur la cible. Les bornes sont sur une
    image entiere (1/25 s) : sinon chaque raccord decalerait le son d'une
    fraction d'image, et sur 15 segments le decalage se verrait.
    """
    if duree <= SEG_MAX_S:
        return [(0.0, duree)]
    n = int(math.ceil(duree / SEG_MAX_S))
    silences = _silences(chemin)
    bornes = [0.0]
    for k in range(1, n):
        cible = duree * k / n
        proches = [m for m in silences
                   if abs(m - cible) <= 6 and m - bornes[-1] >= SEG_MIN_S
                   and duree - m >= SEG_MIN_S]
        coupe = min(proches, key=lambda m: abs(m - cible)) if proches else cible
        bornes.append(round(coupe * IPS) / IPS)
    bornes.append(duree)
    return list(zip(bornes[:-1], bornes[1:]))


def extraire(chemin, debut, fin):
    """Le morceau [debut, fin] du WAV, en octets WAV 48 kHz mono."""
    d = tempfile.mkdtemp(prefix="seg_")
    try:
        sortie = os.path.join(d, "s.wav")
        r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", chemin,
                            "-ss", "%.3f" % debut, "-t", "%.3f" % (fin - debut),
                            "-ac", "1", "-ar", "48000",
                            "-af", "apad=whole_dur=%.2f" % AUDIO_MIN_S, sortie],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError("decoupage audio : %s"
                               % r.stderr.decode(errors="replace")[:200])
        with open(sortie, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _deposer_segment(tid, t, seg, style, image):
    wav = os.path.join(TRAVAUX, "%s.wav" % tid)
    duree = seg["fin"] - seg["debut"]
    entree = {"audio_base64": base64.b64encode(
                  extraire(wav, seg["debut"], seg["fin"])).decode(),
              "image_base64": image,
              "image_ext": t.get("image_ext", "png"),
              "prompt": style["invite"],
              "size": "infinitetalk-480",
              "mode": "clip" if duree <= CLIP_MAX_S else "streaming",
              # Exactement le nombre d'images du segment : les raccords
              # restent alignes sur la voix.
              "max_frame_num": max(1, int(round(duree * IPS)))}
    qualite = QUALITES.get(t.get("qualite"), QUALITES["standard"])
    entree.update(qualite["reglages"])
    entree["audio_guide_scale"] = LEVRES.get(
        t.get("levres"), LEVRES["normale"])["audio_guide_scale"]
    delai = int(delai_job_s(duree) * qualite["facteur"])
    return deposer(IT_ID, CLE, {"input": entree,
                                "policy": {"executionTimeout": delai * 1000}},
                   "video")


def _resume(segments):
    """Une ligne d'avancement lisible, segment par segment."""
    n = len(segments)
    if n == 1:
        return segments[0].get("etape") or "en file d'attente"
    faits = sum(1 for s in segments if s["statut"] == "termine")
    actifs = ["segment %d/%d : %s" % (s["i"] + 1, n, s.get("etape") or "en file")
              for s in segments if s["statut"] != "termine"]
    return "%d/%d segments termines%s" % (
        faits, n, (" · " + " · ".join(actifs[:2])) if actifs else "")


def _video_en_segments(tid, style):
    t = lire(tid)
    if not t.get("segments"):
        wav = os.path.join(TRAVAUX, "%s.wav" % tid)
        bornes = decouper(wav, t["duree_audio"])
        segs = [{"i": i, "debut": a, "fin": b, "statut": "attente",
                 "job": None, "essais": 0} for i, (a, b) in enumerate(bornes)]
        t = maj(tid, segments=segs, statut="video")
        journal(tid, "Animation du visage (InfiniteTalk) : %.1f s d'audio, "
                "%d segment(s)" % (t["duree_audio"], len(segs)))

    with open(os.path.join(TRAVAUX, "%s.img" % tid), "rb") as f:
        image = base64.b64encode(f.read()).decode()

    # Deposer tout ce qui n'a pas encore de job. Apres un redemarrage, les
    # segments deja deposes gardent leur job : rien n'est facture deux fois.
    segs = t["segments"]
    for seg in segs:
        if not seg["job"] and seg["statut"] != "termine":
            seg["job"] = _deposer_segment(tid, t, seg, style, image)
            seg["statut"] = "video"
            maj(tid, segments=segs)

    dernier = None
    while True:
        if lire(tid).get("annule"):
            return
        for seg in segs:
            if seg["statut"] == "termine":
                continue
            s = etat(IT_ID, CLE, seg["job"])
            statut = s.get("status")
            sortie = s.get("output")
            if statut == "COMPLETED":
                if isinstance(sortie, dict) and sortie.get("error"):
                    statut, s = "FAILED", {"error": sortie["error"]}
                else:
                    chemin = os.path.join(TRAVAUX, "%s_seg%02d.mp4" % (tid, seg["i"]))
                    with open(chemin, "wb") as f:
                        f.write(base64.b64decode(sortie["video_base64"]))
                    seg.update(statut="termine", etape="termine",
                               gpu_s=sortie.get("total_s"))
                    continue
            if statut in ("FAILED", "CANCELLED", "TIMED_OUT"):
                detail = s.get("error") or ""
                if seg["essais"] < 1 and statut != "CANCELLED":
                    # Une seconde chance, pour ce segment seulement.
                    seg["essais"] += 1
                    journal(tid, "segment %d : %s, nouvel essai" % (seg["i"] + 1, statut))
                    seg["job"] = _deposer_segment(tid, t, seg, style, image)
                    seg["etape"] = "nouvel essai"
                    continue
                raise RuntimeError("video, segment %d/%d : %s (%s)"
                                   % (seg["i"] + 1, len(segs), statut, str(detail)[:300]))
            if statut == "IN_PROGRESS" and isinstance(sortie, str):
                seg["etape"] = sortie
            elif statut == "IN_QUEUE":
                seg["etape"] = "en file d'attente (demarrage d'un GPU)"
        maj(tid, segments=segs)
        ligne = _resume(segs)
        if ligne != dernier:
            journal(tid, "video : " + ligne)
            dernier = ligne
        if all(sg["statut"] == "termine" for sg in segs):
            break
        time.sleep(10)

    journal(tid, "Assemblage de la video")
    assembler(tid, segs, os.path.join(VIDEOS, "%s.mp4" % tid))
    gpu = sum(sg.get("gpu_s") or 0 for sg in segs)
    maj(tid, statut="termine", video=True, total_gpu_s=round(gpu, 1),
        s_par_seconde=round(gpu / max(t["duree_audio"], 0.1), 2))
    for sg in segs:
        try:
            os.remove(os.path.join(TRAVAUX, "%s_seg%02d.mp4" % (tid, sg["i"])))
        except OSError:
            pass
    journal(tid, "Video prete")


def assembler(tid, segs, sortie):
    """Recolle les segments dans l'ordre et remet la voix entiere par-dessus."""
    morceaux = [os.path.join(TRAVAUX, "%s_seg%02d.mp4" % (tid, sg["i"]))
                for sg in sorted(segs, key=lambda s: s["i"])]
    voix = os.path.join(TRAVAUX, "%s.wav" % tid)
    d = tempfile.mkdtemp(prefix="assemblage_")
    try:
        liste = os.path.join(d, "liste.txt")
        with open(liste, "w", encoding="utf-8") as f:
            for m in morceaux:
                f.write("file '%s'\n" % m.replace("\\", "/").replace("'", "'\\''"))
        r = subprocess.run([_ffmpeg(), "-y", "-v", "error",
                            "-f", "concat", "-safe", "0", "-i", liste,
                            "-i", voix, "-map", "0:v", "-map", "1:a",
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                            "-movflags", "+faststart", sortie],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError("assemblage : %s"
                               % r.stderr.decode(errors="replace")[-300:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _ancien_suivi(tid):
    """Suivi d'un travail depose en un seul job, avant le decoupage."""
    t = lire(tid)
    sortie = attendre(tid, IT_ID, CLE, t["job_video"], "video")
    with open(os.path.join(VIDEOS, "%s.mp4" % tid), "wb") as f:
        f.write(base64.b64decode(sortie["video_base64"]))
    maj(tid, statut="termine", video=True, total_gpu_s=sortie.get("total_s"),
        s_par_seconde=sortie.get("s_par_seconde"))
    journal(tid, "Video prete")


def lancer(tid):
    threading.Thread(target=executer, args=(tid,), daemon=True).start()


@app.on_event("startup")
def reprendre():
    """Reprend les travaux interrompus par un redemarrage du serveur.

    Les identifiants de job RunPod sont dans le JSON : on reprend le suivi,
    on ne redepose rien -- redeposer facturerait deux fois.
    """
    for nom in os.listdir(TRAVAUX):
        if nom.endswith(".json"):
            t = lire(nom[:-5])
            if t.get("statut") in ("attente", "voix", "video") and not t.get("annule"):
                lancer(t["id"])


# ------------------------------------------------------------------ routes
@app.get("/", response_class=HTMLResponse)
def page(request: Request):
    if not _authentifie(request):
        return RedirectResponse("/connexion", status_code=303)
    with open(os.path.join(ICI, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/connexion", response_class=HTMLResponse)
def connexion_page():
    return PAGE_CONNEXION.replace("__ERREUR__", "")


@app.post("/connexion")
def connexion(request: Request, mot_de_passe: str = Form("")):
    if not MOT_DE_PASSE:
        raise HTTPException(503, "SITE_PASSWORD non defini sur le serveur.")
    if not secrets.compare_digest(mot_de_passe, MOT_DE_PASSE):
        # Une seconde de penalite : rend l'essai systematique de mots de
        # passe trop lent pour etre utile.
        time.sleep(1)
        return HTMLResponse(PAGE_CONNEXION.replace(
            "__ERREUR__", '<p class="err">Mot de passe incorrect.</p>'),
            status_code=401)
    expiration = int(time.time()) + DUREE_SESSION
    reponse = RedirectResponse("/", status_code=303)
    # Derriere un hebergeur, le TLS s'arrete au proxy : c'est l'en-tete
    # X-Forwarded-Proto qui dit si le visiteur est en https.
    https = (request.headers.get("x-forwarded-proto", request.url.scheme)
             == "https")
    reponse.set_cookie(COOKIE, "%d.%s" % (expiration, _signer(expiration)),
                       max_age=DUREE_SESSION, httponly=True, samesite="lax",
                       secure=https)
    return reponse


@app.get("/deconnexion")
def deconnexion():
    reponse = RedirectResponse("/connexion", status_code=303)
    reponse.delete_cookie(COOKIE)
    return reponse


# ------------------------------------------------------------------ avatars
# Les visages proposes par defaut : un fichier image par avatar dans
# site/avatars (ou AVATARS_DIR). Le nom du fichier sert de nom affiche :
# « Inna.png » donne « Inna ». Ajouter ou retirer un avatar = deposer ou
# supprimer un fichier, sans toucher au code.
AVATARS = os.environ.get("AVATARS_DIR", os.path.join(ICI, "avatars"))
TYPES_IMAGE = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp"}


def _avatars():
    """{identifiant: chemin} des images de la galerie, triees par nom."""
    if not os.path.isdir(AVATARS):
        return {}
    trouves = {}
    for nom in sorted(os.listdir(AVATARS), key=str.lower):
        base, ext = os.path.splitext(nom)
        if ext.lower().lstrip(".") in TYPES_IMAGE:
            ident = "".join(c if c.isalnum() else "-" for c in base.lower()).strip("-")
            trouves[ident or "avatar"] = os.path.join(AVATARS, nom)
    return trouves


@app.get("/api/avatars")
def liste_avatars(_=Depends(verifier)):
    # « leila.png » s'affiche « Leila » ; « jean_pierre » devient « Jean Pierre ».
    return [{"id": i, "nom": os.path.splitext(os.path.basename(c))[0].replace("_", " ").title(),
             "image": "/api/avatars/%s" % i} for i, c in _avatars().items()]


@app.get("/api/avatars/{ident}")
def image_avatar(ident: str, _=Depends(verifier)):
    chemin = _avatars().get(ident)
    if not chemin:
        raise HTTPException(404, "introuvable")
    ext = os.path.splitext(chemin)[1].lower().lstrip(".")
    return FileResponse(chemin, media_type=TYPES_IMAGE[ext])


@app.get("/api/options")
def options(_=Depends(verifier)):
    return {"styles": {k: v["nom"] for k, v in STYLES.items()},
            "voix": VOIX, "langues": LANGUES,
            "audio_max_s": AUDIO_MAX_S, "texte_max": TEXTE_MAX,
            "qualites": {k: v["nom"] for k, v in QUALITES.items()},
            "levres": {k: v["nom"] for k, v in LEVRES.items()},
            "segment_s": SEG_CIBLE_S,
            "voix_disponible": bool(CB_ID)}


@app.post("/api/travaux")
async def creer(
    visage: UploadFile = File(None),
    avatar: str = Form(""),
    style: str = Form(...),
    source: str = Form(...),
    texte: str = Form(""),
    voix: str = Form("inna"),
    langue: str = Form("fr"),
    audio: UploadFile = File(None),
    titre: str = Form(""),
    qualite: str = Form("standard"),
    levres: str = Form("normale"),
    utilisateur=Depends(verifier),
):
    if not (CLE and IT_ID):
        raise HTTPException(503, "RUNPOD_KEY ou INFINITETALK_ENDPOINT manquant.")
    if style not in STYLES:
        raise HTTPException(400, "Style inconnu.")
    if source not in ("texte", "audio"):
        raise HTTPException(400, "Source inconnue.")
    if qualite not in QUALITES:
        raise HTTPException(400, "Qualite inconnue.")
    if levres not in LEVRES:
        raise HTTPException(400, "Reglage des levres inconnu.")

    if avatar:
        # Un visage de la galerie : on lit le fichier cote serveur, jamais
        # un chemin envoye par la page.
        chemin_avatar = _avatars().get(avatar)
        if not chemin_avatar:
            raise HTTPException(400, "Avatar inconnu.")
        with open(chemin_avatar, "rb") as f:
            image = f.read()
        ext = os.path.splitext(chemin_avatar)[1].lower().lstrip(".")
    elif visage is not None:
        image = await visage.read()
        ext = os.path.splitext(visage.filename or "")[1].lower().lstrip(".")
    else:
        raise HTTPException(400, "Choisir un avatar ou importer une photo.")
    if not image or len(image) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image absente ou trop lourde (15 Mo max).")
    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise HTTPException(400, "Image attendue en PNG, JPG ou WEBP.")

    tid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    t = {"id": tid, "cree": time.time(), "maj": time.time(),
         "titre": (titre or "").strip()[:80], "style": style,
         "source": source, "statut": "attente", "image_ext": ext,
         "qualite": qualite, "levres": levres, "avatar": avatar or None,
         "par": utilisateur, "journal": []}

    if source == "texte":
        texte = (texte or "").strip()
        if not texte:
            raise HTTPException(400, "Texte vide.")
        if len(texte) > TEXTE_MAX:
            raise HTTPException(400, "Texte trop long (%d caracteres max)."
                                % TEXTE_MAX)
        if not CB_ID:
            raise HTTPException(503, "Synthese vocale non configuree.")
        if voix not in VOIX or langue not in LANGUES:
            raise HTTPException(400, "Voix ou langue inconnue.")
        t.update(texte=texte, voix=voix, langue=langue)
    else:
        if audio is None:
            raise HTTPException(400, "Fichier audio manquant.")
        brut = await audio.read()
        if len(brut) > 30 * 1024 * 1024:
            raise HTTPException(400, "Audio trop lourd (30 Mo max).")
        try:
            wav, duree = preparer_audio(brut)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if duree > AUDIO_MAX_S + 0.5:
            raise HTTPException(400, "Audio de %.0f s : la limite est %.0f s."
                                % (duree, AUDIO_MAX_S))
        with open(os.path.join(TRAVAUX, "%s.wav" % tid), "wb") as f:
            f.write(wav)
        t.update(audio_pret=True, duree_audio=round(duree, 2))

    with open(os.path.join(TRAVAUX, "%s.img" % tid), "wb") as f:
        f.write(image)
    ecrire(t)
    journal(tid, "Demande enregistree")
    lancer(tid)
    return {"id": tid}


def _public(t):
    """Ce que la page a le droit de voir d'un travail."""
    garde = ("id", "cree", "maj", "titre", "style", "source", "statut",
             "etape", "erreur", "duree_audio", "texte", "voix", "langue",
             "gpu", "total_gpu_s", "s_par_seconde", "journal", "par")
    sortie = {k: t[k] for k in garde if k in t}
    sortie["style_nom"] = STYLES.get(t.get("style"), {}).get("nom")
    sortie["qualite_nom"] = QUALITES.get(t.get("qualite"), QUALITES["standard"])["nom"]
    sortie["levres_nom"] = "lèvres " + LEVRES.get(t.get("levres"), LEVRES["normale"])["nom"].lower()
    if t.get("segments"):
        sortie["segments"] = len(t["segments"])
    if t.get("video"):
        sortie["video"] = "/api/travaux/%s/video" % t["id"]
    return sortie


@app.get("/api/travaux")
def liste(_=Depends(verifier)):
    tous = []
    for nom in os.listdir(TRAVAUX):
        if nom.endswith(".json"):
            try:
                tous.append(_public(lire(nom[:-5])))
            except (OSError, ValueError):
                pass
    return sorted(tous, key=lambda t: t["cree"], reverse=True)[:100]


def _id_valide(tid):
    # L'identifiant sert a composer un chemin : rien d'autre que ce que
    # creer() fabrique ne doit passer.
    if not tid or any(c not in "0123456789abcdef-" for c in tid):
        raise HTTPException(404, "introuvable")
    if not os.path.exists(_chemin(tid)):
        raise HTTPException(404, "introuvable")
    return tid


@app.get("/api/travaux/{tid}")
def detail(tid: str, _=Depends(verifier)):
    return _public(lire(_id_valide(tid)))


@app.get("/api/travaux/{tid}/video")
def video(tid: str, _=Depends(verifier)):
    chemin = os.path.join(VIDEOS, "%s.mp4" % _id_valide(tid))
    if not os.path.exists(chemin):
        raise HTTPException(404, "pas encore prete")
    return FileResponse(chemin, media_type="video/mp4",
                        filename="infinitetalk-%s.mp4" % tid)


@app.get("/api/travaux/{tid}/visage")
def visage_vignette(tid: str, _=Depends(verifier)):
    t = lire(_id_valide(tid))
    chemin = os.path.join(TRAVAUX, "%s.img" % tid)
    types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
             "webp": "image/webp"}
    return FileResponse(chemin, media_type=types.get(t.get("image_ext"),
                                                     "image/png"))


@app.post("/api/travaux/{tid}/annuler")
def annuler(tid: str, _=Depends(verifier)):
    t = lire(_id_valide(tid))
    # Les jobs d'une video decoupee sont dans ses segments : sans eux,
    # l'annulation laissait tourner (et facturer) le calcul sur RunPod.
    jobs = [(t.get("job_voix"), CB_ID, CB_CLE), (t.get("job_video"), IT_ID, CLE)]
    jobs += [(sg.get("job"), IT_ID, CLE) for sg in t.get("segments") or []
             if sg.get("statut") != "termine"]
    if t.get("statut") in ("attente", "voix", "video"):
        for job, endpoint, cle in jobs:
            if not job:
                continue
            try:
                requests.post("%s/%s/cancel/%s" % (API, endpoint, job),
                              headers=_entetes(cle), timeout=30)
            except requests.RequestException:
                pass
    maj(tid, statut="erreur", erreur="Annule a la demande.", annule=True)
    return {"ok": True}
