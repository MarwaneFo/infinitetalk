"""Client InfiniteTalk, a brancher sur le banc d'essai.

Meme forme que les clients MuseTalk, LatentSync et Chatterbox : deux
fonctions, `lancer()` qui renvoie un identifiant de job, et `status()`
qui le suit.

Deux regles de resilience, apprises a leurs frais :

1. On ne rejoue un POST /run QUE si la requete a echoue AVANT toute
   reponse du serveur. Un POST abouti a cree un job facture ; le
   rejouer le facture deux fois.

2. En revanche /status se reessaie volontiers. RunPod renvoie
   regulierement un 404 transitoire sur un job qui existe bel et bien,
   le temps que son etat se propage.
"""
import base64
import os
import time

import requests

BASE = "https://api.runpod.ai/v2"


def _entetes(cle):
    return {"Authorization": "Bearer %s" % cle,
            "Content-Type": "application/json"}


def _poster(url, entetes, charge, essais=4):
    """POST qui ne rejoue que les echecs survenus avant toute reponse."""
    derniere = None
    for n in range(essais):
        try:
            r = requests.post(url, headers=entetes, json=charge, timeout=60)
        except requests.RequestException as exc:
            # Rien n'est parti, ou rien n'est revenu : rejouer est sans
            # risque de double facturation.
            derniere = exc
            if n < essais - 1:
                time.sleep(2 ** n)
                continue
            raise
        # Une reponse existe : on ne rejoue plus, meme si elle est en erreur.
        if r.status_code >= 400:
            raise RuntimeError("RunPod %s : %s" % (r.status_code, r.text[:400]))
        return r.json()
    raise derniere


def lancer(cle, endpoint, audio_base64=None, reference=None, *,
           audio_url=None, image_base64=None, image_url=None,
           video_base64=None, video_url=None, prompt=None,
           size="infinitetalk-480", mode="streaming", sample_steps=None,
           text_guide_scale=None, audio_guide_scale=None, motion_frame=9,
           max_frame_num=1000, seed=42, color_correction_strength=1.0,
           use_teacache=False, teacache_thresh=0.2, use_apg=False,
           benchmark=False, webhook=None, delai_s=3600):
    """Met un job en file. Renvoie son identifiant."""
    entree = {
        "size": size,
        "mode": mode,
        "motion_frame": int(motion_frame),
        "max_frame_num": int(max_frame_num),
        "seed": int(seed),
        "color_correction_strength": float(color_correction_strength),
        "use_teacache": bool(use_teacache),
        "teacache_thresh": float(teacache_thresh),
        "use_apg": bool(use_apg),
    }
    # Laisses vides, le worker choisit : 40 pas / 5 / 4 sans LoRA, et
    # 8 pas / 1 / 2 avec la LoRA FusionX. Forcer 40 pas ici annulerait
    # tout le gain de la LoRA.
    for cle_e, valeur in (("sample_steps", sample_steps),
                          ("text_guide_scale", text_guide_scale),
                          ("audio_guide_scale", audio_guide_scale)):
        if valeur is not None:
            entree[cle_e] = valeur
    for cle_e, valeur in (("audio_base64", audio_base64),
                          ("audio_url", audio_url),
                          ("reference", reference),
                          ("image_base64", image_base64),
                          ("image_url", image_url),
                          ("video_base64", video_base64),
                          ("video_url", video_url),
                          ("prompt", prompt)):
        if valeur:
            entree[cle_e] = valeur
    if benchmark:
        entree["benchmark"] = True

    # Delai porte par le job lui-meme : il prime sur celui de l'endpoint.
    # Le premier job d'un worker neuf telecharge ~32 Go avant de generer,
    # et le delai par defaut de la console (600 s) ne suffit pas.
    charge = {"input": entree,
              "policy": {"executionTimeout": int(delai_s * 1000)}}
    if webhook:
        # Le rappel se place A COTE de input, jamais dedans.
        charge["webhook"] = webhook

    reponse = _poster("%s/%s/run" % (BASE, endpoint), _entetes(cle), charge)
    jid = reponse.get("id")
    if not jid:
        raise RuntimeError("Reponse sans identifiant de job : %r" % reponse)
    return jid


def status(cle, endpoint, job_id, essais=6, pause=10):
    """Etat d'un job. Reessaie les 404 et 5xx, qui sont souvent transitoires."""
    url = "%s/%s/status/%s" % (BASE, endpoint, job_id)
    dernier = None
    for n in range(essais):
        try:
            r = requests.get(url, headers=_entetes(cle), timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404 or r.status_code >= 500:
                dernier = "HTTP %s" % r.status_code
            else:
                raise RuntimeError("RunPod %s : %s"
                                   % (r.status_code, r.text[:400]))
        except requests.RequestException as exc:
            dernier = str(exc)
        if n < essais - 1:
            time.sleep(pause)
    raise RuntimeError("Etat illisible apres %d essais (%s)" % (essais, dernier))


def attendre(cle, endpoint, job_id, pause=10, plafond=7200, trace=None,
             suivi=None):
    """Suit un job jusqu'a son terme et renvoie sa sortie.

    Leve une exception si la sortie porte une erreur applicative : un job
    peut terminer en COMPLETED tout en ayant echoue a l'interieur.
    """
    depart = time.time()
    while True:
        etat = status(cle, endpoint, job_id)
        statut = etat.get("status")
        if trace:
            trace(statut, round(time.time() - depart, 1))
        # Pendant IN_PROGRESS, `output` porte le dernier message publie par
        # le worker (progress_update) : etape en cours, pas de diffusion.
        if suivi and statut == "IN_PROGRESS" and etat.get("output"):
            suivi(etat["output"])
        if statut in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        if time.time() - depart > plafond:
            raise TimeoutError("job %s toujours %s apres %d s"
                               % (job_id, statut, plafond))
        time.sleep(pause)

    if statut != "COMPLETED":
        raise RuntimeError("job %s : %s (%s)"
                           % (job_id, statut, etat.get("error", "")))
    sortie = etat.get("output") or {}
    if "error" in sortie:
        raise RuntimeError("%s\n%s" % (sortie["error"], sortie.get("trace", "")))
    return sortie


def ecrire_video(sortie, chemin):
    """Ecrit le MP4 renvoye. Renvoie sa taille en octets."""
    donnees = base64.b64decode(sortie["video_base64"])
    with open(chemin, "wb") as f:
        f.write(donnees)
    return len(donnees)


def _b64(chemin):
    with open(chemin, "rb") as f:
        return base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    # Usage : python client.py voix.wav visage.png [dossier_sortie]
    #         [--pas 20] [--teacache] [--delai 5400]
    # La reference peut etre une image ou une video (.mp4), sur ce PC.
    # Variables : RUNPOD_KEY et INFINITETALK_ENDPOINT.
    import argparse
    from datetime import datetime

    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("ref")
    p.add_argument("dossier", nargs="?", default="videos")
    p.add_argument("--pas", type=int, default=None,
                   help="pas d'echantillonnage (defaut du worker : 8 avec "
                        "la LoRA, 40 sans)")
    p.add_argument("--mode", choices=("streaming", "clip"), default="streaming",
                   help="clip : un seul morceau de 81 images (~3,2 s), pour un essai")
    p.add_argument("--teacache", action="store_true",
                   help="saute les calculs redondants : plus vite, un peu moins net")
    p.add_argument("--delai", type=int, default=5400,
                   help="duree maximale du job, en secondes")
    a = p.parse_args()

    cle = os.environ["RUNPOD_KEY"]
    endpoint = os.environ["INFINITETALK_ENDPOINT"]
    wav, ref, dossier = a.wav, a.ref, a.dossier
    os.makedirs(dossier, exist_ok=True)

    reference = {}
    if os.path.isfile(ref):
        cle_ref = ("video_base64" if ref.lower().endswith((".mp4", ".mov"))
                   else "image_base64")
        reference[cle_ref] = _b64(ref)
    else:
        reference["reference"] = ref     # nom de fichier sur un volume

    jid = lancer(cle, endpoint, audio_base64=_b64(wav), sample_steps=a.pas,
                 use_teacache=a.teacache, delai_s=a.delai, mode=a.mode,
                 **reference)
    print("job", jid)
    print("(premier job d'un worker neuf : telechargement des poids, "
          "compter 15 a 30 min)")
    vu = {"dernier": None}

    def _suivi(message):
        if message != vu["dernier"]:
            vu["dernier"] = message
            print("           > %s" % message)

    sortie = attendre(cle, endpoint, jid,
                      trace=lambda s, t: print("  %6.1f s  %s" % (t, s)),
                      suivi=_suivi)

    nom = "%s_%s.mp4" % (os.path.splitext(os.path.basename(wav))[0],
                         datetime.now().strftime("%Y%m%d-%H%M%S"))
    chemin = os.path.join(dossier, nom)
    octets = ecrire_video(sortie, chemin)
    print("%s : %.1f Mo" % (os.path.abspath(chemin), octets / 1e6))
    print("cout : %s s de GPU par seconde de video"
          % sortie.get("s_par_seconde"))
    print("etapes :", sortie.get("etapes"))
