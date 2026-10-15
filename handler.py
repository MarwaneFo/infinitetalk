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

# A poser avant tout import de torch. Essai reel sur RTX 6000 Ada : OOM au
# premier pas avec 1,25 Go "reserves mais non alloues" -- de la
# fragmentation que les segments extensibles recuperent.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import runpod

# ---------------------------------------------------------------- 1. chemins

INFINITETALK_ROOT = os.environ.get("INFINITETALK_ROOT", "/opt/InfiniteTalk")

# Sans volume reseau, le worker depose ses poids ici, sur le disque du
# conteneur. Ce disque est EPHEMERE : il disparait avec le worker, et le
# worker suivant retelecharge. C'est le prix de l'absence de volume.
_LOCAL = os.environ.get("IT_WEIGHTS_LOCAL", "/opt/weights")

# Racines ou chercher les poids, dans l'ordre. Le volume reseau RunPod est
# monte sur /runpod-volume en serverless et sur /workspace sur un pod.
_RACINES = [
    os.environ.get("IT_WEIGHTS", ""),
    "/runpod-volume/infinitetalk/weights",
    "/runpod-volume/weights",
    "/workspace/infinitetalk/weights",
    _LOCAL,
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

    quant = os.environ.get("IT_QUANT") or None          # fp8 | None
    variante = os.environ.get("IT_VARIANTE", "single")  # single | multi
    lora = os.environ.get("IT_LORA") or None

    # int8 existe pour le DiT mais pas pour le T5 : le depot amont ne
    # publie que t5_fp8.safetensors, et le chargeur T5 va le chercher des
    # que la quantification est active. int8 echouerait sur un fichier
    # absent, apres plusieurs minutes de chargement.
    if quant and quant != "fp8":
        raise RuntimeError(
            "IT_QUANT=%s non utilisable : le depot amont ne publie pas de "
            "t5_%s.safetensors, et le chargeur T5 l'exige des que la "
            "quantification est active. Seul fp8 est complet." % (quant, quant))

    # La quantification et la LoRA s'excluent cote amont
    # (multitalk.py : `if lora_dir is not None and quant is None`). Le
    # dire ici plutot que de laisser la LoRA etre ignoree en silence.
    if quant and lora:
        raise RuntimeError(
            "IT_QUANT et IT_LORA sont incompatibles : le pipeline amont "
            "ignore la LoRA quand la quantification est active. Choisir "
            "soit fp8 (~32 Go de poids, 40 pas), soit bf16 + LoRA "
            "(~93 Go, 8 pas).")

    poids = {
        "ckpt_dir": os.path.join(racine, "Wan2.1-I2V-14B-480P"),
        "wav2vec_dir": os.path.join(racine, "chinese-wav2vec2-base"),
        "infinitetalk_dir": os.path.join(
            racine, "InfiniteTalk", variante, "infinitetalk.safetensors"),
        "quant_dir": None,
        "quant": quant,
        "lora_dir": lora,
    }

    # En mode quantifie, infinitetalk_dir n'est jamais lu : on ne le
    # verifie donc pas, sinon le jeu compact serait refuse a tort.
    a_verifier = ["ckpt_dir", "wav2vec_dir"]
    if quant:
        quant_models = os.path.join(racine, "InfiniteTalk", "quant_models")
        poids["quant_dir"] = os.path.join(
            quant_models, "infinitetalk_%s_%s.safetensors" % (variante, quant))
        a_verifier += ["quant_dir", "t5_quant"]
        poids["t5_quant"] = os.path.join(quant_models, "t5_%s.safetensors" % quant)
    else:
        a_verifier.append("infinitetalk_dir")

    for cle in a_verifier:
        chemin = poids.get(cle)
        if chemin and not os.path.exists(chemin):
            raise RuntimeError("%s manquant : %s" % (cle, chemin))
    poids.pop("t5_quant", None)
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
    if surcharges.get("sample_shift") is not None:
        argv += ["--sample_shift", str(float(surcharges.pop("sample_shift")))]
    surcharges.pop("sample_shift", None)

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

    _installer_suivi()
    if not poids["quant"]:
        _installer_chargement_econome()

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

    if _bool(os.environ.get("IT_T5_CPU"), False):
        _accelerer_t5_cpu(pipeline)

    _clip_hors_gpu(pipeline)
    _delester(pipeline)

    # wav2vec reste sur le CPU : c'est ce que fait le script amont, et ces
    # 380 Mo n'ont rien a gagner a occuper de la VRAM.
    _ETAT["wav2vec"] = gi.custom_init("cpu", poids["wav2vec_dir"])
    _ETAT["pipeline"] = pipeline
    _ETAT["pret"] = True
    _ETAT["chargement_s"] = round(time.time() - depart, 1)
    print("[init] pret en %.1f s" % _ETAT["chargement_s"], flush=True)


# ---------------------------------------------------------------- 2 bis. suivi

# Le job en cours et l'avancement de la diffusion. Sans cela, un job de
# quarante minutes reste un IN_PROGRESS muet : impossible de distinguer un
# calcul lent d'un processus fige sans aller lire la jauge GPU.
_SUIVI = {"job": None, "morceau": 0, "prevus": None}


def _publier(message):
    """Ecrit l'avancement dans les journaux ET dans le statut du job.

    progress_update rend le message lisible par GET /status pendant que le
    job tourne : le client l'affiche sans ouvrir la console.
    """
    print("[suivi] %s" % message, flush=True)
    if _SUIVI["job"] is not None:
        try:
            runpod.serverless.progress_update(_SUIVI["job"], message)
        except Exception as exc:
            print("[suivi] progress_update a echoue : %s" % exc, flush=True)


def _installer_suivi():
    """Remplace la barre tqdm du pipeline amont par une barre qui parle.

    La barre d'origine se redessine avec des retours chariot : dans les
    journaux RunPod, rien n'apparait avant la fin d'un morceau. wan/multitalk
    resout `tqdm` au moment de l'appel, on peut donc la substituer ici.
    """
    import wan.multitalk as wm
    from tqdm import tqdm as tqdm_amont

    class _Barre(tqdm_amont):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True        # plus de barre illisible
            super().__init__(*args, **kwargs)
            _SUIVI["morceau"] += 1

        def __iter__(self):
            depart = time.time()
            total = self.total or 0
            for pas, element in enumerate(super().__iter__(), 1):
                yield element
                ecoule = time.time() - depart
                prevus = _SUIVI["prevus"]
                # La VRAM occupee, a chaque pas : sur une video de 30 s, un
                # OOM est survenu au 5e morceau sur 11, la memoire montant
                # d'un morceau a l'autre. Ce chiffre le rend visible tot.
                try:
                    import torch
                    vram = " | VRAM %.1f Go" % (torch.cuda.memory_allocated()
                                                / 1e9)
                except Exception:
                    vram = ""
                _publier("morceau %d%s | pas %d/%d | %.1f s/pas | reste ~%d s "
                         "sur ce morceau%s"
                         % (_SUIVI["morceau"],
                            "/%d" % prevus if prevus else "",
                            pas, total, ecoule / pas,
                            (total - pas) * ecoule / pas, vram))

    wm.tqdm = _Barre


def _clip_hors_gpu(pipeline):
    """Renvoie CLIP en RAM apres chaque usage.

    CLIP ne sert qu'une fois par morceau, a encoder l'image de reference.
    Le code amont ne le decharge qu'avec offload_model, qui decharge aussi
    le DiT entre deux morceaux -- 33 Go de va-et-vient. Ici, seul CLIP part :
    ~2,4 Go de VRAM rendus au DiT pendant la diffusion.
    """
    import torch
    clip = pipeline.clip
    visuel_amont = clip.visual

    def visuel_econome(*args, **kwargs):
        try:
            return visuel_amont(*args, **kwargs)
        finally:
            clip.model.cpu()
            torch.cuda.empty_cache()

    clip.visual = visuel_econome
    clip.model.cpu()
    torch.cuda.empty_cache()


def _delester(pipeline):
    """Laisse une partie des poids du DiT en RAM, envoyes au GPU a la demande.

    Mesure sur RTX 6000 Ada (44,4 Go utilisables) : DiT + CLIP + VAE
    occupaient 42,4 Go, et le premier pas en demandait 0,9 de plus.

    IT_DELESTAGE_GO fixe le volume de poids laisses en RAM. Par defaut :
    0 au-dela de 60 Go de VRAM, 12 en dessous. Six suffisaient pour un
    morceau, pas pour une video de 30 s : OOM dans le decodage VAE au 5e
    morceau sur 11 (L40S, 42,7 Go alloues). Douze Go par passe ajoutent
    ~20 s a un morceau de 4 min, et n'en prennent que douze en RAM, la ou
    l'ancien IT_LOW_VRAM=1 (tout le DiT, ~38 Go) declencherait un OOM de
    RAM a cote du T5.
    """
    import torch
    valeur = os.environ.get("IT_DELESTAGE_GO")
    if _bool(os.environ.get("IT_LOW_VRAM"), False) and valeur is None:
        valeur = "all"
    if valeur is None:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        valeur = "12" if vram < 60 else "0"
    if valeur == "0":
        return

    total = sum(p.numel() for p in pipeline.model.parameters())
    if valeur == "all":
        persistants = 0
    else:
        octets = next(pipeline.model.parameters()).element_size()
        persistants = max(0, total - int(float(valeur) * 1e9 / octets))
    pipeline.vram_management = True
    pipeline.enable_vram_management(num_persistent_param_in_dit=persistants)
    torch.cuda.empty_cache()
    print("[init] delestage DiT : %.1f G parametres sur %.1f restent en VRAM"
          % (persistants / 1e9, total / 1e9), flush=True)


def _coeurs_alloues():
    """Coeurs reellement alloues au conteneur, pas ceux de la machine hote.

    torch.get_num_threads() voit souvent tous les coeurs de l'hote (parfois
    plus de cent) quand le conteneur n'en a que seize : le calcul se disperse
    sur des fils qui se disputent seize coeurs, et ralentit au lieu
    d'accelerer. On lit le quota cgroup, a defaut l'affinite.
    """
    try:
        quota, periode = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(periode)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


def _ram_libre_go():
    """RAM encore disponible dans le conteneur, cache de fichiers exclu.

    Le compteur que montre RunPod inclut le cache des 95 Go de poids lus
    depuis le disque ; ce cache se libere a la demande et ne declenche pas
    d'OOM. On compare donc la limite cgroup a la memoire anonyme seule.
    """
    try:
        limite = open("/sys/fs/cgroup/memory.max").read().strip()
        if limite == "max":
            return None
        anonyme = 0
        for ligne in open("/sys/fs/cgroup/memory.stat"):
            cle, valeur = ligne.split()
            if cle == "anon":
                anonyme = int(valeur)
                break
        return (int(limite) - anonyme) / 1e9
    except (OSError, ValueError):
        return None


def _accelerer_t5_cpu(pipeline):
    """Rend l'encodeur T5 supportable quand il tourne sur le CPU.

    Mesure sur RTX 6000 Ada, IT_T5_CPU=1 : plus de cinq minutes avant le
    premier pas de diffusion, passees a encoder deux phrases. Le code amont
    encode l'invite puis l'invite negative a CHAQUE job, en bf16, sur CPU.

    Trois remedes :
      - limiter les fils au quota reel du conteneur ;
      - passer T5 en fp32 : les processeurs sans bf16 materiel calculent
        bien plus vite en fp32 (le double de RAM, ~23 Go, supportable
        maintenant que le DiT est charge directement sur le GPU) ;
      - memoriser les encodages : l'invite change rarement, le second job
        d'un meme worker ne paie plus rien.
    """
    import torch

    coeurs = _coeurs_alloues()
    torch.set_num_threads(coeurs)
    print("[init] T5 sur CPU : %d fils" % coeurs, flush=True)

    encodeur = pipeline.text_encoder
    marge = _ram_libre_go()
    if not _bool(os.environ.get("IT_T5_FP32"), True):
        pass
    elif marge is not None and marge < 20:
        # La conversion ajoute ~11 Go. Un OOM tue le worker sans message :
        # mieux vaut un T5 lent qu'un worker mort.
        print("[init] T5 garde en bf16 : %.0f Go de RAM libres seulement"
              % marge, flush=True)
    else:
        encodeur.model.float()
        print("[init] T5 converti en fp32 pour le CPU (marge RAM : %s Go)"
              % ("?" if marge is None else "%.0f" % marge), flush=True)

    memoire = {}
    appel_amont = encodeur.__call__

    class _ModeleImmobile:
        """Le T5 reste sur le CPU : .to(cuda) et .cpu() deviennent sans effet.

        La branche GPU amont commence par text_encoder.model.to(device) ;
        sans ce garde-fou, elle y enverrait 11 a 23 Go de T5.
        """

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def __getattr__(self, nom):
            return getattr(encodeur.model, nom)

    class _T5Memorise:
        """Remplace pipeline.text_encoder ; delegue tout le reste.

        Le pipeline est force sur sa branche GPU (t5_cpu = False) : la
        branche CPU amont est fausse. Elle construit 'context': [context]
        avec un context qui est deja une liste, d'ou une liste de listes et
        "'list' object has no attribute 'dtype'" au premier pas -- constate
        sur L40S. La branche GPU, elle, depaquete les deux encodages. On
        calcule ici sur le CPU, et on rend des tenseurs bf16 sur le GPU.
        """

        model = _ModeleImmobile()

        def __getattr__(self, nom):
            return getattr(encodeur, nom)

        def __call__(self, textes, device):
            sortie = []
            for texte in textes:
                if texte not in memoire:
                    depart = time.time()
                    with torch.no_grad():
                        t = appel_amont([texte], torch.device("cpu"))[0]
                    # Le DiT attend du bf16 : un T5 en fp32 ne doit pas
                    # changer le type de ce qu'il rend.
                    memoire[texte] = t.to(torch.bfloat16)
                    _publier("encodage du texte : %.1f s"
                             % (time.time() - depart))
                sortie.append(memoire[texte].to(device))
            return sortie

    pipeline.text_encoder = _T5Memorise()
    pipeline.t5_cpu = False


def _installer_chargement_econome():
    """Charge les poids bf16 tenseur par tenseur, sans dictionnaire geant.

    En mode non quantifie, le code amont cree le modele (~33 Go en bf16),
    puis charge les sept fragments fp32 ET infinitetalk.safetensors dans un
    seul dictionnaire (~75 Go) avant load_state_dict : pic d'environ 110 Go
    de RAM. Les workers RTX 6000 Ada / L40S n'en ont que 62 : le processus
    serait tue apres vingt minutes de telechargement.

    Ici, chaque fichier est ouvert paresseusement et chaque tenseur copie
    directement dans le parametre du modele (avec conversion en bf16). Le
    pic retombe a la taille du modele plus un tenseur. La rigueur de
    load_state_dict(strict=True) est conservee : cles manquantes ou
    inattendues levent une erreur explicite.
    """
    import wan.multitalk as wm
    from safetensors import safe_open

    etat = {"modele": None, "copiees": set()}
    load_file_amont = wm.load_file

    import torch

    class _WanEconome(wm.WanModel):
        def __init__(self, *args, **kwargs):
            # Construit SANS allouer (device meta) : le constructeur amont
            # cree les 14 milliards de parametres en fp32, soit ~66 Go de
            # RAM avant meme la conversion en bf16. Mesure : OOM a 57,7 Go
            # sur RTX 6000 Ada. On materialise ensuite chaque parametre
            # vide, en bf16, directement sur le GPU : la RAM ne porte plus
            # le modele, et les valeurs viennent des fichiers.
            with torch.device("meta"):
                super().__init__(*args, **kwargs)
            cible = "cuda:0" if torch.cuda.is_available() else "cpu"
            for module in self.modules():
                for nom, p in list(module._parameters.items()):
                    if p is None:
                        continue
                    dt = torch.bfloat16 if p.is_floating_point() else p.dtype
                    module._parameters[nom] = torch.nn.Parameter(
                        torch.empty(p.shape, dtype=dt, device=cible),
                        requires_grad=False)
                for nom, b in list(module._buffers.items()):
                    if b is None:
                        continue
                    dt = torch.bfloat16 if b.is_floating_point() else b.dtype
                    module._buffers[nom] = torch.empty(b.shape, dtype=dt,
                                                       device=cible)
            # freqs est un simple attribut calcule a la construction : il
            # est reste sur meta. Meme remede que le chemin quantifie amont.
            self.init_freqs()
            print("[init] modele materialise vide en bf16 sur %s" % cible,
                  flush=True)
            etat["modele"] = self
            etat["copiees"] = set()

        def load_state_dict(self, state_dict, strict=True, assign=False):
            if state_dict or self is not etat["modele"]:
                return super().load_state_dict(state_dict, strict=strict,
                                               assign=assign)
            attendues = set(self.state_dict().keys())
            manquantes = attendues - etat["copiees"]
            if strict and manquantes:
                raise RuntimeError("Poids manquants apres chargement : %s"
                                   % sorted(manquantes)[:10])
            return None

    def load_file_econome(chemin, device="cpu"):
        modele = etat["modele"]
        if modele is None:
            return load_file_amont(chemin, device=device)
        cibles = dict(modele.named_parameters())
        cibles.update(dict(modele.named_buffers()))
        inattendues = []
        with safe_open(chemin, framework="pt", device="cpu") as f:
            for cle in f.keys():
                if cle not in cibles:
                    inattendues.append(cle)
                    continue
                with torch.no_grad():
                    cibles[cle].data.copy_(f.get_tensor(cle))
                etat["copiees"].add(cle)
        if inattendues:
            raise RuntimeError("Cles inattendues dans %s : %s"
                               % (os.path.basename(chemin), inattendues[:10]))
        print("[init] %s copie dans le modele" % os.path.basename(chemin),
              flush=True)
        gc.collect()
        return {}

    wm.WanModel = _WanEconome
    wm.load_file = load_file_econome


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


# ------------------------------- 3 bis. amorcage des poids depuis le worker

# Jeu minimal en fp8 : environ 32 Go. En mode quantifie, le pipeline ne
# lit jamais les sept fragments bf16 (65,6 Go) ni infinitetalk.safetensors
# (9,9 Go) -- seul config.json l'est, et le T5 prend sa version fp8.
_COMPACT = {
    "Wan-AI/Wan2.1-I2V-14B-480P": [
        "config.json", "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "google/umt5-xxl/*", "xlm-roberta-large/*",
    ],
    "MeiGen-AI/InfiniteTalk": [
        "quant_models/infinitetalk_%(variante)s_fp8.safetensors",
        "quant_models/infinitetalk_%(variante)s_fp8.json",
        "quant_models/t5_fp8.safetensors",
        "quant_models/t5_map_fp8.json",
        "quant_models/quant.json",
    ],
}

# Jeu complet en bf16 : environ 93 Go, mais la LoRA FusionX devient
# possible, et avec elle 8 pas d'echantillonnage au lieu de 40.
_COMPLET = {
    "Wan-AI/Wan2.1-I2V-14B-480P": None,       # tout, sauf les illustrations
    "MeiGen-AI/InfiniteTalk": ["%(variante)s/infinitetalk.safetensors"],
}

_WAV2VEC = "TencentGameMate/chinese-wav2vec2-base"


def _octets(chemin):
    """Taille totale d'une arborescence, en octets."""
    total = 0
    for racine, _, fichiers in os.walk(chemin):
        for f in fichiers:
            try:
                total += os.path.getsize(os.path.join(racine, f))
            except OSError:
                pass
    return total


def _telecharger(entree):
    """Depose les poids sur le volume reseau, depuis le worker lui-meme.

    Sans pod, il n'existe aucun terminal ou lancer download_weights.sh.
    Or le worker a le volume monte en ecriture : il peut s'amorcer seul.
    C'est factur au tarif GPU, donc plus cher qu'un pod CPU -- mais pour
    une heure, la difference ne justifie pas le detour.

    Le telechargement est REPRENABLE : huggingface_hub saute ce qui est
    deja la. Si le travail depasse le delai maximum de l'endpoint, il
    suffit de le relancer, il continuera ou il s'est arrete.
    """
    from huggingface_hub import snapshot_download, hf_hub_download

    cible = entree.get("cible") or "/runpod-volume/infinitetalk/weights"
    mode = entree.get("mode_poids", "compact")
    if mode not in ("compact", "complet"):
        return {"error": "mode_poids doit valoir compact ou complet."}
    variante = os.environ.get("IT_VARIANTE", "single")

    # hf_transfer sature la bande passante : sur des dizaines de Go, la
    # difference se compte en heures. Mais huggingface_hub LEVE une
    # erreur si on active le reglage sans que le paquet soit installe --
    # on verifie donc au lieu de supposer.
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        print("[poids] hf_transfer absent : telechargement standard",
              flush=True)

    os.makedirs(cible, exist_ok=True)
    plan = dict(_COMPACT if mode == "compact" else _COMPLET)
    etapes, total = {}, time.time()

    for depot, motifs in plan.items():
        depart = time.time()
        dossier = os.path.join(cible, depot.split("/")[-1])
        if depot == "MeiGen-AI/InfiniteTalk":
            dossier = os.path.join(cible, "InfiniteTalk")
        args = {"repo_id": depot, "local_dir": dossier,
                "max_workers": int(entree.get("parallelisme", 8))}
        if motifs:
            args["allow_patterns"] = [m % {"variante": variante}
                                      for m in motifs]
        else:
            args["ignore_patterns"] = ["assets/*", "examples/*",
                                       "*.png", "*.jpg", "*.JPG"]
        _publier("telechargement %s" % depot)
        snapshot_download(**args)
        etapes[depot] = round(time.time() - depart, 1)

    # L'encodeur audio, moins le point de controle fairseq (1,1 Go) que
    # le code n'utilise pas. Le safetensors vient d'une pull request
    # encore ouverte : c'est ce que demande le README amont.
    depart = time.time()
    dossier = os.path.join(cible, "chinese-wav2vec2-base")
    _publier("telechargement %s" % _WAV2VEC)
    snapshot_download(repo_id=_WAV2VEC, local_dir=dossier,
                      ignore_patterns=["*fairseq*"])
    try:
        hf_hub_download(repo_id=_WAV2VEC, filename="model.safetensors",
                        revision="refs/pr/1", local_dir=dossier)
    except Exception as exc:
        print("[poids] safetensors de la PR indisponible : %s" % exc,
              flush=True)
    etapes[_WAV2VEC] = round(time.time() - depart, 1)

    attendus = ["Wan2.1-I2V-14B-480P/config.json",
                "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
                "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-"
                "large-vit-huge-14.pth",
                "chinese-wav2vec2-base/config.json"]
    if mode == "compact":
        attendus += ["InfiniteTalk/quant_models/infinitetalk_%s_fp8."
                     "safetensors" % variante,
                     "InfiniteTalk/quant_models/t5_fp8.safetensors",
                     "InfiniteTalk/quant_models/t5_map_fp8.json"]
    else:
        attendus += ["Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                     "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-"
                     "of-00007.safetensors",
                     "InfiniteTalk/%s/infinitetalk.safetensors" % variante]
    manquants = [a for a in attendus
                 if not os.path.exists(os.path.join(cible, a))]

    return {
        "action": "telecharger",
        "mode_poids": mode,
        "cible": cible,
        "complet": not manquants,
        "manquants": manquants,
        "taille_go": round(_octets(cible) / 1e9, 2),
        "etapes": etapes,
        "total_s": round(time.time() - total, 1),
        "suite": ("Poids en place. Renseigner IT_WEIGHTS=%s%s puis relancer "
                  "un vrai travail." % (cible, " et IT_QUANT=fp8"
                                        if mode == "compact" else ""))
        if not manquants else
        "Incomplet : relancer ce meme travail, le telechargement reprend.",
    }


# LoRA FusionX : distillation qui ramene l'echantillonnage a 8 pas, avec
# un guidage texte a 1 (deux passes du modele par pas au lieu de trois).
_FUSIONX_DEPOT = "vrgamedevgirl84/Wan14BT2VFusioniX"
_FUSIONX_FICHIER = "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"


def _amorcer_local():
    """Telecharge les poids sur le disque du conteneur s'ils manquent.

    Ne fait rien quand un volume porte deja les poids. Sinon, premier job
    d'un worker neuf : les poids sont rapatries avant le chargement, ce
    qui compte dans le delai d'execution du job. Renvoie la duree, ou None.

    IT_MODE_POIDS choisit le jeu :
      compact -- ~32 Go, fp8, 40 pas. Mesure sur A40 : 210 s par pas, soit
                 plus de deux heures par morceau. Inutilisable en production.
      rapide  -- ~95 Go, bf16 + LoRA FusionX, 8 pas. Demande 80 Go de VRAM
                 et ~120 Go de Container Disk.
    """
    hors_local = [r for r in _RACINES if r and r != _LOCAL]
    if _premiere_racine(hors_local, "Wan2.1-I2V-14B-480P",
                        "chinese-wav2vec2-base"):
        return None
    if not _bool(os.environ.get("IT_AUTO_POIDS"), True):
        return None

    mode = os.environ.get("IT_MODE_POIDS", "compact")
    if mode not in ("compact", "rapide"):
        raise RuntimeError("IT_MODE_POIDS doit valoir compact ou rapide.")
    variante = os.environ.get("IT_VARIANTE", "single")
    lora = os.path.join(_LOCAL, _FUSIONX_FICHIER)

    if mode == "compact":
        # Le jeu compact ne se charge qu'en fp8 : le poser d'office, sinon
        # le worker irait chercher les fragments bf16 qui ne sont pas la.
        if not os.environ.get("IT_QUANT"):
            os.environ["IT_QUANT"] = "fp8"
        attendus = ["Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
                    "chinese-wav2vec2-base/config.json",
                    "InfiniteTalk/quant_models/infinitetalk_%s_fp8.safetensors"
                    % variante,
                    "InfiniteTalk/quant_models/t5_fp8.safetensors"]
        besoin_go = 36
    else:
        # La LoRA est refusee en presence de quantification : l'oter.
        os.environ.pop("IT_QUANT", None)
        if not os.environ.get("IT_LORA"):
            os.environ["IT_LORA"] = lora
        attendus = ["Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
                    "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                    "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-"
                    "of-00007.safetensors",
                    "chinese-wav2vec2-base/config.json",
                    "InfiniteTalk/%s/infinitetalk.safetensors" % variante,
                    _FUSIONX_FICHIER]
        besoin_go = 100
    if all(os.path.exists(os.path.join(_LOCAL, a)) for a in attendus):
        return None

    # Verifier la place AVANT : un disque plein en fin de telechargement
    # coute vingt minutes de GPU pour rien, et l'erreur est obscure.
    os.makedirs(_LOCAL, exist_ok=True)
    libre = shutil.disk_usage(_LOCAL).free / 1e9
    deja = _octets(_LOCAL) / 1e9
    if libre + deja < besoin_go:
        raise RuntimeError(
            "Disque du conteneur trop petit : %.0f Go libres, il en faut "
            "environ %d pour le mode %s. Monter le Container Disk de "
            "l'endpoint a %d Go." % (libre, besoin_go, mode,
                                     60 if mode == "compact" else 130))

    print("[poids] aucun volume : mode %s, telechargement sur %s "
          "(%.0f Go libres)" % (mode, _LOCAL, libre), flush=True)
    depart = time.time()
    r = _telecharger({"cible": _LOCAL, "mode_poids":
                      "compact" if mode == "compact" else "complet"})
    if not r.get("complet"):
        raise RuntimeError("Telechargement incomplet, manquent : %s"
                           % r.get("manquants"))
    if mode == "rapide" and not os.path.exists(lora):
        from huggingface_hub import hf_hub_download
        _publier("telechargement LoRA FusionX")
        hf_hub_download(repo_id=_FUSIONX_DEPOT, filename=_FUSIONX_FICHIER,
                        local_dir=_LOCAL)
    return round(time.time() - depart, 1)


def _diagnostic(entree):
    """Dit ce qui se trouve reellement sur le volume, sans rien charger.

    Evite de deduire l'etat du volume d'un echec de chargement.
    """
    racines = [r for r in _RACINES if r] + ["/runpod-volume", "/workspace"]
    vu = {}
    for r in racines:
        if not os.path.isdir(r):
            vu[r] = "absent"
            continue
        try:
            entrees = sorted(os.listdir(r))[:20]
        except OSError as exc:
            vu[r] = "illisible : %s" % exc
            continue
        vu[r] = {"contenu": entrees, "taille_go": round(_octets(r) / 1e9, 2)}
    try:
        disque = shutil.disk_usage(_LOCAL if os.path.isdir(_LOCAL) else "/")
        vu["disque_conteneur"] = {"libre_go": round(disque.free / 1e9, 1),
                                  "total_go": round(disque.total / 1e9, 1)}
    except OSError:
        pass
    return {"action": "diagnostic", "racines": vu,
            "IT_WEIGHTS": os.environ.get("IT_WEIGHTS"),
            "IT_QUANT": os.environ.get("IT_QUANT"),
            "IT_VARIANTE": os.environ.get("IT_VARIANTE", "single"),
            "IT_LOW_VRAM": os.environ.get("IT_LOW_VRAM"),
            "IT_DELESTAGE_GO": os.environ.get("IT_DELESTAGE_GO"),
            "IT_T5_CPU": os.environ.get("IT_T5_CPU"),
            "IT_MODE_POIDS": os.environ.get("IT_MODE_POIDS", "compact"),
            "IT_LORA": os.environ.get("IT_LORA")}


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
    _SUIVI.update(job=job, morceau=0, prevus=None)

    try:
        # Deux modes de service, traites AVANT tout chargement de modele :
        # ils doivent fonctionner precisement quand les poids manquent.
        if _bool(entree.get("diagnostic"), False):
            return _diagnostic(entree)
        if _bool(entree.get("telecharger"), False):
            return _telecharger(entree)

        premier = not _ETAT["pret"]
        if premier:
            _publier("verification des poids")
            duree = _amorcer_local()
            if duree is not None:
                etapes["telechargement"] = duree

        depart = time.time()
        if premier:
            _publier("chargement du modele")
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

            # Avec la LoRA FusionX, les reglages de reference changent :
            # 8 pas, guidage texte 1 (deux passes par pas), audio 2, shift 2.
            # Une requete peut toujours les surcharger.
            if _ETAT["poids"]["lora_dir"]:
                d_pas, d_texte, d_audio, d_shift = 8, 1.0, 2.0, 2.0
            else:
                d_pas, d_texte, d_audio, d_shift = 40, 5.0, 4.0, None
            shift = entree.get("sample_shift", d_shift)
            args = _args_amont(
                size=taille,
                mode=mode,
                sample_steps=_borne(entree.get("sample_steps", d_pas),
                                    1, 60, d_pas),
                sample_shift=(None if shift is None
                              else _borne(shift, 0.5, 20.0, 2.0)),
                sample_text_guide_scale=_borne(
                    entree.get("text_guide_scale", d_texte), 1.0, 10.0, d_texte),
                sample_audio_guide_scale=_borne(
                    entree.get("audio_guide_scale", d_audio), 1.0, 10.0, d_audio),
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
            # Morceaux prevus : 81 images, dont motion_frame de recouvrement.
            images = int(len(parole) / 16000 * 25)
            pas_morceau = max(1, 81 - args.motion_frame)
            _SUIVI["prevus"] = (1 if images <= 81 or mode == "clip"
                                else 1 + -(-(images - 81) // pas_morceau))
            _publier("diffusion : %d morceau(x) de %d pas prevus"
                     % (_SUIVI["prevus"], args.sample_steps))
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
            _publier("encodage du MP4")
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
