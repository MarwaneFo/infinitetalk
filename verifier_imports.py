"""Verifie les imports du code amont, un par un, a la construction.

Le but n'est pas de tester : c'est de rendre un echec lisible du premier
coup. Importer tout en une seule ligne produit une trace qui remonte a
une dependance lointaine sans dire quelle etape du pipeline la reclamait,
et il faut alors un cycle de build complet -- plusieurs dizaines de
minutes -- pour le deviner.

Ici chaque import est annonce AVANT d'etre tente. La derniere ligne
affichee dans le journal nomme donc le coupable, meme si la trace qui
suit est illisible.
"""
import sys
import traceback

RACINE = "/opt/InfiniteTalk"
sys.path.insert(0, RACINE)


def _feindre_un_gpu():
    """Laisse passer le code amont qui interroge CUDA des l'import.

    wan/modules/t5.py ecrit, en valeur par defaut d'argument :

        device=torch.cuda.current_device(),

    Une valeur par defaut est evaluee quand la classe est definie, donc
    a l'import du module. Sur une machine pourvue d'un GPU cela marche ;
    sur un runner d'integration continue, qui n'en a pas, cela leve

        RuntimeError: Found no NVIDIA driver on your system

    Le code amont n'est pas en faute et le worker, lui, aura un GPU.
    C'est la verification qui doit s'adapter : son travail est de
    controler que les DEPENDANCES sont la et s'importent, pas qu'un GPU
    soit present. La presence du GPU est verifiee a l'execution, par
    _garde_gpu() dans le handler, qui sait aussi refuser les cartes trop
    recentes.

    On rend donc les quelques interrogations faites a l'import
    inoffensives. is_available() reste honnete a False : le code qui
    teste reellement la disponibilite continue de voir la verite.
    """
    import torch
    if torch.cuda.is_available():
        return False
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_capability = lambda *a, **k: (8, 6)
    torch.cuda.get_device_name = lambda *a, **k: "verification sans GPU"
    return True


def _est_une_absence_de_gpu(exc):
    """Vrai si l'echec vient de l'absence de GPU, pas d'une dependance."""
    texte = str(exc).lower()
    return any(m in texte for m in (
        "no nvidia driver", "no cuda gpus", "cuda driver",
        "torch not compiled with cuda", "cuda unavailable"))

# (etiquette, instruction). L'ordre suit la chaine reelle d'imports :
# les socles d'abord, le point d'entree en dernier.
ETAPES = [
    ("socle numerique",      "import numpy, torch"),
    ("diffusers",            "import diffusers"),
    ("diffusers : les deux noms qui ont manque",
     "from diffusers.models.modeling_utils import no_init_weights, ContextManagers"),
    ("transformers",         "import transformers"),
    ("optimum-quanto",       "from optimum.quanto import quantize, freeze, requantize"),
    ("xfuser (remplacant mono-GPU)",
     "from xfuser.core.distributed import get_sequence_parallel_rank, "
     "get_sequence_parallel_world_size, get_sp_group"),
    ("flash-attn",           "import flash_attn"),
    ("xformers",             "import xformers.ops"),
    ("audio",                "import librosa, soundfile, pyloudnorm"),
    ("video",                "import cv2, imageio, decord, moviepy"),
    ("decoupage de scenes",  "import scenedetect"),
    ("wan.configs",          "from wan.configs import WAN_CONFIGS, SIZE_CONFIGS"),
    ("wan (paquet complet)", "import wan"),
    ("pipeline InfiniteTalk", "from wan import InfiniteTalkPipeline"),
    ("kokoro",               "from kokoro import KPipeline"),
    ("wav2vec2 local",       "from src.audio_analysis.wav2vec2 import Wav2Vec2Model"),
    ("decoupage video",      "from wan.utils.segvideo import shot_detect"),
    ("ecriture video",       "from wan.utils.multitalk_utils import save_video_ffmpeg"),
    ("point d'entree",       "import generate_infinitetalk"),
]

largeur = max(len(e) for e, _ in ETAPES)
echecs = 0
differes = 0

if _feindre_un_gpu():
    print("  (aucun GPU ici : les appels CUDA faits a l'import sont neutralises)\n")

for etiquette, instruction in ETAPES:
    # Affiche AVANT de tenter : si l'import tue le processus (segfault,
    # OOM), la derniere ligne du journal designe quand meme l'etape.
    print("  %-*s  ... " % (largeur, etiquette), end="", flush=True)
    try:
        exec(instruction, {})
    except BaseException as exc:
        if _est_une_absence_de_gpu(exc):
            # La dependance est bien la : elle reclame un GPU, que le
            # worker aura. Ce n'est pas un defaut de l'image.
            print("differe (demande un GPU)", flush=True)
            differes += 1
            continue
        print("ECHEC")
        print("    instruction : %s" % instruction, flush=True)
        traceback.print_exc()
        echecs += 1
        break
    print("ok", flush=True)

if echecs:
    print("\nVerification interrompue : l'etape ci-dessus a echoue.")
    sys.exit(1)

if differes:
    print("\n%d etape(s) differee(s) faute de GPU sur la machine de "
          "construction. Les dependances sont en place ; le reste se "
          "verifie au premier job." % differes)

# Le bilan ne doit pas rater ce que les etapes ont tolere : si
# wan.configs a ete differe faute de GPU, l'importer ici releverait la
# meme erreur et ferait echouer le build pour rien.
try:
    from wan.configs import WAN_CONFIGS  # noqa: E402
    print("\nTous les imports passent. Taches declarees :",
          list(WAN_CONFIGS)[:4])
except BaseException as exc:
    if not _est_une_absence_de_gpu(exc):
        raise
    print("\nDependances completes. Les taches declarees ne sont "
          "lisibles qu'avec un GPU.")
