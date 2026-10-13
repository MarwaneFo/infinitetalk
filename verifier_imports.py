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

for etiquette, instruction in ETAPES:
    # Affiche AVANT de tenter : si l'import tue le processus (segfault,
    # OOM), la derniere ligne du journal designe quand meme l'etape.
    print("  %-*s  ... " % (largeur, etiquette), end="", flush=True)
    try:
        exec(instruction, {})
    except BaseException:
        print("ECHEC")
        print("    instruction : %s" % instruction, flush=True)
        traceback.print_exc()
        echecs += 1
        break
    print("ok", flush=True)

if echecs:
    print("\nVerification interrompue : l'etape ci-dessus a echoue.")
    sys.exit(1)

from wan.configs import WAN_CONFIGS  # noqa: E402
print("\nTous les imports passent. Taches declarees :", list(WAN_CONFIGS)[:4])
