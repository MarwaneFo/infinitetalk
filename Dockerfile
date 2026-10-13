########################################################################
#  InfiniteTalk - image GPU pour RunPod Serverless
#
#  L'image ne contient QUE le code et les dependances. Les poids font
#  environ 93 Go : ils vivent sur un volume reseau, monte sur
#  /runpod-volume. Voir download_weights.sh et le README.
#
#  La recette suit a la lettre celle du depot amont : torch 2.4.1 en
#  cu121, xformers 0.0.28, flash-attn 2.7.4.post1. S'en ecarter casse
#  les noyaux d'attention de Wan2.1.
########################################################################
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    USE_TF=0 \
    TRANSFORMERS_NO_TF=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    INFINITETALK_ROOT=/opt/InfiniteTalk

# ---------- 1. Systeme ----------
# libgl1 et libglib2.0 : opencv et decord. libsndfile1 : soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      git curl ca-certificates \
      ffmpeg libsndfile1 libgl1 libglib2.0-0 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# ---------- 2. pip d'abord, et on le prouve ----------
# Ubuntu 22.04 livre pip 22.0.2 et setuptools 59. Or setuptools ignore en
# SILENCE la section [project] du format moderne avant la version 61 : le
# paquet sort alors en UNKNOWN-0.0.0, sans nom ni dependances, et le
# probleme se deguise en erreur numpy deux etapes plus loin. On met donc
# pip a jour en premier, et on affiche les versions obtenues.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -c "import setuptools, pip; print('pip', pip.__version__, '| setuptools', setuptools.__version__)"

# ---------- 3. torch, exactement la version amont ----------
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && pip install -U xformers==0.0.28 \
      --index-url https://download.pytorch.org/whl/cu121 \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ---------- 4. flash-attn par roue precompilee ----------
# Compiler flash-attn depuis les sources demande plus d'une heure et
# depasse la memoire des runners GitHub. La roue officielle correspond
# exactement a cp310 / torch 2.4 / cu12 / abiFALSE, qui est l'ABI des
# roues torch de l'index PyTorch.
ARG FLASH_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install ninja psutil packaging \
 && pip install "${FLASH_WHL}" \
 && python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

# ---------- 5. Code InfiniteTalk, fige sur un commit ----------
# On epingle volontairement : un depot amont suivi sur `main` change de
# comportement au prochain rebuild sans que rien ne l'annonce. Pour
# monter de version, modifier INFINITETALK_REF et relire le journal.
ARG INFINITETALK_REPO=https://github.com/MeiGen-AI/InfiniteTalk.git
ARG INFINITETALK_REF=50aa0a94184315407a991ae804d9b58d6d311ba8
RUN git clone "${INFINITETALK_REPO}" ${INFINITETALK_ROOT} \
 && cd ${INFINITETALK_ROOT} \
 && git checkout "${INFINITETALK_REF}" \
 && rm -rf .git

WORKDIR ${INFINITETALK_ROOT}

# ---------- 6. Dependances amont ----------
# misaki[en] est reclame par kokoro (la synthese vocale integree). On le
# garde : generate_infinitetalk l'importe au chargement du module.
RUN pip install "misaki[en]" \
 && pip install -r requirements.txt \
 && pip install librosa

# ---------- 7. Dependances du worker, et etat final deterministe ----------
# Trois pieges, tous deja payes sur les workers precedents :
#
#  - gradio arrive par requirements.txt pour app.py, dont le serverless
#    n'a aucun usage. Il traine fastapi, uvicorn et un numpy recent.
#  - requirements.txt demande numpy<2, mais une dependance transitive le
#    fait souvent remonter en 2.x. L'ABI change entre les deux : les
#    extensions compilees contre numpy 1 cessent de se charger.
#  - xfuser (multi-GPU, inutilise ici) peut remonter torch. On verifie.
#
# D'ou ce bloc final : on desinstalle, on repingle en --no-deps, puis on
# affirme. Si l'etat n'est pas celui attendu, le build echoue ICI, avec
# un message d'une ligne, au lieu de laisser le worker mourir en silence.
RUN pip install runpod requests soundfile \
 && (pip uninstall -y gradio gradio-client || true) \
 && pip install --force-reinstall --no-deps "numpy==1.26.4" \
 && python -c "import numpy, torch; \
assert numpy.__version__.startswith('1.'), 'numpy remonte en ' + numpy.__version__; \
assert torch.__version__.startswith('2.4.1'), 'torch deplace en ' + torch.__version__; \
print('etat final : numpy', numpy.__version__, '| torch', torch.__version__)"

# ---------- 8. Verification du code amont ----------
# Un import du pipeline a la construction : si une dependance manque, on
# l'apprend maintenant et par son nom, pas au premier job sur GPU.
RUN python -c "import sys; sys.path.insert(0, '${INFINITETALK_ROOT}'); \
import wan; from wan.configs import WAN_CONFIGS; \
import generate_infinitetalk; \
print('pipeline importe, taches :', list(WAN_CONFIGS)[:4])"

# ---------- 9. Handler ----------
COPY handler.py ${INFINITETALK_ROOT}/handler.py

# Reglages par defaut, surchargeables dans l'endpoint RunPod.
ENV IT_VARIANTE=single \
    IT_LOW_VRAM=0 \
    IT_T5_CPU=0

CMD ["python", "-u", "/opt/InfiniteTalk/handler.py"]
