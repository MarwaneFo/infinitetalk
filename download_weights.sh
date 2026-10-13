#!/usr/bin/env bash
# Telecharge les poids d'InfiniteTalk sur un volume reseau RunPod.
#
# A lancer UNE SEULE FOIS, depuis un pod GPU ou CPU ayant le volume monte
# -- pas depuis le worker serverless, qui n'a ni le temps ni le reseau
# pour cela a chaque demarrage.
#
#   bash download_weights.sh /runpod-volume/infinitetalk/weights
#
# Volumetrie du jeu minimal (une seule personne, sans quantification) :
#
#   Wan2.1-I2V-14B-480P        82,3 Go   modele de base
#     diffusion (7 fragments)  65,6 Go
#     encodeur T5 umt5-xxl     11,4 Go
#     CLIP                      4,8 Go
#     VAE                       0,5 Go
#   InfiniteTalk single/         9,9 Go   poids de conditionnement audio
#   chinese-wav2vec2-base        0,4 Go   encodeur audio
#   ------------------------------------
#   total                      ~92,6 Go
#
# Prevoir un volume de 120 Go : les telechargements passent par un cache
# temporaire avant d'atterrir a leur place.
#
# Options par variables d'environnement :
#   VARIANTE=single|multi   jeu de poids (defaut single, une personne)
#   QUANT=fp8|int8          ajoute un modele quantifie (+19,5 Go).
#                           Utile sous 80 Go de VRAM.
#   LORA=1                  ajoute la LoRA FusionX (+1,2 Go), qui permet
#                           de descendre a 8 pas au lieu de 40.
set -euo pipefail

CIBLE="${1:-/runpod-volume/infinitetalk/weights}"
VARIANTE="${VARIANTE:-single}"
QUANT="${QUANT:-}"
LORA="${LORA:-0}"

# Le point d'entree s'appelle "hf" depuis huggingface_hub 0.34 ;
# "huggingface-cli" fonctionne encore mais affiche un avertissement.
if command -v hf >/dev/null 2>&1; then
  HF="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli download"
else
  echo "Installation de huggingface_hub..."
  pip install -q "huggingface_hub[cli]" hf_transfer
  HF="hf download"
fi

# hf_transfer sature la bande passante : sur 93 Go, la difference se
# compte en heures. S'il n'est pas installe, le reglage est ignore.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "${CIBLE}"
echo "Cible : ${CIBLE}"
echo "Variante : ${VARIANTE}   Quantification : ${QUANT:-aucune}   LoRA : ${LORA}"
echo

# ---------- 1. Modele de base Wan2.1 (82,3 Go) ----------
# On exclut assets/ et examples/ : des illustrations du depot, inutiles
# a l'inference.
echo "[1/3] Wan2.1-I2V-14B-480P (82,3 Go)..."
${HF} Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir "${CIBLE}/Wan2.1-I2V-14B-480P" \
  --exclude "assets/*" "examples/*" "*.png" "*.jpg" "*.JPG"

# ---------- 2. Encodeur audio (0,4 Go) ----------
# On exclut le point de controle fairseq (1,1 Go) : le code charge le
# modele transformers, pas celui-la. Le safetensors vient d'une pull
# request encore ouverte ; c'est ce que demande le README amont.
echo "[2/3] chinese-wav2vec2-base (0,4 Go)..."
${HF} TencentGameMate/chinese-wav2vec2-base \
  --local-dir "${CIBLE}/chinese-wav2vec2-base" \
  --exclude "*fairseq*"
${HF} TencentGameMate/chinese-wav2vec2-base model.safetensors \
  --revision refs/pr/1 \
  --local-dir "${CIBLE}/chinese-wav2vec2-base"

# ---------- 3. Poids InfiniteTalk ----------
echo "[3/3] InfiniteTalk ${VARIANTE} (9,9 Go)..."
${HF} MeiGen-AI/InfiniteTalk \
  --include "${VARIANTE}/infinitetalk.safetensors" \
  --local-dir "${CIBLE}/InfiniteTalk"

if [ -n "${QUANT}" ]; then
  echo "       modele quantifie ${QUANT} (19,5 Go)..."
  ${HF} MeiGen-AI/InfiniteTalk \
    --include "quant_models/infinitetalk_${VARIANTE}_${QUANT}.safetensors" \
               "quant_models/infinitetalk_${VARIANTE}_${QUANT}.json" \
               "quant_models/quant.json" \
    --local-dir "${CIBLE}/InfiniteTalk"
fi

if [ "${LORA}" = "1" ]; then
  echo "       LoRA FusionX (1,2 Go) -- permet 8 pas au lieu de 40..."
  ${HF} vrgamedevgirl84/Wan14BT2VFusioniX \
    --include "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" \
    --local-dir "${CIBLE}"
fi

# ---------- Verification ----------
echo
echo "--- Verification ---"
manque=0
for f in "Wan2.1-I2V-14B-480P/config.json" \
         "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
         "Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
         "Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors" \
         "chinese-wav2vec2-base/config.json" \
         "InfiniteTalk/${VARIANTE}/infinitetalk.safetensors" ; do
  if [ -f "${CIBLE}/${f}" ]; then
    printf '  ok      %s\n' "${f}"
  else
    printf '  MANQUE  %s\n' "${f}"
    manque=1
  fi
done

echo
du -sh "${CIBLE}" 2>/dev/null || true
if [ "${manque}" = "1" ]; then
  echo
  echo "Des fichiers manquent : relancer le script, il reprend ou il s'est arrete."
  exit 1
fi
echo
echo "Poids complets. Renseigner dans l'endpoint RunPod :"
echo "  IT_WEIGHTS=${CIBLE}"
