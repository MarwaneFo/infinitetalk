#!/usr/bin/env bash
# Telecharge les poids d'InfiniteTalk sur un volume reseau RunPod.
#
# A lancer UNE SEULE FOIS, depuis un pod ayant le volume monte -- pas
# depuis le worker serverless, qui n'a ni le temps ni le reseau pour le
# refaire a chaque demarrage.
#
#   MODE=compact bash download_weights.sh /runpod-volume/infinitetalk/weights
#
# ----------------------------------------------------------------------
#  DEUX JEUX POSSIBLES, ET LE CHOIX N'EST PAS QUE DE PLACE
# ----------------------------------------------------------------------
#
#  MODE=compact   ~32 Go   modele quantifie fp8
#
#    Le pipeline ne lit alors QUE le fichier quantifie : les 7 fragments
#    bf16 (65,6 Go) et infinitetalk.safetensors (9,9 Go) ne sont jamais
#    ouverts, seul config.json l'est (multitalk.py, branche quant).
#    Le T5 prend lui aussi sa version fp8, 6,73 Go au lieu de 11,36.
#
#      InfiniteTalk/quant_models/infinitetalk_<v>_fp8.safetensors  19,50
#      InfiniteTalk/quant_models/t5_fp8.safetensors                 6,73
#      Wan2.1/models_clip_...pth                                    4,77
#      Wan2.1/Wan2.1_VAE.pth                                        0,51
#      Wan2.1/config.json + jetonniseurs                            0,05
#      chinese-wav2vec2-base                                        0,40
#      -----------------------------------------------------------------
#      total                                                      ~31,96
#
#    MAIS : la LoRA FusionX est refusee quand la quantification est
#    active (multitalk.py : `if lora_dir is not None and quant is None`).
#    On reste donc a 40 pas d'echantillonnage, sans le levier qui les
#    ramene a 8. Compact sur le disque, lent au rendu.
#
#  MODE=complet   ~93 Go   modele bf16
#
#    Les poids d'origine. Plus lourds, mais la LoRA FusionX devient
#    possible (LORA=1) et avec elle 8 pas au lieu de 40.
#
#      Wan2.1-I2V-14B-480P                                         82,30
#      InfiniteTalk/<v>/infinitetalk.safetensors                    9,95
#      chinese-wav2vec2-base                                        0,40
#      -----------------------------------------------------------------
#      total                                                      ~92,65
#
#  Le stockage reseau se compte en quelques dollars par mois ; le temps
#  de GPU, en dizaines. Si le but est de produire du volume, `complet`
#  avec la LoRA coute moins cher a l'arrivee. Verifier le tarif du
#  volume dans la console avant de trancher.
#
# ----------------------------------------------------------------------
#  Variables
#    MODE=compact|complet    defaut compact
#    VARIANTE=single|multi   single = une personne (defaut)
#    LORA=1                  LoRA FusionX (+1,2 Go). MODE=complet requis.
# ----------------------------------------------------------------------
set -euo pipefail

CIBLE="${1:-/runpod-volume/infinitetalk/weights}"
MODE="${MODE:-compact}"
VARIANTE="${VARIANTE:-single}"
LORA="${LORA:-0}"

if [ "${MODE}" != "compact" ] && [ "${MODE}" != "complet" ]; then
  echo "MODE doit valoir compact ou complet (recu : ${MODE})" >&2
  exit 1
fi
if [ "${LORA}" = "1" ] && [ "${MODE}" = "compact" ]; then
  echo "LORA=1 est incompatible avec MODE=compact : le pipeline refuse" >&2
  echo "une LoRA quand la quantification est active. Choisir MODE=complet." >&2
  exit 1
fi

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

# hf_transfer sature la bande passante : sur des dizaines de Go, la
# difference se compte en heures. Ignore s'il n'est pas installe.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "${CIBLE}"
echo "Cible    : ${CIBLE}"
echo "Mode     : ${MODE}  ($([ "${MODE}" = compact ] && echo '~32 Go, fp8, sans LoRA' || echo '~93 Go, bf16, LoRA possible'))"
echo "Variante : ${VARIANTE}"
echo

WAN="${CIBLE}/Wan2.1-I2V-14B-480P"

if [ "${MODE}" = "compact" ]; then

  # ---------- 1. De Wan2.1, seulement ce qui n'a pas d'equivalent fp8 ----------
  # VAE et CLIP ne sont pas quantifies ; config.json et les jetonniseurs
  # sont lus tels quels. Les 65,6 Go de fragments de diffusion et les
  # 11,36 Go de T5 bf16 ne sont PAS telecharges.
  echo "[1/3] Wan2.1 : VAE, CLIP et configuration (5,3 Go)..."
  ${HF} Wan-AI/Wan2.1-I2V-14B-480P \
    --include "config.json" \
              "Wan2.1_VAE.pth" \
              "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
              "google/umt5-xxl/*" \
              "xlm-roberta-large/*" \
    --local-dir "${WAN}"

  # ---------- 2. Les poids quantifies (26,2 Go) ----------
  # t5_fp8 est obligatoire : le chargeur T5 va le chercher dans le meme
  # dossier que le modele quantifie des que quant est actif. C'est aussi
  # pourquoi int8 n'est pas propose -- le depot ne publie pas de t5_int8.
  echo "[2/3] InfiniteTalk ${VARIANTE} fp8 + T5 fp8 (26,2 Go)..."
  ${HF} MeiGen-AI/InfiniteTalk \
    --include "quant_models/infinitetalk_${VARIANTE}_fp8.safetensors" \
              "quant_models/infinitetalk_${VARIANTE}_fp8.json" \
              "quant_models/t5_fp8.safetensors" \
              "quant_models/t5_map_fp8.json" \
              "quant_models/quant.json" \
    --local-dir "${CIBLE}/InfiniteTalk"

else

  # ---------- 1. Modele de base complet (82,3 Go) ----------
  # assets/ et examples/ sont des illustrations du depot.
  echo "[1/3] Wan2.1-I2V-14B-480P complet (82,3 Go)..."
  ${HF} Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir "${WAN}" \
    --exclude "assets/*" "examples/*" "*.png" "*.jpg" "*.JPG"

  # ---------- 2. Poids de conditionnement audio (9,9 Go) ----------
  echo "[2/3] InfiniteTalk ${VARIANTE} (9,9 Go)..."
  ${HF} MeiGen-AI/InfiniteTalk \
    --include "${VARIANTE}/infinitetalk.safetensors" \
    --local-dir "${CIBLE}/InfiniteTalk"

  if [ "${LORA}" = "1" ]; then
    echo "      LoRA FusionX (1,2 Go) -- 8 pas au lieu de 40..."
    ${HF} vrgamedevgirl84/Wan14BT2VFusioniX \
      --include "FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors" \
      --local-dir "${CIBLE}"
  fi
fi

# ---------- 3. Encodeur audio (0,4 Go) ----------
# Le point de controle fairseq (1,1 Go) est exclu : le code charge le
# modele transformers. Le safetensors vient d'une pull request encore
# ouverte, c'est ce que demande le README amont.
echo "[3/3] chinese-wav2vec2-base (0,4 Go)..."
${HF} TencentGameMate/chinese-wav2vec2-base \
  --local-dir "${CIBLE}/chinese-wav2vec2-base" \
  --exclude "*fairseq*"
${HF} TencentGameMate/chinese-wav2vec2-base model.safetensors \
  --revision refs/pr/1 \
  --local-dir "${CIBLE}/chinese-wav2vec2-base"

# ---------- Verification ----------
echo
echo "--- Verification ---"
attendus="Wan2.1-I2V-14B-480P/config.json
Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
chinese-wav2vec2-base/config.json"

if [ "${MODE}" = "compact" ]; then
  attendus="${attendus}
InfiniteTalk/quant_models/infinitetalk_${VARIANTE}_fp8.safetensors
InfiniteTalk/quant_models/t5_fp8.safetensors
InfiniteTalk/quant_models/t5_map_fp8.json"
else
  attendus="${attendus}
Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth
Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors
InfiniteTalk/${VARIANTE}/infinitetalk.safetensors"
fi

manque=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ -f "${CIBLE}/${f}" ]; then
    printf '  ok      %s\n' "${f}"
  else
    printf '  MANQUE  %s\n' "${f}"
    manque=1
  fi
done <<< "${attendus}"

echo
du -sh "${CIBLE}" 2>/dev/null || true
if [ "${manque}" = "1" ]; then
  echo
  echo "Des fichiers manquent. Relancer : le script reprend ou il s'est arrete."
  exit 1
fi

echo
echo "Poids complets. A renseigner dans l'endpoint RunPod :"
echo "  IT_WEIGHTS=${CIBLE}"
if [ "${MODE}" = "compact" ]; then
  echo "  IT_QUANT=fp8          <-- indispensable, sinon le worker cherchera"
  echo "                            les fragments bf16 absents"
fi
if [ "${LORA}" = "1" ]; then
  echo "  IT_LORA=${CIBLE}/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
  echo "  puis dans les requetes : sample_steps=8, text_guide_scale=1,"
  echo "  audio_guide_scale=2"
fi
