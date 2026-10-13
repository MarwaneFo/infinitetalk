"""Remplacant minimal de xfuser pour l'inference mono-GPU.

Le vrai paquet est incompatible avec torch 2.4.1, que la recette amont
impose. Voir xfuser/core/distributed/__init__.py pour le detail.

Seul xfuser.core.distributed est fourni, avec les trois noms que le code
d'InfiniteTalk importe. Tout autre sous-module de xfuser est absent, et
le dira franchement.
"""
__version__ = "0.0.0+shim-mono-gpu"
