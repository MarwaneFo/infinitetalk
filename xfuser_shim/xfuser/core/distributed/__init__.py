"""Remplacant minimal de xfuser.core.distributed, pour l'inference mono-GPU.

POURQUOI CE FICHIER EXISTE
--------------------------
Le depot amont se contredit. Son README impose torch 2.4.1, mais son
requirements.txt demande xfuser>=0.4.1 -- et TOUTES les versions de
xfuser depuis 0.4.0 commencent par :

    from torch.distributed.tensor.experimental._attention import _templated_ring_attention

Ce chemin n'existe pas avant torch 2.5 : il s'appelait
torch.distributed._tensor.experimental. L'import de xfuser casse donc
net sur la version de torch que le projet lui-meme exige :

    ModuleNotFoundError: No module named 'torch.distributed.tensor.experimental'

Aucune version de xfuser ne satisfait les deux contraintes a la fois.

CE QUE XFUSER SERT REELLEMENT ICI
----------------------------------
Deux fichiers l'importent au niveau module, et tous deux ne demandent
que trois noms :

    wan/modules/attention.py        lignes 6-10
    wan/utils/multitalk_utils.py    lignes 7-11

Et ces trois noms ne sont appeles que dans des branches `if enable_sp:`,
c'est-a-dire sous parallelisme de sequence -- du multi-GPU. Le worker
tourne en mono-GPU, avec use_usp=False : ces branches ne sont jamais
prises. xfuser n'est donc present que pour satisfaire une ligne
d'import, au prix d'un framework entier et de ses dependances.

CE QUE FAIT CE REMPLACANT
--------------------------
Il fournit les trois noms, et rien d'autre.

Pour les deux qui ont une reponse juste en mono-GPU, il la donne : un
seul rang, une seule taille de groupe. Ce ne sont pas des valeurs
bouchon, ce sont les bonnes.

Pour get_sp_group(), il n'existe pas de reponse en mono-GPU : la
fonction leve donc une exception explicite. Si une branche non protegee
l'appelait un jour, l'echec serait immediat et nomme, jamais un resultat
silencieusement faux.

POUR REVENIR AU VRAI XFUSER
----------------------------
Il faudrait passer torch a 2.5 ou plus, donc changer aussi la roue
flash-attn, et revalider l'ensemble. Tant qu'on reste en mono-GPU, ce
fichier rend cette montee inutile.
"""

__all__ = [
    "get_sequence_parallel_rank",
    "get_sequence_parallel_world_size",
    "get_sp_group",
]


def get_sequence_parallel_world_size():
    """Taille du groupe de parallelisme de sequence.

    En mono-GPU elle vaut 1 : ce n'est pas une approximation, c'est la
    valeur exacte.
    """
    return 1


def get_sequence_parallel_rank():
    """Rang dans le groupe de parallelisme de sequence.

    En mono-GPU il vaut 0, pour la meme raison.
    """
    return 0


def get_sp_group():
    """Groupe de communication inter-GPU -- sans equivalent en mono-GPU.

    Appelee uniquement dans les branches `if enable_sp:` du code amont,
    donc jamais atteinte avec use_usp=False. Si elle l'etait, mieux vaut
    une erreur nommee qu'un resultat faux.
    """
    raise RuntimeError(
        "get_sp_group() appelee alors que le worker tourne en mono-GPU. "
        "Ce remplacant de xfuser ne fournit pas de groupe de "
        "communication : il n'y en a pas a fournir. Pour du multi-GPU, "
        "installer le vrai xfuser, ce qui impose torch >= 2.5 et une "
        "roue flash-attn correspondante. Voir le commentaire en tete de "
        "ce fichier.")
