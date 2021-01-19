# -*- coding:utf-8 -*-
from bib_neurones import *

P_test = init_P(10,5,3)

def A(im, k, P):
    """ Quantité de signal reça par le neurone k lors de la lecture de im."""
    res=0
    n = len(im)
    p = len(im[0])
    for i in range(n):
        for j in range(p):
            res += P[k][i][j] * im[i][j]
    return res

def sorties_activées(im, P):
    """ Tableaux des neurones de sortie activés lors de la lecture de im."""
    res = []
    for k in range(len(P)):
        if A(im, k, P) >1:
            res.append(k)
    return res


def err(im, P, k0, k):
    """ Erreur commise par le neurone k en lisant im qui est supposée activer le neurone k0."""
    if k==k0:
        valeur_voulue = 2
    else:
        valeur_voulue = -2
    return valeur_voulue - A(im,k,P)


def lecture_image(im, P, k0, η):
    """ Procédure qui corrige les coeff de P lors de la lecture de im, qui est supposée activer le neurone k0."""
    for k in range(len(P)):
        for i in range(len(im)):
            for j in range(len(im[0])):
                P[k][i][j] += η*im[i][j]*err(im,P,k0,k)

                
def lecture_banque(B, P, η):
    """ Corrige les poids en lisant une banque d'images. La banque doit être un tableau de couples (image, neurone à activer)."""
    for (im, k0) in B:
        lecture_image(im, P, k0, η)


def tout_juste(B,P):
    """ Indique si le bon neurone est activé pour chaque image de la banque B."""
    res=True
    for (im,k0) in B:
        if sorties_activées(im, P) != [k0]:
            res=False
    return res

def entrainement(B,Ns, η):
    n=len(B[0][0])
    p=len(B[0][0][0])
    P=init_P(Ns, n, p)
    

    while not tout_juste(B,P):
        lecture_banque(B,P,η)
    return P
