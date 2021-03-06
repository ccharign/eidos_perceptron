#! /usr/bin/python3
# -*- coding: utf-8 -*-

un = [ [0, 1, 0],
       [1, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0]]

zéro=[ [0, 1, 0],
       [1, 0, 1],
       [1, 0, 1],
       [1, 0, 1],
       [0, 1, 0]]
       
deux =[[1, 1, 0],
       [0, 0, 1],
       [0, 1, 0],
       [1, 0, 0],
       [1, 1, 1]]

trois=[[1, 1, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 0, 1],
       [1, 1, 0]]

quatre=[[0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0]]

cinq=[ [1, 1, 1],
       [1, 0, 0],
       [1, 1, 0],
       [0, 0, 1],
       [1, 1, 0]]

six= [ [0, 1, 0],
       [1, 0, 0],
       [1, 1, 1],
       [1, 0, 1],
       [0, 1, 0]]

sept  =[[1, 1, 1],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0]]

huit=[ [1, 1, 1],
       [1, 0, 1],
       [0, 1, 0],
       [1, 0, 1],
       [1, 1, 1]]

neuf=[ [0, 1, 0],
       [1, 0, 1],
       [0, 1, 1],
       [0, 0, 1],
       [1, 1, 0]]

TOUT=[(zéro, 0), (un,1), (deux,2), (trois,3), (quatre,4), (cinq,5), (six,6), (sept,7), (huit,8), (neuf,9)]
# Listes des couples (image, chiffre qu'elle représente)


CATÉGORIES=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


import numpy.random as rd

def nvLigne(n):
    res=[]
    for i in range(n):
        res.append(2*rd.rand()-1)
    return res

def nvMat(n,p):
    res=[]
    for i in range(n):
        res.append(nvLigne(p))
    return res

def init_P(n,p,q):
    """ Renvoie une matrice de format (n,p,q) remplie de flottant aléatoires dans [0,1[."""
    res=[]
    for i in range(n):
        res.append( nvMat(p,q))
    return res

P=init_P(10, 5, 3)


