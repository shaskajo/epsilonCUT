# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:30:06 2024

@author: jshas
"""

import numpy as np
from scipy.linalg import expm

class GraphGenerator():
    def __init__(self,m,n):
        self.n = n
        self.m = m
        self.max = n*m
        self.EDGES = [[] for i in range(m*n)]
        self.PAIRS = []
        self.GRAPHS = []
        self.getall()
        
    def getGRAPHS(self, S, k):
        if len(S)==k:
            self.EDGES[k-1].append(S[:])
            if k==2 and S[-1] < self.n:
                self.PAIRS.append(S[:])
            A =np.zeros((self.m,self.n))
            for i in range(k):
                A[S[i]//self.n,S[i]%self.n]=1.
            self.GRAPHS.append(A[:])
        else:
            if S:
                z = S[-1]
            else:
                z=-1
            for x in range(z+1,self.max):
                S.append(x)
                self.getGRAPHS(S[:],k)
                S.pop()
        return

    def getall(self):
        for k in range(self.max):
            #print(k)
            P = []
            self.getGRAPHS(P, k)