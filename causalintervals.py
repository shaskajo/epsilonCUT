# -*- coding: utf-8 -*-
"""
I am first just going to focus on detecting the presence or absence of an edge
before I try to implement the causal effect cofidence interval

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import lstsq
from scipy.stats import chi2
from scipy.io import savemat
from Graph import GraphGenerator

class Detector:
    def __init__(self, df= None, sig = None, tau = None, PAIRS = None):
        if df is None or sig is None or tau is None or PAIRS is None:
            return
        else:
            self.d = len(df.columns)
            self.n = len(df)
            self.sig = sig
            self.tau = tau
            self.Pairs = PAIRS
            self.order = self.getMLorder(df)
            self.findgraph(df)
            #self.orderings = self.plausibleorders(df)
            
            
    def Cov(self, df):
        X = np.array(df).T
        self.Kxx = (X) @ np.transpose(X)
        self.Kxx /= len(df)
    
    def likelihood(self, B):
        I = np.eye(self.d)
        L = ((self.n*self.d)/2)*np.log(2*np.pi * self.sig) \
            + (self.n/(2*self.sig))*np.trace((I-B).T@(I-B)@self.Kxx)
        return -L
    
    def getMLorder(self, df):
        self.Cov(df)
        varsdiag = np.diag(self.Kxx)
        remaining = np.arange(self.d)
        B = np.zeros((self.d,self.d))
        conditioned = []
        theta = np.argmin(varsdiag)
        conditioned.append(theta)
        remaining = np.delete(remaining, theta)
        Lik =  -((self.n*self.d)/2)*np.log(2*np.pi * self.sig) - (self.n/(2*self.sig))*self.Kxx[theta,theta]
        while len(remaining) > 0:
            Y = df[remaining]
            X = df[conditioned]
            A, varsdiag, _, _ = lstsq(X,Y)
            theta = np.argmin(varsdiag)
            B[remaining[theta], conditioned] = A[:,theta]
            conditioned.append(remaining[theta])
            remaining = np.delete(remaining, theta)
            Lik -= (1/(2*self.sig))*varsdiag[theta]
        self.L1 = Lik
        return conditioned
    
    def plausibleorders(self, df):
        orderings = list(itertools.permutations(np.arange(self.d)))
        possibleorderings = []
        for order in orderings:
            B = np.zeros((d,d))
            conditioned = [order[0]]
            Lik =  -((self.n*self.d)/2)*np.log(2*np.pi * self.sig) - (self.n/(2*self.sig))*self.Kxx[order[0],order[0]]
            for i in range(1,self.d):
                Y = df[order[i]]
                X = df[conditioned]
                A, RSS, _, _ = lstsq(X,Y)
                Lik -= (1/(2*self.sig))*RSS
                if 2*(self.L1-Lik) > self.tau:
                    break
                B[order[i], conditioned] = A
                conditioned.append(order[i])
                if i == self.d-1:
                    possibleorderings.append(order)
        return possibleorderings
    
    def TestEdge(self, i, j, df):
        orderings = list(itertools.permutations(np.arange(self.d)))
        for order in orderings:
            B = np.zeros((self.d,self.d))
            conditioned = [order[0]]
            Lik =  -((self.n*self.d)/2)*np.log(2*np.pi * self.sig) - (self.n/(2*self.sig))*self.Kxx[order[0],order[0]]
            for k in range(1,self.d):
                Y = df[order[k]]
                if order[k] == i and j in conditioned:
                    nuconditioned = conditioned[:]
                    nuconditioned.remove(j)
                    #updates = nuconditioned
                    X = df[nuconditioned]
                elif order[k] == j and i in conditioned:
                    nuconditioned = conditioned[:]
                    nuconditioned.remove(i)
                    X = df[nuconditioned]
                    #updates = nuconditioned
                else:
                    X = df[conditioned]
                    updates = conditioned
                A, RSS, _, _ = lstsq(X,Y)
                Lik -= (1/(2*self.sig))*RSS
                #We can immediately reject this ordering
                if 2*(self.L1-Lik) > self.tau:
                    break
                #B[order[k], updates] = A
                conditioned.append(order[k])
                if k == self.d-1:
                    #If we made it here, we are within the threshold to declare no edge
                    return 0
        #If we are here then we can declare an edge (reject no edge) with 1-eps confidence
        return 1
    
    def findgraph(self, df):#Find a locally optimaly graph
        A = np.zeros((self.d, self.d))
        for pair in self.Pairs:
            A[pair[0],pair[1]] = self.TestEdge(pair[0], pair[1], df)
            A[pair[1],pair[0]] = A[pair[0],pair[1]]
        self.A = A

def generate(sig, power, n ,d):
    X = np.random.normal(0,sig,(n,d))
    Triu = np.triu(np.ones((d,d)),1)
    A = np.random.binomial(1,.5,(d,d)) * Triu
    P = np.zeros((d,d))
    pi = np.random.permutation(d)
    for l in range(d): P[l][pi[l]] = 1
    A = P.T@A@P
    Asym = A+A.T
    positives = np.sum(Asym)/2
    negatives = (np.sum(1-Asym)-d)/2
    B = np.random.random((d,d))
    B[B<.5] = -1
    B[B>=.5] = 1
    A *= B
    A *= power
    X = (np.linalg.inv(np.eye(d)-A)@X.T).T
    return X, A, positives, negatives, Asym

if __name__=="__main__":
    sig = 1
    n, d, power = 10, 3, 2
    graphs = GraphGenerator(2,d)
    tau_list = np.linspace(2,6,10)
    falsepositives = np.zeros_like(tau_list)
    falsenegatives = np.zeros_like(tau_list)
    for i in range(len(tau_list)):
        print(i)
        tau = tau_list[i]
        total_positives, total_negatives = 0, 0
        for j in range(1500):
            if j%100 == 0:
                print(j)
            X, A, positives, negatives, Asym = generate(sig, power, n, d)
            total_positives += positives
            total_negatives += negatives
            df = pd.DataFrame(X)
            E = Detector(df, sig, tau, graphs.PAIRS).A
            Errors = E - Asym
            falsepositives[i] += np.sum(Errors[Errors>0])/2
            falsenegatives[i] -= np.sum(Errors[Errors<0])/2
        falsepositives[i] /= total_negatives
        falsenegatives[i] /= total_positives
    plt.plot(falsepositives, falsenegatives, label='Algorithm 1')
    plt.xlabel('false-positive rate')
    plt.ylabel('false-negative rate')
    plt.legend()
    plt.show()
        