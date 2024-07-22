# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:21:53 2024

@author: jshas
"""
import numpy as np
from scipy.linalg import lstsq
from scipy.stats import chi2
from scipy.io import savemat
import pandas as pd
import matplotlib.pyplot as plt
from RenyiClass import Renyi
from Graph import GraphGenerator


class Estimator:
    def __init__(self, df= None, Pairs=None, Graphs = None, TAU=None, KEK = False):
        if df is None or Pairs is None or Graphs is None or TAU is None:
            return
        else:
            self.d = len(df.columns)
            self.n = len(df)
            self.Graphs = Graphs
            self.Pairs = Pairs
            self.TAU = TAU
            self.KEK = KEK
            self.findgraph(df)
            
    def Cov(self, df):
        X = np.array(df).T
        self.Kxx = (X) @ np.transpose(X)
        #self.Kxx /= len(df)
        
    def findedge(self, pair, df):
        for i in range(len(self.Graphs)):#have to make sure the indexing is right with size
            condSet = self.Graphs[i]
            #Be careful with tau now that conditioning sets are different between the two nodes
            if self.KEK and pair[0]==0:
                condSet[0] = np.zeros(self.d)
            p1, p2 = np.sum(condSet[0]), np.sum(condSet[1])
            if condSet[0,pair[0]]==1 or condSet[0,pair[1]]==1 or condSet[1,pair[1]] == 1 or condSet[1,pair[0]]==1:
                continue
            else:
                tau = self.TAU[int(p1),int(p2)]
                Y1, Y2 = df[pair[0]], df[pair[1]]
                #check the case if either p1 or p2 are 0
                if p1 == 0:
                    Rs1 = self.Kxx[pair[0],pair[0]]
                else:
                    S1 = np.nonzero(condSet[0])
                    X1 = df[S1[0]]
                    A1, Rs1, _, _ = lstsq(X1, Y1)
                if p2 == 0:
                    Rs2 = self.Kxx[pair[1],pair[1]]
                else:
                    S2 = np.nonzero(condSet[1])
                    X2 = df[S2[0]]
                    A2, Rs2, _, _ = lstsq(X2, Y2)
                #REMEMBER TO CHANGE THIS FOR DIFFERENT SIGMA
                if np.abs(Rs1-Rs2) <= tau:
                    return 0
        return 1
                    #find the OLS estimates
    def findtau(self, p1, p2):
        l=0
        u=2
        tau = np.linspace((l*self.n-p1-p2), (u*self.n-p1-p2),10000)
        eps = ((1-chi2.cdf(tau/(2*self.sig) + self.n-p1, self.n-p1))+ chi2.cdf(-tau/(2*self.sig) + self.n-p1, self.n-p1)) +\
            ((1-chi2.cdf(tau/(2*self.sig) + self.n-p2, self.n-p2))+ chi2.cdf(-tau/(2*self.sig) + self.n-p2, self.n-p2))
        while eps[-1] > self.tol:
            l=u
            u*=2
            tau = np.linspace((l*self.n-p1-p2), (u*self.n-p1-p2),10000)
            eps = ((1-chi2.cdf(tau/(2*self.sig) + self.n-p1, self.n-p1))+ chi2.cdf(-tau/(2*self.sig) + self.n-p1, self.n-p1)) +\
                ((1-chi2.cdf(tau/(2*self.sig) + self.n-p2, self.n-p2))+ chi2.cdf(-tau/(2*self.sig) + self.n-p2, self.n-p2))
        self.tau = tau[np.argmin(np.abs(eps-self.tol))]
        
    
    def findgraph(self, df):#Find a locally optimaly graph
        self.Cov(df)#compute the correlations
        A = np.zeros((self.d, self.d))
        for pair in self.Pairs:
            A[pair[0],pair[1]] = self.findedge(pair, df)
            A[pair[1],pair[0]] = A[pair[0],pair[1]]
        self.A = A
        
def createTAU(n, d, sig, tol):
    TAU = np.zeros((d,d))
    for p1 in range(d):
        for p2 in range(p1,d):
            l=0
            u=2
            tau = np.linspace((l*n-p1-p2), (u*n-p1-p2),10000)
            eps = ((1-chi2.cdf(tau/(2*sig) + n-p1, n-p1))+ chi2.cdf(-tau/(2*sig) + n-p1, n-p1)) +\
                ((1-chi2.cdf(tau/(2*sig) + n-p2, n-p2))+ chi2.cdf(-tau/(2*sig) + n-p2, n-p2))
            while eps[-1] > tol:
                l=u
                u*=2
                tau = np.linspace((l*n-p1-p2), (u*n-p1-p2),10000)
                eps = ((1-chi2.cdf((tau-(p2-p1)*sig)/(2*sig) + n-p1, n-p1))+ chi2.cdf(-(tau-(p2-p1)*sig)/(2*sig) + n-p1, n-p1)) +\
                    ((1-chi2.cdf((tau-(p2-p1)*sig)/(2*sig) + n-p2, n-p2))+ chi2.cdf(-(tau-(p2-p1)*sig)/(2*sig) + n-p2, n-p2))
            TAU[p1, p2] = tau[np.argmin(np.abs(eps-tol))]
            TAU[p2, p1] = TAU[p1, p2]
    return TAU
            
"""
X = np.random.normal(0,1,(100,3))
X[:,0] += 3*X[:,1]
#X[:,2] += -5*X[:,1]
df = pd.DataFrame(X)
R = Estimator(df)
"""
if __name__=="__main__":
    sig=[1]
    #tol = [.1, .2, .3, .4, .5, .6,.7,.8,.9,1]
    tol = np.linspace(6,10,10)
    lambda1 = np.linspace(1,.5,6)
    #tol = [.01, .05, .1, .15, .2]
    falsepositives = np.zeros((len(tol),len(sig)))
    falsenegatives = np.zeros((len(tol),len(sig)))
    N = 1500
    n = 10
    d=3
    power = 2
    graphs = GraphGenerator(2,d)
    for i in range(len(sig)):
        for j in range(len(tol)):
            positives = 0
            negatives = 0
            #TAU = createTAU(n, d, sig[i]**2, tol[j])
            TAU  = tol[j]*np.ones((d,d))
            for k in range(N):
                if k%100==0:
                    print(k)
                X = np.random.normal(0,sig[i],(n,d))
                Triu = np.triu(np.ones((d,d)),1)
                A = np.random.binomial(1,.5,(d,d)) * Triu
                P = np.zeros((d,d))
                pi = np.random.permutation(d)
                for l in range(d): P[l][pi[l]] = 1
                A = P.T@A@P
                Asym = A+A.T
                positives += np.sum(Asym)/2
                negatives += (np.sum(1-Asym)-d)/2
                B = np.random.random((d,d))
                B[B<.5] = -1
                B[B>=.5] = 1
                #REMBER TO UNCCOMMENT THIS
                A *= B
                A *= power
                X = (np.linalg.inv(np.eye(d)-A)@X.T).T
                df = pd.DataFrame(X)
                E = Estimator(df, graphs.PAIRS, graphs.GRAPHS, TAU).A
                Errors = E - Asym
                falsepositives[j][i] += np.sum(Errors[Errors>0])/2
                falsenegatives[j][i] -= np.sum(Errors[Errors<0])/2
        #R = Renyi(n,d,sig[i],power)
        #LASSOerrors = R.LASSO(9000, np.linspace(10,0,10))
        #LASSOerrors = R.LASSO(10, np.linspace(1,.5,6))
            falsepositives[j][i] /= negatives
            falsenegatives[j][i] /= positives
    
    plt.plot(falsepositives[:,0], falsenegatives[:,0], label='Algorithm 1')
    #plt.plot(LASSOerrors[0,:4], LASSOerrors[1,:4], '--^', label="LASSO")
    plt.xlabel('false-positive rate')
    plt.ylabel('false-negative rate')
    plt.legend()
    
    #plt.plot(falsepositives[:,1], falsenegatives[:,1])
    #plt.plot(falsepositives[:,2], falsenegatives[:,2])
    plt.show()



    
        
        