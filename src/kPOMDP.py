import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import norm
import warnings
from gaussianMixtures import GM, Gaussian
from random import shuffle
import random

'''
***********************************************************
File: kPOMDP.py

One dimensional trolley testing problem for continuous
observation and state space POMDPs


Trolley can move on -inf,inf
goal located at 7 meters
initial location random on -20,20
initial belief highly uncertain
3 actions, 2 noisy left right, 1 stay
observations continuous


***********************************************************
'''


__author__ = "Luke Burks"
__copyright__ = "Copyright 2019"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"


class kPOMDPSolver:

    def __init__(self):
        self.A = 1
        self.B = 1
        # self.R = 1
        self.Q = 1
        self.C = 1

        self.acts = [-1, 1, 0]
        self.R = [1, 1, 0]
        self.wind = -0.2

        # self.rew = [7, 0.5]
        self.rew = GM()
        self.rew.addG(Gaussian(7, 0.5, 1))
        # self.rew.display()

        self.discount = 0.9

    def filter(self, mu, sig, a, o):
        muBar = self.A*mu + self.B*self.acts[a]
        sigBar = self.A*sig*self.A + self.R[a]

        K = sigBar*self.C*(self.C*sigBar*self.C + self.Q)**-1
        newMu = muBar + K*(o-self.C*muBar)
        newSig = (1-K*self.C)*sigBar

        return newMu, newSig

    def update(self, s, a):
        s = np.random.normal(s+self.acts[a], self.R[a])
        return s

    def measure(self, s):
        return np.random.normal(self.C*s, self.Q)

    def gatherBeliefs(self):
        allBels = []
        sigs = [.1, 1, 5, 10]
        #points = [-20,-15,-10,-5,0,5,6,6.5,7,7.5,8,9,10,15,20];
        for i in range(-20, 21):
            for s in sigs:
                allBels.append([i, s])
        return allBels

    def precomputeAls(self):
        G = self.Gamma
        rew = self.rew
        numActs = 3

        als1 = [[0 for j in range(0, numActs)]
                for k in range(0, len(G))]

        for j in range(0, len(G)):
            for a in range(0, numActs):
                als1[j][a] = GM()

                for k in range(0, G[j].size):
                    weight = G[j][k].weight/self.A
                    D = ((self.C**-1*self.Q*self.C**-1)
                         ** -1 + G[j][k].var**-1)**-1
                    E = self.C**-1*self.Q*self.C**-1

                    mean = self.A**-1 * \
                        (D*(E**-1*self.C*G[j][k].mean + G[j]
                            [k].var**-1*G[j][k].mean)-self.B*self.acts[a])

                    var = self.A**-1*(D*(E**-1*(E*(D**-1*(D+self.R[a])*D**-1)*E +
                                                self.Q + self.C*G[j][k].var*self.C)*E**-1)*D)*self.A**-1

                    als1[j][a].addG(Gaussian(mean, var, weight))
        return als1

    def backup(self, b, als1):
        G = self.Gamma
        R = self.rew
        numActs = 3

        bestVal = -10000000000
        bestAct = 0
        bestGM = []

        for a in range(0, numActs):
            suma = GM()
            suma.addGM(als1[np.argmax([self.continuousDot(als1[j][a], b)
                                       for j in range(0, len(als1))])][a])
            suma.scalarMultiply(self.discount)
            suma.addGM(R)

            tmp = self.continuousDot(suma, b)
            # print(a,tmp);
            if(tmp > bestVal):
                bestAct = a
                bestGM = deepcopy(suma)
                bestVal = tmp

        bestGM.action = bestAct

        return bestGM

    def continuousDot(self, a, b):
        # return norm.pdf(a[0], b[0], a[1]+b[1])

        suma = 0

        for k in range(0, a.size):
            # suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean, a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var));
            suma += a[k].weight*norm.pdf(b[0], a[k].mean, b[1]+a[k].var)
        return suma

    def findB(self, b):
        for beta in self.B:
            if(beta.fullComp(b)):
                return self.B.index(beta)

    def solve(self, numIter, allBels):
        # Mean,sigma,action
        self.Gamma = [deepcopy(self.rew)]
        self.Gamma[0].scalarMultiply(0.001)

        # self.Gamma[0].plot(low=-20, high=20)
        for count in range(numIter):
            print("Iteration: " + str(count+1))

            GammaNew = []
            bestAlphas = [GM()]*len(allBels)
            Value = [0]*len(allBels)
            for b in allBels:
                bestAlphas[allBels.index(b)] = self.Gamma[np.argmax(
                    [self.continuousDot(self.Gamma[j], b) for j in range(0, len(self.Gamma))])]
                Value[allBels.index(b)] = self.continuousDot(
                    bestAlphas[allBels.index(b)], b)

            preAls = self.precomputeAls()

            BTilde = deepcopy(allBels)

            while(len(BTilde) > 0):
                # print(allBels.index(b))
                b = random.choice(BTilde)
                BTilde.remove(b)

                al = self.backup(b, preAls)
                bIndex = 0
                for h in allBels:
                    if(b == h):
                        bIndex = allBels.index(h)
                if(self.continuousDot(al, b) < Value[bIndex]):
                    al = bestAlphas[bIndex]
                else:
                    bestAlphas[bIndex] = al
                    Value[bIndex] = self.continuousDot(al, b)

                GammaNew += [al]

                for bprime in BTilde:
                    bIndex = 0
                    for h in allBels:
                        if(bprime == h):
                            bIndex = allBels.index(h)

                    if(self.continuousDot(al, bprime) >= Value[bIndex]):
                        BTilde.remove(bprime)

            for i in range(0, len(GammaNew)):
                if(GammaNew[i].size > 5):
                    GammaNew[i].condense(5)
                    self.cleaning(GammaNew[i])

            self.Gamma = deepcopy(GammaNew)

            print(len(self.Gamma))
            allHist = [0, 0, 0]
            for p in self.Gamma:
                allHist[p.action] += 1
            print(allHist)
            # print(len(self.Gamma[0].Gs))
        # f = open('KalmanPolicy.npy', 'w+')
        np.save('KalmanPolicy.npy', self.Gamma)

    def cleaning(self, a):
        for g in a.Gs:
            if(isinstance(g.var, list)):
                g.var = g.var[0][0]

    def getAction(self, b, Gamma):
        bestVal = -100000
        bestInd = 0

        for j in range(0, len(Gamma)):
            tmp = self.continuousDot(Gamma[j], b)
            if(tmp > bestVal):
                bestVal = tmp
                bestInd = j
        return int(Gamma[bestInd].action)


def testFilter():
    sol = kPOMDPSolver()
    s = 7
    mu = 7
    sig = 10
    allS = [s]
    allMu = [mu]
    allSig = [sig]

    # acts = np.concatenate(np.ones(shape=(10)), -
    # np.ones(shape=(10)), np.zeros(shape=(10)))
    acts = []
    for i in range(0, 10):
        acts.append(0)
    for i in range(0, 5):
        acts.append(1)
    for i in range(0, 10):
        acts.append(2)

    for i in range(0, 25):
        s = sol.update(s, acts[i])
        allS.append(s)
        o = sol.measure(s)
        mu, sig = sol.filter(mu, sig, acts[i], o)
        allMu.append(mu)
        allSig.append(sig)

    # print(allS)

    t = [i for i in range(0, 26)]
    plt.plot(allS, color='green')
    plt.plot(allMu, color='green', linestyle='--')
    plt.fill_between(t, np.array(allMu)-2*np.sqrt(np.array(allSig)),
                     np.array(allMu)+2*np.sqrt(np.array(allSig)), alpha=0.25, color='green')
    plt.show()


def makePolicy():
    sol = kPOMDPSolver()
    b = sol.gatherBeliefs()
    shuffle(b)
    sol.solve(10, b)


def testPolicy():
    pol = np.load("KalmanPolicy.npy")
    sol = kPOMDPSolver()
    s = -10
    mu = -10
    sig = 10
    allS = [s]
    allMu = [mu]
    allSig = [sig]

    for i in range(0, 50):
        act = sol.getAction([mu, sig], pol)
        s = sol.update(s, act)
        allS.append(s)
        o = sol.measure(s)
        mu, sig = sol.filter(mu, sig, act, o)
        allMu.append(mu)
        allSig.append(sig)

    t = [i for i in range(0, 51)]
    plt.plot(allS, color='green')
    plt.plot(allMu, color='green', linestyle='--')
    plt.fill_between(t, np.array(allMu)-2*np.sqrt(np.array(allSig)),
                     np.array(allMu)+2*np.sqrt(np.array(allSig)), alpha=0.25, color='green')
    plt.show()

    # print(len(pol))
    # allActs = [0, 0, 0]
    # for p in pol:
    #     allActs[p.action] += 1
    #     #print(p.action)
    #     # [a, b] = p.plot(low=-20, high=20, vis=False)
    #     # plt.figure()
    #     # plt.plot(a, b)
    #     # plt.title(p.action)


if __name__ == '__main__':
    # makePolicy()
    testPolicy()
    # testFilter()
