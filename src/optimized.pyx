#!/usr/bin/env python

import argparse
import logging
import os, sys
from os.path import join as pjoin
from logging import debug, info
from itertools import product
from pathlib import Path
import socket
import time

import string
import igraph

import numpy as np
cimport numpy as np
# DTYPE = np.int
# ctypedef np.int_t DTYPE_t
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import pandas as pd
import copy
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import math
from subprocess import Popen, PIPE
from datetime import datetime
from multiprocessing import Pool
import pickle as pkl
# import torch


########################################################## Defines
cdef long SUSCEPTIBLE = 0
cdef long INFECTED = 1
cdef long RECOVERED = 2

########################################################## Defines
def fast_random_choice(lst, probs, double randnum):
    return lst[np.searchsorted(probs.cumsum(), randnum)]

##########################################################
def step_mobility(g, particles, nagents):
    """Give a step in the mobility dynamic

    Args:
    g(igraph.Graph): instance of a graph
    particles(list of list): the set of particle ids for each vertex

    Returns:
    list of list: indices of the particles in each vertex
    """
    particles_fixed = copy.deepcopy(particles) # copy to avoid being altered

    randnum = np.random.rand(nagents)
    cdef long acc = 0
    cdef long destv
    cdef long n

    for i, _ in enumerate(g.vs): # For each vertex
        neighids = g.neighbors(i)
        if neighids == []: continue
        neighids = neighids + [i] # Auto-loop

        n = len(neighids)
        gradients = g.vs[neighids]['gradient']

        if np.sum(gradients) == 0: # Transform into a prob. distribution
            gradients = np.ones(n) / n
        else:
            gradients /= np.sum(gradients)

        for partic in particles_fixed[i]: # For each particle in this vertex
            # neighid = np.random.choice(neighids, p=gradients) # slow
            destv = fast_random_choice(neighids, gradients, randnum[acc])
            acc += 1

            if destv == i: continue

            particles[i].remove(partic)
            particles[destv].append(partic)

    return particles

##########################################################
cpdef step_transmission(long nvertices, long[:] status, double beta, double gamma, particles):
# cpdef step_transmission(g, status, double beta, double gamma, particles):
    """Give a step in the transmission dynamic

    Args:
    g(igraph.Graph): instance of a graph
    status(list): status of each particle
    beta(float): contagion chance
    gamma(float): recovery chance
    particles(list of list): the set of particle ids for each vertex

    Returns:
    list: updated status
    """

    cdef long nlocalparticles, nsusceptible, ninfected, nrecovered
    cdef long numnewinfected, numnewrecovered
    cdef int i, j, acc

    cdef long[:] status_fixed = status.copy()
    cdef int nparticles = len(status)
    cdef long[:] ntransmissions = np.zeros((nvertices), dtype=np.int_)

    for i in range(nvertices):
        nlocalparticles = len(particles[i]) # number of particles in vertex i
        localparticles = particles[i]

        nsusceptible = 0
        ninfected =  0
        nrecovered =  0
        for j in range(nlocalparticles):
            if status[localparticles[j]] == 0: # status of particle j in vertex i
                nsusceptible += 1
            elif status[localparticles[j]] == 1:
                ninfected += 1
            elif status[localparticles[j]] == 2:
                nrecovered += 1

        x  = np.random.rand(nsusceptible*ninfected)
        y  = np.random.rand(ninfected)

        numnewinfected = np.sum(x <= beta)
        numnewrecovered = np.sum(y <= gamma)

        if numnewinfected > nsusceptible: numnewinfected = nsusceptible
        if numnewrecovered > ninfected: numnewrecovered = ninfected

        ntransmissions[i] = numnewinfected

        if numnewinfected > 0:
            acc = 0
            for j in range(nlocalparticles):
                if status_fixed[localparticles[j]] == SUSCEPTIBLE:
                    status[localparticles[j]] = INFECTED # Update the modifiable
                    acc += 1
                    if acc == numnewinfected: break

        # We use a copy to avoid considering the newly infected agents because
        # this is an offline update step
        if numnewrecovered > 0:
            acc = 0
            for j in range(nlocalparticles):
                if status_fixed[localparticles[j]] == INFECTED:
                    status[localparticles[j]] = RECOVERED
                    acc += 1
                    if acc == numnewrecovered: break

    return status, ntransmissions

##########################################################
cpdef generate_waxman_adj(long n, long avgdegree, float alpha, float beta,
                          long xmin, long ymin, long xmax, long ymax):

    # cdef long maxnumvertices = n*avgdegree//2
    cdef int maxnumvertices = n*n//2
    cdef int[:, :] adj = np.ones((maxnumvertices, 2), dtype=np.intc)
    cdef double[:] x = np.zeros(n, dtype=np.double)
    cdef double[:] y = np.zeros(n, dtype=np.double)
    cdef int u, v, nodeid, i
    cdef double l

    for nodeid in range(n):
        x[nodeid] = xmin + ((xmax-xmin)*np.random.rand())
        y[nodeid] = ymin + ((ymax-ymin)*np.random.rand())

    l = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)

    i = 0
    for u in range(n):
        x1, y1 = x[u], y[u]
        for v in range(u + 1, n):
            x2, y2 = x[v], y[v]
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2)

            if np.random.rand() < alpha * math.exp(-d/(beta*l)):
                # adj[u, v] = 1 # just fill upper part of the matrix
                adj[i, 0] = u
                adj[i, 1] = v
                i += 1
        # if u % 10 == 0:
            # print(u)
    adj = adj[:i]
    # print(adj.tolist())
    return np.asarray(adj), np.asarray(x), np.asarray(y)
