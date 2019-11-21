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
    # randnum = torch.rand((nagents,))
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
def step_transmission(g, status, double beta, double gamma, particles):
    """Give a step in the transmission dynamic

    Args:
    g(igraph.Graph): instance of a graph
    status(list): statuses of each particle
    beta(float): contagion chance
    gamma(float): recovery chance
    particles(list of list): the set of particle ids for each vertex

    Returns:
    list: updated statuses
    """

    statuses_fixed = copy.deepcopy(status)
    ntransmissions = np.zeros((g.vcount()), dtype=int)

    cdef long N, nsusceptible, ninfected, nrecovered
    cdef long numnewinfected, numnewrecovered

    for i, _ in enumerate(g.vs):
        statuses = statuses_fixed[particles[i]]
        N = len(statuses)
        nsusceptible = len(statuses[statuses==SUSCEPTIBLE])
        ninfected = len(statuses[statuses==INFECTED])
        nrecovered = len(statuses[statuses==RECOVERED])

        indsusceptible = np.where(statuses_fixed==SUSCEPTIBLE)[0]
        indinfected = np.where(statuses_fixed==INFECTED)[0]
        indrecovered = np.where(statuses_fixed==RECOVERED)[0]

        x  = np.random.rand(nsusceptible*ninfected)
        y  = np.random.rand(ninfected)
        # x  = torch.rand((nsusceptible*ninfected,)).numpy()
        # y  = torch.rand((ninfected,)).numpy()
        numnewinfected = np.sum(x <= beta)
        numnewrecovered = np.sum(y <= gamma)
        if numnewinfected > nsusceptible: numnewinfected = nsusceptible
        if numnewrecovered > ninfected: numnewrecovered = ninfected

        ntransmissions[i] = numnewinfected
        status[indsusceptible[0:numnewinfected]] = INFECTED
        status[indinfected[0:numnewrecovered]] = RECOVERED
    return status, ntransmissions
##########################################################
