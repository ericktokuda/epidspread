#!/usr/bin/env python3
"""Test step_transmission
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
from optimized import step_transmission
import igraph
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    nvertices = 2
    nparticles = 2*nvertices
    g = igraph.Graph.Full(nvertices)
# def step_transmission(g, status, double beta, double gamma, particles):
    status = np.zeros(nparticles, dtype=int)
    status[-1] = 1
    beta = 0.7
    gamma = 0.7
    particles = [np.array([], dtype=int) for i in range(nvertices)]
    particles[1] = np.array(list(range(nparticles)))

    print('Input: nvertices:{}, particlesloc:{}, status:{}'.format(
          nvertices, particles, status))

    newstatus, newtransmissions = step_transmission(g, status, beta, gamma, particles)
    print('Newstatus:{}, newtransm:{}'.format(np.asarray(newstatus),
                                              np.asarray(newtransmissions)))

##########################################################
    print('##########################################################')
    nvertices = 3
    nparticles = 2*nvertices
    g = igraph.Graph.Full(nvertices)
    status = np.zeros(nparticles, dtype=int)
    status[-1] = 1
    beta = 0.5
    gamma = 0.5
    particles = [np.array([], dtype=int) for i in range(nvertices)]
    particles[1] = np.array(list(range(nparticles)))

    print('Input: nvertices:{}, particlesloc:{}, status:{}'.format(
          nvertices, particles, status))

    newstatus, newtransmissions = step_transmission(g, status, beta, gamma, particles)
    print('Newstatus:{}, newtransm:{}'.format(np.asarray(newstatus),
                                              np.asarray(newtransmissions)))

##########################################################
    print('##########################################################')
    nvertices = 10
    nparticles = 2*nvertices
    g = igraph.Graph.Full(nvertices)
    status = np.zeros(nparticles, dtype=int)
    status[-1] = 1
    status[int(nparticles/2)] = 1
    status[0] = 2
    beta = 0.5
    gamma = 0.5
    particles = [np.array([], dtype=int) for i in range(nvertices)]
    # particles[1] = np.array(list(range(nparticles)))

    print('Input: nvertices:{}, particlesloc:{}, status:{}'.format(
          nvertices, particles, status))

    newstatus, newtransmissions = step_transmission(g, status, beta, gamma, particles)
    print('Newstatus:{}, newtransm:{}'.format(np.asarray(newstatus),
                                              np.asarray(newtransmissions)))

if __name__ == "__main__":
    main()

