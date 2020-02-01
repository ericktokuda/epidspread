#!/usr/bin/env python3
"""Find best params
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info

from multiprocessing import Pool
import numpy as np
import scipy
import scipy.optimize
import igraph
import networkx as nx
import time
import pandas as pd
from optimized import generate_waxman_adj


def run_one_experiment(r):
    nvertices = 625
    avgdegree = 6
    alpha = 0.0025

    time.sleep(np.random.rand()*2)
    maxnedges = nvertices * nvertices //2
    domain = [0, 0, 1, 1]
    n = nvertices

    def rgr(r):
        # '625,6': 0.056865545,
        # '10000,6': 0.0139,
        # '22500,6': 0.00925,
        g = igraph.Graph.GRG(nvertices, r)
        err =  np.mean(g.degree()) - avgdegree
        with open('/home/keiji/temp/grg_params.csv', 'a') as fh:
            fh.write('{},{}\n'.format(r, err))
        # print(r, err)

    def waxman(b):
        adjlist, x, y = generate_waxman_adj(nvertices, maxnedges, alpha, b,
                                            domain[0], domain[1], domain[2], domain[3])
        adjlist = adjlist.astype(int).tolist()

        g = igraph.Graph(n, adjlist)
        err =  np.mean(g.degree()) - avgdegree
        return err

    err = waxman(r)
    print('r:{}, err:{}'.format(r, err))


def generate_waxman(n, maxnedges, alpha, beta, domain=(0, 0, 1, 1)):
    adjlist, x, y = generate_waxman_adj(n, maxnedges, alpha, beta,
                                        domain[0], domain[1], domain[2], domain[3])
    adjlist = adjlist.astype(int).tolist()

    g = igraph.Graph(n, adjlist)
    g.vs['x'] = x
    g.vs['y'] = y
    return g

def get_waxman_params(nvertices, avgdegree, alpha):
    maxnedges = nvertices * nvertices // 2

    radiuscatalog = {
    }

    k = '{},{}'.format(nvertices, avgdegree)
    if k in radiuscatalog.keys():
        return radiuscatalog[k], alpha

    def f(b):
        g = generate_waxman(nvertices, maxnedges, alpha=alpha, beta=b)
        return np.mean(g.degree()) - avgdegree

    b1 = 0.0001
    b2 = 10000
    beta = scipy.optimize.brentq(f, b1, b2, xtol=0.001, rtol=0.05)
    return beta, alpha

def main():
    
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    # First option
    nvertices = 625
    for alpha in [0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.0250]:
        beta = []
        for i in range(10):
            try:
                beta_, alpha_ = get_waxman_params(nvertices, 6, alpha)
                beta.append(beta_)
            except:
                pass
        print(alpha, len(beta), np.mean(beta), np.std(beta))
    return

    # Second option
    params = [1,2,3]
    n = len(params)
    pool = Pool(n)
    pool.map(run_one_experiment, params)

if __name__ == "__main__":
    main()

