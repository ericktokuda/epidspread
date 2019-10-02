#!/usr/bin/env python
""" Simulation of two dynamics: mobility and infection over a lattice
"""

import argparse
import logging
import os, sys
from os.path import join as pjoin
from logging import debug, info
from itertools import product
from pathlib import Path
import socket

import string
import igraph
import networkx as nx
import numpy as np
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


########################################################## Defines
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
EPSILON = 1E-5
MAX = sys.maxsize


#############################################################
def get_4connected_neighbours_2d(i, j, n, thoroidal=False):
    """Get 4-connected neighbours. It does not check if there are repeated entries (2x2 or 1x1)

    Args:
    i(int): row of the matrix
    j(int): column of the matrix
    n(int): side of the square matrix

    Returns:
    ndarray 4x2: 4 neighbours indices
    """
    inds = []
    if j > 0: # left
        inds.append([i, j-1])
    elif thoroidal:
        inds.append([i, n-1])

    if j < n-1: # right
        inds.append([i, j+1])
    elif thoroidal:
        inds.append([i, 0])

    if i > 0: # top
        inds.append([i-1, j])
    elif thoroidal:
        inds.append([n-1, j])

    if i < n-1: # bottom
        inds.append([i+1, j])
    elif thoroidal:
        inds.append([0, j])

    return np.array(inds)

#############################################################
def generate_lattice(n, thoroidal=False, s=10):
    """Generate 2d lattice of side n

    Args:
    n(int): side of the lattice
    thoroidal(bool): thoroidal lattice
    s(float): edge size

    Returns:
    ndarray nx2, ndarray nxn: positions and adjacency matrix (triangular)
    """
    n2 = n*n
    pos = np.ndarray((n2, 2), dtype=float)
    adj = np.zeros((n2, n2), dtype=int)

    k = 0
    for j in range(n):
        for i in range(n): # Set positions
            pos[k] = [i*s, j*s]
            k += 1

    for i in range(n): # Set connectivity
        for j in range(n):
            neighs2d = get_4connected_neighbours_2d(i, j, n, thoroidal)
            neighids = np.ravel_multi_index((neighs2d[:, 0], neighs2d[:, 1]), (n, n))
            curidx = np.ravel_multi_index((i, j), (n, n))

            for neigh in neighids:
                adj[curidx, neigh] = 1
    return pos, adj
#############################################################
def run_one_experiment_given_list(l):
    run_lattice_sir(*l)

#############################################################
def generate_graph(graphtopology, graphsize, graphparam1, graphparam2,
                   graphparam3, graphlayout, layoutparam1, layoutparam2, layoutparam3,
                   plotarea):
    """Generate graph with given topology

    Args:
    graphsize(int): number of vertices
    graphtopology(str): topology, consult igraph layout options
    graphparam1, graphparam2, graphparam3: topology options

    Returns:
    igraph.Graph, np.ndarray: graph and the layout
    """

    if graphtopology == 'lattice':
        # 1: neigh, 2: thoroidal
        mapside = int(np.sqrt(graphsize))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=graphparam2)
    elif graphtopology == 'erdos':
        # 1: probability
        g = igraph.Graph.Erdos_Renyi(graphsize, graphparam1)
    elif graphtopology == 'watts':
        # 1: dimension, 2: size of the lattice, 3: rewiring prob
        # graphsize =  x^k
        # g = igraph.graph.(x, k, graphparam2, graphparam3)
        # Watts_Strogatz(dim, size, nei, p
        pass

    if graphlayout == 'grid':
        layout = g.layout(graphlayout)
    elif graphlayout == 'fr' or graphlayout == 'fruchterman_reingold':
        layout = g.layout(graphlayout, layoutparam1, layoutparam2, area=plotarea)

    aux = np.array(layout.coords)
    coords = (aux - np.mean(aux, 0))/np.std(aux, 0)
    return g, coords
##########################################################
def run_lattice_sir(graphtopology, graphsize, graphparam1, graphparam2, graphparam3,
                    graphlayout, layoutparam1, layoutparam2, layoutparam3,
                    nepochs, s0, i0, r0,
                    beta, gamma, graddist , gradparam1, gradparam2, gradparam3,
                    autoloop_prob, plotzoom, plotrate, outdir,
                    nprocs, randomseed, expidx):
    """Main function

    Args:
    params

    Returns:
    ret
    """


    cfgdict = {}
    keys = ['graphtopology', 'graphsize', 'graphparam1' , 'graphparam2' , 'graphparam3',
            'graphlayout', 'layoutparam1', 'layoutparam2', 'layoutparam3',
            'nepochs', 's0' , 'i0' , 'r0' ,
            'beta', 'gamma' , 'graddist' , 'gradparam1', 'gradparam2', 'gradparam3',
            'autoloop_prob' , 'plotzoom' , 'plotrate' , 'outdir' ,
            'nprocs' , 'randomseed']
    args = [graphtopology, graphsize, graphparam1, graphparam2, graphparam3,
            graphlayout, layoutparam1, layoutparam2, layoutparam3,
            nepochs, s0, i0, r0,
            beta, gamma, graddist, gradparam1, gradparam2, gradparam3,
            autoloop_prob, plotzoom, plotrate, outdir,
            nprocs , randomseed]
    for i, k in enumerate(keys):
        cfgdict[k] = args[i]

    # args = [mapside, nei, istoroid , nepochs , s0 , i0 , r0 ,
    cfg = pd.DataFrame.from_dict(cfgdict, 'index', columns=['data'])

    ########################################################## Cretate outdir
    # expidxstr = '{:03d}'.format(expidx)
    outdir = pjoin(outdir, expidx)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ##########################################################
    info('exp:{} Copying config file ...'.format(expidx))
    cfg['data'].to_json(pjoin(outdir, 'config.json'), force_ascii=False)

    mapside = int(np.sqrt(graphsize))
    nei = graphparam1
    istoroid = graphparam2

    dim = [mapside, mapside]
    N = s0 + i0 + r0
    nvertices = mapside**2 # square lattice
    status = np.ndarray(N, dtype=int)
    status[0: s0] = SUSCEPTIBLE
    status[s0:s0+i0] = INFECTED
    status[s0+i0:] = RECOVERED
    np.random.shuffle(status)
    info('exp:{} Generated random distribution of S, I, R ...'.format(expidx))

    visual = {}
    visual["bbox"] = (mapside*10*plotzoom, mapside*10*plotzoom)
    visual["margin"] = mapside*plotzoom
    visual["vertex_size"] = 10*plotzoom

    totalnsusceptibles = [s0]
    totalninfected = [i0]
    totalnrecovered = [r0]

    aux = '' if istoroid else 'non-'
    info('exp:{} Generating {}toroidal lattice with dim ({}, {}) ...'.format(expidx,
                                                                             aux,
                                                                             mapside,
                                                                             mapside,
                                                                             ))
    plotarea = 36   # Square of the center surrounded by radius 3
                    # (equiv to 99.7% of the points of a gaussian)
    g, coords = generate_graph(graphtopology, graphsize, graphparam1,
                               graphparam2, graphparam3, graphlayout,
                               layoutparam1, layoutparam2, layoutparam3, plotarea)

    # visualize_static_graph_layouts(g, 'config/layouts_lattice.txt', outdir);

    ntransmissions = np.zeros(nvertices, dtype=int)
    ########################################################## Distrib. of particles
    info('exp:{} Generating uniform distribution of agents in the lattice ...'.format(expidx))
    nparticles = np.ndarray(nvertices, dtype=int)
    aux = np.random.rand(nvertices) # Uniform distrib
    nparticles = np.round(aux / (np.sum(aux)) *N).astype(int)

    diff = N - np.sum(nparticles) # Correct rounding differences on the final number
    for i in range(np.abs(diff)):
        idx = np.random.randint(nvertices)
        nparticles[idx] += np.sign(diff) # Initialize number of particles per vertex

    particles = [None]*nvertices # Initialize indices of particles per vertex
    aux = 0
    for i in range(nvertices):
        particles[i] = list(range(aux, aux+nparticles[i]))
        aux += nparticles[i]
    nparticlesstds = [np.std([len(x) for x in particles])]

    ########################################################## Distrib. of gradients
    gradstd = gradparam2
    info('exp:{} Initializing gradients distribution ...'.format(expidx))
    g = initialize_gradients(g, coords, graddist, gradstd)
    info('exp:{} Exporting relief map...'.format(expidx))

    aux = pd.DataFrame()

    aux['x'] = coords[:, 0]
    aux['y'] = coords[:, 1]
    aux['gradient'] = g.vs['gradient']
    aux.to_csv(pjoin(outdir, 'attraction.csv'), index=False, header=['x', 'y', 'gradient'])

    ########################################################## Plot gradients
    if plotrate > 0:
        info('exp:{} Generating plots for epoch 0'.format(expidx))

        aux = np.sum(g.vs['gradient'])
        gradientscolors = [ [c, c, c] for c in g.vs['gradient']]
        # gradientscolors = [1, 1, 1]*g.vs['gradient']
        gradsum = float(np.sum(g.vs['gradient']))
        gradientslabels = [ '{:2.3f}'.format(x/gradsum) for x in g.vs['gradient']]
        outgradientspath = pjoin(outdir, 'gradients.png')
        igraph.plot(g, target=outgradientspath, layout=coords.tolist(),
                    vertex_shape='rectangle', vertex_color=gradientscolors,
                    vertex_frame_width=0.0, **visual)      

        b = 0.1 # For colors definition
        ########################################################## Plot epoch 0
        nsusceptibles, ninfected, nrecovered, \
            _, _, _  = compute_statuses_sums(status, particles, nvertices, [], [], [])
        plot_epoch_graphs(-1, g, coords, visual, status, nvertices, particles,
                          N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                          totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    maxepoch = nepochs if nepochs > 0 else MAX
    for ep in range(maxepoch):
        if ep % 10 == 0:
            info('exp:{}, t:{}'.format(expidx, ep))
        particles = step_mobility(g, particles, autoloop_prob)
        aux = np.std([len(x) for x in particles])
        nparticlesstds.append(aux)
        status, ntransmissions = step_transmission(g, status, beta, gamma, particles,
                                                      ntransmissions)
      
        nsusceptibles, ninfected, nrecovered, \
            totalnsusceptibles, totalninfected, \
            totalnrecovered  = compute_statuses_sums(status, particles, nvertices,
                                                     totalnsusceptibles, totalninfected,
                                                     totalnrecovered)

        if nepochs == -1 and np.sum(status==INFECTED) == 0: break

        if plotrate > 0 and ep % plotrate == 0:
            plot_epoch_graphs(ep, g, coords, visual, status, nvertices, particles,
                              N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                              totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    ########################################################## Enhance plots
    if plotrate > 0:
        # cmd = "mogrify -gravity south -pointsize 24 " "-annotate +50+0 'GRADIENT' " \
            # "-annotate +350+0 'S' -annotate +650+0 'I' -annotate +950+0 'R' " \
            # "{}/concat*.png".format(outdir)
        # proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        # stdout, stderr = proc.communicate()

        animationpath = pjoin(outdir, 'animation.gif')
        cmd = 'convert -delay 120 -loop 0  {}/concat*.png "{}"'.format(outdir, animationpath)
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        print(stderr)

    ########################################################## Export to csv
    info('exp:{} Exporting transmissions locations...'.format(expidx))
    aux = pd.DataFrame(ntransmissions)
    aux.to_csv(pjoin(outdir, 'ntranmissions.csv'), index=False, header=['ntransmission'])
    ########################################################## Export to csv
    info('exp:{} Exporting S, I, R data'.format(expidx))
    aux = np.array([totalnsusceptibles, totalninfected, totalnrecovered, nparticlesstds]).T
    pd.DataFrame(aux).to_csv(pjoin(outdir, 'sir.csv'), header=['S', 'I', 'R', 'nparticlesstd'],
                             index=True, index_label='t')

    ########################################################## Plot SIR over time
    info('exp:{} Generating plots for counts of S, I, R'.format(expidx))
    fig, ax = plt.subplots(1, 1)
    plot_sir(totalnsusceptibles, totalninfected, totalnrecovered, fig, ax, outdir)
    info('exp:{} Finished. Results are in {}'.format(expidx, outdir))

def visualize_static_graph_layouts(g, layoutspath, outdir):
    layouts = [line.rstrip('\n') for line in open(layoutspath)]
    for l in layouts:
        info(l)
        try:
            igraph.plot(g, target=pjoin(outdir, l + '.png'), layout=g.layout(l),
                        vertex_color='lightgrey',
                        vertex_label=list(range(g.vcount())))
        except Exception:
            pass

########################################################## Distrib. of gradients
def initialize_gradients_peak(g):
    """Initizalition of gradients with a peak at 0

    Args:
    g(igraph.Graph): graph instance

    Returns:
    igraph.Graph: graph instance with attribute 'gradient' updated
    """
    g.vs['gradient'] = 0.1
    g.vs[0]['gradient'] = 1
    return g

##########################################################
def multivariate_normal(x, d, mean, cov):
    """pdf of the multivariate normal when the covariance matrix is positive definite.
    Source: wikipedia"""
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(cov))) *
            np.exp(-(np.linalg.solve(cov, x - mean).T.dot(x - mean)) / 2))

##########################################################
def gaussian(xx, mu, sig):
    """pdf of the normal distrib"""
    x = np.array(xx)
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

##########################################################
def set_gaussian_weights_recursive(g, curid, nextvs, dist, mu, sigma):
    supernewgrad = gaussian(dist+1, mu, sigma)
    visitted.add(curid)
    for v in g.neighbors(curid):
        g.vs[v]['gradient'] = supernewgrad 

    visitted.remove(curid)

##########################################################
def initialize_gradients_gaussian_on_graph(g, mu=0, sigma=1):
    """Initizalition of gradients with a single gaussian

    Args:
    g(igraph.Graph): graph instance
k
    Returns:
    igraph.Graph: graph instance with attribute 'gradient' updated
    """

    # centeridx = int((g.vcount())/2)
    if g.vcount() % 2 == 0:
        centeridx = int((g.vcount())/2 - np.sqrt(g.vcount())/2) 
    else:
        centeridx = int((g.vcount())/2)
    dists = g.shortest_paths(centeridx)
    gauss = gaussian(dists, mu, sigma).flatten()
    for v in range(len(gauss)):
        g.vs[v]['gradient'] = gauss[v]

    return g

##########################################################
def initialize_gradients_gaussian(g, coords, mu, cov):
    """Initizalition of gradients with a single gaussian

    Args:
    g(igraph.Graph): graph instance
k
    Returns:
    igraph.Graph: graph instance with attribute 'gradient' updated
    """

    for i, v in enumerate(g.vs):
        g.vs[i]['gradient'] = multivariate_normal(coords[i, :], 2, mu, cov)

    return g

##########################################################
def initialize_gradients(g, coords, method='peak', sigma=1):
    """Initialize gradients with some distribution

    Args:
    g(igraph.Graph): graph instance

    Returns:
    igraph.Graph: graph instance with attribute 'gradient' updated
    """


    if method == 'uniform':
        g.vs['gradient'] = 0.1
        return g
    if method == 'peak':
        return initialize_gradients_peak(g)
    elif method == 'gaussian':
        mu = (np.max(coords, 0) + np.min(coords, 0)) / 2
        cov = np.eye(2) * sigma
        return initialize_gradients_gaussian(g, coords, mu, cov)

def step_mobility(g, particles, autoloop_prob):
    """Give a step in the mobility dynamic

    Args:
    g(igraph.Graph): instance of a graph
    particles(list of list): the set of particle ids for each vertex
    autoloop_prob(float): probability of staying in the same place

    Returns:
    list of list: indices of the particles in each vertex
    """
    particles_fixed = copy.deepcopy(particles) # copy to avoid being altered

    for i, _ in enumerate(g.vs): # For each vertex
        numvparticles = len(particles_fixed[i])
        neighids = g.neighbors(i)
        n = len(neighids)
        gradients = g.vs[neighids]['gradient']

        if np.sum(gradients) == 0:
            gradients = np.ones(n) / n
        else:
            gradients /= np.sum(gradients)

        for j, partic in enumerate(particles_fixed[i]): # For each particle in this vertex
            if np.random.rand() <= autoloop_prob: continue
            if neighids == []: continue
            neighid = np.random.choice(neighids, p=gradients)
            particles[i].remove(partic)
            particles[neighid].append(partic)
    return particles

##########################################################
def step_transmission(g, status, beta, gamma, particles, ntransmissions):
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
        numnewinfected = np.sum(x <= beta)
        numnewrecovered = np.sum(y <= gamma)
        if numnewinfected > nsusceptible: numnewinfected = nsusceptible
        if numnewrecovered > ninfected: numnewrecovered = ninfected

        ntransmissions[i] += numnewinfected
        status[indsusceptible[0:numnewinfected]] = INFECTED
        status[indinfected[0:numnewrecovered]] = RECOVERED
    return status, ntransmissions

##########################################################
def compute_statuses_sums(status, particles, nvertices, totalnsusceptibles,
                          totalninfected, totalnrecovered):
    """Compute the sum of each status

    Args:
    params

    Returns:
    nsusceptibles(list of int): number of susceptibles per vertex
    ninfected(list of int): number of infected per vertex
    nrecovered(list of int): number of recovered per vertex
    """

    nsusceptibles = np.array([ np.sum(status[particles[i]]==SUSCEPTIBLE) for i in range(nvertices)] )
    ninfected = np.array([ np.sum(status[particles[i]]==INFECTED) for i in range(nvertices)] )
    nrecovered = np.array([ np.sum(status[particles[i]]==RECOVERED) for i in range(nvertices)] )
    totalnsusceptibles.append(np.sum(nsusceptibles))
    totalninfected.append(np.sum(ninfected))
    totalnrecovered.append(np.sum(nrecovered))
    return nsusceptibles, ninfected, nrecovered, totalnsusceptibles, totalninfected, totalnrecovered
##########################################################
def plot_epoch_graphs(ep, g, coords, visual, status, nvertices, particles,
                      N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                      totalnsusceptibles, totalninfected, totalnrecovered, outdir):
    susceptiblecolor = []
    infectedcolor = []
    recoveredcolor = []

    for z in nsusceptibles:
        zz = [0, math.log(z, N), 0] if z*N > 1 else [0, 0, 0] # Bug on log(1,1)
        susceptiblecolor.append(zz)
    for z in ninfected:
        zz = [math.log(z, N), 0, 0] if z*N > 1 else [0, 0, 0]
        infectedcolor.append(zz)
    for z in nrecovered:
        zz = [0, 0,  math.log(z, N)] if z*N > 1 else [0, 0, 0]
        recoveredcolor.append(zz)  
        
    outsusceptiblepath = pjoin(outdir, 'susceptible{:02d}.png'.format(ep+1))
    outinfectedpath = pjoin(outdir, 'infected{:02d}.png'.format(ep+1))
    outrecoveredpath = pjoin(outdir, 'recovered{:02d}.png'.format(ep+1))

    igraph.plot(g, target=outsusceptiblepath, layout=coords.tolist(), vertex_shape='rectangle', vertex_color=susceptiblecolor, vertex_frame_width=0.0, **visual)
    igraph.plot(g, target=outinfectedpath, layout=coords.tolist(), vertex_shape='rectangle', vertex_color=infectedcolor, vertex_frame_width=0.0, **visual)
    igraph.plot(g, target=outrecoveredpath, layout=coords.tolist(), vertex_shape='rectangle', vertex_color=recoveredcolor, vertex_frame_width=0.0, **visual)

    outconcatpath = pjoin(outdir, 'concat{:02d}.png'.format(ep+1))
    proc = Popen('convert {} {} {} {} +append {}'.format(outgradientspath,
                                                         outsusceptiblepath,
                                                         outinfectedpath,
                                                         outrecoveredpath,
                                                         outconcatpath),
                 shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

##########################################################
def plot_sir(s, i, r, fig, ax, outdir):
    ax.plot(s, 'g', label='Susceptibles')
    ax.plot(i, 'r', label='Infected')
    ax.plot(r, 'b', label='Recovered')
    ax.legend()
    fig.savefig(pjoin(outdir, 'sir.png'))

def random_string(length=8):
    """Generate a random string of fixed length """
    letters = np.array(list(string.ascii_lowercase + string.digits))
    # aux = ''.join(np.random.choice(letters, size=length))
    # return aux.replace(' ', 'z')
    return ''.join(np.random.choice(letters, size=length))
##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/toy01.json')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    cfg = pd.read_json(args.config, typ='series', precise_float=True) # Load config

    outdir = pjoin(cfg.outdir[0], datetime.now().strftime('%Y%m%d_%H%M') + '-agentssir')

    info('Files will be generated in {}/...'.format(outdir))
    if os.path.exists(outdir):
        ans = input(outdir + ' exists. Do you want to continue? ')
        if ans.lower() not in ['y', 'yes']:
            info('Aborting')
            return
    else:
        os.mkdir(outdir)

    cfg.outdir = [outdir]
    
    aux = list(product(*cfg))
    params = []
    fh = open(pjoin(outdir, 'exps.csv'), 'w')
    colnames = ['idx'] + (list(cfg.index)) + ['hostname']
    fh.write(','.join(colnames) + '\n')

    hashsz = 6
    hashes = []
    fixedchar = random_string(1)
    for i in range(len(aux)):
        while True:
            hash = fixedchar + random_string(hashsz - 1)
            if hash not in hashes: break
        hashes.append(hash)
        params.append(list(aux[i]) + [hash])
        pstr = [str(x) for x in [hash] + list(aux[i])]
        pstr += [socket.gethostname()]
        fh.write(','.join(pstr) + '\n')
    fh.close()

    if cfg.nprocs[0] <= 1:
        [ run_one_experiment_given_list(p) for p in params ]
    else:
        pool = Pool(cfg.nprocs[0])
        pool.map(run_one_experiment_given_list, params)


if __name__ == "__main__":
    main()
