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
import time

import string
import igraph
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
MAXITERS = 100000

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

def fast_random_choice(lst, probs):
    return lst[np.searchsorted(probs.cumsum(), np.random.rand())]

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

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
    # print(type(l))
    # print(len(l))
    run_experiment(l)

#############################################################
def generate_graph(topologymodel, nvertices,
                   latticethoroidal, erdosavgdegree,
                   erdosloops, layoutmodel,
                   frmaxiter, frmaxdelta,
                   kkmaxiter, kkstd):
    """Generate graph with given topology

    Args:
    graphsize(int): number of vertices
    graphtopology(str): topology, consult igraph layout options
    graphparam1, graphparam2, graphparam3: topology options

    Returns:
    igraph.Graph, np.ndarray: graph and the layout
    """

    if topologymodel == 'lattice':
        mapside = int(np.sqrt(nvertices))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=latticethoroidal)
    elif topologymodel == 'erdos':
        erdosprob = erdosavgdegree / nvertices
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(nvertices, erdosprob)
    elif topologymodel == 'watts':
        pass

    layout = g.layout(layoutmodel) # To be overwritten if parameters

    if layoutmodel == 'grid':
        layout = g.layout(layoutmodel)
    elif layoutmodel == 'fr' or layoutmodel == 'fruchterman_reingold':
        if frmaxiter != -1 or frmaxdelta != -1:
            layout = g.layout(layoutmodel, maxiter=frmaxiter, maxdelta=frmaxdelta)
    elif layoutmodel == 'kk' or layoutmodel == 'kamada_kawai':
        if kkmaxiter != -1 and kksigma != -1:
            layout = g.layout(graphlayout, maxiter=kkmaxiter, sigma=kksigma)

    aux = np.array(layout.coords)
    coords = (aux - np.mean(aux, 0))/np.std(aux, 0) # stndard normalization
    return g, coords
##########################################################
def run_experiment(cfg):
    """Main function

    Args:
    params

    Returns:
    ret
    """

    t0 = time.time()
    cfgdf = pd.DataFrame.from_dict(cfg, 'index', columns=['data'])
    
    ##########################################################  Local vars
    outdir = cfg['outdir']
    nvertices = cfg['nvertices']
    topologymodel = cfg['topologymodel']
    erdosavgdegree = cfg['erdosavgdegree']
    erdosloops = cfg['erdosloops']
    latticethoroidal = cfg['latticethoroidal']
    layoutmodel = cfg['layoutmodel']
    frmaxiter = cfg['frmaxiter']
    frmaxdelta= cfg['frmaxdelta']
    kkmaxiter = cfg['kkmaxiter']
    kkstd     = cfg['kkstd']
    nepochs   = cfg['nepochs']
    s0        = cfg['s0']
    i0        = cfg['i0']
    r0        = cfg['r0']
    beta      = cfg['beta']
    gamma     = cfg['gamma']
    graddist  = cfg['graddist']
    ngaussians = cfg['ngaussians']
    gaussianstds = cfg['gaussianstds']
    autoloop_prob = cfg['autoloop_prob']
    plotzoom  = cfg['plotzoom']
    plotrate  = cfg['plotrate']
    nprocs    = cfg['nprocs']
    randomseed= cfg['randomseed']
    expidx= cfg['expidx']


    ########################################################## 
    outdir = pjoin(outdir, expidx)
    summarypath = pjoin(outdir, 'sir.csv')
    elapsedpath = pjoin(outdir, 'elapsed.csv')
    if os.path.exists(summarypath): return
    os.makedirs(outdir, exist_ok=True) # Create outdir

    ##########################################################
    info('exp:{} Copying config file ...'.format(expidx))
    cfgdf['data'].to_json(pjoin(outdir, 'config.json'), force_ascii=False)


    mapside = int(np.sqrt(nvertices))
    istoroid = latticethoroidal
    N = s0 + i0 + r0
    status = np.ndarray(N, dtype=int)
    status[0: cfg['s0']] = SUSCEPTIBLE
    status[cfg['s0']:cfg['s0']+cfg['i0']] = INFECTED
    status[cfg['s0']+cfg['i0']:] = RECOVERED
    np.random.shuffle(status)
    info('exp:{} Generated random distribution of S, I, R ...'.format(cfg['expidx']))

    visual = {}
    visual["bbox"] = (mapside*10*cfg['plotzoom'], mapside*10*cfg['plotzoom'])
    visual["margin"] = mapside*cfg['plotzoom']
    visual["vertex_size"] = 10*cfg['plotzoom']

    statuscountsum = np.zeros((MAXITERS, 3), dtype=int)
    statuscountsum[0, :] = np.array([cfg['s0'], cfg['i0'], cfg['r0']])


    aux = '' if cfg['latticethoroidal'] else 'non-'
    info('exp:{} Generating {}toroidal lattice with dim ({}, {}) ...'.format(expidx,
                                                                             aux,
                                                                             mapside,
                                                                             mapside,
                                                                             ))

    plotarea = 36   # Square of the center surrounded by radius 3
                    # (equiv to 99.7% of the points of a gaussian)

    g, coords =  generate_graph(topologymodel, nvertices,
                                latticethoroidal, erdosavgdegree,
                                erdosloops, layoutmodel,
                                frmaxiter, frmaxdelta,
                                kkmaxiter, kkstd)

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
    # nparticlesstds = [np.std([len(x) for x in particles])]
    nparticlesstds = np.zeros((MAXITERS,), dtype=float)

    ########################################################## Distrib. of gradients
    info('exp:{} Initializing gradients distribution ...'.format(expidx))
    g = initialize_gradients(g, coords, graddist, gaussianstds)
    info('exp:{} Exporting relief map...'.format(expidx))

    # print(np.max(coords))
    # print(np.min(coords))
    # print(np.max(g.vs['gradient']))
    # print(np.min(g.vs['gradient']))
    # print(np.mean(g.vs['gradient']))
    # print(np.std(g.vs['gradient']))
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
        statuscount, _  = compute_statuses_sums(status, particles, nvertices, )

        plot_epoch_graphs(-1, g, coords, visual, status, nvertices, particles,
                          N, b, outgradientspath,
                          statuscount[:, 0], statuscount[:, 1], statuscount[:, 2],
                          outdir)

    maxepoch = nepochs if nepochs > 0 else MAX

    for ep in range(maxepoch):
        lastepoch = ep
        if ep % 10 == 0:
            info('exp:{}, t:{}'.format(expidx, ep))
        particles = step_mobility(g, particles, autoloop_prob)
        nparticlesstds[ep] = np.std([len(x) for x in particles])
        status, newtransmissions = step_transmission(g, status, beta, gamma, particles)
        ntransmissions += newtransmissions
      
        dist, distsum  = compute_statuses_sums(status, particles, nvertices)
        statuscountsum[ep, :] = distsum

        if nepochs == -1 and np.sum(status==INFECTED) == 0: break

        if plotrate > 0 and ep % plotrate == 0:
            plot_epoch_graphs(ep, g, coords, visual, status, nvertices, particles,
                              N, b, outgradientspath,
                              dist[:, 0], dist[:, 1], dist[:, 2],
                              outdir)

    ########################################################## Enhance plots
    if plotrate > 0:
        animationpath = pjoin(outdir, 'animation.gif')
        cmd = 'convert -delay 120 -loop 0  {}/concat*.png "{}"'.format(outdir, animationpath)
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        print(stderr)

    statuscountsum = statuscountsum[:lastepoch+1, :]
    nparticlesstds = nparticlesstds[:lastepoch+1]
    ########################################################## Export to csv
    info('exp:{} Exporting transmissions locations...'.format(expidx))
    aux = pd.DataFrame(ntransmissions)
    aux.to_csv(pjoin(outdir, 'ntranmissions.csv'), index=False, header=['ntransmission'])
    ########################################################## Export to csv
    info('exp:{} Exporting S, I, R data'.format(expidx))
    outdf = pd.DataFrame({
        'S': statuscountsum[:, 0],
        'I': statuscountsum[:, 1],
        'R': statuscountsum[:, 2],
        'nparticlesstd': nparticlesstds
    })
    outdf.to_csv(summarypath, index=True, index_label='t')

    ########################################################## Plot SIR over time
    info('exp:{} Generating plots for counts of S, I, R'.format(expidx))
    fig, ax = plt.subplots(1, 1)
    plot_sir(statuscountsum[:, 0], statuscountsum[:, 1], statuscountsum[:, 2],
             fig, ax, outdir)

    elapsed = time.time() - t0
    info('exp:{} Elapsed time: {:.2f}h'.format(expidx, elapsed/3600))
    with open(elapsedpath, 'w') as fh: fh.write(str(elapsed))
    info('exp:{} Finished. Results are in {}'.format(expidx, outdir))

########################################################## Plot SIR over time
def visualize_static_graph_layouts(g, layoutspath, outdir):
    layouts = [line.rstrip('\n') for line in open(layoutspath)]
    print(layouts)
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

            # neighid = np.random.choice(neighids, p=gradients) # slow
            neighid = fast_random_choice(neighids, gradients)

            particles[i].remove(partic)
            particles[neighid].append(partic)
    return particles

##########################################################
def step_transmission(g, status, beta, gamma, particles):
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

        ntransmissions[i] = numnewinfected
        status[indsusceptible[0:numnewinfected]] = INFECTED
        status[indinfected[0:numnewrecovered]] = RECOVERED
    return status, ntransmissions

##########################################################
def compute_statuses_sums(status, particles, nvertices, nclasses=3):
    """Compute the sum of each status

    Args:
    status(np.ndarray): size nparticlesx1, with an int corresponding to the status
    particles(list of list): list of ids of the neighbour particles
    nvertices(int): number of vertices of the map
    totalnsusceptibles(int): number of vertices of the map

    Returns:
    nsusceptibles(list of int): number of susceptibles per vertex
    ninfected(list of int): number of infected per vertex
    nrecovered(list of int): number of recovered per vertex
    """


    dist = np.zeros((nvertices, 3))
    for i in range(nvertices):
        x = np.bincount(status[particles[i]], minlength=nclasses)
        dist[i, :] = x

    sums = np.sum(dist, 0)
    return dist, np.sum(dist, 0)
##########################################################
def plot_epoch_graphs(ep, g, coords, visual, status, nvertices, particles,
                      N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                      outdir):
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

    # Delete individual files
    proc = Popen('rm {} {} {} {}'.format(outgradientspath,
                                         outsusceptiblepath,
                                         outinfectedpath,
                                         outrecoveredpath
                                         ),
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
    return ''.join(np.random.choice(letters, size=length))

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite')
    parser.add_argument('--shuffle', action='store_true', help='Shuffled traversing of config parameters')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    cfg = pd.read_json(args.config, typ='series', precise_float=True) # Load config

    # outdir = pjoin(cfg.outdir[0], datetime.now().strftime('%Y%m%d_%H%M') + '-agentssir')
    outdir = cfg.outdir[0]

    info('Files will be generated in {}/...'.format(outdir))
    existing = os.path.exists(outdir)

    if existing and not args.overwrite:
        print('Folder {} exists. Change the outdir parameter or use --overwrite'. \
              format(outdir))
        return

    os.makedirs(outdir, exist_ok=True)
    cfg.outdir = [outdir]

    expspath = pjoin(outdir, 'exps.csv')

    cfgkeys = list(cfg.index)

    if existing and os.path.exists(expspath): # Load config from exps
        df = pd.read_csv(expspath)
        params = df.to_dict(orient='records')
    else:
        aux = list(product(*cfg))
        params = []
        fh = open(expspath, 'w')
        colnames = ['expidx'] + (list(cfg.index)) + ['hostname']
        fh.write(','.join(colnames) + '\n')

        hashsz = 8
        hashes = []
        hostname = socket.gethostname()
        fixedchars = hostname[:2]
        for i in range(len(aux)):
            while True:
                hash = fixedchars + random_string(hashsz - 2)
                if hash not in hashes: break
            hashes.append(hash)
            param = {}
            for j, key in enumerate(cfgkeys):
                param[key] = aux[i][j]
            param['expidx'] = hash
            params.append(param)
            pstr = [str(x) for x in [hash] + list(aux[i])]
            pstr += [hostname]
            fh.write(','.join(pstr) + '\n')
        fh.close()

    if args.shuffle: np.random.shuffle(params)

    if cfg.nprocs[0] == 1:
        [ run_one_experiment_given_list(p) for p in params ]
    else:
        pool = Pool(cfg.nprocs[0])
        pool.map(run_one_experiment_given_list, params)

if __name__ == "__main__":
    main()
