#!/usr/bin/env python
""" Simulation of two dynamics: mobility and infection over a lattice
"""

import argparse
import logging
import os, sys
from os.path import join as pjoin
from logging import debug, info

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


########################################################## Defines
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
EPSILON = 1E-5
MAX = sys.maxsize

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
    g.vs[0]['gradient'] = 100
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
def initialize_gradients_gaussian(g, mu=0, sigma=1):
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
def initialize_gradients(g, method='peak', mu=0, sigma=1):
    """Initialize gradients with some distribution

    Args:
    g(igraph.Graph): graph instance

    Returns:
    igraph.Graph: graph instance with attribute 'gradient' updated
    """

    g.vs['gradient'] = 10

    if method == 'uniform':
        return g
    if method == 'peak':
        return initialize_gradients_peak(g)
    elif method == 'gaussian':
        return initialize_gradients_gaussian(g, mu, sigma)

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

        status[indsusceptible[0:numnewinfected]] = INFECTED
        status[indinfected[0:numnewrecovered]] = RECOVERED
    return status

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
def plot_epoch_graphs(ep, g, layout, visual, status, nvertices, particles,
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

    igraph.plot(g, target=outsusceptiblepath, layout=layout, vertex_shape='rectangle', vertex_color=susceptiblecolor, vertex_frame_width=0.0, **visual)      
    igraph.plot(g, target=outinfectedpath, layout=layout, vertex_shape='rectangle', vertex_color=infectedcolor, vertex_frame_width=0.0, **visual)      
    igraph.plot(g, target=outrecoveredpath, layout=layout, vertex_shape='rectangle', vertex_color=recoveredcolor, vertex_frame_width=0.0, **visual)      

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

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/toy01.json')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    cfg = pd.read_json(args.config, typ='series') # Load config

    outdir = pjoin(cfg.outdir, datetime.now().strftime('%Y%m%d_%H%M') + '-latticesir')
    if os.path.exists(outdir):
        ans = input(outdir + ' exists. Do you want to continue? ')
        if ans.lower() not in ['y', 'yes']:
            info('Aborting')
            return
    else:
        os.mkdir(outdir)

    info('Copying config file ...')
    cfg.to_json(pjoin(outdir, os.path.basename(args.config)), force_ascii=False)
    dim = [cfg.mapside, cfg.mapside]
    N = cfg.s0 + cfg.i0 + cfg.r0
    nvertices = cfg.mapside**2 # square lattice
    status = np.ndarray(N, dtype=int)
    status[0: cfg.s0] = SUSCEPTIBLE
    status[cfg.s0:cfg.s0+cfg.i0] = INFECTED
    status[cfg.s0+cfg.i0:] = RECOVERED
    np.random.shuffle(status)
    info('Generated random distribution of S, I, R ...')

    visual = {}
    visual["bbox"] = (cfg.mapside*10*cfg.plotzoom, cfg.mapside*10*cfg.plotzoom)
    visual["margin"] = cfg.mapside*cfg.plotzoom
    visual["vertex_size"] = 10*cfg.plotzoom

    totalnsusceptibles = [cfg.s0]
    totalninfected = [cfg.i0]
    totalnrecovered = [cfg.r0]

    aux = '' if cfg.istoroid else 'non-'
    info('Generating {}toroidal lattice with dim ({}, {}) ...'.format(aux,
                                                                  cfg.mapside,
                                                                  cfg.mapside,
                                                                  ))
    g = igraph.Graph.Lattice(dim, cfg.nei, directed=False, mutual=True, circular=cfg.istoroid)

    # visualize_static_graph_layouts(g, 'config/layouts_lattice.txt', outdir);
    layout = g.layout(cfg.plotlayout)

    ########################################################## Distrib. of particles
    info('Generating uniform distribution of agents in the lattice ...')
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
    info('Initializing gradients distribution ...')
    g = initialize_gradients(g, cfg.graddist, cfg.gradmean, cfg.gradstd)

    ########################################################## Plot gradients
    if cfg.plotrate > 0:
        info('Generating plots for epoch 0')

        aux = np.sum(g.vs['gradient'])
        gradientscolors = [ [c, c, c] for c in g.vs['gradient']]
        # gradientscolors = [1, 1, 1]*g.vs['gradient']
        gradsum = float(np.sum(g.vs['gradient']))
        gradientslabels = [ '{:2.3f}'.format(x/gradsum) for x in g.vs['gradient']]
        outgradientspath = pjoin(outdir, 'gradients.png')
        igraph.plot(g, target=outgradientspath, layout=layout,
                    vertex_shape='rectangle', vertex_color=gradientscolors,
                    vertex_frame_width=0.0, **visual)      

        b = 0.1 # For colors definition
        ########################################################## Plot epoch 0
        nsusceptibles, ninfected, nrecovered, \
            _, _, _  = compute_statuses_sums(status, particles, nvertices, [], [], [])
        plot_epoch_graphs(-1, g, layout, visual, status, nvertices, particles,
                          N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                          totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    for ep in range(cfg.nepochs):
        if ep % 10 == 0:
            info('t={}'.format(ep))
        particles = step_mobility(g, particles, cfg.autoloop_prob)
        aux = np.std([len(x) for x in particles])
        nparticlesstds.append(aux)
        status = step_transmission(g, status, cfg.beta, cfg.gamma, particles)
      
        nsusceptibles, ninfected, nrecovered, \
            totalnsusceptibles, totalninfected, \
            totalnrecovered  = compute_statuses_sums(status, particles, nvertices,
                                                     totalnsusceptibles, totalninfected,
                                                     totalnrecovered)
        if cfg.plotrate > 0 and ep % cfg.plotrate == 0:
            plot_epoch_graphs(ep, g, layout, visual, status, nvertices, particles,
                              N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                              totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    ########################################################## Enhance plots
    if cfg.plotrate > 0:
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
    info('Exporting S, I, R data')
    aux = np.array([totalnsusceptibles, totalninfected, totalnrecovered, nparticlesstds]).T
    pd.DataFrame(aux).to_csv(pjoin(outdir, 'sir.csv'), header=['S', 'I', 'R', 'nparticlesstd'],
                             index=True, index_label='t')

    ########################################################## Plot SIR over time
    info('Generating plots for counts of S, I, R')
    fig, ax = plt.subplots(1, 1)
    plot_sir(totalnsusceptibles, totalninfected, totalnrecovered, fig, ax, outdir)
    info('Finished. Results are in {}'.format(outdir))

if __name__ == "__main__":
    main()
