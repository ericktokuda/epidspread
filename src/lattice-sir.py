#!/usr/bin/env python
""" Simulation of two dynamics: mobility and infection over a lattice
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug

import igraph
import networkx as nx
import numpy as np
import pandas as pd
import copy
from matplotlib import cm
from matplotlib import pyplot as plt
import math
from subprocess import Popen, PIPE
import json

########################################################## Defines
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

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
        gradients = g.vs[neighids]['gradient']
        gradients /= np.sum(gradients)

        for j, partic in enumerate(particles_fixed[i]): # For each particle in this vertex
            if np.random.rand() <= autoloop_prob: continue
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
        susceptible = statuses[statuses==SUSCEPTIBLE]            
        infected = statuses[statuses==INFECTED]            
        recovered = statuses[statuses==RECOVERED]
        numnewinfected = round(beta * len(susceptible) * len(infected))
        numnewrecovered = round(gamma*len(infected))
        indsusceptible = np.where(statuses_fixed==SUSCEPTIBLE)[0]
        indinfected = np.where(statuses_fixed==INFECTED)[0]
        indrecovered = np.where(statuses_fixed==RECOVERED)[0]

        if numnewinfected > 0: status[indsusceptible[0:numnewinfected]] = INFECTED  # I don't like this "if"
        if numnewrecovered > 0: status[indinfected[0:numnewrecovered]] = RECOVERED
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
                      N, b, outgradientspath, nsusceptibles, ninfected, nrecovered, totalnsusceptibles,
                      totalninfected, totalnrecovered, outdir):
    susceptiblecolor = []
    infectedcolor = []
    recoveredcolor = []
    
    for z in nsusceptibles:
        zz = [math.log(z, b)/math.log(N, b), 0, 0] if z != 0 else [0, 0, 0]
        susceptiblecolor.append(zz)
    for z in ninfected:
        zz = [0, math.log(z, b)/math.log(N, b), 0] if z != 0 else [0, 0, 0]
        infectedcolor.append(zz)
    for z in nrecovered:
        zz = [0, 0, math.log(z, b)/math.log(N, b)] if z != 0 else [0, 0, 0]
        recoveredcolor.append(zz)  
        
    outsusceptiblepath = pjoin(outdir, 'susceptible{:02d}.png'.format(ep+1))
    igraph.plot(g, target=outsusceptiblepath, layout=layout, vertex_label=nsusceptibles,
                vertex_label_color='white', vertex_color=susceptiblecolor, **visual)      

    outinfectedpath = pjoin(outdir, 'infected{:02d}.png'.format(ep+1))
    igraph.plot(g, target=outinfectedpath, layout=layout, vertex_label=ninfected,
                vertex_color=infectedcolor, vertex_label_color='white', **visual)      

    outrecoveredpath = pjoin(outdir, 'recovered{:02d}.png'.format(ep+1))
    igraph.plot(g, target=outrecoveredpath, layout=layout, vertex_label=nrecovered,
                vertex_color=recoveredcolor, vertex_label_color='white', **visual)      

    outconcatpath = pjoin(outdir, 'concat{:02d}.png'.format(ep+1))
    proc = Popen('convert {} {} {} {} +append {}'.format(outgradientspath,
                                                         outsusceptiblepath,
                                                         outinfectedpath,
                                                         outrecoveredpath,
                                                         outconcatpath),
                 shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/toy01.json')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)


    ########################################################## PARAMETERS
    with open(args.config) as fh:
        cfg = json.load(fh)

    mapw = cfg['mapw']
    maph = cfg['maph']
    nei = cfg['nei']
    istoroid = cfg['istoroid']
    nepochs = cfg['nepochs']
    s0 = cfg['s0']
    i0 = cfg['i0']
    r0 = cfg['r0']
    beta = cfg['beta']
    gamma = cfg['gamma']
    autoloop_prob = cfg['autoloop_prob']
    plotw = cfg['plotw']
    ploth = cfg['ploth']
    plotmargin = cfg['plotmargin']
    plotvsize = cfg['plotvsize']
    plotlayout = cfg['plotlayout']
    plotrate = cfg['plotrate']
    outdir = cfg['outdir']

    ########################################################## Load config
    dim = [mapw, maph]
    N = s0 + i0 + r0
    nvertices = mapw*maph # square lattice
    status = np.ndarray(N, dtype=int)
    status[0: s0] = SUSCEPTIBLE
    status[s0:s0+i0] = INFECTED
    status[s0+i0:] = RECOVERED
    np.random.shuffle(status)

    visual = {}
    visual["bbox"] = (plotw, ploth)
    visual["margin"] = plotmargin
    visual["vertex_size"] = plotvsize

    totalnsusceptibles = [s0]
    totalninfected = [i0]
    totalnrecovered = [r0]

    g = igraph.Graph.Lattice(dim, nei, directed=False, mutual=True, circular=istoroid)
    layout = g.layout(plotlayout)

    ########################################################## Distrib. of particles
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

    ########################################################## Distrib. of gradients
    g.vs['gradient'] = 5
    g.vs[1]['gradient'] = 1
    g.vs[3]['gradient'] = 25

    ########################################################## Plot gradients
    maxgradients = np.max(g.vs['gradient'])
    gradientscolors = [[1, 1, 1]]*nvertices
    gradientslabels = [ str(x/50) for x in g.vs['gradient']]
    outgradientspath = pjoin(outdir, 'gradients.png')
    igraph.plot(g, target=outgradientspath, layout=layout,
                vertex_label=gradientslabels,
                vertex_color=gradientscolors, **visual)      

    b = 0.1 # For colors definition
    ########################################################## Plot epoch 0
    nsusceptibles, ninfected, nrecovered, \
        totalnsusceptibles, totalninfected, \
        totalnrecovered  = compute_statuses_sums(status, particles, nvertices,
                                                 totalnsusceptibles, totalninfected,
                                                 totalnrecovered)
    plot_epoch_graphs(-1, g, layout, visual, status, nvertices, particles,
                      N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                      totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    for ep in range(nepochs):
        particles = step_mobility(g, particles, autoloop_prob)
        status = step_transmission(g, status, beta, gamma, particles)
      
        nsusceptibles, ninfected, nrecovered, \
            totalnsusceptibles, totalninfected, \
            totalnrecovered  = compute_statuses_sums(status, particles, nvertices,
                                                     totalnsusceptibles, totalninfected,
                                                     totalnrecovered)
        if ep % plotrate == 0:
            plot_epoch_graphs(ep, g, layout, visual, status, nvertices, particles,
                              N, b, outgradientspath, nsusceptibles, ninfected, nrecovered,
                              totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    ########################################################## Enhance plots
    cmd = "mogrify -gravity south -pointsize 24 " "-annotate +50+0 'GRADIENT' " \
        "-annotate +350+0 'S' -annotate +650+0 'I' -annotate +950+0 'R' " \
        "{}/concat*.png".format(outdir)
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

    proc = Popen('convert -delay 120 -loop 0  /tmp/concat*.png /tmp/movie.gif',
                 shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    print(stderr)

    ########################################################## Plot SIR over time
    plt.plot(totalnsusceptibles, 'r', label='Susceptibles')
    plt.plot(totalninfected, 'g', label='Infected')
    plt.plot(totalnrecovered, 'b', label='Recovered')
    plt.legend()
    plt.savefig('/tmp/out.png')

if __name__ == "__main__":
    main()
