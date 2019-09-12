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

########################################################## Defines
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

##########################################################
def compute_iteration(ep, g, layout, visual, status, nvertices, particles,
                      N, b, outattractivinesspath,
                      totalnsusceptibles, totalninfected, totalnrecovered, outdir):
    nsusceptibles = np.array([ np.sum(status[particles[i]]==SUSCEPTIBLE) for i in range(nvertices)] )
    ninfected = np.array([ np.sum(status[particles[i]]==INFECTED) for i in range(nvertices)] )
    nrecovered = np.array([ np.sum(status[particles[i]]==RECOVERED) for i in range(nvertices)] )
    totalnsusceptibles.append(np.sum(nsusceptibles))
    totalninfected.append(np.sum(ninfected))
    totalnrecovered.append(np.sum(nrecovered))
    
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
    proc = Popen('convert {} {} {} {} +append {}'.format(outattractivinesspath,
                                                         outsusceptiblepath,
                                                         outinfectedpath,
                                                         outrecoveredpath,
                                                         outconcatpath),
                 shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=False, default='/tmp',
                        help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    outdir = args.outdir

    ########################################################## PARAMETERS
    # TODO: move to external config file
    mapw = 3
    maph = 3
    dim = [mapw, maph]
    nei = 2
    istoroid = True
    nepochs = 100

    s0 = 500
    i0 = 5
    r0 = 10
    beta = 0.4
    gamma = 0.5

    autoloop_prob = 0.5 # probability of staying

    plotw, ploth = (300, 300)
    plotmargin = 20
    plotvsize = 40
    plotlayout = 'kk'

    ########################################################## Load config
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
    nparticles = np.ndarray(nvertices)
    for i in range(nvertices):
        nparticles[i] = np.random.randint(10) # Uniform distrib
        
    nparticles = (nparticles / (np.sum(nparticles))*N).astype(int)

    diff = N - np.sum(nparticles) # Correct rounding differences on the final number
    for i in range(np.abs(diff)):
        idx = np.random.randint(np.absolute(nvertices))
        nparticles[idx] += 1 * np.sign(diff)

    # Initialize distribution of particles (with status)
    particles = [None]*nvertices
    k = 0
    for i in range(nvertices):
        particles[i] = list(range(k, k+nparticles[i]))
        k += nparticles[i]

    ########################################################## Distrib. of gradients
    g.vs['gradient'] = 5
    g.vs[1]['gradient'] = 1
    g.vs[3]['gradient'] = 25

    ########################################################## Plot gradients
    maxattractiviness = np.max(g.vs['gradient'])
    attractivinesscolors = [[1, 1, 1]]*nvertices
    attractivinesslabels = [ str(x/50) for x in g.vs['gradient']]
    outattractivinesspath = pjoin(outdir, 'gradient.png')
    igraph.plot(g, target=outattractivinesspath, layout=layout,
                vertex_label=attractivinesslabels,
                vertex_color=attractivinesscolors, **visual)      

    b = 0.1 # For colors definition
    ########################################################## Plot epoch 0
    compute_iteration(-1, g, layout, visual, status, nvertices, particles,
                      N, b, outattractivinesspath,
                      totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    for ep in range(nepochs):
        ########################################################## Mobility
        particles_fixed = copy.deepcopy(particles) # copy to avoid being altered

        
        for i, _ in enumerate(g.vs): # For each vertex
            numvparticles = len(particles_fixed[i])
            neighids = g.neighbors(i)
            gradient = g.vs[neighids]['gradient']
            gradient /= np.sum(gradient)

            for j, partic in enumerate(particles_fixed[i]): # For each particle in this vertex
                if np.random.rand() <= autoloop_prob: continue
                neighid = np.random.choice(neighids, p=gradient)
                particles[i].remove(partic)
                particles[neighid].append(partic)
      
        ########################################################## Infection
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
       
        compute_iteration(ep, g, layout, visual, status, nvertices, particles,
                          N, b, outattractivinesspath,
                          totalnsusceptibles, totalninfected, totalnrecovered, outdir)

    ########################################################## Enhance plots
    cmd = "mogrify -gravity south -pointsize 24 " "-annotate +50+0 'GRADIENT' " \
        "-annotate +350+0 'S' -annotate +650+0 'I' -annotate +950+0 'R'" \
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
