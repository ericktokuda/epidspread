#!/usr/bin/env python
""" Simulation of two dynamics: mobility and infection over a lattice
"""

import argparse
import logging
import os, sys, string, random, math, time, socket
from os.path import join as pjoin
from logging import debug, info
from itertools import product
from pathlib import Path

import igraph
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
from datetime import datetime
from multiprocessing import Pool
import pickle as pkl
import scipy
import scipy.optimize
from optimized import step_mobility, step_transmission, generate_waxman_adj
from optimized import get_matrix_index_from_triu, get_linear_index_from_triu
from optimized import update_contacts_list

from make_rw_matrix import calc_rw_transition_matrix # PauloCVS
from make_rw_matrix import calc_matrix_leading_eigenvector
from make_rw_matrix import distribute_agents_by_weights

########################################################## Defines
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
EPSILON = 1E-5
MAX = sys.maxsize
MAXITERS = 100000

#############################################################
def get_4connected_neighbours_2d(i, j, n, thoroidal=False):
    """Get the indices of the 4-connected neighbours of element (i, j). It does not check if there are repeated entries (2x2 or 1x1)"""
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
def fast_random_choice(lst, probs, randnum):
    return lst[np.searchsorted(probs.cumsum(), randnum)]

#############################################################
def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

#############################################################
def generate_lattice(n, thoroidal=False, s=10):
    """Generate 2d lattice of side @n. Returns a tuple containing the positions and the adjacency matrix """
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
def get_rgg_params(nvertices, avgdegree):
    rggcatalog = {
        '625,6': 0.056865545,
        '10000,6': 0.0139,
        '22500,6': 0.00925,
    }

    if '{},{}'.format(nvertices, avgdegree) in rggcatalog.keys():
        return rggcatalog['{},{}'.format(nvertices, avgdegree)]

    def f(r):
        g = igraph.Graph.GRG(nvertices, r)
        return np.mean(g.degree()) - avgdegree

    return scipy.optimize.brentq(f, 0.0001, 10000)

#############################################################
def generate_waxman(n, maxnedges, alpha, beta, domain=(0, 0, 1, 1)):
    adjlist, x, y = generate_waxman_adj(n, maxnedges, alpha, beta,
                                        domain[0], domain[1], domain[2], domain[3])
    adjlist = adjlist.astype(int).tolist()

    g = igraph.Graph(n, adjlist)
    g.vs['x'] = x
    g.vs['y'] = y
    return g

#############################################################
def get_waxman_params(nvertices, avgdegree, alpha, wxparamspath):
    maxnedges = nvertices * nvertices // 2
    if os.path.exists(wxparamspath):
        wxparams = pd.read_csv(wxparamspath)
        row = wxparams[(wxparams.nvertices == nvertices) & \
                       (wxparams.avgdegree == avgdegree) & \
                       (wxparams.alpha == alpha) ]
        if len(row) == 1:
            return row.beta.values[0], alpha

    def f(b):
        g = generate_waxman(nvertices, maxnedges, alpha=alpha, beta=b)
        return np.mean(g.degree()) - avgdegree

    beta = scipy.optimize.brentq(f, 0.0001, 1000, xtol=0.00001, rtol=0.01)
    return beta, alpha

#############################################################
def generate_graph(topologymodel, nvertices, avgdegree, latticethoroidal,
                   baoutpref, wsrewiring, wxalpha, expidx,
                   graphseed, wxparamspath, tmpdir):
    """Generate graph with given topology

    Args:
    graphsize(int): number of vertices
    graphtopology(str): topology, consult igraph layout options
    graphparam1, graphparam2, graphparam3: topology options

    Returns:
    igraph.Graph, np.ndarray: graph and the layout
    """

    info('exp:{} Generating graph with topology {}...'.format(expidx, topologymodel))
    if topologymodel == 'la':
        mapside = int(np.sqrt(nvertices))
        g = igraph.Graph.Lattice([mapside, mapside], nei=1, circular=latticethoroidal)
    elif topologymodel == 'er':
        erdosprob = avgdegree / nvertices
        if erdosprob > 1: erdosprob = 1
        g = igraph.Graph.Erdos_Renyi(nvertices, erdosprob)
    elif topologymodel == 'ba':
        m = round(avgdegree/2)
        if m == 0: m = 1
        g = igraph.Graph.Barabasi(nvertices, m)
    elif topologymodel == 'ws':
        mapside = int(np.sqrt(nvertices))
        m = round(avgdegree/2)
        g = igraph.Graph.Lattice([mapside, mapside], nei=1,
                                 circular=False)
        g.rewire_edges(wsrewiring)
    elif topologymodel == 'gr':
        radius = get_rgg_params(nvertices, avgdegree)
        g = igraph.Graph.GRG(nvertices, radius)
    elif topologymodel == 'wx':
        bufwaxmanpath = pjoin(tmpdir, 'waxman_{:02d}_{:01.4f}_{:02d}.pkl'.\
                              format(avgdegree, wxalpha, graphseed))
        try:
            with open(bufwaxmanpath, 'rb') as fh:
                g = pkl.load(fh)
        except:
            beta, alpha = get_waxman_params(nvertices, avgdegree, wxalpha, wxparamspath)
            maxnedges = nvertices * nvertices // 2
            g = generate_waxman(nvertices, maxnedges, beta=beta, alpha=alpha)
            with open(bufwaxmanpath, 'wb') as fh:
                pkl.dump(g, fh)

    g = g.clusters().giant()

    if topologymodel in ['gr', 'wx']:
        aux = np.array([ [g.vs['x'][i], g.vs['y'][i]] for i in range(g.vcount()) ])
        # layoutmodel = 'grid'
    else:
        if topologymodel in ['la', 'ws']:
            layoutmodel = 'grid'
        else:
            layoutmodel = 'random'
        aux = np.array(g.layout(layoutmodel).coords)
    # coords = (aux - np.mean(aux, 0))/np.std(aux, 0) # standardization
    coords = -1 + 2*(aux - np.min(aux, 0))/(np.max(aux, 0)-np.min(aux, 0)) # minmax
    return g, coords

##########################################################
def copy_experiment_config(cfgdf, outjsonpath, expidx):
    """Copy @configs

    Args:
    cfgdf(pd.DataFrame): dataframe with the index column as field name and the
    data column containing the value
    expidx(int): experiment index
    """
    info('exp:{} Copying config file ...'.format(expidx))
    for k in cfgdf['data'].keys():
        cfgdf['data'][k] = [cfgdf['data'][k]]
    cfgdf['data'].to_json(outjsonpath, force_ascii=False)

##########################################################
def generate_distribution_of_status(N, s0, i0, expidx):
    """Generate a random distribution of status according with @s0 susceptibles
    and @i0 infected

    Args:
    s0(int): number of susceptibles
    i0(int): number of infected
    expidx(int): experiment index

    Returns:
    status(np.ndarray): array with length @N and values corresponding to the status
    """

    info('exp:{} Generating random distribution of S, I, R ...'.format(expidx))
    status = np.ndarray(N, dtype=int)
    status[0: s0] = SUSCEPTIBLE
    status[s0:] = INFECTED
    np.random.shuffle(status)
    return status

##########################################################
def define_plot_layout(mapside, plotzoom, expidx):
    """Create a dict containing the layout settings"""
    # Square of the center surrounded by radius 3
    #  (equiv to 99.7% of the points of a gaussian)
    visual = dict(
        bbox = (mapside*10*plotzoom, mapside*10*plotzoom),
        margin = mapside*plotzoom,
        vertex_size = 5*plotzoom,
        vertex_shape = 'circle',
        # vertex_frame_width = 0
        vertex_frame_width = 0.1*plotzoom,
        edge_width=1.0
    )
    return visual

##########################################################
def distribute_agents(nvertices, nagents, expidx):
    """Initialize the location of the agents. The nagents per vertex is random but
    the agents id is NOT. The ids of a vertex will be all sequential

    Args:
    nvertices(int): number of vertices in the map
    nagents(int): number of agents
    expidx(int): experiment index

    Returns:
    list of list: each element corresponds to a vertex and contains the indices of the
    vertices
    """

    info('exp:{} Distributing agents in the map...'.format(expidx))
    nparticles = np.ndarray(nvertices, dtype=int)
    aux = np.random.rand(nvertices) # Uniform distrib
    nparticles = np.floor(aux / np.sum(aux) *nagents).astype(int)

    diff = nagents - np.sum(nparticles) # Correct rounding differences on the final number
    for i in range(np.abs(diff)):
        idx = np.random.randint(nvertices)
        nparticles[idx] += np.sign(diff) # Initialize number of particles per vertex

    particles = [None]*nvertices # Initialize indices of particles per vertex

    particlesidx = 0
    for i in range(nvertices):
        particles[i] = list(range(particlesidx, particlesidx+nparticles[i]))
        particlesidx += nparticles[i]

    return particles

##########################################################
def export_map(coords, gradients, mappath, expidx):
    """Export the map along with the gradient map

    Args:
    coords(np.ndarray(nnodes, 2)): coordinates of each node
    gradients(np.ndarray(nnodes,)): gradient of each node
    mappath(str): output path
    expidx(int): experiment index
    """

    info('exp:{} Exporting relief map...'.format(expidx))
    aux = pd.DataFrame()
    aux['x'] = coords[:, 0]
    aux['y'] = coords[:, 1]
    aux['gradient'] = gradients
    aux.to_csv(mappath, index=False, header=['x', 'y', 'gradient'])


##########################################################
def plot_gradients(g, coords, gradiestsrasterpath, visualorig, plotalpha):
    """Plot the gradients map

    Args:
    g(igraph.Graph): graph
    outgradientspath(str): output path
    visual(dict): parameters of the layout of the igraph plot
    """
    visual = visualorig.copy()
    aux = np.sum(g.vs['gradient'])
    gradientscolors = [ [c, c, c, plotalpha] for c in g.vs['gradient']]
    gradsum = float(np.sum(g.vs['gradient']))
    gradientslabels = [ '{:2.3f}'.format(x/gradsum) for x in g.vs['gradient']]
    visual['edge_width'] = 0

    igraph.plot(g, target=gradiestsrasterpath, layout=coords.tolist(),
                vertex_color=gradientscolors,
                vertex_label=[str(x) for x in range(g.vcount())],
                **visual)

##########################################################
def plot_topology(g, coords, toprasterpath, visualorig, plotalpha):
    """Plot the gradients map

    Args:
    g(igraph.Graph): graph
    outgradientspath(str): output path
    visual(dict): parameters of the layout of the igraph plot
    """

    visual = visualorig.copy()
    visual['vertex_size'] = 0
    gradientscolors = [1, 1, 1]
    gradsum = float(np.sum(g.vs['gradient']))
    gradientslabels = [ '{:2.3f}'.format(x/gradsum) for x in g.vs['gradient']]
    igraph.plot(g, target=toprasterpath, layout=coords.tolist(),
                vertex_color=gradientscolors, **visual)

##########################################################
def delete_individual_frames(outdir):
    """Delete individual frames in @outdir"""
    cmd = "rm {}/concat*.png".format(outdir)
    print(cmd)
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    print(stderr)

##########################################################
def export_summaries(ntransmpervertex, ntransmpervertexpath, transmstep,
                     ntransmpath, elapsed, statuscountsum, nparticlesstds,
                     lastepoch, mobstep, ncomponents, nvertices, nedges,
                     coordsrms, avgpathlen, sirplotpath, summarypath, expidx):
    aux = pd.DataFrame(ntransmpervertex)
    aux.to_csv(ntransmpervertexpath, index=False, header=['ntransmission'])

    outdf = pd.DataFrame({
        'transmstep': transmstep.astype(int),
        'S': statuscountsum[:, 0],
        'I': statuscountsum[:, 1],
        'nparticlesstd': nparticlesstds
    })
    outdf.to_csv(ntransmpath, index=True, index_label='t')

    # Plot SIR over time
    info('exp:{} Generating plots for counts of S, I'.format(expidx))
    fig, ax = plt.subplots(1, 1)
    plot_sis(statuscountsum[:, 0], statuscountsum[:, 1], fig, ax, sirplotpath)

    info('exp:{} Elapsed time: {:.2f}min'.format(expidx, elapsed/60))
    summary = dict(
        server = socket.gethostname(),
        elapsed = '{:.2f}'.format(elapsed),
        nsteps = lastepoch,
        stepsmobility = np.sum(mobstep),
        ncomponents = ncomponents,
        nvertices = nvertices,
        nedges = nedges,
        coordsrms = coordsrms,
        avgpathlen = avgpathlen,
    )
    with open(summarypath, 'w') as fh:
        fh.write(','.join(summary.keys()) + '\n')
        fh.write(','.join(str(x) for x in summary.values()))

##########################################################
def run_experiment_given_list(cfg):
    """Execute an experiment given the parameters defined in the @cfg dict."""

    plotalpha = .9
    DELAYTIME = 3600
    store_count_per_vertex = True

    t0 = time.time()
    cfgdf = pd.DataFrame.from_dict(cfg, 'index', columns=['data'])

    # Local vars
    outdir = cfg['outdir']
    wxparamspath= cfg['wxparamspath']
    nvertices = cfg['nvertices']
    # nagents   = cfg['nagentspervertex'] * nvertices
    topologymodel = cfg['topologymodel']
    avgdegree = cfg['avgdegree']
    latticethoroidal = cfg['lathoroidal']
    baoutpref = cfg['baoutpref']
    wsrewiring = cfg['wsrewiring']
    wxalpha = cfg['wxalpha']
    mobilityratio   = cfg['mobilityratio']
    nepochs   = cfg['nepochs']
    beta      = cfg['beta']
    gamma     = cfg['gamma']
    ngaussians = cfg['ngaussians']
    gaussianstd = cfg['gaussianstd']
    gaussianpower= cfg['gaussianpower']
    plotzoom  = cfg['plotzoom']
    plotrate  = cfg['plotrate']
    nprocs    = cfg['nprocs']
    randomseed= cfg['randomseed']
    expidx = cfg['expidx']

    outdir = pjoin(outdir, expidx)
    ntransmpath = pjoin(outdir, 'ntransmperepoch.csv') # Stats per epoch
    summarypath = pjoin(outdir, 'summary.csv') # General info from the run
    runningpath = pjoin(outdir, 'RUNNING') # Lock file
    outjsonpath = pjoin(outdir, 'config.json')
    mappath = pjoin(outdir, 'attraction.csv')
    animationpath = pjoin(outdir, 'animation.mp4')
    ntransmpervertexpath = pjoin(outdir, 'ntransmpervertex.csv')
    gradsrasterpath = pjoin(outdir, 'gradients.png')
    toporasterpath = pjoin(outdir, 'topology.png')
    sirplotpath = pjoin(outdir, 'sir.png')

    if os.path.exists(summarypath):
        return
    elif os.path.exists(runningpath):
        startedtime = float(open(runningpath).read().strip())
        if (time.time() - startedtime) < DELAYTIME:
            info('Skipping {} (recently started)'.format(expidx))
            return

    os.makedirs(outdir, exist_ok=True) # Create outdir
    open(runningpath, 'w').write(str(time.time()))

    copy_experiment_config(cfgdf, outjsonpath, expidx)

    mapside = int(np.sqrt(nvertices))
    istoroid = latticethoroidal

    visual = define_plot_layout(mapside, plotzoom, expidx)

    if mobilityratio == -1: # Steps occur in parallel
        transmstep = np.ones(MAXITERS, dtype=bool)
        mobstep = np.ones(MAXITERS, dtype=bool)
    else: # They occur in an interleaved way
        transmstep = np.zeros(MAXITERS, dtype=bool)
        mobstep = np.zeros(MAXITERS, dtype=bool)

    graphseed = 0 # Force graphs and gradients to always be the same
    np.random.seed(graphseed); random.seed(graphseed)
    g, coords = generate_graph(topologymodel, nvertices, avgdegree,
                               latticethoroidal, baoutpref, wsrewiring, wxalpha,
                               expidx, graphseed,
                               cfg['wxparamspath'], cfg['outdir'])

    g = initialize_gradients(g, coords, ngaussians, gaussianstd, gaussianpower)

    np.random.seed(randomseed); random.seed(randomseed)

    nvertices = g.vcount()
    nedges = g.ecount()
    avgpathlen = g.average_path_length(directed=False, unconn=True)
    coordsrms = np.sqrt(np.mean(np.square(coords)))

    nagents   = int(cfg['nagentspervertex'] * nvertices)
    s0        = int(nagents*cfg['s0'])
    i0        = int(nagents*cfg['i0'])

    status = generate_distribution_of_status(nagents, s0, i0, expidx)

    statuscountperepoch = np.zeros((MAXITERS, 2), dtype=int)
    statuscountperepoch[0, :] = np.array([s0, i0])

    # visualize_static_graph_layouts(g, 'config/layouts_lattice.txt', outdir);

    ntransmpervertex = np.zeros(nvertices, dtype=int)

    rw_transmat = calc_rw_transition_matrix(g)
    node_probabs = calc_matrix_leading_eigenvector(rw_transmat, sparse=True)
    particles = distribute_agents_by_weights(nvertices, nagents, expidx,
                                             weights=node_probabs)
    nparticlesstds = np.zeros((MAXITERS,), dtype=float)


    export_map(coords, g.vs['gradient'], mappath, expidx)
    g.write_adjacency(pjoin(outdir, 'graph.csv'), sep=',')

    plot_gradients(g, coords, gradsrasterpath, visual, plotalpha)
    plot_topology(g, coords, toporasterpath, visual, plotalpha)
    statuscountpervertex  = sum_status_per_vertex(status, particles, nvertices, )
    visual["edge_width"] = 0.0

    maxepoch = nepochs if nepochs > 0 else MAXITERS
    transmstep[0] = 0; mobstep[0] = 0 # Nobody either move or transmit in epoch 0

    if store_count_per_vertex:
        statuscountpervertexall = - np.ones((maxepoch, nvertices, 2), dtype=int)
        # particlpervertex[0, :] = [len(x) for x in particles]

    for ep in range(1, maxepoch):
        lastepoch = ep

        if plotrate > 0 and ep % plotrate == 0:
            plot_epoch_graphs(ep-1, g, coords, visual, status, nvertices, particles,
                              nagents, statuscountpervertex[:, 0],
                              statuscountpervertex[:, 1],
                              outdir, expidx)

        if ep % 10 == 0: info('exp:{}, t:{}'.format(expidx, ep))

        nparticlesstds[ep] = np.std([len(x) for x in particles])

        if store_count_per_vertex:
            statuscountpervertexall[ep - 1, :, :] = statuscountpervertex

        # if mobilityratio == -1 or np.random.random() < mobilityratio:
        if mobilityratio == -1 or ep % mobilityratio == 0:
            particles = step_mobility(g, particles, nagents)

            if mobilityratio != -1: # If interleaved steps
                # Keep the prev. ep value
                statuscountperepoch[ep, :] = statuscountperepoch[ep-1, :]
                statuscountpervertex = sum_status_per_vertex(status, particles,
                                                             nvertices)
                mobstep[ep] = 1
                continue # Do NOT transmit in this step

        transmstep[ep] = 1
        status, newtransm = step_transmission(nvertices, status, beta, gamma, particles)
        status = np.asarray(status)
        ntransmpervertex += newtransm

        statuscountpervertex = sum_status_per_vertex(status, particles, nvertices)
        statuscountperepoch[ep, :] = np.sum(statuscountpervertex, 0)

        if nepochs == -1 and np.sum(status==INFECTED) == 0: break

    # if plotrate > 0 and os.path.exists('/usr/bin/ffmpeg'):
            # generate_plots_animation(outdir, animationpath)
            # delete_individual_frames(outdir)

    statuscountperepoch = statuscountperepoch[:lastepoch+1, :]
    transmstep = transmstep[:lastepoch+1]
    mobstep = mobstep[:lastepoch+1]
    nparticlesstds = nparticlesstds[:lastepoch+1]

    elapsed = time.time() - t0

    if store_count_per_vertex:
        statuscountpervertexall[-1, :, :] = statuscountpervertex

        cols = ['v{:03d}'.format(x) for x in range(nvertices)]
        myidx = ['ep{:03d}'.format(x) for x in range(maxepoch)]

        for j, stat in enumerate(['S', 'I']):
            countsdf = pd.DataFrame(statuscountpervertexall[:, :, j], columns=cols,
                                    index=myidx)
            countsdf.to_csv(pjoin(outdir, 'count{}pervertex.csv'.format(stat)))

    export_summaries(ntransmpervertex, ntransmpervertexpath, transmstep, ntransmpath,
                     elapsed, statuscountperepoch, nparticlesstds, lastepoch, mobstep,
                     len(g.components()), nvertices, nedges, coordsrms, avgpathlen,
                     sirplotpath, summarypath, expidx)

    os.remove(runningpath) # Remove lock
    info('exp:{} Finished. Results are in {}'.format(expidx, outdir))

########################################################## Plot SIR over time
def visualize_static_graph_layouts(g, layoutspath, outdir, plotalpha=.9):
    layouts = [line.rstrip('\n') for line in open(layoutspath)]
    print(layouts)
    for l in layouts:
        info(l)
        try:
            igraph.plot(g, target=pjoin(outdir, l + '.png'),
                        layout=g.layout(l),
                        bbox=(1200,1200),
                        vertex_frame_width=0,
                        # vertex_color=[.5, .5, .5, plotalpha],
                        vertex_color='gray',
                        )
                        # vertex_label=list(range(g.vcount())))
        except Exception as e:
            print('Error generating {}'.format(l))
            pass

########################################################## Distrib. of gradients
def initialize_gradients_peak(g):
    """Initizalition of gradients with a peak in the 0-th vertex. Returns the graph with updated gradients"""
    g.vs['gradient'] = 0.1
    g.vs[0]['gradient'] = 1
    return g

##########################################################
def multivariate_normal(x, mean, cov):
    """P.d.f. of the multivariate normal when the covariance matrix is positive definite.  Source: wikipedia"""
    ndims = len(mean)
    B = x - mean
    return (1. / (np.sqrt((2 * np.pi)**ndims * np.linalg.det(cov))) *
            np.exp(-0.5*(np.linalg.solve(cov, B).T.dot(B))))

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
    """Initizalition of gradients with a gaussian *not* considering the location. Returns the an igraph instance with updated gradients """

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
    """Initizalition of gradients with a gaussian considering the location.
    Returns the an igraph instance with updated gradients """

    for i, v in enumerate(g.vs):
        g.vs[i]['gradient'] = multivariate_normal(coords[i, :], mu, cov)

    return g

##########################################################
def gradient_to_the_power(g, power):
    """ Returns the an igraph instance with updated gradients """
    grads = np.array(g.vs['gradient']) ** power
    g.vs['gradient'] = grads.tolist()
    return g

##########################################################
def initialize_gradients(g, coords, ngaussians, sigma, gradpower):
    """Initialize gradients with some distribution """

    if ngaussians == 0 or sigma > 998:
        g.vs['gradient'] = 0.1
        return g

    # mu = (np.max(coords, 0) + np.min(coords, 0)) / 2
    mu = np.random.rand(2) * 2 - 0.9999 # Not 1 because rand includes 0

    cov = np.eye(2) * sigma
    g = initialize_gradients_gaussian(g, coords, mu, cov)
    g = gradient_to_the_power(g, gradpower)
    return g

##########################################################
def sum_status_per_vertex(status, particles, nvertices, nclasses=2):
    """Compute the sum of each status

    Args:
    status(np.ndarray): size nparticlesx1, with an int corresponding to the status
    particles(list of list): list of ids of the neighbour particles
    nvertices(int): number of vertices of the map
    totalnsusceptibles(int): number of vertices of the map

    Returns:
    dist(np.ndarray(nvertices, 3)): number of susceptibles, infected and recovered per vertex
    nrecovered(list of int): number of recovered per vertex
    """

    dist = np.zeros((nvertices, nclasses))
    for i in range(nvertices):
        dist[i, :] = np.bincount(status[particles[i]], minlength=nclasses)

    return dist

##########################################################
def plot_epoch_graphs(ep, g, coords, visual, status, nvertices, particles,
                      N, nsusceptibles, ninfected,
                      outdir, expidx, plotalpha=.9):
    info('exp:{} Generating plots'.format(expidx))
    susceptiblecolor = []
    infectedcolor = []

    for z in nsusceptibles:
        zz = [0, 1, 0, math.log(z, N) + 0.2] if z*N > 1 else [0, 0, 0, 0] # Bug on log(1,1)
        susceptiblecolor.append(zz)
    for z in ninfected:
        zz = [1, 0, 0, math.log(z, N) + 0.2] if z*N > 1 else [0, 0, 0, 0] # Bug on log(1,1)
        infectedcolor.append(zz)

    outsusceptiblepath = pjoin(outdir, 'susceptible{:02d}.png'.format(ep))
    outinfectedpath = pjoin(outdir, 'infected{:02d}.png'.format(ep))

    igraph.plot(g, target=outsusceptiblepath, layout=coords.tolist(),
                vertex_color=susceptiblecolor, **visual)
    igraph.plot(g, target=outinfectedpath, layout=coords.tolist(),
                vertex_color=infectedcolor, **visual)

    outconcatpath = pjoin(outdir, 'concat{:02d}.png'.format(ep))
    proc = Popen('convert {} {} +append {}'.format(outsusceptiblepath,
        outinfectedpath, outconcatpath),
        shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

    # Delete individual files
    proc = Popen('rm {} {}'.format(outsusceptiblepath, outinfectedpath,
                                         ),
                 shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()

##########################################################
def plot_sis(s, i, fig, ax, sirpath):
    ax.plot(s, 'g', label='Susceptibles')
    ax.plot(i, 'r', label='Infected')
    ax.legend()
    fig.savefig(sirpath)

##########################################################
def random_string(length=8):
    """Generate a random string of fixed length """
    letters = np.array(list(string.ascii_lowercase + string.digits))
    return ''.join(np.random.choice(letters, size=length))

##########################################################
def generate_params_combinations(origcfg):
    """Generate a random string of fixed length. It is dependent on the order of
    the columns in the dataframe"""
    cfg = origcfg.copy()
    cfg.lathoroidal = [-1]
    cfg.baoutpref = [-1]
    cfg.wsrewiring = [-1]
    cfg.wxalpha = [-1]

    params = []
    if 'la' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['la']
        aux.lathoroidal = origcfg.lathoroidal
        aux.avgdegree = [4]
        params += list(product(*aux))

    if 'er' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['er']
        params += list(product(*aux))

    if 'ba' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['ba']
        aux['baoutpref'] = origcfg.baoutpref
        params += list(product(*aux))

    if 'ws' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['ws']
        aux['wsrewiring'] = origcfg.wsrewiring
        params += list(product(*aux))

    if 'gr' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['gr']
        params += list(product(*aux))

    if 'wx' in cfg.topologymodel:
        aux = cfg.copy()
        aux.topologymodel = ['wx']
        aux['wxalpha'] = origcfg.wxalpha
        params += list(product(*aux))

    return params

##########################################################
def convert_list_to_df(mylist):
    """Convert list to pd.DataFrame and assign expidx if it does not exist"""

    hashsz = 8
    for i in range(len(mylist)):
        while True:
            hash = random_string(hashsz)
            if hash not in hashes: break
        hashes.append(hash)

        param = {}
        for j, key in enumerate(cfgkeys):
            param[key] = mylist[i][j]
        param['expidx'] = hash
    return param
        # params.append(param)
        # pstr = [str(x) for x in [hash] + list(mylist[i])]
        # fh.write(','.join(pstr) + '\n')

##########################################################
def load_df_from_json(myjson):
    """Load a pandas dataframe from a cfg file"""
    aux = generate_params_combinations(myjson)
    nrows = len(aux)
    colnames = list(myjson.keys())
    df = pd.DataFrame(index=np.arange(0, nrows), columns=colnames)

    for i in np.arange(0, nrows):
        df.loc[i] = aux[i]

    return df

##########################################################
def prepend_random_ids_columns(df):
    n = df.shape[0]
    hashsz = 8

    hashes = []
    for i in range(n):
        while True:
            hash = random_string(hashsz)
            if hash not in hashes: break
        hashes.append(hash)

    df.insert(0, 'expidx', hashes)
    return df

##########################################################
def get_experiments_table(configpath, expspath):
    """Merge requested experiments from @configpath and already executed ones
    (@expspath)

    Args:
    configpath(str): path to the config file in json format
    expspath(str): path to the exps file in csv format

    Returns:
    pd.DataFrame: merged experiments directives
    """
    configdf = load_df_from_json(configpath)
    cols = configdf.columns.tolist()

    # if not 'expidx' in configdf.columns:
    configdf = prepend_random_ids_columns(configdf)
    expsdf = configdf
    if os.path.exists(expspath):
        try:
            loadeddf = pd.read_csv(expspath)
            aux = pd.concat([loadeddf, configdf], sort=False, ignore_index=True)
            cols.remove('outdir')
            cols.remove('wxparamspath')
            cols.remove('nprocs')
            expsdf = aux.drop_duplicates(cols, keep='first')
            expsdf = expsdf.assign(outdir = configdf.outdir[0])
            expsdf = expsdf.assign(wxparamspath = configdf.wxparamspath[0])
            expsdf = expsdf.assign(nprocs = configdf.nprocs[0])
        except Exception as e:
            info('Error occurred when merging exps')
            info(e)
            expsdf = configdf

    expsdf.set_index('expidx')
    if not os.path.exists(expspath) or len(loadeddf) != len(expsdf):
        rewriteexps = True
    else: rewriteexps = False
    return expsdf, rewriteexps

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Config file')
    parser.add_argument('--continue_', action='store_true', help='Continue execution')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffled traversing of config parameters')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.INFO)

    cfg = pd.read_json(args.config, typ='series', precise_float=True) # Load config

    outdir = cfg.outdir[0]

    existing = os.path.exists(outdir)

    if existing and not args.continue_:
        print('Folder {} exists. Change the outdir parameter or use --continue_'. \
              format(outdir))
        return

    os.makedirs(outdir, exist_ok=True)
    cfg.outdir = [outdir]

    expspath = pjoin(outdir, 'exps.csv')
    expsdf, rewriteexps = get_experiments_table(cfg, expspath)

    if os.path.exists(expspath) and rewriteexps:
        os.rename(expspath, expspath.replace('exps.csv', 'exps_orig.csv'))

    if not os.path.exists(expspath) or rewriteexps:
        expsdf.drop(columns=['outdir', 'nprocs']).to_csv(expspath, index=False)

    params = expsdf.to_dict(orient='records')
    if args.shuffle: np.random.shuffle(params)

    if cfg.nprocs[0] == 1:
        [ run_experiment_given_list(p) for p in params ]
    else:
        info('Running in parallel ({})'.format(cfg.nprocs[0]))
        pool = Pool(cfg.nprocs[0])
        pool.map(run_experiment_given_list, params)

##########################################################
if __name__ == "__main__":
    main()
