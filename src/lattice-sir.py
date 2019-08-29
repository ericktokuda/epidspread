#!/usr/bin/env python3
"""SIR model on a lattice
"""

import argparse
import logging
from logging import debug
import numpy as np
import igraph
import pandas as pd
import pickle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    dim = [2, 4]
    g = igraph.Graph.Lattice(dim, nei=1, directed=False, mutual=True, circular=True)
    for v in g.vs:
        r = np.random.rand()
        v['nparticles'] = round(r*10)
        mycolor = list([0,0,r])
        v['color'] = mycolor
    igraph.plot(g, )
if __name__ == "__main__":
    main()

