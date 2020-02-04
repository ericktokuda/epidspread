#!/usr/bin/env python3
"""Re-generate exps.csv from individual experiments
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info
import pandas as pd
import os

def create_exps_from_folders(expsdir, dffolderspath):
    files = sorted(os.listdir(expsdir))

    df = pd.DataFrame()
    # for i, d in enumerate(files[:10]):
    for i, d in enumerate(files[:100]):
        dpath = pjoin(expsdir, d)
        if not os.path.isdir(dpath): continue
        expspath = pjoin(dpath, 'config.json')
        dfaux = pd.read_json(expspath)
        df = pd.concat([df, dfaux], axis=0, sort=False)
        if i % 100 == 0: info(i)

    df.to_csv(dffolderspath, index=False, float_format='%g')

def merge_exps(dfexpspath, dffolderspath):
    dffolders = pd.read_csv(dffolderspath)
    dffolders = dffolders.drop(['outdir', 'nprocs'], axis=1)
    dfexps = pd.read_csv(dfexpspath)
    merged = pd.concat([dffolders, dfexps], axis=0, sort=False)
    return merged.drop_duplicates()

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expsdir', help='Experiments dir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
    datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    dfexpspath = pjoin(args.expsdir, 'exps.csv')
    dffolderspath = '/tmp/aux.csv'
    create_exps_from_folders(args.expsdir, dffolderspath)
    dfmerged = merge_exps(dfexpspath, dffolderspath)
    dfmerged.to_csv('/tmp/del.csv', index=False)

if __name__ == "__main__":
    main()

