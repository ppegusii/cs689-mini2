#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

import parse
import clf


def main():
    args = parseArgs(sys.argv)
    trainXY, testXY = parse.trainTestData(args.data)
    clf.SVM(trainXY, testXY)
    clf.KNN(trainXY, testXY)


def parseArgs(args):
    parser = argparse.ArgumentParser(
        description='Gate classifier. Written in Python 2.7.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data',
                        default='../../data',
                        type=filesInDir,
                        help='The directory containing the data.')
#    parser.add_argument('analogies',
#                        type=validDirectory,
#                        help='The directory containing analogy files.')
#    parser.add_argument('output',
#                        type=validDirectory,
#                        help='The directory for output files.')
#    parser.add_argument('-d', '--distance_measure',
#                        default='cosadd',
#                        type=distMeasure,
#                        help='''The distance measure to use.
#                             {"cosadd", "cosmult"}''')
    return parser.parse_args()


def validFile(fileName):
    if os.path.isfile(fileName):
        return fileName
    msg = 'File "{:s}" does not exist.'.format(fileName)
    raise argparse.ArgumentTypeError(msg)


def validDirectory(dirName):
    if os.path.isdir(dirName):
        return dirName
    msg = 'Directory "{:s}" does not exist.'.format(dirName)
    raise argparse.ArgumentTypeError(msg)


def filesInDir(dirName):
    if not os.path.isdir(dirName):
        msg = 'Directory "{:s}" does not exist.'.format(dirName)
        raise argparse.ArgumentTypeError(msg)
    fns = [os.path.join(dirName, fn) for fn in next(os.walk(dirName))[2]]
    if len(fns) < 1:
        msg = 'Directory "{:s}" contains no files.'.format(dirName)
        raise argparse.ArgumentTypeError(msg)
    return fns


# def distMeasure(dm):
#    if dm == 'cosadd':
#        return dist.cosadd
#    if dm == 'cosmult':
#        return dist.cosmult
#    msg = 'Distance measuere "{:s}" is not defined.'.format(dm)
#    raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    main()
