import numpy
import sys

def compute_parameters(nClass, typeObj, nPics=None, typeFeat=None, patchSize=None):
    param = {}

    param['nClass'] = nClass

    # OBJ + IMAGE

    param['nObj'] = len(typeObj)

    if nPics is None:
        print('Warning: only one image loaded')
        nPics = 1

    if not isinstance(nPics, list) or len(nPics) == 1:
        param['nPics'] = param['nObj'] * nPics
        nPics = nPics * numpy.ones(param['nObj'])
    elif len(nPics) != param['nObj']:
        sys.exit('Error: Ambiguous number of images')
    else:
        param['nPics'] = sum(nPics)

    param['listObj'] = typeObj
    param['listNPics'] = nPics

    param['picMaxSize'] = 160

    # Features
    if typeFeat is None:
        typeFeat = 'sift'

    if patchSize is None and typeFeat is 'color':
        patchSize = 2
    elif patchSize is None:
        patchSize = 16

    return param
