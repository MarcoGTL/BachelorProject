import slic
import sift
import histogram
import argparse
import copy
import numpy
import sys


def compute_parameters(num_classes, type_objects, num_pics=None):
    param = {
        'num_classes': num_classes,
        'num_objects': len(type_objects)
    }

    if num_pics is None:
        print('Warning: only one image loaded')
        num_pics = [1]

    if len(num_pics) == 1:
        param['num_pics'] = param['num_objects'] * num_pics
        num_pics = num_pics * numpy.ones(param['num_objects'])
    elif len(num_pics) != param['num_objects']:
        sys.exit('Error: Ambiguous number of images')
    else:
        param['num_pics'] = sum(num_pics)

    param['list_objects'] = type_objects
    param['list_num_pics'] = num_pics

    param['pic_max_size'] = 160

    return param


def open_feature(param):
    print('open_feature not implemented')
    descr = {'data': 1}
    return [descr, param]


def fun_kernel(data):
    print('funKernel not implemented')
    return data


def compute_regularization_param(data_complex_conjugate_transpose, df):
    print('compute_regularization_param not implemented')
    return 1, 2


def main():
    # Compute solution from Joulin et. al. cvpr'10 and Joulin et al. cvpr'12

    # Setting parameters
    param = compute_parameters(2, ['elephant'], [5])
    print(param)
    print(param['num_classes'])

    # Create Mask
    print('Mask not yet implemented')

    # Open Features
    [descr, param] = open_feature(param)

    # Compute map related to kernel
    descr['data'] = fun_kernel(descr['data'])
    param['n_descr'], param['dim_descr'] = len(descr['data'])

    # Compute lambda (regularization parameter)
    param['lambda'], x_dif = compute_regularization_param(descr['data'].getH(), param['df'])


if __name__ == '__main__':
    main()
