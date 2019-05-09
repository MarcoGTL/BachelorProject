import numpy

def createFeatFileList(param):
    if 'imFileList' not in param or ('reboot' in param and param['reboot'] is 1):
        param = createImageFileList(param)

    param['featFileList'] = numpy.ones(param['nPics'])

    featEmptyFileList = numpy.zeros()

    return param, featEmptyFileList


def generateAllFeatures(param, featEmptyFileList):
    return None


def open_feature(param):

    if 'featFileList' not in param or ('reboot' in param and param['reboot'] is 1):
        param, featEmptyFileList = createFeatFileList(param)
        #
        if sum(featEmptyFileList) != 0:
            generateAllFeatures(param, featEmptyFileList)

    fsift = None  # cellfun(@(L) importdata(L), param.featFileList)

    descr = {}
    descr['data'] = None  # double(cell2mat(convertData({fsift(:).data}',3)))

    descr['x'] = None  # convertData({fsift(:).x}, 2)
    descr['y'] = None  # convertData({fsift(:).y}, 2)

    param['lW_px'] = None  # cellfun(@(L) firstSize(L, 3), {fsift.data})'

    return [descr, param]



