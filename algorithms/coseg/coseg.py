import computeParameters as computeParameters
import openFeature as openFeature

def main():
    # Compute solution from Joulin et. al. cvpr'10 and Joulin et al. cvpr'12

    # Setting parameters
    param = computeParameters.compute_parameters(2, ['elephant'], [5])

    # Create mask

    # Open Features
    [descr, param] = openFeature.open_feature(param)

    # Compute map related to kernel

    # Compute lambda (regularization parameter)

    # Compute the binary term (Laplacian matrix)

    # Open superpixels

    # CVPR'10

    # CVPR'12

if __name__ == '__main__':
    main()
