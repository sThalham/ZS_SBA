import numpy as np

'''
Stolen from: https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
'''

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m - my))), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    #if my < m:
    #    T = T[:my, :]
    #c = muX - b * np.dot(muY, T)
    #c_r = muX - np.dot(muY, T)
    #c_b = muX - b * muY
    #c = np.dot(muY, T) + muX
    #c = np.dot((muX - muY), T)
    c = muX - muY

    #print("translations: ", c, c_r, c_b, c_c)
    #print("muX: ", muX, "muY: ", muY, "muY.T", np.dot(muY, T))

    #X0 = scale * Y0 dot T + c[n, :]
    #scale = (X0 - c) / np.dot(Y0, T)
    #U0, s0, Vt0 = np.linalg.svd(X0, full_matrices=False)
    #U1, s1, Vt1 = np.linalg.svd(Y0, full_matrices=False)

    norm = None
    #norm = 'nuc'  # same as normY/normX of OPA
    X0_v = np.linalg.norm((X - muX), ord=norm, axis=1)  # vector length query
    Y0_v = np.linalg.norm((Y - muY), ord=norm, axis=1)  # vector length template

    mKPd = np.sum(Y0_v) / np.sum(X0_v)
    KP_l2 = normY / normX
    KP_l1 = np.sum(np.abs((Y - muY))) / np.sum(np.abs((X - muX)))

    #sd = np.sum((X - muX) * -1) / np.sum(np.dot((Y - muY), T) * -1)
    print("L2 of vector: ", mKPd, "l2 GPA: ", KP_l2, "l1: ", KP_l1)

    # transformation values
    #tform = {'rotation': T, 'scale': s, 'translation': c}
    tform = {'rotation': T, 'scale': KP_l1, 'translation': c}

    return d, Z, tform