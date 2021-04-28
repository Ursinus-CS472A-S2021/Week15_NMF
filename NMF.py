import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

def getKLError(V, WH, eps = 1e-10):
    """
    Return the Kullback-Liebler diverges between V and W*H
    """
    denom = np.array(WH)
    denom[denom == 0] = 1
    arg = V/denom
    arg[arg < eps] = eps
    return np.sum(V*np.log(arg)-V+WH)

def getEuclideanError(V, WH):
    """
    Return the Frobenius norm between V and W*H
    """
    return np.sum((V-WH)**2)

def plotNMFSpectra(V, W, H, iter, errs, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Convergence errors
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    plt.subplot(151)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("V")
    plt.subplot(152)
    WH = W.dot(H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W*H Iteration %i"%iter)  
    plt.subplot(153)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(W), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')        
    else:
        plt.imshow(W, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W")
    plt.subplot(154)
    plt.imshow(np.log(H + np.min(H[H > 0])), cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("H Iteration %i"%iter)
    plt.subplot(155)
    plt.semilogy(errs[1::])
    plt.scatter([iter-1], errs[iter])
    plt.xlim([-int(errs.size*0.05), errs.size+int(errs.size*0.05)])
    plt.title("Errors")
    plt.xlabel("Iteration")
    plt.tight_layout()             



def doNMF(V, K, L, W = np.array([]), plotfn = None):
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size > 0:
        WFixed = True
        K = W.shape[1]
    else:
        W = np.random.rand(N, K)
    H = np.random.rand(K, M)

    errs = np.zeros(L+1)
    errs[0] = getKLError(V, W.dot(H))
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
    Ws = [np.array(W)]
    Hs = [np.array(H)]
    for l in range(L):  
        if l%20 == 0:
            print("Doing iteration {} of {}".format(l, L))    
        #KL Divergence Version
        VLam = V/(W.dot(H))
        H *= (W.T).dot(VLam)/np.sum(W, 0)[:, None]
        if not WFixed:
            VLam = V/(W.dot(H))
            W *= (VLam.dot(H.T))/np.sum(H, 1)[None, :]
        errs[l+1] = getKLError(V, W.dot(H))
        if plotfn:
            Ws.append(np.array(W))
            Hs.append(np.array(H))
    if plotfn:
        for l, (W, H) in enumerate(zip(Ws, Hs)):
            plt.clf()
            plotfn(V, W, H, l, errs)
            plt.savefig("NMF_%i.png"%(l+1), bbox_inches = 'tight', transparent=False, facecolor='w')
    return (W, H)


def doNMFKL(V, K, L, W = np.array([]), plotfn = None):
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size > 0:
        WFixed = True
        K = W.shape[1]
    else:
        W = np.random.rand(N, K)
    H = np.random.rand(K, M)

    errs = np.zeros(L+1)
    errs[0] = getKLError(V, W.dot(H))
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
    Ws = [np.array(W)]
    Hs = [np.array(H)]
    for l in range(L):  
        if l%20 == 0:
            print("Doing iteration {} of {}".format(l, L))    
        #KL Divergence Version
        VLam = V/(W.dot(H))
        H *= (W.T).dot(VLam)/np.sum(W, 0)[:, None]
        if not WFixed:
            VLam = V/(W.dot(H))
            W *= (VLam.dot(H.T))/np.sum(H, 1)[None, :]
        errs[l+1] = getKLError(V, W.dot(H))
        if plotfn:
            Ws.append(np.array(W))
            Hs.append(np.array(H))
    if plotfn:
        for l, (W, H) in enumerate(zip(Ws, Hs)):
            plt.clf()
            plotfn(V, W, H, l, errs)
            plt.savefig("NMF_%i.png"%(l+1), bbox_inches = 'tight', transparent=False, facecolor='w')
    return (W, H)


