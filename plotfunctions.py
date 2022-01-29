# auxiliary functions for PCA with spectral data
import matplotlib.pyplot as plt
import numpy as np

def scree_plot(PCA):
    """
    In:
        PCA : sklearn.decomposition.PCA
            Fitted PCA object
    Out:
        fig : matplotlib.pyplot.figure object
            the scree plot object
    """

    expl_var_1 = PCA.explained_variance_ratio_

    with plt.style.context(('ggplot')):
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
        fig.set_tight_layout(True)
    
        ax1.plot(np.arange(1, len(expl_var_1) + 1, 1), expl_var_1,'-o', label="Explained Variance %")
        ax1.plot(np.arange(1, len(expl_var_1) + 1, 1), np.cumsum(expl_var_1),'-o', label = 'Cumulative variance %')
        ax1.set_xlabel("PC number")
        ax1.set_ylabel("Explained Variance %")
        ax1.set_title('Scree plot')                
        plt.legend()
    return fig
    


def scores_plot(scores, y, PCs=(1,2)):
    """
    In:
        scores : np.array of shape (Nrows, Ncomp)
            scores matrix
        y : np.array of shape (Nrows,)
            vector of class labels; 
        PCs [optional] :  2-tuple of integers
            Principle Components to be plotted
    Out:
        fig : matplotlib.pyplot.figure object
            scores plot object
    """
    
    unique = np.unique(y) # list of unique labels
    colors = [plt.cm.jet(float(i)/len(unique)) for i in range(len(unique))]
    
    PCx = PCs[0]
    PCy = PCs[1]
    
    fig = plt.figure()

    with plt.style.context(('ggplot')):
        for i, u in enumerate(unique):
            col = np.expand_dims(np.array(colors[i]), axis=0)
            xi = [scores[j,PCx-1] for j in range(len(scores[:,PCx-1])) if y[j] == u]
            yi = [scores[j,PCy-1] for j in range(len(scores[:,PCy-1])) if y[j] == u]
            plt.scatter(xi, yi, c=col, s=60, edgecolors='k',label=str(u))
    
        plt.xlabel('PC'+str(PCx))
        plt.ylabel('PC'+str(PCy))
        plt.legend(unique,loc='lower right')
        plt.title('Scores Plot')
        #plt.show()

    return fig


def loading_plot(loadings, dim, PCs=[1], xlabel='wave number'):
    """
    In:
        loadings : np.array of shape (Nrows, Ncomp)
            loadings matrix
        dim : numpy.ndarray of shape (n_features,)
            Wavelength, -number etc.
        PCs [optional] :  list of integers
            Principle Components to be plotted
    Out:
        fig : matplotlib.pyplot.figure object
            loadings plot object
    """

    fig = plt.figure()

    with plt.style.context(('ggplot')):
        for PC in PCs:
            plt.plot(dim, loadings[:,PC], label='PC'+str(PC))

    plt.xlabel(xlabel)
    plt.ylabel('Loading [a.u.]')
    plt.legend()

    

def Tsq_Q_plot(X, scores, loadings, conf=0.95):
    """
    T^2-Q-Plot of PCA results
    adapted from https://nirpyresearch.com/outliers-detection-pls-regression-nir-spectroscopy-python/ 
    
    In:
    X : np.array of shape (Nrows, Ncomp)
        data matrix
    scores : np.array of shape (Nrows, Ncomp)
        scores matrix
    loadings : np.array of shape (Nrows, Ncomp)
        loadings matrix
    conf [optional]: float
        confidence level

    Out:
        fig : matplotlib.pyplot.figure object
            T^2-Q plot object
    """

    ncomp = scores.shape[1]

    # residuals ("errors")
    Err = X - np.dot(scores,loadings.T)

    # Calculate Q-residuals (sum over the rows of the error array)
    Q = np.sum(Err**2, axis=1)

    # Calculate Hotelling's T-squared (note that data are normalised by default)
    Tsq = np.sum((scores/np.std(scores, axis=0))**2, axis=1)
    
    from scipy.stats import f
    # Calculate confidence level for T-squared from the ppf of the F distribution
    Tsq_conf =  f.ppf(q=conf, dfn=ncomp, \
                dfd=X.shape[0])*ncomp*(X.shape[0]-1)/(X.shape[0]-ncomp)

    # Estimate the confidence level for the Q-residuals
    i = np.max(Q)+1
    while 1-np.sum(Q>i)/np.sum(Q>0) > conf:
        i -= 1
    Q_conf = i

    fig = plt.figure()

    with plt.style.context(('ggplot')):
        fig = plt.plot(Tsq, Q, 'o')
    
        plt.plot([Tsq_conf,Tsq_conf],[plt.axis()[2],plt.axis()[3]],  '--')
        plt.plot([plt.axis()[0],plt.axis()[1]],[Q_conf,Q_conf],  '--')
        plt.xlabel("Hotelling's T-squared")
        plt.ylabel('Q residuals')
    
    #plt.show()

    return fig, Q, Tsq