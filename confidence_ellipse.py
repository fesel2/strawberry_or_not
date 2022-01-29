import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse_interv(x, y, ax, facecolor='none', conf_interv="90", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Basic function from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    adapted by Lukas Fesenmeier with help of 
    https://www.cs.utah.edu/~tch/CS6640F2020/resources/How%20to%20draw%20a%20covariance%20error%20ellipse.pdf
    https://www.datasciencelearner.com/find-eigenvalues-eigenvectors-numpy/
    https://www.youtube.com/watch?v=0GzMcUy7ZI0
    https://www.youtube.com/watch?v=-8q-yKY-vg4&t=1082s

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    conf_interval : string
        The confidence interval wich is needed, avaiable in this function: '90', '95', '99'

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    if conf_interv == "90":
        s = 4.605
    elif conf_interv == "95":
        s = 5.991
    elif conf_interv == "99":
        s = 9.210
    else:
        raise ValueError("must be string with value '90', '95' or '99', other values must be calculated individually or use other function")

    #get Eigenvalues and Eigenvectors
    evalues, evectors = np.linalg.eig(cov) 
    evalues = np.sqrt(evalues)


    #Ellipsengleichung
    ellipse = Ellipse(xy = (np.mean(x), np.mean(y)), 
    width = evalues[0]*np.sqrt(s) *2, height = evalues[1]* np.sqrt(s)* 2,
    angle = np.rad2deg(np.arccos((evectors[0,0]))), alpha = 0.3, facecolor=facecolor, **kwargs)

    return ax.add_patch(ellipse)

def confidence_ellipse_std(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)