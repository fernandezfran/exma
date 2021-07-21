import numpy as np

class statistics:
    """
    class with statistical calculations of interest
    """


class block_average(statistics):
    """
    a method to find an estimation of the error when the data of a time series
    are correlated
    
    (H. Flyvbjerg and H. G. Petersen: Averages of correlated data (1989))

    Parameters
    ----------
    x : array
        where the time series is
    """

    def __init__(self, x):
        self.x = x
    
    def estimate_error(self):
        """
        Returns
        -------
        idx : numpy array of integers
            number of times that block sums where applied

        ds : numpy array of integers
            data size

        mean : numpy array of floats
            with the mean value of the data considered

        var : numpy array of floats
            variance of the data considered

        varerr : numpy array of floats
            error of the variance
        """
        
        x = self.x

        ds     = []
        mean   = []
        var    = []
        varerr = []

        idx = 0
        ds.append(len(x))
        mean.append(np.mean(x))
        var.append(np.var(x) / (ds[idx] - 1))
        varerr.append(np.sqrt(2.0 / (ds[idx] - 1)) * var[idx])

        oldx = x
        while (np.intc(len(oldx)/2) > 2):
            newx = np.zeros(np.intc(len(oldx)/2))

            for k in range(len(newx)):
                newx[k] = 0.5 * (oldx[2*k - 1] + oldx[2*k])

            idx += 1
            ds.append(len(newx))
            mean.append(np.mean(newx))
            var.append(np.var(newx) / (ds[idx] - 1))
            varerr.append(np.sqrt(2.0 / (ds[idx] - 1)) * var[idx])

            oldx = newx

        idx = np.array(list(range(0, idx+1)))
        ds = np.array(ds)
        mean = np.array(mean)
        var = np.array(var)
        varerr = np.array(varerr)

        return idx, ds, mean, var, varerr
