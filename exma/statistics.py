import pandas as pd
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
    x : array (default = 0.0)
        where the time series is

    file_data : file (default = None)
        where the time series is if not x

    column : integer (default = 0)
        number of the column for which you want to calculate the error

    comment : str (default = '#')
        how are the comments denoted in the file

    dtype : str (default = np.float32)
        type of data
    """

    def __init__(self, x=0.0, file_data=None, column=0, comment='#',
                 dtype=np.float32):

        self.x = x
        self.file_data = file_data
        self.column = column
        self.comment = comment
        self.dtype = dtype

    
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
        
        if self.file_data is not None:
            self.x = pd.read_table(self.file_data, delim_whitespace=True, \
                                   comment=self.comment, dtype=self.dtype)
            self.x = pd.DataFrame.to_numpy(self.x)
            self.x = self.x[:, self.column]

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

            for k in range(0, len(newx)):
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
