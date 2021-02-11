import pandas as pd
import numpy as np

class statistics:
    """
    class with statistical calculations of interest
    """
    pass


class block_average(statistics):
    """
    a method to find an estimation of the error when the data of a time series
    are correlated
    
    (H. Flyvbjerg and H. G. Petersen: Averages of correlated data (1989))
    """


    def __init__(self, file_data, column, comment='#', dtype=np.float32):
        """
        Parameters
        ----------
        file_data : file
            where the time series is

        column : integer
            number of the column for which you want to calculate the error

        comment : str
            how are the comments denoted in the file

        dtype : str
            type of data
        """
        self.file_data = file_data
        self.column = column
        self.comment = comment
        self.dtype = dtype

    
    def estimate_error(self):
        """
        Returns
        -------
        idx : list of integers
            number of times that block sums where applied

        ds : list of integers
            data size

        mean : list of floats
            with the mean value of the data considered

        var : list of floats
            variance of the data considered

        varerr : list of floats
            error of the variance
        """
        x = pd.read_table(self.file_data, delim_whitespace=True, \
                comment=self.comment, dtype=self.dtype)
        x = pd.DataFrame.to_numpy(x)
        x = x[:, self.column]

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

        idx = list(range(0, idx+1))

        return idx, ds, mean, var, varerr
