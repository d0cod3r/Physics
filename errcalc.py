"""
Mathmatical methods for the practical physics
Version 0.2.3
@author Nicolas Essing
"""

#IMPORTS

import numpy as np


#QUANTITIES WITH ERRORS

class Quantity:
    """
    Represents a value X with statistic error Y and systematic error Z,
    X+-Y(sys)+-Z(stat)
    Supports default arithmetics, log, exp, sqrt and can be displayed as a string.
    CAUTION: ALL CORRELATIONS ARE THOUGHT TO BE ZERO! a*a will have a lower
    error than a**2!
    Use Methods getValue(), getStatisticalError(), getSystematicalError and
    getFullError() to access seperatly.
    Please use Methods below insted of directly creating an instance.
    """
    def __init__(self, value, statistical_error, systematical_error):
        """
        Takes variances
        """
        self.val = np.longdouble(value)
        self.stat = np.longdouble(statistical_error)
        self.sys = np.longdouble(systematical_error)
    
    def __add__(self, other):
        if type(other)==Quantity:
            return q2(self.val + other.val,
                      self.stat+other.stat,
                      self.sys+other.sys)
        else:
            return q2(self.val+other, self.stat, self.sys)
    
    def __sub__(self, other):
        if type(other)==Quantity:
            return q2(self.val - other.val,
                      self.stat+other.stat,
                      self.sys + other.sys)
        else:
            return q2(self.val-other, self.stat, self.sys)

    def __mul__(self, other):
        if type(other)==Quantity:
            return q2(self.val * other.val,
                      self.val**2 * other.stat + other.val**2 * self.stat,
                      self.val**2 * other.sys + other.val**2 * self.sys)
        else:
            return q2(self.val*other, self.stat*other**2, self.sys*other**2)

    def __truediv__(self, other):
        if type(other)==Quantity:
            return q2(self.val / other.val,
                      self.stat/other.val**2 + self.val**2*other.stat/other.val**4,
                      self.sys/other.val**2 + self.val**2*other.sys/other.val**4)
        else:
            return q2(self.val/other, self.stat/other**2, self.sys/other**2)

    def __neg__(self):
        return q2(-self.val, self.stat, self.sys)

    def __pow__(self, exp):
        return q2(self.val**exp,
                  (exp*self.val**(exp-1))**2 * self.stat,
                  (exp*self.val**(exp-1))**2 * self.sys)
    
    def getValue(self):
        """
        Returns value as double.
        """
        return self.val
    
    def getStatisticalError(self):
        """
        Returns statistacal standart deviation.
        """
        return np.sqrt(self.stat)
    
    def getSystematicalError(self):
        """
        Returns systematical standart deviation.
        """
        return np.sqrt(self.sys)
    
    def getStatisticalVariance(self):
        """
        Returns statistical variance
        """
        return self.stat
    
    def getSystematicalVariance(self):
        """
        Returns systematical variance
        """
        return self.sys
    
    def getFullError(self):
        """
        Returns estimated full error
        """
        return np.sqrt(self.stat + self.sys)
    
    def __float__(self):
        return float(self.val)
    
    def __str__(self):
        return str(self.val)+"+-"+str(np.sqrt(self.stat))+"(stat)+-"+str(np.sqrt(self.sys))+"(sys)"


def q(value, statistical_error=0, systematical_error=0):
    """
    Use insted of creating instances directly.
    Takes a value and (optional) standard deviations.
    """
    return Quantity(value, statistical_error**2, systematical_error**2)

def q2(value, statistical_error=0, systematical_error=0):
    """
    Use insted of creating instances directly.
    Takes a value and (optional) variances.
    """
    return Quantity(value, statistical_error, systematical_error)


def sqrt(x):
    return q2(np.sqrt(x.val),
                  x.stat/(4*x.val),
                  x.sys/(4*x.val))

def exp(x):
    return q2(np.exp(x.val),
                  np.exp(2*x.val)*x.stat,
                  np.exp(2*x.val)*x.sys)

def log(x):
    return q2(np.log(x.val),
                  x.stat/x.val**2,
                  x.sys/x.val**2)


def join(*args):
    """
    Gives an estimated best value, a wighted average
    """
    res = q(0)
    div = 0
    for arg in args:
        res = res + arg / arg.getStatisticalVariance()
        div += 1 / arg.getStatisticalVariance()
    return res / div
    
