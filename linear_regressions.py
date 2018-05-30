from numpy import sqrt
import scipy.odr

def linear_regression_0(x, y):
    """
    Linear regression without uncertainties.
    Fits to y(x)=m*x+c and returns m and c.
    """
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(x**2)
    sum_xy = sum(x*y)
    delta = n*sum_x2-sum_x**2
    m = (n*sum_xy-sum_x*sum_y)/delta
    c = (sum_x2*sum_y-sum_x*sum_xy)/delta
    return m, c

def linear_regression_1(x, y, uncert_y=None):
    """
    Linear regression with uncertainty only in y.
    Takes an array of x-values, an array of y-values and an array of
    uncertainties of the y-values.
    Fits to y(x)=m*x+c and returns at tuple containing
      (m, c, sigma_m, sigma_c, correlation, chisquared)
    """
    if uncert_y is None:
        m, c = linear_regression_0(x,y)
        return m, 0., c, 0., 0., float("inf")
    sum_w   = sum(1./uncert_y**2)
    sum_x  = sum(x/uncert_y**2)
    sum_y  = sum(y/uncert_y**2)
    sum_x2 = sum(x**2/uncert_y**2)
    sum_xy = sum(x*y/uncert_y**2)
    delta = sum_w*sum_x2-sum_x**2
    m = (sum_w*sum_xy-sum_x*sum_y)/delta
    c = (sum_x2*sum_y-sum_x*sum_xy)/delta
    uncert_m = sqrt(sum_w/delta)
    uncert_c = sqrt(sum_x2/delta)
    cov = -sum_x/delta
    corr = cov/(uncert_m*uncert_c)
    chiq  = sum(((y-(m*x+c))/uncert_y)**2)
    return m, uncert_m, c, uncert_c, corr, chiq

def linear_regression(x, y, uncert_x=None, uncert_y=None):
    """
    Linear regression with uncertainties in both directions.

    Parameters
    ----------
    x : array_like
        x-values
    y : array_like
        y-values
    uncert_x : array_like or None
        uncertainty of x-values
    uncert_y : array_like
        uncertainty of y-values

    Takes two arrays for x-values and y-values and optional two arrays of
    uncertainties. The uncertainty arguments are default to None, which is
    treated as a zero uncertainty in every value.
    Fits a function y=m*x+c to the values and returns the results for m and c
    with uncertainty as well as their correlation and the chisquared of the
    fit. It returns a tuple containing:
        m, sigma_m, c, sigma_c, correlation, chisquared
    
    If both uncertainties are unequal zero, the calculation can not be
    performed analytically and the package scipy.odr is used.
    """
    # if only one uncertainty is given, calculate analytically
    if uncert_x is None:
        return linear_regression_1(x, y, uncert_y)
    elif uncert_y is None:
        # as linear_regression_1 assumes uncertainties in y, switch the axes
        m, um, c, uc, corr, chisq = linear_regression_1(y,x,uncert_x)
        sigma_c = sqrt(uc**2/m**2+c**2*um**2/m**4-c/m**3*corr*um*uc)
        return 1/m, um/m**2, -c/m, sigma_c, corr, chisq
    
    # For a first guess, assume a slope around 1 and weight both uncertainties
    # equal. Calculate initial values analytically.
    uncert_sum = uncert_x + uncert_y
    m0, um0, c0, uc0, corr0, chisq0 = linear_regression_1(x, y, uncert_sum)

    def f(B, x):
        return B[0]*x + B[1]
    
    model  = scipy.odr.Model(f)
    data   = scipy.odr.RealData(x, y, sx=uncert_x, sy=uncert_y)
    odr    = scipy.odr.ODR(data, model, beta0=[m0, c0])
    output = odr.run()
    ndof = len(x)-2
    chiq = output.res_var*ndof
    sigma_m = sqrt(output.cov_beta[0,0])
    sigma_c = sqrt(output.cov_beta[1,1])
    corr = output.cov_beta[0,1] /sigma_m /sigma_c
    
    return (output.beta[0], sigma_m, output.beta[1], sigma_c, corr, chiq)


# when imported, only give last method, as it uses the other two if necessary
__all__ = ["linear_regression"]


try:
    import uncertainties
    
    def linear_regression_u(x, y, uncert_x=None, uncert_y=None):
        """
        Linear regression with uncertainties in both directions, returning
        UncertainVariables from the uncertainties package.

        Parameters
        ----------
        x : array_like
            x-values
        y : array_like
            y-values
        uncert_x : array_like or None
            uncertainty of x-values
        uncert_y : array_like
            uncertainty of y-values

        Takes two arrays for x-values and y-values and optional two arrays of
        uncertainties. The uncertainty arguments are default to None, which is
        treated as a zero uncertainty in every value.
        Fits a function y=m*x+c to the values and returns the results for m and c
        as UncertainVariables from the package uncertainties (considering their
        correlation), and the chisquared of the fit. It returns a tuple containing:
            m, c, chisquared

        If both uncertainties are unequal zero, the calculation can not be
        performed analytically and the package scipy.odr is used.
        """
        m,sigma_m,c,sigma_c,corr,chisq = linear_regression(x,y,uncert_x, uncert_y)
        cov = sigma_m * sigma_c * corr
        cov_mat = [[sigma_m**2, cov],[cov, sigma_c**2]]

        m2, c2 = uncertainties.correlated_values([m, c], cov_mat)
        return m2, c2, chisq
    
    __all__.append("linear_regression_u")
