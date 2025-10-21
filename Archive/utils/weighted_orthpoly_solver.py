# Utilities for generating orthogonal polynomials.

import numpy as np
from numpy import pi, sqrt, sin, dot, sum, sign, arange
from numpy.linalg import inv, qr, cholesky

import numpy.polynomial.chebyshev  as cheb
import numpy.polynomial.legendre   as lege

def get_orthpoly(n_deg, f_weighting, n_extra_point = 10, return_coef = True, representation='chebyshev', integral_method='legendre'):
    """
     Get orthogonal polynomials with respect to the weighting (f_weighting).
     The polynomials are represented by coefficients of Chebyshev polynomials.
     Evaluate it as: np.dot(cheb.chebvander(set_of_x, n_deg), repr_coef)
     See weighted_orthpoly1.py for validation of this algorithm.
    """
    n_sampling = n_deg + 1 + n_extra_point
    # Do the intergration by discrete sampling at good points.
    if integral_method == 'chebyshev':
        # Here we use (first kind) Chebyshev points.
        x_sampling = sin( pi * (-(n_sampling-1)/2 + np.arange(n_sampling))/n_sampling )
        # Put together the weighting and the weighting of the Gauss-Chebyshev quadrature.
        diag_weight = np.array([f_weighting(x)/cheb.chebweight(x) for x in x_sampling]) * (pi/n_sampling)
    elif integral_method == 'legendre':
        # Use Gauss-Legendre quadrature.
        x_sampling, int_weight = lege.leggauss(n_sampling)
        diag_weight = np.array([f_weighting(x)*w for x, w in zip(x_sampling, int_weight)])

    if representation == 'chebyshev':
        V = cheb.chebvander(x_sampling, n_deg)
    else:
        V = lege.legvander(x_sampling, n_deg)
    
    # Get basis from chol of the Gramian matrix.
#    inner_prod_matrix = dot(V.T, diag_weight[:, np.newaxis] * V)
#    repr_coef = inv(cholesky(inner_prod_matrix).T)
    # QR decomposition should give more accurate result.
    repr_coef = inv(qr(sqrt(diag_weight[:, np.newaxis]) * V, mode='r'))
    repr_coef = repr_coef * sign(sum(repr_coef,axis=0))
    
    if return_coef:
        return repr_coef
    else:
        if representation == 'chebyshev':
            polys = [cheb.Chebyshev(repr_coef[:,i]) for i in range(n_deg+1)]
        else:
            polys = [lege.Legendre (repr_coef[:,i]) for i in range(n_deg+1)]
        return polys

def orthpoly_coef(f, f_weighting, n_deg, **kwarg):
    if f_weighting == 'chebyshev':
        x_sample = sin( pi * (-n_deg/2 + arange(n_deg+1))/(n_deg+1) )
        V = cheb.chebvander(x_sample, n_deg)
    elif f_weighting == 'legendre':
        x_sample = lege.legroots(1*(arange(n_deg+2)==n_deg+1)) # 先生成一个n_deg+2个项的勒让德多项式多项式，最高项系数为1，并求解其根
        V = sqrt(arange(n_deg+1)+0.5) * lege.legvander(x_sample, n_deg)
    else:
        x_sample = lege.legroots(1*(arange(n_deg+2)==n_deg+1))
        basis_repr_coef = get_orthpoly(n_deg, f_weighting, 100, **kwarg)
        V = dot(cheb.chebvander(x_sample, n_deg), basis_repr_coef)

    if f == None:
        # Return Pseudo-Vandermonde matrix.
        return V

    y_sample = f(x_sample)
    coef = np.linalg.solve(V, y_sample)
    return coef

# vim: et sw=4 sts=4
