import pymc
import numpy as np

# Priors on unknown parameters
init = 0.5; minv = 0.; maxv = 1.
theta = pymc.TruncatedNormal('theta', value=init, mu=0, tau=1., a=minv, b=maxv)

# Binomial likelihood for data
d = pymc.Binomial('d', n=100, p=theta, value=75,observed=True)