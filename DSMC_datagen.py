import numpy as np
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import stats

def dsmc_coll(Nc, frac_mft, U, rng=None):
    ## here I use random to generate samples
    if not rng:
        rng = np.random.default_rng()
    # the parameters
    n = 1.0 # normalized distribution
    kT_m = 1.0  # variance of normalized distribution
    sigma = 1.0 # diameter is used as normalizer for x
    l = 1.0/(np.sqrt(2.0)*np.pi*n*sigma**2) # mean free path
    mft = l/kT_m #mean free time
    dt = frac_mft*mft # time step size as frac of mean free time

    # How many candidates for collisions?
    crm = 10*np.sqrt(kT_m);
    upd_crm = False
    M_cand = int(math.pi * sigma ** 2 * Nc * n * crm * dt / 2.0) # number of collision candidates

    # number of actual collisions to be evaluated during collisions
    Ncol = 0
    # make sure to perform collision one by one (jump process)
    for i in range(M_cand):
        # id_p1,2 are ids of colliding pair
        id_p1 = rng.integers(0,Nc-1,endpoint=True)#random.randint(0, Nc-1)
        id_p2 = rng.integers(0,Nc-1,endpoint=True)#random.randint(0, Nc-1)
        while id_p2 == id_p1:# if we pick same particles, we should resample
            id_p2 = rng.integers(0,Nc-1,endpoint=True)#random.randint(1, Nc-1)
        vr = np.linalg.norm(U[:,id_p1]-U[:,id_p2]) # relative velocity
        # update estimate of maximum velocity if it happens
        if vr - crm > 0.0:
            crm = vr
            upd_crm = True

        # acceptance rejection for collision
        if vr / crm > rng.uniform(0.0,1.0):#np.random.rand():
            # find a random point on the surface of sphere with unif. distr.
            alpha1 = rng.random()#np.random.rand()
            alpha2 = rng.random()#np.random.rand()
            theta = math.acos(2.0 * alpha1 - 1.0);
            phi = 2.0 * math.pi * alpha2;

            # find central mass and post rel velocity
            vc = np.array(0.5 * (U[:,id_p1] + U[:,id_p2]))
            vrp = np.array([vr*np.cos(theta), vr*np.sin(theta)*np.cos(phi), vr*np.sin(theta)*np.sin(phi)])

            # update velocities of colliding pair
            U[:,id_p1] = vc + 0.5 * vrp
            U[:,id_p2] = vc - 0.5 * vrp

            # count number of actual collisions
            Ncol = Ncol + 1
    # return number of collisions Nc, updated maximum rel. velocity crm, and a boolen saying if crm has been updated or not
    return Ncol, upd_crm, crm

## gmix() is used to sample particles of initial distribution
## we will use the samples to intialize the weights of KDE on quad points
def gmix(meanvars, size, mix_weights=None, rng=None):
    """
    Samples from a gaussian mixture.
    """
    if not rng:
        rng = np.random.default_rng()
    means, vars = rng.choice(meanvars, p=mix_weights, size=size).T
    sample = rng.standard_normal(size=size).T
    return np.transpose(means + np.sqrt(vars) * sample)

##
def find_weights(x0, ids, r, dr, theta, dtheta, phi, dphi):
    f0 = np.zeros((Nth * Nphi * Nr ))
    for ii in range(len(x0[0])):
        x = x0[0][ii]; y = x0[1][ii]; z = x0[2][ii];

        # finding distribution from samples using hist
        rr = np.sqrt(x**2+y**2+z**2)
        tt = np.arctan(y/x)
        if (x<0.0 and y>0.0) or (x<0.0 and y<0.0):
            tt = tt +np.pi
        elif (x>0.0 and y<0.0):
            tt = tt + 2.0*np.pi
        pp = np.arccos(z/rr)
        iir = [i for i in range(len(r)) if r[i]+0.5*dr[i] > rr]
        if len(iir)>0:
            ir = iir[0]
        else:
            ir = len(r)-1
        it = [i for i in range(len(theta)) if theta[i]+0.5*dtheta[i]  > tt][0]
        ip = [i for i in range(len(phi)) if phi[i]+0.5*dphi[i]  > pp ][0]
        id = ids[ir][it][ip]
        f0[id] = f0[id] + 1.0
    #normalize distr.
    f0 = f0 / (1.0*len(x0[0]))
    return f0

### input paramters
N_data = 100 # number of data points to generate
Np = 1*10**4 # number of samples, increase it to get better results with less noise (~O(1/Np))
Nt = 50 # number of time steps to perform DSMC
frac_mft = 0.1 # delta = frac_mft*(mean free time)
L= 5.0 # size of domain: r \in (0,L] (r is the radius in spherical coord.)
Nth = 10 # number of grid points in theta direction (spherical coord.), increase it to get more accurate results
Nphi = 10 # number of grid points in phi direction (spherical coord.), |---|
Nr = 10 # number of grid points in r direction (spherical coord.), |---|
address = 'data_hist' # the name of file/directory to save data

### change the following paramteres carefully
sp_mu = [-1.0,1.0] # the space to sample mean of bi-modal distr
sp_sig = [0.5,2.0] # the space to sample variance of bi-modal distr
prior_mix_weights = [.5, .5] # used for sampling bi-modal
nreps = 1 # number of realization. for now, let's keep it =1
# kde is used for resampling particles at grid points
eps = 0.01 # tolerance for fixing the bias where bdw = eps/Np**(1.0/5.0)*1: eq 12 from: Phys. Fluids 31, 062008 (2019)
#seed = 3570
#rng = np.random.default_rng(seed)
rng = np.random.default_rng()
intt = rng.integers(10**3, size=10**3);  # warm up of RNG

# what moment to look at for checking bias of weight-to-grid
p=4

# allocate samples x0
x0 = np.zeros((3,Np))# samples
# grid points
theta = 2.0*np.pi*np.linspace(1.0/(2.0*Nth), 1.0-1.0/(2.0*Nth), Nth)
phi = 1.0*np.pi*np.linspace(1.0/(2.0*Nphi), 1.0-1.0/(2.0*Nphi), Nphi)
r, w_quad = np.polynomial.hermite.hermgauss(2*Nr)
w_quad = w_quad*np.exp(r**2)
idd = r>1e-13
r = r[idd]
w_quad= w_quad[idd]
r = r *L/np.max(r)
w_quad = w_quad *L/np.max(r)

# grid size in each direction
dphi = np.diff(phi); dphi = np.append(dphi,dphi[-1])
dtheta = np.diff(theta); dtheta = np.append(dtheta,dtheta[-1])
ddr = np.diff(r);
dr = np.zeros(Nr)
for i in range(Nr):
    if i == 0:
        delp1 = ddr[i]
        delm1 = (r[0]-0.0)*2.0
    elif i == Nr-1:
        delp1 = ddr[i-1]
        delm1 = ddr[i-1]
    else:
        delp1 = ddr[i]
        delm1 = ddr[i-1]
    dr[i] = 0.5*(delp1+delm1)

# q0: the grid points all put together
# f0 is the value of distribution at grid pionts
# w0 is the weight of quadrature at grid points, can be used for integration
# ids stores the node id of each cell
q0 = np.zeros((Nth*Nphi*Nr,3))
f0 = np.zeros((Nth*Nphi*Nr))
w0 = np.zeros((Nth*Nphi*Nr))
ids = [[[0 for x in range(Nphi)] for x in range(Nth)] for x in range(Nr)]

### build the grid
id = 0
for j in range(Nth):
    for k in range(Nphi):
        for i in range(Nr):
            x = r[i] * np.cos(theta[j] ) * np.sin(phi[k] )
            y = r[i] * np.sin(theta[j] ) * np.sin(phi[k] )
            z = r[i] * np.cos(phi[k])
            q0[id] = [x, y, z]
            w0[id] = r[i]**2 * np.sin(phi[k]) * dr[i] * dtheta[j] * dphi[k]
            ids[i][j][k] = id
            id = id+1;
q0 = q0.T

## generating data
data = []
for nps in range(N_data):
    sig2 = [-1.0,-1.0,-1.0]
    # first we sample bimodal distributions
    while any(sig < 1e-3 for sig in sig2):
        mu1 = sp_mu[0] + rng.random(3) * (sp_mu[1] - sp_mu[0])
        mu2 = -mu1
        sig1 = sp_sig[0] + rng.random(3) * (sp_sig[1] - sp_sig[0])
        sig2 = 2.0 - (sig1 + 2 * mu1 ** 2)
    for dim in range(3):
        meanvars = [(mu1[dim], sig1[dim]), (mu2[dim], sig2[dim])]
        x0[dim] = gmix(meanvars, Np, mix_weights=prior_mix_weights, rng=rng)

    # find the values of weights at grid points
    f0 = find_weights(x0, ids, r, dr, theta, dtheta, phi, dphi)
    # resample particles based on the kernel with poles at grid points and weights of f0
    bdw = eps / Np ** (1.0 / 5.0) * 1  # kt_m = 1 here, eq 12 from: Phys. Fluids 31, 062008 (2019)
    kdefit = stats.gaussian_kde(q0, bw_method=bdw, weights=f0)
    x0 = kdefit.resample(Np)

    ## check the relative error in moments
    print(r"rel_err(m_" + str(p) + ")=" + str(
        abs((np.average(x0[1] ** p) - np.sum(q0[1] ** p * f0)) / (np.average(x0[1] ** p)))))

    fprint = [f0]
    for id_t in range(1,Nt+1):
        # do DSMC
        Ncol, upd_crm, crm = dsmc_coll(Np, frac_mft, x0, rng=rng)

        #compute weights at grid points again
        f0 = find_weights(x0, ids, r, dr, theta, dtheta, phi, dphi)

        # print the relative error in pth moments
        print("t_"+str(id_t)+", rel_err(m_"+str(p)+")="+str(abs((np.average(x0[1] ** p) - np.sum(q0[1] ** p * f0)) / (np.average(x0[1] ** p)))))

        fprint.append(f0)
    data.append(fprint)
    print("data " + str(nps) + " done!")
np.savez_compressed(address, a=data)
