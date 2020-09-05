# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-09-26
# Lecture 3 - Parametric and nonparametric methods
# Dependencies: numpy, matplotlib.pyplot, matplotlib patches,
#               scipy.stats, scipy.io
# Python version: 3.6
# Functionality: Unsupervised grouping for the dimensional reduced data set
# by the EM step using a mixed gausian model.
# =============================================================================
import numpy as np
import scipy.io as io
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =============================================================================
# Function definition
# =============================================================================
def E_estimation(data, k, mu, cov=None, prior=None, eps=1e-10):
    """
    Calculated model for the multivariate normal distribution
    the maximum prosterior probability is calculated .

    The probability for the point belonging to the different groups is
    calculated. This together with the prior probability for the group gives
    the likelihood.

    If the posterior is larger for one group than the other the function
    assign the group to the given data point.

    -----

    Input is the data, the amount of groups, and initial covariance and mean
    for the gausian model used.

    -----
    Output is a (k X len(data)) matrix h.

    """
    if prior is None:
        prior = np.array([1/k for i in range(k)])
    if cov is None:
        cov = np.array([np.identity(data.shape[0]) for i in range(k)])

    mods = np.array([ss.multivariate_normal(m, C) for m, C in zip(mu, cov)])
    h = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(k):
            props = [mods[t].pdf(data.T[i])*prior[t] for t in range(k)]
            h[j, i] = props[j]/(sum(props) + eps)

    return h


def M_estimation(h, data, k):
    """
    The M step.

    The new Prior, M and Sigma is calculated using the appropriate formulass

    -----

    Input is the h matrix, the data (coullums-C#) and the amount of groups.

    -----

    Ouput is the new prior M and S.
    """
    Prior = np.array([np.sum(h[i])/data.shape[1] for i in range(k)])
    Mean = np.array([(np.sum(h[i]*data, axis=1)/np.sum(h[i]))
                     for i in range(k)])

    Sigma = np.zeros((k, data.shape[0], data.shape[0]))  # Init empty arrays
    for i in range(k):
        Mcentered = data.T - Mean[i]
        for j in range(data.shape[1]):
            Sigma[i] += h[i][j] * np.outer(Mcentered[j].T, Mcentered[j].T)
        Sigma[i] = Sigma[i] * (1/np.sum(h[i]))

    return Prior, Mean, Sigma

# =============================================================================
# Data import and setup.
# =============================================================================
data = io.loadmat("Data/2D3classes.mat")  # Import
trn5 = data['trn5'].T
trn6 = data['trn6'].T
trn8 = data['trn8'].T
Mdata = np.hstack((trn5, trn6, trn8))  # Stacked a single vector
plt.scatter(Mdata[0], Mdata[1])

# =============================================================================
# Initilisation variables
# =============================================================================
k = 3
cov = np.array([np.identity(2) * 500**2 for i in range(k)])
mu = Mdata.T[np.random.randint(0, Mdata.shape[1], k)]
prior = np.array([1/k for i in range(k)])
h = E_estimation(Mdata, k, mu, cov, prior)
P, M, Sigma = M_estimation(h, Mdata, k)  # First itteration

# =============================================================================
# Calculating the Gaussian models
# =============================================================================
for i in range(19):
    print(i)
    h = E_estimation(Mdata, k, M, Sigma, P)
    P, M, Sigma = M_estimation(h, Mdata, k)

# =============================================================================
# Visualisation
# =============================================================================
" Singular plot "
fig, ax = plt.subplots()
ax.set_title("Data after clustering")
ax.scatter(Mdata[0], Mdata[1], marker="o", c=h[0], cmap="Blues",
           s=.4)
ax.scatter(Mdata[0], Mdata[1], marker=".", alpha=0.6, c=h[1], cmap="Reds",
           s=.4)
ax.scatter(Mdata[0], Mdata[1], marker=".", alpha=0.6, c=h[2], cmap="Greens",
           s=.4)
ax.set_xlabel(r"$z_1$")
ax.set_ylabel(r"$z_2$")
ax.legend(["Cluster 1", "Cluster 2", "Cluster 3"])
plt.savefig("clusters.png", dpi=500)
plt.show()

" Multiple segmentations "
fig, ax = plt.subplots(1, 3)
ax[1].set_title("Colormap of individual clusters")
ax[0].scatter(Mdata[0], Mdata[1], marker="o", c=h[0], cmap="Blues",
              s=.4)
ax[1].scatter(Mdata[0], Mdata[1], marker=".", alpha=0.6, c=h[1], cmap="Reds",
              s=.4)
ax[2].scatter(Mdata[0], Mdata[1], marker=".", alpha=0.6, c=h[2], cmap="Greens",
              s=.4)
ax[1].set_yticks([])
ax[2].set_yticks([])
#plt.savefig("Figures/sepeate_clusters.png", dpi=500)
plt.show()

# =============================================================================
# Results from last exercise for comparison
# =============================================================================
S_mean = np.array([[0344.77717395, -466.28694530],
                   [-599.03018878, -157.48233593],
                   [0478.06926910, -516.13708070]])

S_cov = np.array([[[116661.15939656,  51442.92504458],
                   [051442.92504458, 409168.06664259]],

                  [[178382.89890439, 121634.69810585],
                   [121634.69810585, 200921.25794665]],

                  [[098389.64401342,  57402.96842938],
                   [057402.96842938, 223047.99202654]]])

# =============================================================================
# Plotting resutlts for visualisation.
# =============================================================================
def plot_bivar_normal_elipse(mu, cov, color):
    """
    Quick function for plotting an elipse representing the bivariat normal pdfs
    """
    table_value = 1.5   # 95% confidence interval
    var_x = cov[0, 0]
    var_y = cov[1, 1]
    width = 2*np.sqrt(var_x * table_value)
    hight = 2*np.sqrt(var_y * table_value)
    eigval, eigvec = np.linalg.eig(cov)
    eigval_max = np.argmax(eigval)
    eigvec_max = eigvec[eigval_max]
    angle = 360*np.arctan2(eigvec_max[1], eigvec_max[0])/(2*np.pi)
    return Ellipse(mu, width, hight, angle=angle, fill=False, edgecolor=color,
                   linewidth=2)

" Color Codes"
green0 = (0, 1, 0)
green1 = (0, .6, 0)
red0 = (1, 0, 0)
red1 = (0.6, 0, 0)
blue0 = (0, 0, 1)
blue1 = (0, 0, .6)

# Plotting the mean and variances for visualisation of the GMM,
fig, ax = plt.subplots()
ax.set_title("Mean and variances visualised")

ax.scatter(trn5[0], trn5[1], label="5", color="blue", alpha=.5, s=0.1)
ax.scatter(trn6[0], trn6[1], label="6", color="red", alpha=.5, s=0.1)
ax.scatter(trn8[0], trn8[1], label="8", color="green", alpha=.5, s=0.1)

ax.add_patch(plot_bivar_normal_elipse(M[0], Sigma[0], blue0))
ax.add_patch(plot_bivar_normal_elipse(M[1], Sigma[1], red0))
ax.add_patch(plot_bivar_normal_elipse(M[2], Sigma[2], green0))

ax.plot(M[0, 0], M[0, 1], "o", color=blue0)
ax.plot(M[1, 0], M[1, 1], "o", color=red0)
ax.plot(M[2, 0], M[2, 1], "o", color=green0)

ax.add_patch(plot_bivar_normal_elipse(S_mean[0], S_cov[0], blue1))
ax.add_patch(plot_bivar_normal_elipse(S_mean[1], S_cov[1], red1))
ax.add_patch(plot_bivar_normal_elipse(S_mean[2], S_cov[2], green1))

ax.plot(S_mean[0, 0], S_mean[0, 1], "o", color=blue1)
ax.plot(S_mean[1, 0], S_mean[1, 1], "o", color=red1)
ax.plot(S_mean[2, 0], S_mean[2, 1], "o", color=green1)

ax.legend(['Group_1', 'Group_2', 'Group_3', "5", "6", "8"])
ax.set_xlabel(r"$z_1$")
ax.set_ylabel(r"$z_2$")
