"""
Hyperbolic K-means implementation from https://github.com/drewwilimitis/hyperbolic-learning/blob/master/hyperbolic_kmeans/hkmeans.py
- Should be able to fit on poincare data
"""

import numpy as np
import sys
from tqdm import tqdm
# from args import args
import numpy as np


def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

#-------------------------
#----- Poincaré Disk -----
#-------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)
    
# distance in poincare disk
def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

# compute symmetric poincare distance matrix
def poincare_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array from poincare disk to hyperboloid
def poincare_pts_to_hyperboloid(Y, eps=1e-6, metric='lorentz'):
    mink_pts = np.zeros((Y.shape[0], Y.shape[1]+1))
    # print('Minkowski pts shape: {}'.format(mink_pts.shape))
    r = norm(Y, axis=1)
    if metric == 'minkowski':
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(1, mink_pts.shape[1]):
            mink_pts[:, i] = 2/(1 - r**2 + eps) * Y[:, i - 1]
    else:
        mink_pts[:, Y.shape[1]] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(0, Y.shape[1]):
            mink_pts[:, i] = 2/(1 - r**2 + eps) * Y[:, i]
    """
    if metric == 'minkowski':
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * Y[:, 1]
    else:
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 1]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    """
    return mink_pts

# convert single point to hyperboloid
def poincare_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    mink_pt = np.zeros((y.shape[0] + 1, ))
    # print('mink_pt.shape: {}'.format(mink_pt.shape))
    r = norm(y)

    if metric == 'minkowski':
        mink_pt[0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(1, mink_pt.shape[0]):
            mink_pt[i] = 2/(1 - r**2 + eps) * y[i - 1]
    else:
        mink_pt[y.shape[0]] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        for i in range(0, y.shape[0]):
            mink_pt[i] = 2/(1 - r**2 + eps) * y[i]
    """
    mink_pt = np.zeros((3, ))
    r = norm(y)
    if metric == 'minkowski':
        mink_pt[0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pt[1] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[2] = 2/(1 - r**2 + eps) * y[1]
    else:
        mink_pt[0] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[1] = 2/(1 - r**2 + eps) * y[1]
        mink_pt[2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    """
    return mink_pt

#------------------------------
#----- Hyperboloid Model ------
#------------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)

# define hyperboloid bilinear form
def hyperboloid_dot(u, v):
    # print('U dim: {}'.format(u.shape))
    # print('V dim: {}'.format(v.shape))
    return np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]

# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 

# hyperboloid distance function
def hyperboloid_dist(u, v, eps=1e-6, metric='lorentz'):
    if metric == 'minkowski':
        dist = np.arccosh(-1*minkowski_dot(u, v))
    else:
        dist = np.arccosh(-1*hyperboloid_dot(u, v))
    if np.isnan(dist):
        #print('Hyperboloid dist returned nan value')
        return eps
    else:
        return dist

# compute symmetric hyperboloid distance matrix
def hyperboloid_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = hyperboloid_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array to poincare disk
def hyperboloid_pts_to_poincare(X, eps=1e-6, metric='lorentz'):
    poincare_pts = np.zeros((X.shape[0], X.shape[1]-1))
    if metric == 'minkowski':
        for i in range(0, X.shape[1]):
            poincare_pts[:, i] = X[:, i + 1] / ((X[:, 0] + 1) + eps)
    else:
        for i in range(0, X.shape[1]):
            poincare_pts[:, i] = X[:, i] / ((X[:, -1] + 1) + eps)
    """        
    if metric == 'minkowski':
        poincare_pts[:, 0] = X[:, 1] / ((X[:, 0]+1) + eps)
        poincare_pts[:, 1] = X[:, 2] / ((X[:, 0]+1) + eps)
    else:
        poincare_pts[:, 0] = X[:, 0] / ((X[:, 2]+1) + eps)
        poincare_pts[:, 1] = X[:, 1] / ((X[:, 2]+1) + eps)
    """
    return poincare_pts

# project within disk
def proj(theta,eps=1e-3):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

# convert single point to poincare
def hyperboloid_pt_to_poincare(x, eps=1e-6, metric='lorentz'):
    poincare_pt = np.zeros((x.shape[0] - 1, ))
    if metric == 'minkowski':
        for i in range(0, poincare_pt.shape[0]):
            poincare_pt[i] = x[i + 1] / ((x[0] + 1) + eps)
    else:
        for i in range(0, poincare_pt.shape[0]):
            poincare_pt[i] = x[i] / ((x[-1] + 1) + eps)
    """
    poincare_pt = np.zeros((2, ))
    if metric == 'minkowski':
        poincare_pt[0] = x[1] / ((x[0]+1) + eps)
        poincare_pt[1] = x[2] / ((x[0]+1) + eps)
    else:
        poincare_pt[0] = x[0] / ((x[2]+1) + eps)
        poincare_pt[1] = x[1] / ((x[2]+1) + eps)
    """
    return proj(poincare_pt)
    
# helper function to generate samples
def generate_data(n, radius=0.7, hyperboloid=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    if hyperboloid:
        return poincare_pts_to_hyperboloid(init_data)
    else:
        return init_data

def exp_map(v, theta, eps=1e-6):
    # v: tangent vector in minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # project vector v from tangent minkowski space -> hyperboloid
    return np.cosh(norm(v))*theta + np.sinh(norm(v)) * v / (norm(v) + eps)

def minkowski_distance_gradient(u, v):
    # u,v in hyperboloid
    # returns gradient with respect to u
    return -1*(hyperboloid_dot(u,v)**2 - 1)**-1/2 * v

def minkowski_loss_gradient(theta, X):
    # X : array with points in hyperboloid cluster
    # theta: parameter vector in hyperboloid with centroid coordinates
    # returns gradient vector
    # print('POINCARE KMEANS DEBUG: theta.shape: {}'.format(theta.shape))
    # print('POINCARE KMEANS DEBUG: X.shape: {}'.format(X.shape))
    distances = np.array([-1*hyperboloid_dist(theta, x) for x in X]).reshape(-1,1)
    distance_grads = np.array([minkowski_distance_gradient(theta, x) for x in X])
    grad_loss = 2*np.mean(distances*distance_grads, axis=0)
    if np.isnan(grad_loss).any():
        print('Hyperboloid dist returned nan value')
        return eps
    else:
        return grad_loss

def project_to_tangent(theta, minkowski_grad):
    # grad: gradient vector in ambient Minkowski space
    # theta: parameter vector in hyperboloid with centroid coordinates
    # projects to hyperboloid gradient in tangent space
    return minkowski_grad + hyperboloid_dot(theta, minkowski_grad)*theta

def update_theta(theta, hyperboloid_grad, alpha=0.1):
    # theta: parameter vector in hyperboloid with centroid coordinates
    return exp_map(-1*alpha*hyperboloid_grad, theta)

def frechet_loss(theta, X):
    s = X.shape[0]
    dist_sq = np.array([hyperboloid_dist(theta, x)**2 for x in X])
    return np.sum(dist_sq) / s

def compute_mean(theta, X, num_rounds = 10, alpha=0.3, tol = 1e-4, verbose=False):
    centr_pt = theta.copy()
    centr_pts = []
    losses = []
    for i in range(num_rounds):
        gradient_loss = minkowski_loss_gradient(centr_pt, X)
        tangent_v = project_to_tangent(centr_pt, -gradient_loss)
        centr_pt = update_theta(centr_pt, tangent_v, alpha=alpha)
        centr_pts.append(centr_pt)
        losses.append(frechet_loss(centr_pt, X))
        if verbose:
            print('Epoch ' + str(i+1) + ' complete')
            print('Loss: ', frechet_loss(centr_pt, X))
            print('\n')
    return centr_pt

#-----------------------------------------------
#----- Hyperbolic K-Means Clustering Model -----
#-----------------------------------------------

class HyperbolicKMeans():
    """
    Perform K-Means clustering in hyperbolic space. Applies gradient descent in
    the hyperboloid model to iteratively compute Fréchet means, and the Poincaré disk
    model for visualization.
    
    API design is modeled on the standard scikit-learn Classifier API
    """
    
    def __init__(self, n_clusters=6, random_state=123, max_iter=300, tol=1e-8, verbose=False):
        np.random.seed(random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels = None
        self.cluster_centers_ = None
        
    def init_centroids(self, radius=0.3, dim=3):
        # randomly sample starting points on small uniform ball
        """
        theta = np.random.uniform(0, 2*np.pi, self.n_clusters)
        u = np.random.uniform(0, radius, self.n_clusters)
        r = np.sqrt(u)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        centers = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        self.centroids = centers
        """
        N = self.n_clusters
        norm = np.random.normal
        normal_deviates = norm(size=(dim, N))
        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        points = normal_deviates / radius
        points = points.reshape(-1, dim)
        self.centroids = points 
        # print('POINCARE KMEANS DEBUG: centroids shape: {}'.format(self.centroids.shape))
        
    def init_assign(self, labels=None):
        # cluster assignments as indicator matrix
        assignments = np.zeros((self.n_samples, self.n_clusters))
        for i in range(self.n_samples):
            if labels is not None:
                # assign to classes with ground truth input labels
                assignments[i][labels[i]] = 1
            else:
                # randomly initialize each binary vector
                j = np.random.randint(0, self.n_clusters)
                assignments[i][j] = 1
        self.assignments = assignments
        
    def update_centroids(self, X):
        """Updates centroids with Fréchet means in Hyperboloid model
        Parameters
        ----------
        X : array of shape (n_samples, dim) with input data.
        First convert X to hyperboloid points
        """
        dim = X.shape[1]
        new_centroids = np.empty((self.n_clusters, dim)) 
        H = poincare_pts_to_hyperboloid(X)
        # print('Centroids shape: {}'.format(self.centroids.shape))
        for i in range(self.n_clusters):
            if np.sum(self.assignments[:, i] ==1) == 0:
                new_centroids[i] = self.centroids[i]
            else:
                # find subset of observations in cluster
                H_k = H[self.assignments[:, i] ==1]
                # print(H_k.shape)
                theta_k = poincare_pt_to_hyperboloid(self.centroids[i])
                # print('Theta_k shape: {}'.format(theta_k.shape))
                # solve for frechet mean
                fmean_k = compute_mean(theta_k, H_k, alpha=0.1)
                # convert back to Poincare disk
                # print('HYPERBOLOID TO POINCARE SHAPE: {}'.format(hyperboloid_pt_to_poincare(fmean_k).shape))
                # print('NEW CENTROIDS SHAPE: {}'.format(new_centroids[i]))
                new_centroids[i] = hyperboloid_pt_to_poincare(fmean_k)
        self.centroids = new_centroids
        
    def cluster_var(self, X):
        n = self.centroids.shape[0]
        var_C = []
        for i in range(n):
            var_C.append(np.mean(np.array([poincare_dist(self.centroids[i], x) for x in X])))
        self.variances = np.sort(var_C)[-2]

    def fit_predict(self, X, y=None, max_epochs=40, verbose=False):
        self.fit(X, y, max_epochs, verbose)
        self.labels_ = self.predict(X)
        return self.labels_
    
    def fit(self, X, y=None, max_epochs=10, verbose=False):
        """
        Fit the K centroids from X, and return the class assignments by nearest centroid
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        
        # make sure X within poincaré ball
        #X = Normalizer().fit_transform(X)
        if (norm(X, axis=1) > 1).any():
            X = X / (np.max(norm(X, axis=1)))
        
        # initialize random centroids and assignments
        self.n_samples = X.shape[0]
        self.init_centroids()
        
        if y is not None:
            self.init_assign(y)
            self.update_centroids(X)
        else:
            self.init_assign()
        
        # loop through the assignment and update steps
        for j in tqdm(range(max_epochs)):
            self.inertia_ = 0
            self.update_centroids(X)
            for i in range(self.n_samples):
                # zero out current cluster assignment
                self.assignments[i, :] = np.zeros((1, self.n_clusters))
                # find closest centroid (in Poincare disk)
                centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
                cx = np.argmin(centroid_distances)
                self.inertia_ += centroid_distances[cx]**2
                self.assignments[i][cx] = 1
            if self.verbose:
                print('Epoch ' + str(j) + ' complete')
#                 print(self.centroids)
        self.labels = np.argmax(self.assignments, axis=1)
        self.labels_ = self.labels
        self.cluster_var(X)
        return
    
    def predict(self, X):
        """
        Predict class labels for given data points by nearest centroid rule
        Parameters
        ----------
        X : array, shape (n_samples, n_features). Observations to be assigned to the
        class represented by the nearest centroid.
        """
        # zero out current cluster assignment
        n = X.shape[0]
        labels = np.zeros((n, self.n_clusters))
        # find closest centroid (in Poincare disk)
        for i in tqdm(range(n)):
            centroid_distances = np.array([poincare_dist(X[i], centroid) for centroid in self.centroids])
            cx = np.argmin(centroid_distances)
            labels[i][cx] = 1
        return labels
