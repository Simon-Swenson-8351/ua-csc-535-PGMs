import random as rng
import math
import numpy as np
import matplotlib.pyplot as plt

max_iterations = 200
min_log_likelihood_delta = 1.0e-8
filenames = [
    'gaussian-3-clusters-A.txt',
    'independent-gauss-3-clusters-A.txt',
    'independent-gauss-3-clusters-B.txt',
    'independent-gauss-5-clusters-A.txt',
    'independent-uniform-5-clusters-A.txt',
    'independent-uniform-5-clusters-B.txt'
]

# https://en.wikipedia.org/wiki/Multivariate_normal_distribution
# PDF = det(2 * pi * covar_matrix)^(-1/2) * e^(-1/2 * (x - mu)^T * inverse(covar_matrix) * (x - mu))
def gaussian_pdf(x, mean, covar_matrix, inv_covar_matrix = []):
    x_diff = x - mean
    # Can't do a none check or numpy complains
    if len(inv_covar_matrix) == 0:
        inv_covar_matrix = np.linalg.inv(covar_matrix)
    return np.linalg.det(2 * math.pi * covar_matrix) ** (-1 / 2) * math.e ** (-1 / 2 * np.transpose(x_diff) @ inv_covar_matrix @ x_diff)

def parseInputRecord(record):
    result = []
    split = record.split()
    for feature in split:
        result.append(float(feature))
    return result

def parseInputFile(inputFile):
    result = []
    for line in inputFile:
        result.append(parseInputRecord(line))
    result = np.array(result)
    return result

class Gmm:

    def __init__(self, X, k, observer, diagonal = False):
        # Either initialize random model parameters or assign expectations.
        # We choose to assign expectations here. Requires less lines of code.
        self.k = k
        self.X = X
        self.observer = observer
        self.diagonal = diagonal

        self.responsibilities = np.random.rand(self.X.shape[0], self.k)
        self.responsibilities /= np.sum(self.responsibilities, axis = 1).reshape((self.responsibilities.shape[0], 1))
        self.priors = None
        self.means = None
        self.covar_matrices = None
        self.inv_covar_matrices = None
        
        while True:
            # Maximization
            self.calc_priors()
            self.calc_means()
            if self.diagonal:
                self.calc_diagonal_covar_matrices()
            else:
                self.calc_covar_matrices()

            # Expectation
            (self.responsibilities, self.most_likely_clusters) = self.calc_responsibilities(self.X)

            self.observer.broadcast_iteration(self.X, self.priors, self.means, self.covar_matrices, self.responsibilities)
            if self.observer.stop_iteration(self.X, self.priors, self.means, self.covar_matrices, self.responsibilities):
                break

    # Provide X to allow use with external X, for predictions.
    def calc_responsibilities(self, X):
        r1 = np.zeros((X.shape[0], self.means.shape[0]))
        for x_idx in range(X.shape[0]):
            normalizer = 0.0
            for c_idx in range(self.means.shape[0]):
                r1[x_idx, c_idx] = self.priors[c_idx] * gaussian_pdf(X[x_idx], self.means[c_idx], self.covar_matrices[c_idx], self.inv_covar_matrices[c_idx])
                normalizer += r1[x_idx, c_idx]
            r1[x_idx] /= normalizer
        r2 = np.argmax(r1, axis = 1)
        return (r1, r2)

    def calc_priors(self):
        self.priors = np.zeros((self.responsibilities.shape[1]))
        for c_idx in range(self.responsibilities.shape[1]):
            for x_idx in range(self.responsibilities.shape[0]):
                self.priors[c_idx] += self.responsibilities[x_idx, c_idx]

        self.priors /= self.responsibilities.shape[0]

    def calc_means(self):
        self.means = np.zeros((self.responsibilities.shape[1], self.X.shape[1]))
        for c_idx in range(self.responsibilities.shape[1]):
            normalizer = 0.0
            for x_idx in range(self.X.shape[0]):
                self.means[c_idx] += self.X[x_idx] * self.responsibilities[x_idx, c_idx]
                normalizer += self.responsibilities[x_idx, c_idx]
            self.means[c_idx] /= normalizer

    def calc_diagonal_covar_matrices(self):
        self.covar_matrices = np.zeros((self.means.shape[0], self.means.shape[1], self.means.shape[1]))
        for c_idx in range(self.covar_matrices.shape[0]):
            for feature_idx in range(self.covar_matrices.shape[1]):
                normalizer = 0.0
                for x_idx in range(self.X.shape[0]):
                    self.covar_matrices[c_idx, feature_idx, feature_idx] += self.responsibilities[x_idx, c_idx] * (self.X[x_idx, feature_idx] - self.means[c_idx, feature_idx]) ** 2
                    normalizer += self.responsibilities[x_idx, c_idx]
                self.covar_matrices[c_idx, feature_idx, feature_idx] /= normalizer
        # Might as well calculate the inverses at the same time we calculate the 
        # matrices to keep things consistent outside of this function.
        self.inv_covar_matrices = np.zeros(self.covar_matrices.shape)
        for c_idx in range(self.covar_matrices.shape[0]):
            self.inv_covar_matrices[c_idx] = np.linalg.inv(self.covar_matrices[c_idx])

    def calc_covar_matrices(self):
        self.covar_matrices = np.zeros((self.means.shape[0], self.means.shape[1], self.means.shape[1]))
        for c_idx in range(self.covar_matrices.shape[0]):
            for feature_1_idx in range(self.covar_matrices.shape[1]):
                for feature_2_idx in range(self.covar_matrices.shape[2]):
                    normalizer = 0.0
                    for x_idx in range(self.X.shape[0]):
                        self.covar_matrices[c_idx, feature_1_idx, feature_2_idx] += self.responsibilities[x_idx, c_idx] * (self.X[x_idx, feature_1_idx] - self.means[c_idx, feature_1_idx]) * (self.X[x_idx, feature_2_idx] - self.means[c_idx, feature_2_idx])
                        normalizer += self.responsibilities[x_idx, c_idx]
                    self.covar_matrices[c_idx, feature_1_idx, feature_2_idx] /= normalizer
        # Might as well calculate the inverses at the same time we calculate the 
        # matrices to keep things consistent outside of this function.
        self.inv_covar_matrices = np.zeros(self.covar_matrices.shape)
        for c_idx in range(self.covar_matrices.shape[0]):
            self.inv_covar_matrices[c_idx] = np.linalg.inv(self.covar_matrices[c_idx])

    def predict(self, X):
        (responsibilities, result) = self.calc_responsibilities(X)
        return result
        

# Abstract observer class to define interactions with the GMM creation process.
class GmmObserver:
    # Don't need X_tr, since the make_gmm callback interface is designed to take X.
    def __init__(self):
        self.iteration_callbacks = []
        self.stopping_condition_callbacks = []

    def broadcast_iteration(self, X, priors, means, covar_matrices, responsibilities):
        for iter_callback in self.iteration_callbacks:
            iter_callback(X, priors, means, covar_matrices, responsibilities)

    def stop_iteration(self, X, priors, means, covar_matrices, responsibilities):
        # This is an "or" condition. Maybe could consider refactoring to allow 
        # for other boolean operators. That would basically require a tree 
        # structure, though. Not something I want to consider for now.
        for stop_cond_callback in self.stopping_condition_callbacks:
            if stop_cond_callback(X, priors, means, covar_matrices, responsibilities):
                return True
        return False

# Default implementation of the above abstract class, as per the assignment 
# specifications.
class DefaultGmmObserver(GmmObserver):
    # Don't need X_tr, since the make_gmm callback interface is designed to take X.
    def __init__(self, X_te, max_iterations, min_log_likelihood_delta):
        GmmObserver.__init__(self)

        self.iteration_callbacks.append(self.iteration_callback)
        self.stopping_condition_callbacks.append(self.stopping_condition_callback)

        self.i = 0
        self.n = max_iterations
        self.X_te = X_te
        self.tr_log_likelihoods = []
        self.te_log_likelihoods = []

        # TODO remove this. It's just used for debugging.
        self.problematic_points = []

    def calc_log_likelihood(self, X, priors, means, covar_matrices, responsibilities):
        result = 0.0
        best_guess_clusters = np.argmax(responsibilities, axis = 1)
        for x_idx in range(X.shape[0]):
            try:
                x_diff = X[x_idx] - means[best_guess_clusters[x_idx]]
                inv_covar_matrix = np.linalg.inv(covar_matrices[best_guess_clusters[x_idx]])
                result += math.log(priors[best_guess_clusters[x_idx]]) + math.log(np.linalg.det(2 * math.pi * covar_matrices[best_guess_clusters[x_idx]]) ** (-1 / 2)) + (-1 / 2 * np.transpose(x_diff) @ inv_covar_matrix @ x_diff)
            except:
                # Sometimes, test data wasn't well represented by the distributions.
                # This could be big outliers.
                result = -math.inf
                # TODO remove this. It's for debugging.
                self.problematic_points.append(X[x_idx])
        return result

    def iteration_callback(self, X, priors, means, covar_matrices, responsibilities):
        self.i += 1
        self.tr_log_likelihoods.append(self.calc_log_likelihood(X, priors, means, covar_matrices, responsibilities) / X.shape[0])
        self.te_log_likelihoods.append(self.calc_log_likelihood(self.X_te, priors, means, covar_matrices, responsibilities) / self.X_te.shape[0])

    def stopping_condition_callback(self, X, priors, means, covar_matrices, responsibilities):
        if self.i >= self.n:
            return True
        if len(self.tr_log_likelihoods) > 1:
            former = self.tr_log_likelihoods[len(self.tr_log_likelihoods) - 2]
            diff = self.tr_log_likelihoods[len(self.tr_log_likelihoods) - 1] - former
            if abs(diff / former) <= min_log_likelihood_delta:
                return True
        return False

if __name__ == '__main__':
    rng.seed(42)
    np.random.seed(42)

    # For comparing random initialization
    '''
    for inputFileName in [filenames[0], filenames[2]]:
        X = None
        with open(inputFileName, 'r') as inputFile:
            X = parseInputFile(inputFile)
        for i in range(4):
            np.random.shuffle(X)
            X_tr = X[:(9 * X.shape[0] // 10)]
            X_te = X[(9 * X.shape[0] // 10):]

            obs = DefaultGmmObserver(X_te, max_iterations, min_log_likelihood_delta)
            diag_gmm = Gmm(X_tr, 3, obs, diagonal = True)

            plt.scatter(X[:, 0], X[:, 1], c = diag_gmm.predict(X))
            plt.show()
    '''

    # Different k values
    for i in range(len(filenames)):
        inputFileName = filenames[i]
        X = None
        with open(inputFileName, 'r') as inputFile:
            X = parseInputFile(inputFile)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1])
        fig.savefig('gmm_ds' + str(i) + '_kn'+ '_input.png')
        plt.close(fig)

        for k in [3, 5]:
            for n in range(5):
                np.random.shuffle(X)
                X_tr = X[:(9 * X.shape[0] // 10)]
                X_te = X[(9 * X.shape[0] // 10):]

                obs = DefaultGmmObserver(X_te, max_iterations, min_log_likelihood_delta)
                diag_gmm = Gmm(X_tr, k, obs, diagonal = True)
                print(obs.problematic_points)

                fig, ax = plt.subplots()
                ax.scatter(X[:, 0], X[:, 1], c = diag_gmm.predict(X))
                fig.savefig('gmm_ds' + str(i) + '_k' + str(k) + '_orthog_plt' + '_trial' + str(n) + '.png')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(range(len(obs.tr_log_likelihoods)), obs.tr_log_likelihoods, range(len(obs.tr_log_likelihoods)), obs.te_log_likelihoods)
                fig.savefig('gmm_ds' + str(i) + '_k' + str(k) + '_orthog_ll' + '_trial' + str(n) + '.png')
                plt.close(fig)

                obs = DefaultGmmObserver(X_te, max_iterations, min_log_likelihood_delta)
                cool_gmm = Gmm(X_tr, k, obs)
                print(obs.problematic_points)

                fig, ax = plt.subplots()
                ax.scatter(X[:, 0], X[:, 1], c = cool_gmm.predict(X))
                fig.savefig('gmm_ds' + str(i) + '_k' + str(k) + '_covar_plt' + '_trial' + str(n) + '.png')
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(range(len(obs.tr_log_likelihoods)), obs.tr_log_likelihoods, range(len(obs.tr_log_likelihoods)), obs.te_log_likelihoods)
                fig.savefig('gmm_ds' + str(i) + '_k' + str(k) + '_covar_ll' + '_trial' + str(n) + '.png')
                plt.close(fig)

