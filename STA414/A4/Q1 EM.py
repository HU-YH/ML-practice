import scipy.stats
import numpy as np
import itertools
import matplotlib.pyplot as plt

def normal_density(x, mu, Sigma):
    return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
        / np.sqrt(np.linalg.det(2 * np.pi * Sigma))


def log_likelihood(data, Mu, Sigma, Pi):
    """ Compute log likelihood on the data given the Gaussian Mixture Parameters.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        L: a scalar denoting the log likelihood of the data given the Gaussian Mixture
    """
    # Fill this in:
    N, D = data.shape  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of mixtures
    L, T = 0., 0.
    for n in range(N):
        T = 0
        for k in range(K):
            print(Sigma[k])
            T += Pi[k] * scipy.stats.multivariate_normal(mean=Mu[:, k], cov=Sigma[k]).pdf(data[n])  # Compute the likelihood from the k-th Gaussian weighted by the mixing coefficients
        L += np.log(T)
    return L


def gm_e_step(data, Mu, Sigma, Pi):
    """ Gaussian Mixture Expectation Step.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients

    Returns:
        Gamma: a NxK matrix of responsibilities
    """
    # Fill this in:
    N, D = data.shape  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of mixtures
    Gamma = np.zeros((N, K))  # zeros of shape (N,K), matrix of responsibilities
    for n in range(N):
        for k in range(K):
            Gamma[n, k] = Pi[k] * scipy.stats.multivariate_normal(mean=Mu[:, k], cov=Sigma[k]).pdf(data[n])
        Gamma[n, :] /= np.sum(Pi[k] * scipy.stats.multivariate_normal(mean=Mu[:, k], cov=Sigma[k]).pdf(data[n]))  # Normalize by sum across second dimension (mixtures)
    return Gamma


def gm_m_step(data, Gamma):
    """ Gaussian Mixture Maximization Step.

    Args:
        data: a NxD matrix for the data points
        Gamma: a NxK matrix of responsibilities

    Returns:
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients
    """
    # Fill this in:
    N, D = data.shape  # Number of datapoints and dimension of datapoint
    K = Gamma.shape[1]  # number of mixtures
    Gamma_converted = np.zeros((N, K))
    Gamma_converted[np.arange(0, N, 1), np.argmax(Gamma, axis=1)] = 1
    Nk = np.sum(Gamma_converted, axis=0)  # Sum along first axis
    Mu = np.matmul(data.T, Gamma) / Nk
    Sigma = np.zeros((K, D, D))

    for k in range(K):
        count = np.zeros((D,D))
        for n in range(N):
            residual = data[n]-Mu[:,k]
            residual = np.reshape(residual,(D,-1))
            residual_t = np.reshape(residual,(-1,D))
            squared_residual = np.matmul(residual,residual_t)
            count += Gamma[n,k]*squared_residual
        Sigma[k] = count/Nk[k]
    Pi = Nk / N
    return Mu, Sigma, Pi

if __name__ == "__main__":
    num_samples = 400
    cov = np.array([[1., .7], [.7, 1.]]) * 10
    mean_1 = [.1, .1]
    mean_2 = [6., .1]

    x_class1 = np.random.multivariate_normal(mean_1, cov, num_samples // 2)
    x_class2 = np.random.multivariate_normal(mean_2, cov, num_samples // 2)
    xy_class1 = np.column_stack((x_class1, np.zeros(num_samples // 2)))
    xy_class2 = np.column_stack((x_class2, np.ones(num_samples // 2)))
    data_full = np.row_stack([xy_class1, xy_class2])
    np.random.shuffle(data_full)
    data = data_full[:, :2]
    labels = data_full[:, 2]

    N, D = data.shape
    K = 2
    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    Sigma = [np.eye(2), np.eye(2)]
    Pi = np.ones(K) / K
    Gamma = np.zeros([N, K])  # Gamma is the matrix of responsibilities

    max_iter = 200

    for it in range(2):
        Gamma = gm_e_step(data, Mu, Sigma, Pi)
        Mu, Sigma, Pi = gm_m_step(data, Gamma)
        print(it, log_likelihood(data, Mu, Sigma, Pi)) # This function makes the computation longer, but good for debugging

    class_1 = np.where(Gamma[:, 0] >= .5)
    class_2 = np.where(Gamma[:, 1] >= .5)

    class_1 = np.where(Gamma[:, 0] >= .5)
    class_2 = np.where(Gamma[:, 1] >= .5)

    correct_class_1 = num_samples // 2 - sum(labels[class_1])
    correct_class_2 = sum(labels[class_2])
    num_correct = correct_class_1 + correct_class_2
    print('Missclassification rate of EM algorithm is ', num_correct / 400 * 100, '%')
    x_class1_new = data[class_1]
    x_class2_new = data[class_2]
    print("##########################Plot of EM algorithm#################################")
    plt.plot(x_class1_new[:, 0], x_class1_new[:, 1], 'X')  # first class, x shape
    plt.plot(x_class2_new[:, 0], x_class2_new[:, 1], 'o')  # second class, circle shape
    plt.show()