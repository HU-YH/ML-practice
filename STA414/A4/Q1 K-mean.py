import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt

def cost(data, R, Mu):
    N, D = data.shape
    K = Mu.shape[1]
    J = 0
    for k in range(K):
        J += np.dot(np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2, R[:, k])
    return J


def km_assignment_step(data, Mu):
    """ Compute K-Means assignment step

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the cluster means locations

    Returns:
        R_new: a NxK matrix of responsibilities
    """

    # Fill this in:
    N, D = data.shape
    K = Mu.shape[1]
    r = np.zeros((N, K))
    for k in range(K):
        r[:, k] = np.linalg.norm(data - Mu[:, k], ord=2, axis=1) ** 2
    arg_min = np.argmin(r, axis=1)  # argmax/argmin along dimension 1
    R_new = np.zeros((N, K))  # Set to zeros/ones with shape (N, K)
    R_new[np.arange(0, N, 1), arg_min] = 1  # Assign to 1
    return R_new


def km_refitting_step(data, R, Mu):
    """ Compute K-Means refitting step.

    Args:
        data: a NxD matrix for the data points
        R: a NxK matrix of responsibilities
        Mu: a DxK matrix for the cluster means locations

    Returns:
        Mu_new: a DxK matrix for the new cluster means locations
    """
    N, D = data.shape  # Number of datapoints and dimension of datapoint
    K = Mu.shape[1]  # number of clusters
    Mu_new = np.zeros((D, K))
    for i in range(N):
        Mu_new[:, np.argmax(R[i])] += data[i]
    Mu_new = Mu_new / np.sum(R, axis=0)
    # np.sum(R,axis=0) returns
    # an 1*k vector, number of datapoitns assigned to each class
    return Mu_new


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

    plt.plot(x_class1[:, 0], x_class1[:, 1], 'X')  # first class, x shape
    plt.plot(x_class2[:, 0], x_class2[:, 1], 'o')  # second class, circle shape
    plt.show()

    N, D = data.shape
    K = 2
    max_iter = 100
    class_init = np.random.binomial(1., .5, size=N)
    R = np.vstack([class_init, 1 - class_init]).T

    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    R.T.dot(data), np.sum(R, axis=0)

    for it in range(max_iter):
        R = km_assignment_step(data, Mu)
        Mu = km_refitting_step(data, R, Mu)
        print(it, cost(data, R, Mu))

    class_1 = np.where(R[:, 0])
    class_2 = np.where(R[:, 1])


    correct_class_1 = num_samples // 2 - sum(labels[class_1])
    correct_class_2 = sum(labels[class_2])
    num_correct = correct_class_1 + correct_class_2
    print('Missclassification rate of K-mean clustering is ', round(1-num_correct/400, 3)*100, '%')
    x_class1_new = data[class_1]
    x_class2_new = data[class_2]
    plt.plot(x_class1_new[:, 0], x_class1_new[:, 1], 'X')  # first class, x shape
    plt.plot(x_class2_new[:, 0], x_class2_new[:, 1], 'o')  # second class, circle shape
    plt.show()
