from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()
import numpy as np
from scipy.special import logsumexp
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist(N_data=None):
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    K_data = 10
    train_labels = one_hot(train_labels, K_data)
    test_labels = one_hot(test_labels, K_data)
    if N_data is not None:
        train_images = train_images[:N_data, :]
        train_labels = train_labels[:N_data, :]

    return train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_log_regression(images, labels, learning_rate, max_iter):
    """ Used in Q1
        Inputs: train_images, train_labels, learning rate,
        and max num of iterations in gradient descent
        Returns the trained weights (w/o intercept)"""
    N_data, D_data = images.shape
    K_data = labels.shape[1]
    weights = np.zeros((D_data, K_data))

    # YOU NEED TO WRITE THIS PART
    y = np.zeros(shape=(N_data, K_data))
    for iter in range(max_iter):
        for n in range(N_data):
            for k in range(K_data):
                y[n][k] = np.exp(np.dot(images[n],weights[:,k]))/sum(np.exp(np.dot(images[n],weights)))
        gradient = np.matmul(np.transpose(images),y-labels)
        weights = weights - learning_rate*gradient

    w0 = None # No intercept for log-reg
    return weights, w0


def train_gda(images, labels):
    """ Used in Q2
        Inputs: train_images, train_labels
        Returns the trained weights, the intercept, and D x K class means, 
        and D x D common covariance matrix."""
    N_data, D_data = images.shape
    K_data = labels.shape[1]

    
    # YOU NEED TO WRITE THIS PART
    e = 1 / N_data
    error = e * np.identity(D_data)
    # Calculate estimator of p(C_k)=pi_k
    # pi is a vector of length K
    pi = labels.mean(0)
    # Calculate estimator of Mu
    # Mu is a K*D matrix
    Mu = np.matmul(images.T, labels)
    N_k = np.sum(labels, axis=0)
    for i in range(K_data):
        Mu[:, i] = Mu[:, i] / N_k[i]
    # Calculate estimator of Sigma
    # Sigma is a D*D matrix
    Sigma = np.zeros((D_data, D_data))
    for k in range(K_data):
        sampling_Sigma = np.zeros((D_data, D_data))
        for n in range(N_data):
            # this is not the actual sampling sigma since I omit 1/N_k term
            # reason for that is 1/N_k*(N_k/N) will cancel N_k out, and only 1/N remains
            # so in the next line I can simply divide by N_data, and do not to calculate N_k for every k
            if np.argmax(labels) == k:
                s = images[n] - Mu[:, k]
                s = np.reshape(s, (D_data, -1))
                s_t = np.reshape(s, (-1, D_data))
                s_2 = np.matmul(s, s_t)
                # feels like this complicated to do n*1 vector multiply by 1*n vector to get a n*n matrix
                sampling_Sigma = np.add(sampling_Sigma, s_2)
        Sigma = np.add(Sigma, sampling_Sigma / N_data)
    Sigma = np.add(Sigma, error)
    # Calculate estimator of w0
    # w0 is a vector of length K
    w0 = np.zeros((K_data))
    for k in range(K_data):
        w0[k] = w0[k] + (-1 / 2 * np.matmul(np.matmul(np.transpose(Mu[:, k]), np.linalg.inv(Sigma)), Mu[:, k]) + np.log(
            pi[k]))
    # Calculate estimator of weights
    # Weights is a N*K matrix
    weights = np.matmul(np.linalg.inv(Sigma), Mu)
    return weights, w0, Mu, Sigma


def log_softmax(images, weights, w0=None):
    """ Used in Q1 and Q2
        Inputs: images, and weights
        Returns the log_softmax values."""
    if w0 is None: w0 = np.zeros(weights.shape[1])
    # YOU NEED TO WRITE THIS PART
    # a N*K matrix is supposed to be returned
    N_data = images.shape[0]
    K_data = weights.shape[1]
    # append w0 to weights
    weights = np.concatenate(([w0], weights), axis=0)
    # append a column of 1 to images matrix to accommdate w0
    x0 = np.full((N_data, 1), 1)
    images = np.concatenate((x0, images), axis=1)
    result = np.zeros(shape=(N_data,K_data))

    for n in range(N_data):
        for k in range(K_data):
            # softmax is a fraction,add log at front it will become subtraction
            # and the first term is simply log of e^f, result is the exponent f
            result[n][k] = np.dot(images[n],weights[:,k])-logsumexp(np.dot(images[n],weights))

    return result


def cross_ent(log_Y, train_labels):
    """ Used in Q1
        Inputs: log of softmax values and training labels
        Returns the cross entropy."""

    # YOU NEED TO WRITE THIS PART
    N_data = train_labels.shape[0]
    cross_entropy=0
    for i in N_data:
        cross_entropy+=1*log_Y[i][np.argmax(train_labels)]
    cross_entropy=-cross_entropy
    return cross_entropy


def predict(log_softmax):
    """ Used in Q1 and Q2
        Inputs: matrix of log softmax values
        Returns the predictions"""
    # refer to piazza, the return value of prediction is suppsoed to be vector[N],not a matrix
    # YOU NEED TO WRITE THIS PART

    prediction = np.argmax(log_softmax,axis=1)
    return prediction


def accuracy(log_softmax, labels):
    """ Used in Q1 and Q2
        Inputs: matrix of log softmax values and 1-of-K labels
        Returns the accuracy based on predictions from log likelihood values"""
    
    # YOU NEED TO WRITE THIS PART
    prediction = predict(log_softmax)

    num_correct = 0
    num_sample = len(labels)

    for i in range(num_sample):
        if labels[i][prediction[i]] != 0:
            num_correct+=1
    #just confused about how to write it in a compact way in python
    accuracy=num_correct/num_sample
    return accuracy


def main():
    N_data = None # Num of samples to be used in training
    # Set this to a small number while experimenting.
    # For log reg, finally use the entire training dataset for training (N_data=None).
    # For gda, use as many training samples as your computer can handle.
    
    train_images, train_labels, test_images, test_labels = load_mnist(N_data)

    # Q1: train logistic regression
    #learning_rate, max_iter = .00001, 100
    #weights, w0 = train_log_regression(train_images, train_labels, learning_rate, max_iter)
    #save_images(weights.T, 'weights.png')

    #Q2: train gaussian discriminant
    weights, w0, Mu, Sigma = train_gda(train_images, train_labels)
    save_images(Mu.T, 'means.png')

    new_digit = 3
    new_images = np.random.multivariate_normal(Mu[:, new_digit], Sigma, 10)
    save_images((new_images > .5).astype(float), 'new_images.png')

    # evaluation
    log_softmax_train = log_softmax(train_images, weights, w0)
    log_softmax_test = log_softmax(test_images, weights, w0)
    
    train_accuracy = accuracy(log_softmax_train, train_labels)
    test_accuracy = accuracy(log_softmax_test[:1000], test_labels[:1000])

    print("Training accuracy is ", train_accuracy)
    print("Test accuracy is ", test_accuracy)


if __name__ == '__main__':
    main()