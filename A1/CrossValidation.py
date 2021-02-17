import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(data):
    if len(data['X']) == len(data['t']):
        num_sample = len(data['X'])
    else:
        raise ValueError('The input dataset is not paired')
    permutation = np.random.permutation(num_sample)
    xshf = []
    tshf = []
    for i in permutation:
        xshf.append(data['X'][i])
        tshf.append(data['t'][i])
    data_shf = {'X':np.array(xshf),'t':np.array(tshf)}
    return data_shf


def split_data(data,num_folds,fold):
    num_sample = len(data['X'])
    length_partition = int(num_sample/num_folds)
    if length_partition < num_sample/num_folds:
        raise ValueError("Number of Samples is supposed to be multiple of num_folds")
    data_fold_X = data['X'][(length_partition*(fold-1)):(length_partition*fold)]
    data_fold_t = data['t'][(length_partition*(fold-1)):(length_partition*fold)]
    # this if chunk may be redundent, I don't know if I simply use list subtraction, the index order will be affected
    # or not
    if num_folds == fold:
        data_rest_X = data['X'][0:(length_partition * (fold - 1))]
        data_rest_t = data['t'][0:(length_partition * (fold - 1))]
    elif num_folds > fold:
        data_rest_X_first_half = data['X'][0:(length_partition * (fold - 1))]
        data_rest_t_first_half = data['t'][0:(length_partition * (fold - 1))]
        data_rest_X_second_half = data['X'][(length_partition * fold):num_sample]
        data_rest_t_second_half = data['t'][(length_partition * fold):num_sample]
        data_rest_X = np.append(data_rest_X_first_half, data_rest_X_second_half, axis = 0)
        data_rest_t = np.append(data_rest_t_first_half, data_rest_t_second_half, axis = 0)
    else:
        raise ValueError('Not enough folds')
    data_fold = {'X':data_fold_X, 't':data_fold_t}
    data_rest = {'X':data_rest_X, 't':data_rest_t}
    return data_fold, data_rest


def train_model(data,lambd):
    step_1 = np.matmul(np.transpose(data['X']),data['X'])
    step_2 = step_1 + lambd*np.identity(len(data['X'][0]))
    step_3 = np.linalg.inv(step_2)
    step_4 = np.matmul(step_3,np.transpose(data['X']))
    model =  np.matmul(step_4,data['t'])
    return model


def predict(data,model):
    prediction = np.matmul(data['X'],model)
    return prediction


def loss(data,model):
    if len(data['X']) != len(data['t']):
        raise ValueError('The input dataset is not paired')
    else:
        num_sample = len(data['X'])
        prediction = predict(data,model)
        residuals = data['t'] - prediction
        error = (np.linalg.norm(residuals) ** 2)/num_sample
        return error


def cross_validation(data,num_folds,lambd_seq):
    data = shuffle_data(data)
    cv_error = [None] * len(lambd_seq)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0.
        for fold in range(1,num_folds+1):
            # split data returns data_fold for validation at 1st position, data_rest for training at 2nd position
            # which is consistent with val_cv for validation at 1st position, train_cv for training at 2nd position
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error


if __name__ == '__main__':
    data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
                  't': np.genfromtxt('data_train_y.csv', delimiter=',')}
    data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
                 't': np.genfromtxt('data_test_y.csv', delimiter=',')}

    lambd_seq = np.linspace(0.03,1.5)
    lambd_seq = list(lambd_seq)
    loss_train = [None]*50
    loss_test = [None]*50
    i=0
    for lambd in lambd_seq:
        model_train = train_model(data_train,lambd)
        loss_train[i] = loss(data_train,model_train)
        loss_test[i] = loss(data_test,model_train)
        i+=1
    loss_fold_5 = cross_validation(data_train,5,lambd_seq)
    loss_fold_10 = cross_validation(data_train,10,lambd_seq)

    for index in range(50):
        print('lambda:',round(lambd_seq[index],3),
              ',training error:',round(loss_train[index],5),
              ';testing error:',round(loss_test[index],5),
              ';5-fold cv error:',round(loss_fold_5[index],5),
              ';10-fold cv error:',round(loss_fold_10[index],5),sep='')
    #print('testing error in increasing lambda order:',loss_test)
    #print('5-fold cv error in increasing lambda order:',loss_fold_5)
    #print('10-fold cv error in increasing lambda order:',loss_fold_10)
    plt.plot(lambd_seq,loss_train,label='train')
    plt.plot(lambd_seq,loss_test,label='test')
    plt.plot(lambd_seq,loss_fold_5,label='5-fold CV')
    plt.plot(lambd_seq,loss_fold_10,label='10-fold CV')
    plt.xlabel('lambda')
    plt.ylabel('loss')
    plt.legend()
    plt.show()