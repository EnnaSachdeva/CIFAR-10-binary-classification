"""
Enna Sachdeva
sachdeve@oregonstate.edu
"""


from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np


##########  This code describes linear transformation, ReLU, sigmoid cross-entropy layers as separate classes

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step

class LinearTransform:

    def __init__(self, W, b):
        # W is a nxm matrix
        # n: dimension of input x
        # m: number of hidden units
        # b: 1xm bias vector


        self.weight = W
        self.bias = b

    # DEFINE __init function

    def forward(self, x): # return linear vector, z = Wx+b
        return np.add(np.matmul(x, self.weight), self.bias)



# This is a class for a ReLU layer max(x,0)
class ReLU:

    def forward(self, x):
        return np.maximum(x, np.zeros(x.shape[0]))


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

    def sigmoid(self, x):
        """ Applies sigmoid function to x. Clips the signal to avoid numerical errors. """
        x = np.clip(x, -25, 25)
        return 1 / (1 + np.exp(-x))

    def crossEntropy(self, yhat, y):
        """ Returns the cross entropy error for binary classification. """
        entropy = np.multiply(y, np.log(yhat)) + np.multiply(1 - y, np.log(1 - yhat))
        return -entropy

    def forward(self, x, y):
        z = 1 / (1 + np.exp(-x))
        return -(np.multiply(y, np.log(z)) + np.multiply(1 - y, np.log(1 - z)))

# This is a class for the Multilayer perceptron


#############  A function that evaluates the trained network as well as computes the subgradients of W1 and W2 using backpropagation
#############  Neural Network for Training
class NeuralNetwork:

    def __init__(self, input_dims, hidden_units, output_dims):
        #self.learning_rate = 0.01
        self.dimensions = (input_dims, hidden_units, output_dims) # initialize weights for 1st layer
        self.W1 = np.random.randn(input_dims, hidden_units)* 0.1
        self.b1 = np.random.randn(1, hidden_units)* 0.1

        self.W2 = np.random.randn(hidden_units, output_dims) * 0.1# initialize weights for 2nd layer
        self.b2 = np.random.randn(1, output_dims)* 0.1

        self.dLdW1 = np.zeros((input_dims, hidden_units))
        self.dLdb1 = np.zeros((1, hidden_units))
        self.dLdW2 = np.zeros((hidden_units, output_dims))
        self.dLdb2 = np.zeros((1, output_dims))


        self.prev_dLdW1 = np.zeros((input_dims, hidden_units))
        self.prev_dLdb1 = np.zeros((1, hidden_units))
        self.prev_dLdW2 = np.zeros((hidden_units, output_dims))
        self.prev_dLdb2 = np.zeros((1, output_dims))

    def forward(self, x):
        """ Get output from given input vector. """
        z1 = LinearTransform(self.W1, self.b1).forward(x)
        a1 = ReLU().forward(z1)
        z2 = LinearTransform(self.W2, self.b2).forward(a1)
        yhat = SigmoidCrossEntropy().sigmoid(z2) # just take the sigmoid function
        return yhat


    def calculate_gradients(self, y, yhat, a1, z1, x):

        dLdz2 = yhat -y
        #da2dz2 = np.matmul((1-a2).T, a2)
        dz2dW2 = a1
        dz2db2 = 1


        # da1dz1 : output of RELU
        dz1dW1 = x
        dz1db1 = 1

        dLdW2 = np.matmul(dz2dW2.T, dLdz2)      #dLda2 * da2dz2 * dz2dW2
        dLdb2 = dLdz2                          #dLda2 * da2dz2 * dz2db2
        dLdW1 = np.matmul(np.matmul(x.T, yhat - y), self.W2.T)   #dLda2 * da2dz2 * dz2da1 * dz1dW1
        dLdb1 = np.matmul(yhat - y, self.W2.T)    #dLda2 * da2dz2 * dz2da1 * dz1db1

        #dLW2 = np.matmul(a1.T, a2 - y)
        #dLdb2 = yhat -y
        #dLdW1 = np.matmul(np.matmul(x.T, a2 - y), self.W2.T)
        #dLdb1 = np.matmul(a2 - y, self.W2.T)

        dLdW1 = np.asarray([dLdW1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdW1.T[j])) for j in range(len(dLdW1.T))]).T
        dLdb1 = np.asarray([dLdb1.T[j] if z1[0, j] > 0 else np.zeros(len(dLdb1.T[j])) for j in range(len(dLdb1.T))]).T

        return dLdW1, dLdb1, dLdW2, dLdb2


    def backpropagation(self, x_batch, y_batch, learning_rate, momentum):

        input_dims, hidden_units, output_dims = self.dimensions

        # initialize gradients
        dLdW1 = np.zeros((input_dims, hidden_units))
        dLdW2 = np.zeros((hidden_units, output_dims))
        dLdb1 = np.zeros((1, hidden_units))
        dLdb2 = np.zeros((1, output_dims))

        Loss = 0
        Count = 0
        for m in range(len(x_batch)): # for each training example of the mini batch
            x = x_batch[m]  # take one element at a time
            y = y_batch[m]  # take one element at a time


            ######## Calculate output of Neural Network ###############
            z1 = LinearTransform(self.W1, self.b1).forward(x)
            a1 = ReLU().forward(z1)
            z2 = LinearTransform(self.W2, self.b2).forward(a1)
            yhat = SigmoidCrossEntropy().sigmoid(z2)
            #yhat1= self.forward(x)
            loss= sum(SigmoidCrossEntropy().crossEntropy(yhat, y)[0])
            #loss = SigmoidCrossEntropy().crossEntropy(yhat, y)[0]

            Loss = Loss+loss
            if ((y == 0) and (yhat < 0.5)) or ((y == 1) and (yhat >= 0.5)):
                Count =Count + 1

            grad_L_W1, grad_L_b1, grad_L_W2, grad_L_b2 = self.calculate_gradients(y, yhat, a1, z1, x)


            dLdW1 = dLdW1 + grad_L_W1
            dLdb1 = dLdb1 + grad_L_b1
            dLdW2 = dLdW2 + grad_L_W2
            dLdb2 = dLdb2 + grad_L_b2

        # Get the average gradient
        dLdW2 /= len(x_batch)
        dLdb2 /= len(x_batch)
        dLdW1 /= len(x_batch)
        dLdb1 /= len(x_batch)

        # Add previous gradients for momentum
        dLdW2 = momentum * self.prev_dLdW2 + (1 - momentum) * dLdW2
        dLdb2 = momentum * self.prev_dLdb2 + (1 - momentum) * dLdb2
        dLdW1 = momentum * self.prev_dLdW1 + (1 - momentum) * dLdW1
        dLdb1 = momentum * self.prev_dLdb1 + (1 - momentum) * dLdb1

        # Save previous gradients for momentum
        self.prev_dLdW2 = dLdW2
        self.prev_dLdb2 = dLdb2
        self.prev_dLdW1 = dLdW1
        self.prev_dLdb1 = dLdb1


        # Update weights
        self.W2 -= dLdW2 * learning_rate
        self.b2 -= dLdb2 * learning_rate
        self.W1 -= dLdW1 * learning_rate
        self.b1 -= dLdb1 * learning_rate

        return Loss/len(x_batch), Count/len(x_batch)

    def get_batch(self, X, y, batch_size):
        arrange_data = np.arange(0, X.shape[0])
        indices = np.random.choice(arrange_data, batch_size)  # gives 32 random values out of 10000 data set
        batch_data = X[indices].reshape(len(X[indices]), 1, len(X[indices][0]))  # converts to 32*1*3072
        batch_label = y[indices]

        #batch = zip(X[batch_idx:batch_idx + batch_size],
        #            y[batch_idx:batch_idx + batch_size])
        #batch = zip(batch_data, batch_label)
        return batch_data, batch_label


def train():
   #--------------------- Training -------------------------------#
    mlp = NeuralNetwork(input_dims, hidden_units, output_dims)
    losses = []
    for epoch in range(num_epochs):

        total_loss = 0
        train_accuracy = 0
        test_accuracy = 0
        test_loss = 0
        for b in range(num_batches):

            #batch = batch_iter.__next__()
            batch_data, batch_label = mlp.get_batch(train_data, train_label, batch_size)
            #### calculate the loss from the neural network

            #for m in range(len(batch_data)):  # for each training example of the mini batch
            #    x = batch_data[m]  # take one element at a time
            #    y = batch_label[m]  # take one element at a time
            batch_loss, batch_accuracy = mlp.backpropagation(batch_data, batch_label, learning_rate, momentum)
            total_loss = total_loss + batch_loss
            train_accuracy = train_accuracy + batch_accuracy


            #print(len(losses), " LOSS: ", batch_loss / batch_size)

        losses.append(total_loss / (batch_size*num_batches)) # average loss per epoch

        #print(total_loss)

        ######### testing objective

        for l in range(test_data.shape[0]):
            yhat = mlp.forward(test_data[l])
            y = test_label[l]
            loss = sum(SigmoidCrossEntropy().crossEntropy(yhat, y)[0])
            # loss = SigmoidCrossEntropy().crossEntropy(yhat, y)[0]
            test_loss = test_loss + loss

            if ((y == 0) and (yhat < 0.5)) or ((y == 1) and (yhat >= 0.5)):
                test_accuracy = test_accuracy + 1

        print('Epoch {}  Avg Train Loss = {:.3f}  Train Acc. = {:.2f} Average Test Loss. = {:.2f} Test Acc. = {:.2f}%'.format(epoch + 1, total_loss/num_batches, train_accuracy*100 / num_batches, test_loss/test_data.shape[0], test_accuracy*100/test_data.shape[0] ))

    ######## save model after it has been trained
    pickle.dump(mlp, open("Models/{0}Mom_{1}Batch_{2}LR_{3}_Hid.p".format(momentum, batch_size, learning_rate,
                                                                      hidden_units),
                         "wb"))



def evaluate():
    ######## load model
    #mlp = NeuralNetwork(input_dims, hidden_units, output_dims)
    mlp = pickle.load(open("Models/{0}Mom_{1}Batch_{2}LR_{3}_Hid.p".format(momentum, batch_size, learning_rate, hidden_units), "rb"))

    #--------------------- Testing -------------------------------#
    test_loss = 0
    test_accuracy = 0
    for i in range(test_data.shape[0]):
        yhat = mlp.forward(test_data[i])
        y = test_label[i]
        loss = sum(SigmoidCrossEntropy().crossEntropy(yhat, y)[0])
        #loss = SigmoidCrossEntropy().crossEntropy(yhat, y)[0]
        test_loss = test_loss + loss

        if ((y==0) and (yhat < 0.5)) or ((y==1) and (yhat>=0.5)):
            test_accuracy = test_accuracy + 1


    print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss/(i+1), 100. * test_accuracy/(i+1),))
    return test_loss/(i+1), 100. * test_accuracy/(i+1)

if __name__ == '__main__':

    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_data = data[b'train_data']
    train_label = data[b'train_labels']
    test_data = data[b'test_data']
    test_label = data[b'test_labels']


    ####### I coded this in Pycharm, so while evaluating this assignment, if the above 4 lines show error,
    ####### please comment those and uncomment the below lines.

    #train_data = data['train_data']
    #train_label = data['train_labels']
    #test_data = data['test_data']
    #test_label = data['test_labels']

    train_data = train_data/255
    test_data = test_data/255

    num_examples, input_dims = train_data.shape

    num_epochs = 40
    #batch_size_set= [16, 64, 256, 512, 1024, 2048, 6000, 8000, 9500]

    batch_size = 64
    num_batches = train_data.shape[0] // batch_size
    #num_batches = 100
    learning_rate = 0.01
    momentum = 0.5
    hidden_units = 20
    output_dims = 1


    #hidden_units_set = [10 , 20, 50, 100, 150, 200, 250]
    acc_batchsize = []
    learn = []

    count = 0

    train()
    av_test_loss, av_test_acc = evaluate()

    '''
    for k in range(len(hidden_units_set)):
        hidden_units = hidden_units_set[k]
        train() # function to train the model
        av_test_loss, av_test_acc = evaluate() # function to test the model
        acc_batchsize.append(av_test_acc)

    #### plotting
    # Draw lines
    plt.plot(hidden_units_set, acc_batchsize, linestyle='solid', markerfacecolor='blue',
             markersize=10,  label="Test accuracy ")

    

    # Create plot
    plt.title("Testing accuracy w.r.t hidden layer nodes ")
    plt.xlabel("Hidden Layer nodes"), plt.ylabel("Accuracy percentage"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    '''