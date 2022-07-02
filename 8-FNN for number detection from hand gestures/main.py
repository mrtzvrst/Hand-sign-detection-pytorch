import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch 
from torch import nn, optim
from torch.nn import functional as f

from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.random.seed(1)

torch.set_default_dtype(torch.float64)

class NN(nn.Module):
    def __init__(self, n_x, n_y):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
          nn.Linear(n_x, 25, bias=True),
          nn.BatchNorm1d(25),
          nn.ReLU(),
          
          nn.Linear(25, 12, bias=True),
          nn.BatchNorm1d(12),
          nn.ReLU(),
          
          nn.Linear(12, n_y, bias=True)   
        )
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers(x)
    
    
def model(X_train, Y_train, X_test, Y_test, classes, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, Lambda = 0.000001, print_cost = True):
    
    (m, n_x) = X_train.shape
    n_y = len(classes)  
    
    costs = []
    
    model = NN(n_x, n_y)
    CEF_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    seed = 0
    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        
        for minibatch_X, minibatch_Y in minibatches:
            optimizer.zero_grad()
            Y_out = model.forward(minibatch_X)
            loss = CEF_loss(Y_out,minibatch_Y) #+ Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item() / num_minibatches
        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)
                
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # # lets save the parameters in a variable
    # parameters = np.array(parameters)
    # print ("Parameters have been trained!")

    # # Calculate the correct predictions
    # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    # # Calculate accuracy on the test set
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    # print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    
    return model


X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

# Example of a picture
index = 4
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1)
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

trained_model = model(torch.tensor(X_train), torch.LongTensor(Y_train.flatten()), 
                      torch.tensor(X_test), torch.LongTensor(Y_test.flatten()), 
                      classes)

"""Prediction"""
trained_model.eval()
Y_out = torch.argmax(trained_model(torch.tensor(X_test)), dim=1)
accuracy = (1-torch.sum(Y_out!=torch.LongTensor(Y_test.flatten()))/Y_out.shape[0])*100
print('accuracy is: %f'%accuracy)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    