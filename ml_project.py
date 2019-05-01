import csv
import numpy as np
import matplotlib.pyplot as plt
import ast

# Reads data from Cryotherapy.csv
with open('Cryotherapy.csv','r') as f:
    data = list(csv.reader(f,delimiter=','))
data = np.array(data)
data = data[1:,:]
DATA = data.astype(float)

# Building model


f = open("theta.txt","r")
theta = f.read()
theta= ast.literal_eval(theta)
theta = np.array(theta)


# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Caculates g(x)
def g_x(theta,x):
    return np.dot(x,theta)


def cost(theta,x,y):
    m = len(training_x)
    total_cost = -(1/m)*np.sum(y*np.log(sigmoid(g_x(theta,x)))+(1-y)*np.log(1-sigmoid(g_x(theta,x))))
    return total_cost


def grad_desc(theta,x,y):
    m = len(x)
    return (1/m)*np.dot(x.transpose(),sigmoid(g_x(theta,x))-y)


def train(theta,x,y,iterations,learning_rate):
    print("Training model ...");
    cost_history = []
    for i in range(iterations):
        theta -= learning_rate*grad_desc(theta,x,y)
        err = cost(theta,x,y)
        cost_history.append(err)
    cost_history = np.array(cost_history)
    return theta,cost_history

def score(theta,x,y):
    error_count = 0
    pred_y = sigmoid(g_x(theta,x))
    for i in range(len(pred_y)):
        if pred_y[i] < 0.5:
            pred_y[i] = 0
        else:
            pred_y[i] = 1
        
        if pred_y[i] != y[i]:
            error_count += 1
    return 1-float(error_count/len(y))


iterations = 20000
training_x = DATA[:75,:6]
training_y = DATA[:75,6]
testing_x = DATA[75:,:6]
testing_y = DATA[75:,6]
theta,cost_history = train(theta,training_x,training_y,iterations,0.0003)
x = np.zeros((iterations,1))
for i in range(iterations):
    x[i] = i+1
plt.title("Cost function")
plt.plot(x,cost_history)
plt.show()
print("Testing Accuracy: ",score(theta,testing_x,testing_y))
print("Training Accuracy: ",score(theta,training_x,training_y))
zero = np.zeros((5,1))
one = np.zeros((10,1))
zero_i = 0
one_i = 0
for i in range(len(testing_x)):
    if testing_y[i] == 0:
        zero[zero_i,:] = sigmoid(g_x(theta,DATA[i+75,:6]))
        zero_i += 1
    else:
        one[one_i,:] = sigmoid(g_x(theta,DATA[i+75,:6]))
        one_i += 1
plot_x = np.zeros((len(testing_x),1))
for i in range(len(testing_x)):
    plot_x[i] = i+1
plt.title("Testing Samples")
plt.axhline(y=0.5,color="r",linestyle='-')
plt.scatter(plot_x[:5],zero)
plt.scatter(plot_x[5:],one)
plt.show()
zero_i = 0
one_i = 0
zero = np.zeros((37,1))
one = np.zeros((38,1))
plot_x = np.zeros((75,1))
for i in range(len(training_x)):
    if training_y[i] == 0:
        zero[zero_i] = sigmoid(g_x(theta,training_x[i,:]))
        zero_i +=1
    else:
        one[one_i] = sigmoid(g_x(theta,training_x[i,:]))
        one_i +=1
for i in range(len(training_x)):
    plot_x[i] = i+1
plt.title("Training Samples")
plt.axhline(y=0.5,color='g',linestyle='-')
plt.scatter(plot_x[:37],zero)
plt.scatter(plot_x[37:],one)
plt.show()
