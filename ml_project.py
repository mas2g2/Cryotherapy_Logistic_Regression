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
    return np.dot(x,theta.transpose())

# Calculates error using cost function
def cost(theta,x,y):
    m = len(training_x)
    total_cost =-(1/m)*np.sum(y*np.log(sigmoid(g_x(theta,x)))+(1-y)*np.log(1-sigmoid(g_x(theta,x))))
    return total_cost

# CAlculates gradient descent for updating the weights to minimize the error
def grad_desc(theta,x,y):
    m = len(x)
    return (1/m)*np.dot(x.transpose(),sigmoid(g_x(theta,x))-y)

# TRains model
def train(theta,x,y,iterations,learning_rate):
    print("Training model ...");
    cost_history = []
    theta_history = []
    for i in range(iterations):
        theta -= learning_rate*grad_desc(theta,x,y)
        err = cost(theta,x,y)
        theta_history.append(theta)
        cost_history.append(err)
        print(err)
    cost_history = np.array(cost_history)
    return theta,cost_history

# Import roc_auc_score function to calculate area under the curve
from sklearn.metrics import roc_auc_score

# Calculates evaluation scores to evaluate the performance of the model
# Calculates sensitivity, specificity, accuracy, auc, and f1 measure
def score(theta,x,y):
    error_count = 0
    pred_y = sigmoid(g_x(theta,x))
    positives = 0
    true_pos = 0
    true_neg =0
    negatives = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(pred_y)):
        if pred_y[i] < 0.5:
            pred_y[i] = 0
        else:
            pred_y[i] = 1
        
        if pred_y[i] != y[i]:
            error_count += 1
        if y[i] == 1:
            positives += 1
        if y[i] == 0:
            negatives += 1
        if y[i] == 1 and pred_y[i] == 1:
            true_pos += 1
        elif y[i] == 1 and pred_y[i] == 0:
            false_neg += 1
        if y[i] == 0 and pred_y[i] == 0:
            true_neg += 1
        elif y[i] == 0 and pred_y[i] == 1:
            false_pos += 1
    sensitivity = true_pos/positives
    specificity = true_neg/negatives
    accuracy = 1-float(error_count/len(y))
    precision = true_pos/(true_pos+false_pos)
    f1_measure = (2*precision*sensitivity)/(precision+sensitivity)
    auc = roc_auc_score(y,pred_y)
    return sensitivity,specificity, accuracy, auc, f1_measure

# Cross validation function for evaluating model
def cross_val(theta,DATA,k):
    np.random.shuffle(DATA)
    scores = np.split(DATA,k)
    accuracies = []

    for i in range(len(scores)):
        if i == 0:
            testing_data = scores[0]
            training_data = DATA[k:,:]
        elif i == len(scores) - 1:
            m = len(DATA)-k
            testing_data = scores[len(scores)-1]
            training_data = DATA[:m,:]
        else:
            fpi = (i+1)*k
            spi = (i+2)*k
            first_part = DATA[:fpi,:]
            second_part = DATA[spi:,:]
            testing_data = scores[i]
            training_data = np.vstack((first_part,second_part))
        testing_x = testing_data[:,:6]
        testing_y = testing_data[:,6]
        training_x = training_data[:,:6]
        training_y = training_data[:,6]
        new_theta,cost_history = train(theta,training_x,training_y,20000,0.0001)
        accuracies.append(score(new_theta,testing_x,testing_y))
    return accuracies

iterations = 20000
training_x = DATA[:75,:6]
training_y = DATA[:75,6]
testing_x = DATA[75:,:6]
testing_y = DATA[75:,6]
og_theta = theta
theta,cost_history = train(theta,training_x,training_y,iterations,0.0001)

x = np.zeros((iterations,1))
for i in range(iterations):
    x[i] = i+1
plt.title("Cost function")
plt.plot(x,cost_history)
plt.show()
print("Evaluation scores: ",score(theta,testing_x,testing_y))
zero = []
one = []
for i in range(len(testing_x)):
    if testing_y[i] == 0:
        zero.append(sigmoid(g_x(theta,testing_x[i,:])))
    else:
        one.append(sigmoid(g_x(theta,testing_x[i,:])))
zero = np.array(zero)
one = np.array(one)
plot_x = np.zeros((len(testing_x),1))
for i in range(len(testing_x)):
    plot_x[i] = i+1
plt.title("Testing Samples")
plt.axhline(y=0.5,color="r",linestyle='-')
plt.scatter(plot_x[:len(zero)],zero)
plt.scatter(plot_x[len(zero):],one)
plt.show()
scores = cross_val(og_theta,DATA,10)
res = sum(i[0] for i in scores)/10, sum(i[1] for i in scores)/10,sum(i[2] for i in scores)/10, sum(i[3] for i in scores)/10, sum(i[4] for i in scores)/10
print("Eval Scores: ",res)
