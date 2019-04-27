import csv
import numpy as np
import matplotlib.pyplot as plt
import ast

# Reads data from Cryotherapy.csv
with open('Cryotherapy.csv','r') as f:
    data = list(csv.reader(f,delimiter=','))
data = np.array(data)
data = data[1:,:]


one_counter = 0
zero_counter = 0
zero_arr = []
one_arr = []
for i in range(len(data[:,0])):
    if data[i,6] == '0':
        zero_counter += 1
        zero_arr.append(data[i,:])
    else:
        one_counter += 1
        one_arr.append(data[i,:])
zero_arr = np.array(zero_arr)
one_arr = np.array(one_arr)
data = np.concatenate((zero_arr,one_arr),axis=0)
for i in range(90):
    print(i," => ",data[i,6])
#print(data[:,6],zero_counter)

# Partition into training and testing data
print("TRaining Zeros : ",data[:29,:].shape)
training_zero = data[:30,:]
training_one = data[42:72,:]
training_data = np.concatenate((training_zero,training_one),axis=0)
X0 = np.ones((len(training_data),1))
training_data = np.hstack((X0,training_data))
print(training_data)
#print(training_zero.shape,training_one.shape)
testing_one = data[72:,:]
testing_zero = data[30:42,:]
testing_data = np.concatenate((testing_zero,testing_one),axis=0)
X0 = np.ones((len(testing_data),1))
testing_data = np.hstack((X0,testing_data))
training_x = training_data[:,:7]
training_x = training_x.astype(np.float)
training_y = training_data[:,7]
training_y = training_y.astype(np.float)

testing_x = testing_data[:,:7]
testing_x = testing_x.astype(np.float)
testing_y = testing_data[:,7]
testing_y = testing_y.astype(np.float)

# Building model

zero_mean = np.zeros((1,7))
one_mean = np.zeros((1,7))
zero_cov = np.zeros((7,7))
one_cov = np.zeros((7,7))

for i in range(30):
    zero_mean = zero_mean + training_x[i,:]
    one_mean = one_mean + training_x[i+30,:]
zero_mean = zero_mean/30
one_mean = one_mean/30
for i in range(30):
    zero_cov = zero_cov + np.matmul((training_x[i,:] - zero_mean).transpose(),(training_x[i,:] - zero_mean))
    one_cov = one_cov + np.matmul((training_x[i+30,:] - one_mean).transpose(),(training_x[i+30,:] - one_mean))
zero_cov = zero_cov/30
one_cov = one_cov/30
print("Covariance of 0:",zero_cov)
print("Covariance of 1:",one_cov)
print("Mean of 1: ",one_mean)
print("Mean of 0: ",zero_mean)

f = open("theta.txt","r")
theta_init = f.read()
print(theta_init)
theta_init = ast.literal_eval(theta_init)
theta_init = np.array(theta_init)

iterations = 10000
step = 0.002
print(training_x.shape)
print(testing_x.shape)
# Calculates dot product for predicted value
s_n = np.matmul(theta_init.transpose(),training_x.transpose())

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Caculates g(x)
def g_x(theta,x):
    return np.dot(x,theta)

print("Prediction on index 1:",g_x(training_x[1,:],theta_init))

def cost(theta,x,y):
    m = len(training_x)
    total_cost = -(1/m)*np.sum(y*np.log(sigmoid(g_x(theta,x)))+(1-y)*np.log(1-sigmoid(g_x(theta,x))))
    return total_cost

print("Cost :",cost(theta_init,training_x,training_y))

def grad_desc(theta,x,y):
    m = len(x)
    return (1/m)*np.dot(x.transpose(),sigmoid(g_x(theta,x))-y)

print("Grad dedsc:",grad_desc(theta_init,training_x,training_y))

def train(theta,x,y,iterations,learning_rate):
    for i in range(iterations):
        theta -= learning_rate*grad_desc(theta,x,y)
        err = cost(theta,x,y)
        print("Cost: ",err)
    return theta

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

theta_init = train(theta_init,training_x,training_y,iterations,step)
plot_y =  sigmoid(g_x(testing_x.transpose(),theta_init))
plot_x = np.zeros((len(testing_x),1))
for i in range(len(testing_x)):
    plot_x[i] = i+1
plt.scatter(plot_x[:10],plot_y[:10])
plt.scatter(plot_x[10:],plot_y[10:])
plt.axhline(y=0.5,color="r",linestyle='-')
plt.show()
print("Accuracy rate: ",score(theta_init,testing_x,testing_y))
