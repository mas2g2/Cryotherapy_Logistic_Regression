import csv
import numpy as np
from numpy import linalg,pi
import math
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
print(data[:,6],zero_counter)

# Partition into training and testing data
training_zero = data[:30,:]
training_one = data[42:72,:]
training_data = np.concatenate((training_zero,training_one),axis=0)
print(training_zero.shape,training_one.shape)
testing_one = data[73:,:]
testing_zero = data[31:42,:]
testing_data = np.concatenate((testing_zero,testing_one),axis=0)

training_x = training_data[:,:6]
training_x = training_x.astype(np.float)
training_y = training_data[:,6]
training_y = training_y.astype(np.float)

testing_x = testing_data[:,:6]
testing_x = testing_x.astype(np.float)
testing_y = testing_data[:,6]
testing_y = testing_y.astype(np.float)

# Building model
theta_init = np.random.rand(6,1)
original_theta = theta_init
print(theta_init)
iterations = 10000
step = 0.001
print(training_x)

# Calculates dot product for predicted value
s_n = np.matmul(theta_init.transpose(),training_x.transpose())

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Apply sigmoid function to dot product
for i in range(len(s_n)):
    s_n[i] = sigmoid(s_n[i])

print(s_n)
diff = np.subtract(s_n, training_y)
#print(theta)

mse = np.mean(diff**2)

for i in range(iterations):
    print("MSE: ",mse)
    psv =  step * np.matmul(training_x.transpose(),diff.transpose())/len(training_y)
    theta_init = np.subtract(theta_init,psv)
    s_n = np.matmul(theta_init.transpose(),training_x.transpose())
    for j in range(len(s_n)):
            s_n[j] = sigmoid(s_n[j])
    diff = np.subtract(s_n, training_y)
    mse = np.mean(diff**2)
s_n = np.matmul(theta_init.transpose(),testing_x.transpose())
s_n = s_n[0,:]
for i in range(len(s_n)):
    s_n[i] = sigmoid(s_n[i])
    if s_n[i] >= 0.5:
        s_n[i] = 1
    else:
        s_n[i] = 0
print("Predictions : ")
print(s_n)
print("Actual Label : ")
print(testing_y)
error_count = 0
for i in range(len(testing_y)):
    print("Pred: ",s_n[i]," Label: ",testing_y[i])
    if s_n[i] != testing_y[i]:
        error_count += 1
error_rate = error_count/len(testing_y)
print("Error rate: ",error_rate)
print("Original theta: ",original_theta)
# Theta : [0.70917487] Error rate = 25%
#         [0.23187651]
#         [0.99927246]
#         [0.38097745]
#         [0.05778725]
#         [0.49655668]
"""
Theta : [0.85301419]
 [0.03555556]
  [0.39178867]
   [0.15730367]
    [0.22898207]
     [0.45940889]

     Error rate : 21%
"""
