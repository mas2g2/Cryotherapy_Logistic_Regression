import csv
import numpy as np

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

# Partition into training and testing data
training_zero = data[:30,:]
training_one = data[42:72,:]
training_data = np.concatenate((training_zero,training_one),axis=0)

testing_one = data[73:89,:]
testing_zero = data[31:41,:]
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
print(theta_init)
iterations = 2500
step = 0.0075
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
diff = np.zeros((90,1))
#print(theta)
for i in range(iterations):
    diff = np.subtract(s_n, training_y)
    theta_init = theta_init - step * np.matmul(training_x.transpose(),diff.transpose())
    s_n = np.matmul(theta_init.transpose(),training_x.transpose())
    for j in range(len(s_n)):
            s_n[j] = sigmoid(s_n[j])

s_n = np.matmul(theta_init.transpose(),testing_x.transpose())

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(len(s_n)):
    s_n[i] = sigmoid(s_n[i])
print("Predictions : ")
print(s_n)
print("Actual Label : ")
print(testing_y)
