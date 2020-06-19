# -*- coding: utf-8 -*-


# Importing Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline

torch.manual_seed(0)

"""# Loading Dataset"""

sns.get_dataset_names()

flight_data = sns.load_dataset("flights")
flight_data.head()

"""# Preprocessing"""

# Changing the plot size

figsize = plt.rcParams["figure.figsize"]
figsize[0] = 15
figsize[1] = 5
plt.rcParams["figure.figsize"] = figsize

# Plotting the data

plt.title("Time Series Representation of Data")
plt.xlabel("Months")
plt.ylabel("Passengers")
plt.grid(True)
plt.autoscale(axis = "x",tight=True)
plt.plot(flight_data["passengers"])

#Please note that this is univariate time series data : consisting of one variable passengers
#
data = flight_data["passengers"].values.astype(float)
print(data)
print(len(data))

# Train-Test Split
# Consider the last the 12 months data as evaluation data for testing the model's behaviour

train_window = 12

train_data = data[:-train_window]
test_data = data[-train_window:]
print(len(train_data))
print(len(test_data))

# Normalizing the train-data

scaler = MinMaxScaler(feature_range=(-1,1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))
print(train_data_normalized[:10])

# Converting to Torch Tensor

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
print(train_data_normalized)

# Final step is creating sequences of length 12 (12 months data) from the train-data and 
# the label for each sequence is the passenger_data for the (12+1)th Month

def create_in_sequences(input_data,tw):
    in_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        in_seq.append((train_seq,train_label))
    return in_seq

# Therefore, we get 120 train sequences along with the label value
train_in_seq = create_in_sequences(train_data_normalized,train_window)
print(len(train_in_seq))
print(train_in_seq[:5])

"""# The Model

Please note that the model considered here is:


1.   LSTM layer with a univariate input sequence of length 12 and LSTM's previous hidden cell consisting of previous hidden state and previous cell state of length 100 and also , the size of LSTM's output is 100 
2.   The second layer is a Linear layer of 100 inputs from the LSTM's output and a single output size
"""

class LSTM(nn.Module):    #LSTM Class inheriting the inbuilt nn.Module class for neural networks
    def __init__(self,input_size = 1,hidden_layer_size = 100, output_size = 1):
        super().__init__()   #Calls the init function of the nn.Module superclass for being able to access its features
        self.hidden_layer_size = hidden_layer_size # Defines the size of h(t-1) [Previous hidden output] and c(t-1) [previous cell state] 
        
        self.lstm = nn.LSTM(input_size,hidden_layer_size,dropout = 0.45) # definining the LSTM with univariate input,output and with a dropout regularization of 0.45
        self.linear = nn.Linear(hidden_layer_size,output_size) # Linear layer which returns the weighted sum of 100 outputs from the LSTM
        self.hidden_cell = (torch.ones(1,1,self.hidden_layer_size),   # This is the previous hidden state 
                             torch.ones(1,1,self.hidden_layer_size))  # This is the previous cell state 
    def forward(self,input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq),1,-1),self.hidden_cell) #returns 1200 outputs from each of the 100 output neurons for the 12 valued sequence
        predictions = self.linear(lstm_out.view(len(input_seq),-1)) # Reshaped to make it compatible as an input to the linear layer
        return predictions[-1] # The last element contains the prediction

model = LSTM()
print(model)

"""# Loss Function and Learning Algorithm (Optimizer)

Please note that for this simple model , 
* Loss Function considered is *Mean Squared Error* and
* Optimization Function used is  Stochastic Version of **Adam** *Optimizer*.
"""

loss_fn = nn.MSELoss() # Mean Squared Error Loss Function
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0002) # Adam Learning Algorithm

"""# Training"""

epochs = 450
loss_plot = []

for epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
    for seq,label in train_in_seq:
        optimizer.zero_grad()         # makes the gradients zero for each new sequence
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),    # Initialising the previous hidden state and cell state for each new sequence
                             torch.zeros(1,1,model.hidden_layer_size))
        y_pred = model(seq) # Automatically calls the forward pass 
        
        loss = loss_fn(y_pred,label) # Determining the loss
        loss.backward() # Backpropagation of loss and gradients computation
        optimizer.step() # Weights and Bias Updation

    loss_plot.append(loss.item())   # Some Bookkeeping
plt.plot(loss_plot,'r-')
plt.xlabel("Epochs")
plt.ylabel("Loss : MSE")
plt.show()

print(loss_plot[-1])

"""# Making Prediction

Please note that for comparison purpose we use the training data's values and predicted data values to predict the number of passengers for the test data months and then compare them
"""

fut_pred = 12

test_inputs = train_data_normalized[-train_window: ].tolist()
print(test_inputs)
print(len(test_inputs))

model.eval()        # Makes the model ready for evaluation

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window: ])       # Converting to a tensor

    with torch.no_grad(): # Stops adding to the computational flow graph (stops being prepared for backpropagation)
        model.hidden_cell = (torch.zeros(1,1,model.hidden_layer_size),
                             torch.zeros(1,1,model.hidden_layer_size))
        test_inputs.append(model(seq).item())

predicted_outputs_normalized = []
predicted_outputs_normalized = test_inputs[-train_window: ]
print(predicted_outputs_normalized)
print(len(predicted_outputs_normalized))

"""# Postprocessing"""

predicted_outputs = scaler.inverse_transform(np.array(predicted_outputs_normalized).reshape(-1,1))
print(predicted_outputs)

x = np.arange(132, 144, 1)
print(x)

"""# Final Output"""

figsize = plt.rcParams["figure.figsize"]
figsize[0] = 15
figsize[1] = 5
plt.rcParams["figure.figsize"] = figsize

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x,predicted_outputs)
plt.show()

figsize = plt.rcParams["figure.figsize"]
figsize[0] = 15
figsize[1] = 5
plt.rcParams["figure.figsize"] = figsize

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'][-train_window-5: ])
plt.plot(x,predicted_outputs)
plt.show()

"""**Please observe that the model is able to get the trend of the passengers but it can be further fine-tuned by adding appropriate regularization methods**"""
