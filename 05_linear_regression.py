from torch import nn
import torch
from torch import tensor

# no need to use "Variable" anymore. tensor is already a recognize Variable
# 3x1 data
x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() # always have to call this super
        # After that, we can add/create components to our model
        # In this case, a simple Linear model
        # Why (1,1) ? X is always just 1 value/dimension and y is out just 1 output
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x) # in this case, we get X and use the models we difined on our definition
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') # Choose the Loss you want to use - Good for linear
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Chose the gradiente you want to use

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data) # This is the loss that we choose
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() 
    loss.backward() # based on loss, we get the gradients 
    optimizer.step() # based on the optimizer we chose, we then update/move towards globol minimum


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
