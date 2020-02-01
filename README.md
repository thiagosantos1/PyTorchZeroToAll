# Pytorch Notes

Every operation is saved as a graph.

Thus, we can have the use of special methods, for instance:

l = loss(x,y)
l.backward() # comes back and calculate gradiente descente for all weights

w.data = w.data - 0.01 * w.grad.data - Store the gradiente at w.grad
 0.01 is the learning rate



***** IMPORTANT ******
Gotta zero the gradiente, to 



* Chain Rule --> "easer" with pytorch, since it's a graph 
We can compute the gradient at each node


* Python Variables

w = Variable(torch.Tensor([1.0]), requires_grad=True)

Once Pytorch sees the "Variable" or the "Tensor", it already creates a graph around that Variable.

That is really good, since for each node it already calculate the gradiente for us

# Logic


1) First, we want to do forward pass

2) Second, we want to calculate the Gradiente from the Loss

3) Then, update based on gradient


# Pytorch steps

1) Design your model using class with Variables --> This is the most important part. This would go in your definition
   It can be a very complicated LSTM, or CNN or others.

2) Construct loss and optimizer ( from PyTorch API)

3) Training cycle --> Forward, Backward, Update



# How to construct the model ?

We can use multiple Linear combinations to create our model. Example 07:
The idea is that each linear can be multiplied agains the weights
We can also add layers , as seen below. In that case, we can add squash functions between layers.

self.l1 = nn.Linear(8, 6) # 8 features as input and 6 as ouput

self.l2 = nn.Linear(6, 4)

self.l3 = nn.Linear(4, 1)

self.sigmoid = nn.Sigmoid()

In this case,  Why these numbers ? 

these numbers are arbitrary. Same as a Hidden layers. This is your Neural Network
Thus, you must choose those numbers. 
However, The first one always has to be the same as your input dimension 
and the last must be same as your desired output dimension


