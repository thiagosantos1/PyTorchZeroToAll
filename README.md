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






