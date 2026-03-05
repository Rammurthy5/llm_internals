import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 200   # number for input samples
x = np.linspace(-1, 1, N).reshape(-1, 1)   # converting N to synthetic dataset 
y = 3*x + 2 + 0.1*np.random.randn(N, 1)   # y = wx + b + noise for random variation

w = np.random.randn(1, 1) # weight
b = np.zeros((1,))   # bias

lr = 0.1
epochs = 200 # set by LLM dev as how many iterations we'd like to train the NN

for epoch in range(epochs):
    # Forward
    y_pred = x @ w + b                  # (N,1)
    loss = np.mean((y_pred - y)**2)     # mean sq error. we had to sq. to avoid neg loss

    # backward pass / back propagation
    grad_w = (2/N) * (x.T @ (y_pred - y)) # gradient w is found based on avg (2/N) of actual error (y_pred - y) * size of i/p 
    grad_b = (2/N) * np.sum(y_pred - y) # gradient b is sum of actual err


    # gradient descent step - adjust the weight and bias based on gradients above
    w -= lr * grad_w # we are reducing the weight with gradient calc above and multiply with learning rate
    b -= lr * grad_b

    if (epoch + 1) % 20 == 0: # status update every 20th epoch
        print(f"epoch {epoch+1:3d} | loss {loss:.6f} | w {w.item():.3f} | b {b.item():.3f}")

print("Learned:", "w=", w.item(), "b=", float(b))

# graph
plt.scatter(x,y,label="data")
plt.plot(x,y_pred,color="red",label="model")
plt.legend()
plt.show()
