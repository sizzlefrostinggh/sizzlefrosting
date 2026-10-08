import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
  def __init__(self, learning_rate, epochs):
    """
    Initialize the linear regression model with the learning rate,
    number of epochs, model parameters, optimizer, loss function,
    and lists to store training history.
    """
    super().__init__()
    self.learning_rate = learning_rate
    self.epochs = epochs

    self.w0 = nn.Parameter(torch.zeros(1))
    self.w1 = nn.Parameter(torch.zeros(1))

    self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
    self.loss_fn = nn.MSELoss()

    self.w0_history = []
    self.w1_history = []
    self.loss_history = []

  def forward(self, X):
    """
    Compute the predicted response using the linear model
        y = w0 + w1 * x.
    """
    return self.w0 + self.w1 * X

  def fit(self, X_train, y_train, X_test, y_test):
    """
    Train the linear regression model using gradient descent
    and calculate the R-squared value on the test dataset.
    """
    self._X_train, self._y_train = X_train, y_train
    self._X_test, self._y_test = X_test, y_test

    for epoch in range(self.epochs):
      self.optimizer.zero_grad()
      y_hat = self.forward(X_train)
      loss = self.loss_fn(y_hat, y_train)
      loss.backward()
      self.optimizer.step()

      self.w0_history.append(self.w0.item())
      self.w1_history.append(self.w1.item())
      self.loss_history.append(loss.item())

    with torch.no_grad():
      y_hat = self.forward(X_test)
      sse = torch.sum((y_test - y_hat) ** 2)
      ssto = torch.sum((y_test - torch.mean(y_test)) ** 2)
      r_squared = (1 - sse / ssto).item()

    print(f"R-squared: {r_squared}")

  def predict(self, x_new):
    """
    Predict AnnualProduction for a given new BCR value.
    """
    with torch.no_grad():
      return self.forward(x_new)

  def analysis_plot(self):
    """
    Create subplots showing the training and test data,
    fitted regression line, w0, w1, and loss during training.
    """
    fig, axes = plt.subplots(2, 2, figsize = (12, 9))

    ax = axes[0, 0]
    ax.scatter(self._X_train, self._y_train, alpha = 0.6, label = "Train data")
    ax.scatter(self._X_test, self._y_test, alpha = 0.6, label = "Test data")
    x_all = torch.cat([self._X_train, self._X_test])
    x_line = torch.linspace(x_all.min().item(), x_all.max().item(), 100)
    ax.plot(x_line, self.predict(x_line), color = "orange", label = "Fitted line")
    ax.set_xlabel("BCR")
    ax.set_ylabel("AnnualProduction")
    ax.set_title("Data and Fitted Regression Line")
    ax.legend()

    axes[0, 1].plot(self.w0_history, color = "purple")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("$w_0$")
    axes[0, 1].set_title("$w_0$ during training")

    axes[1, 0].plot(self.w1_history, color= "green")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("$w_1$")
    axes[1, 0].set_title("$w_1$ during training")

    axes[1, 1].plot(self.loss_history, color= "red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE loss")
    axes[1, 1].set_title("Loss during training")

    for ax in axes.flat:
      ax.grid(axis="x", linestyle="--")
      ax.grid(axis="y", linestyle="--")

    fig.tight_layout()
