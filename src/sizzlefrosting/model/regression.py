import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import numpy as np

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


class CauchyRegression(nn.Module):
  def __init__(self, learning_rate = 0.05, epochs = 3000, c = 1.0):
    """
    Initialize the Cauchy regression model with the learning rate,
    number of epochs, model parameters, Cauchy loss parameter,
    optimizer, and lists to store training history.
    """
    super().__init__()
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.c = c

    self.w0 = nn.Parameter(torch.zeros(1))
    self.w1 = nn.Parameter(torch.zeros(1))
    self.w2 = nn.Parameter(torch.zeros(1))
    self.w3 = nn.Parameter(torch.zeros(1))
    self.w4 = nn.Parameter(torch.zeros(1))

    self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

    self.w0_history = []
    self.w1_history = []
    self.w2_history = []
    self.w3_history = []
    self.w4_history = []
    self.loss_history = []

    self.se_ = None
    self.ci_95_ = None

  def forward(self, X):
    """
    Compute the predicted response using the multiple linear model
        y = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4.
    """
    return (
      self.w0
      + self.w1 * X[:, 0]
      + self.w2 * X[:, 1]
      + self.w3 * X[:, 2]
      + self.w4 * X[:, 3]
    )

  def loss_fn(self, y_hat, y):
    """
    Compute the Cauchy loss using the residuals between
    the predicted and actual response values.
    """
    y_hat = y_hat.view(-1)
    y = y.view(-1)
    return torch.mean(0.5 * (self.c ** 2) * torch.log(1.0 + ((y - y_hat) / self.c) ** 2))

  def fit(self, X, y):
    """
    Train the Cauchy regression model using gradient descent
    and calculate the standard errors and 95% confidence intervals.
    """
    if not isinstance(X, torch.Tensor):
      X = torch.tensor(np.asarray(X), dtype = torch.float32)
    else:
      X = X.float()

    if not isinstance(y, torch.Tensor):
      y = torch.tensor(np.asarray(y), dtype = torch.float32).view(-1)
    else:
      y = y.float().view(-1)

    self._X = X
    self._y = y

    # Standardize the features to improve convergence during training
    x_mean = X.mean(dim = 0)
    x_std = X.std(dim = 0)
    X_scaled = (X - x_mean) / x_std

    # Initialize the intercept to the mean response and the slopes to zero
    with torch.no_grad():
      self.w0.fill_(y.mean().item())
      self.w1.zero_()
      self.w2.zero_()
      self.w3.zero_()
      self.w4.zero_()

    for epoch in range(self.epochs):
      self.optimizer.zero_grad()
      y_hat = self.forward(X_scaled)
      loss = self.loss_fn(y_hat, y)
      loss.backward()
      self.optimizer.step()

      # Convert the parameters to the original feature scale and store their history
      with torch.no_grad():
        w_scaled = torch.tensor([
          self.w1.item(),
          self.w2.item(),
          self.w3.item(),
          self.w4.item()
        ])
        w_unscaled = w_scaled / x_std
        w0_unscaled = self.w0.item() - torch.sum(w_scaled * x_mean / x_std).item()

        self.w0_history.append(w0_unscaled)
        self.w1_history.append(w_unscaled[0].item())
        self.w2_history.append(w_unscaled[1].item())
        self.w3_history.append(w_unscaled[2].item())
        self.w4_history.append(w_unscaled[3].item())
        self.loss_history.append(loss.item())

    # Convert the fitted parameters back to the original feature scale
    with torch.no_grad():
      w_scaled = torch.stack([
        self.w1.data,
        self.w2.data,
        self.w3.data,
        self.w4.data
      ]).view(-1)
      w_unscaled = w_scaled / x_std
      w0_unscaled = self.w0.data - torch.sum(w_scaled * x_mean / x_std)

      self.w0.copy_(w0_unscaled.view(1))
      self.w1.copy_(w_unscaled[0].view(1))
      self.w2.copy_(w_unscaled[1].view(1))
      self.w3.copy_(w_unscaled[2].view(1))
      self.w4.copy_(w_unscaled[3].view(1))

    self._compute_confidence_intervals()
    print(f"Final Cauchy Loss: {self.loss_history[-1]:.4f}")

  def predict(self, x_new):
    """
    Predict the response for a given new feature matrix.
    """
    if not isinstance(x_new, torch.Tensor):
      x_new = torch.tensor(np.asarray(x_new), dtype = torch.float32)
    else:
      x_new = x_new.float()

    with torch.no_grad():
      return self.forward(x_new)

  def get_coefficients(self):
    """
    Return the fitted model coefficients as a dictionary.
    """
    return {
      "w0": self.w0.item(),
      "w1": self.w1.item(),
      "w2": self.w2.item(),
      "w3": self.w3.item(),
      "w4": self.w4.item()
    }

  def _compute_confidence_intervals(self):
    """
    Calculate the standard errors and 95% confidence intervals
    using the covariance matrix of the Cauchy regression estimator.
    """
    X_np = self._X.detach().numpy().astype(np.float64)
    y_np = self._y.detach().numpy().astype(np.float64)
    N, D = X_np.shape

    X_design = np.column_stack([np.ones(N), X_np])
    params = np.array([
      self.w0.item(),
      self.w1.item(),
      self.w2.item(),
      self.w3.item(),
      self.w4.item()
    ], dtype = np.float64)

    residuals = y_np - X_design @ params
    u = residuals / self.c

    # Calculate the first and second derivatives of the Cauchy loss
    psi = -(self.c * u) / (1.0 + u ** 2)
    psi_prime = (self.c ** 2) * (1.0 - u ** 2) / ((self.c ** 2 + residuals ** 2) ** 2)

    H = (X_design.T * psi_prime) @ X_design
    G = (X_design.T * (psi ** 2)) @ X_design
    H_inv = np.linalg.inv(H)
    cov_matrix = H_inv @ G @ H_inv

    self.se_ = np.sqrt(np.diag(cov_matrix))
    t_crit = stats.t.ppf(0.975, df = N - D - 1)
    self.ci_95_ = np.column_stack([
      params - t_crit * self.se_,
      params + t_crit * self.se_
    ])

  def residual_plot(self):
    """
    Create subplots showing the residuals against each input feature.
    """
    with torch.no_grad():
      y_hat = self.forward(self._X)
      residuals = (y_hat - self._y).numpy()

    X_np = self._X.numpy()
    feature_labels = [
      "x1: Ambient Temperature (AT) [°C]",
      "x2: Exhaust Vacuum (V) [cm Hg]",
      "x3: Ambient Pressure (AP) [milibar]",
      "x4: Relative Humidity (RH) [%]"
    ]

    fig, axes = plt.subplots(2, 2, figsize = (12, 9))

    for i, ax in enumerate(axes.flat):
      ax.scatter(X_np[:, i], residuals, alpha = 0.25, s = 10, color = "steelblue")
      ax.axhline(0, color = "red", linestyle = "--", linewidth = 1.5)
      ax.set_xlabel(feature_labels[i])
      ax.set_ylabel(r"Residual ($\hat{y} - y$)")
      ax.set_title(f"Residuals vs. $x_{i+1}$")
      ax.grid(axis = "x", linestyle = "--")
      ax.grid(axis = "y", linestyle = "--")

    fig.tight_layout()
    plt.show()


class CauchyModel(GenericLikelihoodModel):
  def __init__(self, endog, exog, c = 1.0, **kwds):
    """
    Initialize the Cauchy regression model with the response,
    input features, and Cauchy loss parameter.
    """
    self.c = c
    super().__init__(endog, exog, **kwds)

  def nloglikeobs(self, params):
    """
    Compute the negative log-likelihood for each observation.
    """
    y_hat = np.dot(self.exog, params)
    residual = (self.endog - y_hat) / self.c
    return 0.5 * (self.c ** 2) * np.log(1.0 + residual ** 2)

  def fit(self, start_params = None, maxiter = 10000, **kwds):
    """
    Fit the Cauchy regression model using maximum likelihood estimation.
    """
    if start_params is None:
      start_params = np.linalg.lstsq(self.exog, self.endog, rcond = None)[0]

    return super().fit(start_params = start_params, maxiter = maxiter, **kwds)
