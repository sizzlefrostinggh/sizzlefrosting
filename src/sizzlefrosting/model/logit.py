import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  roc_auc_score, confusion_matrix, classification_report, roc_curve
)

class LogisticRegression(nn.Module):
  def __init__(self, n_features, learning_rate = 0.05, epochs = 100, batch_size = 128):
    """
    Initialize the logistic regression model with the number of features,
    learning rate, number of epochs, batch size, model parameters,
    optimizer, loss function, and lists to store training history.
    """
    super().__init__()
    self.n_features = n_features
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size

    self.w0 = nn.Parameter(torch.zeros(1))
    self.w = nn.Parameter(torch.zeros(n_features, 1))

    self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate)
    self.loss_history = []

    self.x_mean = None
    self.x_std = None

  def forward(self, X):
    """
    Compute the predicted probability using the logistic regression model
        h(X) = 1 / (1 + exp(-(w0 + X @ w))).
    """
    z = self.w0 + torch.matmul(X, self.w)
    z = torch.clamp(z, -50.0, 50.0)
    return 1.0 / (1.0 + torch.exp(-z))

  def loss_fn(self, y_hat, y):
    """
    Compute the binary cross-entropy loss between the predicted
    probabilities and the actual response values.
    """
    eps = 1e-7
    y_hat = torch.clamp(y_hat.view(-1), eps, 1.0 - eps)
    y = y.view(-1)
    return -torch.mean(y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat))

  def fit(self, X_train, y_train, X_test, y_test):
    """
    Train the logistic regression model using mini-batch stochastic
    gradient descent and evaluate the classification metrics
    on the test dataset.
    """
    if not isinstance(X_train, torch.Tensor):
      X_train = torch.tensor(np.asarray(X_train), dtype = torch.float32)
      y_train = torch.tensor(np.asarray(y_train), dtype = torch.float32).view(-1)
      X_test = torch.tensor(np.asarray(X_test), dtype = torch.float32)
      y_test = torch.tensor(np.asarray(y_test), dtype = torch.float32).view(-1)
    else:
      X_train = X_train.float()
      y_train = y_train.float().view(-1)
      X_test = torch.as_tensor(np.asarray(X_test), dtype = torch.float32)
      y_test = torch.as_tensor(np.asarray(y_test), dtype = torch.float32).view(-1)

    # Standardize the features using the training dataset statistics
    self.x_mean = X_train.mean(dim = 0)
    self.x_std = X_train.std(dim = 0) + 1e-8
    X_train_scaled = (X_train - self.x_mean) / self.x_std

    self._X_test = X_test
    self._y_test = y_test

    dataset = TensorDataset(X_train_scaled, y_train)
    loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

    self.loss_history = []

    for epoch in range(self.epochs):
      epoch_loss = 0.0

      for batch_x, batch_y in loader:
        self.optimizer.zero_grad()
        y_hat = self.forward(batch_x)
        loss = self.loss_fn(y_hat, batch_y)
        loss.backward()
        self.optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

      self.loss_history.append(epoch_loss / len(dataset))

    # Evaluate the fitted model on the test dataset
    y_prob = self.predict_proba(X_test).numpy()
    y_pred = self.predict(X_test).numpy()
    y_true = y_test.numpy()

    print("--- Test Set Evaluation Metrics ---")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits = 4))

  def predict_proba(self, x_new):
    """
    Predict the probability P(y = 1 | X) for given new observations.
    """
    if not isinstance(x_new, torch.Tensor):
      x_new = torch.tensor(np.asarray(x_new), dtype = torch.float32)
    else:
      x_new = x_new.float()

    x_scaled = (x_new - self.x_mean) / self.x_std

    with torch.no_grad():
      return self.forward(x_scaled).view(-1)

  def predict(self, x_new, threshold = 0.5):
    """
    Predict binary class labels (0 or 1) for given new observations
    using the specified classification threshold.
    """
    return (self.predict_proba(x_new) >= threshold).int()

  def analysis_plot(self):
    """
    Create subplots showing the cross-entropy loss during training
    and the receiver operating characteristic (ROC) curve
    on the test dataset.
    """
    y_prob = self.predict_proba(self._X_test).numpy()
    y_true = self._y_test.numpy()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    axes[0].plot(self.loss_history, color = "red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].set_title("Cross-Entropy Loss During SGD Training")
    axes[0].grid(linestyle = "--")

    axes[1].plot(
      fpr, tpr,
      color = "blue",
      linewidth = 2,
      label = f"ROC Curve (AUC = {auc_val:.4f})"
    )
    axes[1].plot([0, 1], [0, 1], color = "gray", linestyle = "--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Test Set ROC Curve")
    axes[1].legend(loc = "lower right")
    axes[1].grid(linestyle = "--")

    fig.tight_layout()
    plt.show()
