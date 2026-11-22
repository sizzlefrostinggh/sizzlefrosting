import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort


class TorchNet(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.loss_history = []

        # Two hidden layers: 4 neurons -> 3 neurons
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # 1.  Data Standardization
    def standardize_data(self, X: np.ndarray, fit: bool = True) -> torch.Tensor:
        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return torch.tensor(X_scaled, dtype=torch.float32)

    # 3.  Training the Model
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.01):
        X_t = self.standardize_data(X, fit=True)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, self.output_dim)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.loss_history = []
        self.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            preds = self.forward(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 200 == 0 or epoch == 1:
                print(f"Epoch [{epoch:4d}/{epochs}] | Loss: {loss.item():.4f}")

    # 4.  Loss Plot
    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.title("Training Loss vs. Epochs (TorchNet)")
        plt.xlabel("Epochs")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    # 5.  Model Saving (ONNX)
    def save_onnx(self, filepath: str = "torchnet_diabetes.onnx"):
        self.eval()
        dummy_input = torch.randn(1, self.input_dim, dtype=torch.float32)
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"x": {0: torch.export.Dim.AUTO}},
        )
        print(f"\nModel saved to {filepath}")

    # 6.  Model Prediction (ONNX)
    def predict_onnx(self, X_sample: np.ndarray, filepath: str = "torchnet_diabetes.onnx"):
        X_scaled = self.scaler.transform(X_sample).astype(np.float32)
        session = ort.InferenceSession(filepath)
        input_name = session.get_inputs()[0].name
        prob = float(session.run(None, {input_name: X_scaled})[0][0][0])
        return prob, int(prob >= 0.5)
