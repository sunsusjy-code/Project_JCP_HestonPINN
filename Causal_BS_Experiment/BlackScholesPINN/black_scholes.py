from model import PINN
from data import generate_synthetic_data, generate_collocation_points
from loss import causal_loss
import torch

class BlackScholesPINN:
    def __init__(self, config):
        self.config = config
        self.model = PINN(config["layers"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # Prepare data
        self.S_data, self.t_data, self.C_data = generate_synthetic_data(config)

    def train(self):
        epsilon_val = self.config["epsilon"]

        for epoch in range(self.config["epochs"]):
            self.optimizer.zero_grad()

            # PDE collocation points
            S_colloc, t_colloc = generate_collocation_points(self.config)

            # Losses
            loss, loss_data, loss_pde, mean_weights = causal_loss(
                self.model,
                self.S_data, self.t_data, self.C_data,
                S_colloc, t_colloc,
                self.config["r"],
                self.config["sigma"],
                epsilon=epsilon_val
            )
            loss.backward()
            self.optimizer.step()

            if epoch % self.config["log_interval"] == 0:
                # [修改] 加上 Mean W 的打印
                print(f"Epoch {epoch} | Total: {loss.item():.6f} | Data: {loss_data.item():.6f} | PDE: {loss_pde.item():.6f} | Mean W: {mean_weights.item():.4f}")
    
    def export(self):
        torch.save(self.model.state_dict(), self.config.get("model_path", "model.pth"))

    def predict(self, S_eval, t_eval):
        with torch.no_grad():
            return self.model(S_eval, t_eval)
