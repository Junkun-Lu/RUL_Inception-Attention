import torch.nn as nn

# ------------------------------------- Residual_Connection & Layer_Norm -----------------------------------
"""
input shape: [batch_size, max_len, d_model]
output shape: [batch_size, max_len, d_model]
"""
class PoswiseFeedForwardNet(nn.Module):
  def __init__(self, d_model, d_ff, device):
    super(PoswiseFeedForwardNet, self).__init__()

    self.d_model = d_model
    self.device = device
    self.fc = nn.Sequential(
        nn.Linear(d_model, d_ff, bias=False),
        nn.ReLU(),
        nn.Linear(d_ff, d_model, bias=False)
    )

  def forward(self, inputs):
    residual = inputs
    output = self.fc(inputs)
    return nn.LayerNorm(self.d_model).double().to(self.device)(output + residual)