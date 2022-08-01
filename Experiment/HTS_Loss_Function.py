import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

"""
To respect the smoothness of the degradation process and avoid oscillations, 
we introduce a regularization term penalizing the squared distance between the 
predictions of adjacent cycles. 
Based on this idea, the training objective consists of 
a weighted mean squared error(Weighted_MSE_Loss) and 
a smoothed regularization term(MSE_Smoothness_Loss)
"""


class MSE_Smoothness_Loss(nn.Module):
    
    def __init__(self):
        super(MSE_Smoothness_Loss, self).__init__()
        self.conv1d = nn.Conv1d(1,1,2,bias=True)
        self.conv1d.weight.data = torch.tensor(np.array([[[-1,1]]])).double()
        self.conv1d.bias.data = torch.tensor([1]).double()
        self.conv1d.weight.requires_grad = False
        self.conv1d.bias.requires_grad = False

        
    def forward(self, pred):
        smooth_error = self.conv1d(pred)**2
        loss = smooth_error.mean()
        return loss
		

class Weighted_MSE_Loss(nn.Module):
    
    def __init__(self, 
          seq_length=20, 
          weight_type="gaussian",
          sigma_faktor=10, 
          anteil=15,
          device = "cuda"):

        super(Weighted_MSE_Loss, self).__init__()
        self.weight_type = weight_type
        self.seq_length = seq_length
        if self.weight_type == "gaussian":

            self.sigma      = seq_length/sigma_faktor
        
            x = np.linspace(1, seq_length, seq_length)

            mu = seq_length
            y = stats.norm.pdf(x, mu, self.sigma)
            y = y + np.max(y)/anteil
            print(anteil, sigma_faktor)
            y = y/np.sum(y)*seq_length
            plt.plot(x, y)
            plt.show()
            with torch.no_grad():
                self.weights = torch.Tensor(y).double().to(device)  
        elif self.weight_type == "last":
            y = np.zeros(self.seq_length)
            y[-1] = self.seq_length
            with torch.no_grad():
                self.weights = torch.Tensor(y).double().to(device)
        else:
            print( weight_type," is not implemented")
        
    def forward(self, pred, target):
        se  = (pred-target)**2
        out = se * self.weights.expand_as(target)
        loss = out.mean() 
        return loss


criterion_dict = {"MSE"            :nn.MSELoss,
          "CrossEntropy"   :nn.CrossEntropyLoss,
          "WeightMSE"      :Weighted_MSE_Loss,
          "smooth_mse"     :MSE_Smoothness_Loss}

class HTSLoss(nn.Module):
    def __init__(self, 
          enc_pred_loss = "WeightMSE", 
          seq_length = 32,
          weight_type = "gaussian",
          sigma_faktor = 10,
          anteil      = 15,
          smooth_loss = "smooth_mse",
          device  = "cpu"):

        super(HTSLoss, self).__init__()
        self.enc_pred_loss          = enc_pred_loss   
        self.smooth_loss            = smooth_loss
        print("enc_pred_criterion")
        if self.enc_pred_loss == "WeightMSE":
            self.enc_pred_criterion     =  criterion_dict["WeightMSE"](seq_length, weight_type, sigma_faktor, anteil, device)
        else:
            self.enc_pred_criterion     =  criterion_dict[self.enc_pred_loss]()
			
        if smooth_loss is not None:
            print("smooth_loss")
            self.smooth_criterion       =  criterion_dict[smooth_loss]().to(device)  

    def forward(self, outputs, batch_y):
        # no decoder , only the prediction from encoder "enc_pred"
        enc_pred              = outputs
        enc_pred_loss         = self.enc_pred_criterion(enc_pred, batch_y)
        loss                  = enc_pred_loss
        if self.smooth_loss is not None:
            smoothloss = self.smooth_criterion(enc_pred.unsqueeze(1))
            loss = loss + smoothloss
        return loss