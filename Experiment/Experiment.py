import sys
sys.path.append("..")
import os 
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from CMAPSS_Related.load_data_CMAPSS import cmapss_data_train_vali_loader
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData
from torch.utils.data import DataLoader
from Model.Inception_Attention import Inception_Attention
from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from Experiment.HTS_Loss_Function import HTSLoss
from Experiment.HTS_Loss_Function import Weighted_MSE_Loss, MSE_Smoothness_Loss


"""
This file only used for CMAPSS Datase
"""


class Exp_Inception_Attention(object):
    def __init__(self, args):
        self.args = args

        # load CMAPSS dataset
        if self.args.dataset_name == "CMAPSS":   
            self.train_data, self.train_loader, self.vali_data, self.vali_loader = self._get_data_CMPASS(flag='train')
            self.test_data, self.test_loader = self._get_data_CMPASS(flag='test')
        
        # load FEMTO dataset
        # if self.args.dataset_name == "FEMTO":  
            # self.train_data, self.train_loader, self.vali_data, self.vali_loader = self._get_data_FEMTO(flag='train')
            # self.test_data, self.test_loader = self._get_data_FEMTO(flag='test')
        
        # load XJTU dataset
        # if self.args.dataset_name == "XJTU":    
            # self.train_data, self.train_loader, self.vali_data, self.vali_loader = self._get_data_XJTU(flag='train')
            # self.test_data, self.test_loader = self._get_data_XJTU(flag='test')

        # build the Inception-Attention Model:
        self.model = self._get_model()

        # What optimisers and loss functions can be used by the model
        self.optimizer_dict = {"Adam": optim.Adam}
        self.criterion_dict = {"MSE": nn.MSELoss, "CrossEntropy": nn.CrossEntropyLoss, "WeightMSE":Weighted_MSE_Loss, "smooth_mse":MSE_Smoothness_Loss}

    # ------------------- function to build model -------------------------------------
    def _get_model(self):
        model = Inception_Attention(dropout        =  self.args.dropout,
                                    d_model        =  self.args.d_model,
                                    max_len        =  self.args.max_len,
                                    d_k            =  self.args.d_k,
                                    d_v            =  self.args.d_v,
                                    n_heads_full   =  self.args.n_heads_full,
                                    n_heads_log    =  self.args.n_heads_log,
                                    n_heads_local  =  self.args.n_heads_local,
                                    n_heads_prob   =  self.args.n_heads_prob,
                                    n_heads_FFT    =  self.args.n_heads_FFT,
                                    n_heads_auto   =  self.args.n_heads_auto,
                                    moving_avg     =  self.args.moving_avg,
                                    d_ff           =  self.args.d_ff,
                                    n_layers       =  self.args.n_layers,
                                    device         =  self.args.device,
                                    predictor_type =  self.args.predictor_type)
        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))
        
        return model.double().to(self.args.device)

    # --------------------------- select optimizer ------------------------------
    def _select_optimizer(self):
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # ---------------------------- select criterion --------------------------------
    def _select_criterion(self):
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError

        criterion = self.criterion_dict[self.args.criterion]()
        return criterion

    # ------------------------ get Dataloader -------------------------------------

    #  funnction of load CMPASS Dataset
    def _get_data_CMPASS(self, flag="train"):
        args = self.args
        if flag == 'train':
            # train and validation dataset
            X_train, y_train, X_vali, y_vali = cmapss_data_train_vali_loader(data_path        =  args.data_path,
                                                                             Data_id          =  args.Data_id,
                                                                             flag             =  "train",
                                                                             sequence_length  =  args.max_len,
                                                                             MAXLIFE          =  args.MAXLIFE,
                                                                             difference       =  args.difference,
                                                                             normalization    =  args.normalization,
                                                                             validation       =  args.validation)
            train_data_set = CMAPSSData(X_train, y_train)
            vali_data_set = CMAPSSData(X_vali, y_vali)

            train_data_loader = DataLoader(dataset     =  train_data_set,
                                           batch_size  =  args.batch_size,
                                           shuffle     =  True,
                                           num_workers =  0,
                                           drop_last   =  False)

            vali_data_loader = DataLoader(dataset      =  vali_data_set,
                                          batch_size   =  args.batch_size,
                                          shuffle      =  False,
                                          num_workers  =  0,
                                          drop_last    =  False)
            return train_data_set, train_data_loader, vali_data_set, vali_data_loader

        else:
            # test dataset
            X_test, y_test = cmapss_data_train_vali_loader(data_path        =  args.data_path,
                                                           Data_id          =  args.Data_id,
                                                           flag             =  "test",
                                                           sequence_length  =  args.max_len,
                                                           MAXLIFE          =  args.MAXLIFE,
                                                           difference       =  args.difference,
                                                           normalization    =  args.normalization,
                                                           validation       =  args.validation)
            test_data_set = CMAPSSData(X_test, y_test)
            test_data_loader = DataLoader(dataset      =  test_data_set,
                                          batch_size   =  args.batch_size,
                                          shuffle      =  False,
                                          num_workers  =  0,
                                          drop_last    =  False)
            return test_data_set, test_data_loader

    
    #  function of load FEMTO Dataset
    # def _get_data_FEMTO(self, flag="train"):


    #  function of load XJTU Dataset
    # def _get_data_XJTU(self, flag="train"):

    # -------------------------------------------- training function -----------------------------------
    def train(self, save_path):

        # save address
        path = './logs/' + save_path
        if not os.path.exists(path):
            os.makedirs(path)

        # how many step of train and validation:
        train_steps = len(self.train_loader)
        vali_steps = len(self.vali_loader)
        print("train_steps: ", train_steps)
        print("validaion_steps: ", vali_steps)

        # initial early stopping
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)

        # initial learning rate
        learning_rate_adapter = adjust_learning_rate_class(self.args, True)
        # choose optimizer
        model_optim = self._select_optimizer()

        # choose loss function
        # loss_criterion = self._select_criterion().to(self.args.device)
        loss_criterion = HTSLoss(enc_pred_loss = self.args.enc_pred_loss, 
                                 seq_length    = self.args.max_len,
                                 weight_type   = self.args.weight_type,
                                 sigma_faktor  = self.args.sigma_faktor,
                                 anteil        = self.args.anteil,
                                 smooth_loss   = self.args.smooth_loss,
                                 device        = self.args.device)

        # training process
        print("start training")
        for epoch in range(self.args.train_epochs):
            start_time = time()

            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.double().to(self.args.device)
                batch_y = batch_y.double().to(self.args.device)

                # model prediction
                outputs = self.model(batch_x)

                loss = loss_criterion(outputs, batch_y)

                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                # ------------------------------------------------

            end_time = time()
            epoch_time = end_time - start_time
            train_loss = np.average(train_loss)  # avgerage loss 

            # validation process:
            vali_loss = self.validation(self.vali_loader, loss_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

            # At the end of each epoch, Determine if we need to stop and adjust the learning rate
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            learning_rate_adapter(model_optim, vali_loss)

        last_model_path = path + '/' + 'last_checkpoint.pth'
        torch.save(self.model.state_dict(), last_model_path)

        average_enc_loss, average_enc_overall_loss = self.test(self.test_loader)
        print("test performace of enc is: ", average_enc_loss, " of enc overall is: ", average_enc_overall_loss)

    # ---------------------------------- validation function -----------------------------------------
    def validation(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.double().to(self.args.device)
            batch_y = batch_y.double().to(self.args.device)

            # prediction
            outputs = self.model(batch_x)

            loss = criterion(outputs, batch_y)

            total_loss.append(loss.item())

        average_vali_loss = np.average(total_loss)

        self.model.train()
        return average_vali_loss

    # ----------------------------------- test function ------------------------------------------
    def test(self, test_loader):
        self.model.eval()
        enc_pred = []
        gt = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.double().double().to(self.args.device)
            batch_y = batch_y.double().double().to(self.args.device)

            outputs = self.model(batch_x)

            batch_y = batch_y.detach().cpu().numpy()
            enc = outputs.detach().cpu().numpy()

            gt.append(batch_y)
            enc_pred.append(enc)

        gt = np.concatenate(gt).reshape(-1, self.args.max_len)
        enc_pred = np.concatenate(enc_pred).reshape(-1, self.args.max_len)
        average_enc_loss = np.sqrt(mean_squared_error(enc_pred[:, -1], gt[:, -1]))
        
        #  ---------------- visualization -------------------------------- 
        # actual_rul = gt[:, -1]
        # pred_rul = enc_pred[:, -1]
        # x = range(1, len(pred_rul)+1)
        # plt.figure(figsize=(12, 3))
        # plt.plot(x, actual_rul, color='r', label="real_rul") 
        # plt.plot(x, pred_rul, color='g', label="predicted_rul") 
        # plt.xlabel("sample_time")
        # plt.ylabel("rul_time")
        # plt.show()

        average_enc_overall_loss = np.sqrt(mean_squared_error(enc_pred, gt))

        return average_enc_loss, average_enc_overall_loss