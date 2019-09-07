# Copyright 2019 Indro Spinelli. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import dgl
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from ginn.utils import *
from ginn.models import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class GINN(object):

    """
    Impute with GINN!
    Functions:
        __init__
        add_data()
        fit()
        fit_transform()
        transform()
    """

    def __init__(
        self,
        features,
        mask,
        num_mask,
        cat_mask,
        oh_cat_cols,
        numerical_columns,
        categorical_columns,
        embedding_dim=128,
        activation=F.relu,
        dropout=0.5,
        percentile=97.72,
        distance_metric="euclidean",
        weight_missing=False,
        graph=True,
        skip=True,
        glob_attr=True,
    ):
        """
            Build the graph-structure of the dataset based on the similarity
            Instantiate the network based on the graph using the dgl library
        """
        self.features = features
        self.mask = mask
        self.num_mask = num_mask
        self.cat_mask = cat_mask
        self.oh_cat_cols = oh_cat_cols

        self.embedding_dim = embedding_dim
        self.activation = activation
        self.dropout = dropout
        self.percentile = percentile
        self.weight_missing = weight_missing
        self.distance_metric = distance_metric
        self.num_cols = numerical_columns
        self.cat_cols = categorical_columns

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GINN is running on", self.device)
        maskT = torch.FloatTensor(self.mask).to(self.device)
        nxg = dataset2nxg(
            self.features,
            self.mask,
            self.percentile,
            self.distance_metric,
            self.weight_missing,
        )
        g = dgl.DGLGraph()
        g.set_e_initializer(dgl.init.zero_initializer)
        g.set_n_initializer(dgl.init.zero_initializer)

        if graph:
            g.from_networkx(nxg)
            if skip:
                f = dgl.DGLGraph()
                f.set_e_initializer(dgl.init.zero_initializer)
                f.set_n_initializer(dgl.init.zero_initializer)
                f.from_networkx(nxg)
        else:
            g.add_nodes(self.features.shape[0])
        g.add_edges(g.nodes(), g.nodes())

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.unsqueeze(1)
        g.ndata["norm"] = torch.FloatTensor(norm).to(self.device)
        self.graph = g

        if skip:
            degs = f.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.unsqueeze(1)
            f.ndata["norm"] = torch.FloatTensor(norm).to(self.device)
            self.f = f
            if glob_attr:
                m = torch.FloatTensor(np.mean(self.features, axis=0)).to(self.device)
                self.autoencoder = ginn_autoencoder_global(
                    self.graph,
                    self.f,
                    m,
                    maskT,
                    self.features.shape[1],
                    self.embedding_dim,
                    self.activation,
                    self.dropout,
                ).to(self.device)
            else:
                self.autoencoder = ginn_autoencoder_skip(
                    self.graph,
                    self.f,
                    maskT,
                    self.features.shape[1],
                    self.embedding_dim,
                    self.activation,
                    self.dropout,
                ).to(self.device)
        else:
            self.autoencoder = ginn_autoencoder(
                self.graph,
                maskT,
                self.features.shape[1],
                self.embedding_dim,
                self.activation,
                self.dropout,
            ).to(self.device)

        self.critic = ginn_critic(self.features.shape[1], self.embedding_dim, 0.25).to(
            self.device
        )

        return

    def add_data(self, new_features, new_mask, new_num_mask, new_cat_mask):
        """
            Inject the new dataset in the previous graph adding nodes and edges
        """

        old_features = self.features.copy()
        old_mask = self.mask.copy()
        old_num_mask = self.num_mask.copy()
        old_cat_mask = self.cat_mask.copy()

        self.features = np.r_[old_features, new_features]
        self.mask = np.r_[old_mask, new_mask]
        self.num_mask = np.r_[old_num_mask, new_num_mask]
        self.cat_mask = np.r_[old_cat_mask, new_cat_mask]
        maskT = torch.FloatTensor(self.mask).to(self.device)
        self.autoencoder.mask = maskT
        edges_to_add = new_edges(
            self.features[: old_features.shape[0]],
            old_mask,
            self.features[old_features.shape[0] :],
            new_mask,
            self.percentile,
            self.distance_metric,
            self.weight_missing,
        )

        src = list(edges_to_add[:, 0] + old_features.shape[0])
        dest = list(edges_to_add[:, 1])
        self_loop = [
            i
            for i in range(
                old_features.shape[0], old_features.shape[0] + new_features.shape[0]
            )
        ]

        self.graph.add_nodes(new_features.shape[0])
        self.graph.add_edges(self_loop, self_loop)
        self.graph.add_edges(src, dest)
        self.graph.add_edges(dest, src)

        degs = self.graph.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.unsqueeze(1)
        self.graph.ndata["norm"] = torch.FloatTensor(norm).to(self.device)
        self.autoencoder.masked_gcn.g = self.graph
        self.autoencoder.output_gcn.g = self.graph
        if self.f:
            self.f.add_nodes(new_features.shape[0])
            self.f.add_edges(src, dest)
            self.f.add_edges(dest, src)

            degs = self.f.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.unsqueeze(1)
            self.f.ndata["norm"] = torch.FloatTensor(norm).to(self.device)
            self.autoencoder.masked_gcn.f = self.f
            self.autoencoder.output_gcn.f = self.f

        return

    def fit(
        self,
        epochs=3000,
        batch_size=32,
        patience=5,
        auto_lr=1e-3,
        crit_lr=1e-5,
        crit_iter=5,
        weight_decay=0,
        adversarial=True,
        fine_tune=False,
        verbose=False,
        glob_attr=True,
    ):
        """
            Trains the network, if fine_tune=True uses the previous state of the optimizer 
            instantiated before.
        """
        if not fine_tune:
            self.optim_auto = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=auto_lr,
                betas=(0.0, 0.9),
                weight_decay=weight_decay,
            )
            if adversarial:
                self.optim_crit = torch.optim.Adam(
                    self.critic.parameters(),
                    lr=crit_lr,
                    betas=(0.0, 0.9),
                    weight_decay=weight_decay,
                )

        cat_loss_weigth = len(self.cat_cols) / (len(self.cat_cols) + len(self.num_cols))
        if cat_loss_weigth > 0.0:
            bce_criterion = nn.BCELoss().to(self.device)
        if cat_loss_weigth < 1.0:
            mse_criterion = nn.MSELoss().to(self.device)
        if glob_attr:
            global_criterion = nn.MSELoss().to(self.device)

        dur = []
        a_losses = []
        c_losses = []
        r_losses = []
        w_losses = []
        t0 = time.time()

        featT = torch.FloatTensor(self.features).to(self.device)
        maskT = torch.FloatTensor(self.mask).to(self.device)
        num_maskT = torch.ByteTensor(self.num_mask.astype(int)).to(self.device)
        cat_maskT = torch.ByteTensor(self.cat_mask.astype(int)).to(self.device)

        b_loss = 999
        patience_cnt = 0

        for epoch in range(epochs + 1):
            self.autoencoder.train()
            self.critic.train()
            # Reconstruction
            iX = self.autoencoder(featT)
            a_loss = 0
            if cat_loss_weigth < 1.0:
                num_loss = mse_criterion(iX[num_maskT], featT[num_maskT])
                a_loss += (1 - cat_loss_weigth) * num_loss
            if cat_loss_weigth > 0.0:
                cat_loss = bce_criterion(iX[cat_maskT], featT[cat_maskT])
                a_loss += cat_loss_weigth * cat_loss
            # Regularization
            if adversarial:
                if glob_attr:
                    ag_loss = a_loss + global_criterion(
                        torch.FloatTensor(np.mean(self.features, axis=0)).to(
                            self.device
                        ),
                        torch.FloatTensor(
                            np.mean(iX.cpu().detach().numpy(), axis=0)
                        ).to(self.device),
                    )
                    self.optim_auto.zero_grad()
                    ag_loss.backward()
                    self.optim_auto.step()
                else:
                    self.optim_auto.zero_grad()
                    a_loss.backward()
                    self.optim_auto.step()

                for _ in range(crit_iter):

                    b_mask = batch_mask(self.features.shape[0], batch_size)
                    batchT = torch.ByteTensor(b_mask.astype(int)).to(self.device)

                    iX = self.autoencoder(featT)

                    x_real = torch.mul(featT[batchT], maskT[batchT])
                    x_fake = torch.mul(iX[batchT], maskT[batchT])

                    x_fake = proper_onehot(
                        x_fake.detach().cpu().numpy(), self.oh_cat_cols
                    )
                    x_fake = torch.FloatTensor(x_fake).to(self.device)
                    c_real = self.critic(x_real)
                    c_fake = self.critic(x_fake)
                    gp = hard_gradient_penalty(self.critic, x_real, x_fake, self.device)
                    w_loss = c_fake.mean() - c_real.mean()
                    c_loss = w_loss + gp

                    self.optim_crit.zero_grad()
                    c_loss.backward()
                    self.optim_crit.step()

                b_mask = batch_mask(self.features.shape[0], batch_size)
                batchT = torch.ByteTensor(b_mask.astype(int)).to(self.device)

                iX = self.autoencoder(featT)

                b_featT = featT[batchT]
                b_iX = iX[batchT]
                b_maskT = maskT[batchT]
                b_num_maskT = num_maskT[batchT]
                b_cat_maskT = cat_maskT[batchT]

                ra_loss = 0
                if cat_loss_weigth < 1.0:
                    num_loss = mse_criterion(b_iX[b_num_maskT], b_featT[b_num_maskT])
                    ra_loss += (1 - cat_loss_weigth) * num_loss

                if cat_loss_weigth > 0.0:
                    cat_loss = bce_criterion(b_iX[b_cat_maskT], b_featT[b_cat_maskT])
                    ra_loss += cat_loss_weigth * cat_loss

                x_fake = torch.mul(b_iX, b_maskT)
                x_fake = torch.FloatTensor(
                    proper_onehot(x_fake.detach().cpu().numpy(), self.oh_cat_cols)
                ).to(self.device)

                rc_fake = self.critic(x_fake)
                r_loss = -rc_fake.mean() + ra_loss
                self.optim_auto.zero_grad()
                r_loss.backward()
                self.optim_auto.step()

            else:
                if glob_attr:
                    ag_loss = a_loss + global_criterion(
                        torch.FloatTensor(np.mean(self.features, axis=0)).to(
                            self.device
                        ),
                        torch.FloatTensor(
                            np.mean(iX.cpu().detach().numpy(), axis=0)
                        ).to(self.device),
                    )
                    self.optim_auto.zero_grad()
                    ag_loss.backward()
                    self.optim_auto.step()
                else:
                    self.optim_auto.zero_grad()
                    a_loss.backward()
                    self.optim_auto.step()

            if epoch % 100 == 0:
                patience_cnt += 1
                if a_loss < b_loss:
                    patience_cnt = 0
                    b_loss = a_loss
                    if adversarial:
                        torch.save(
                            {
                                "auto_state_dict": self.autoencoder.state_dict(),
                                "optim_auto_state_dict": self.optim_auto.state_dict(),
                                "crit_state_dict": self.critic.state_dict(),
                                "optim_crit_state_dict": self.optim_crit.state_dict(),
                            },
                            "ginn.pth",
                        )
                    else:
                        torch.save(
                            {
                                "auto_state_dict": self.autoencoder.state_dict(),
                                "optim_auto_state_dict": self.optim_auto.state_dict(),
                            },
                            "ginn.pth",
                        )

            if patience_cnt > patience:
                break

            if epoch % 1 == 0:
                dur.append(time.time() - t0)

                a_losses.append(a_loss.detach().item())
                if adversarial:
                    c_losses.append(c_loss.detach().item())
                    r_losses.append(r_loss.detach().item())
                    w_losses.append(w_loss.detach().item())
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f," % a_loss.detach().item(),
                            "C_LOSS: %f," % c_loss.detach().item(),
                            "R_LOSS: %f," % r_loss.detach().item(),
                            "W_LOSS: %f " % w_loss.detach().item(),
                            "= (%f" % c_fake.mean().detach().item(),
                            "-(%f))" % c_real.mean().detach().item(),
                            "GP: %f" % gp.detach().item(),
                        )
                else:
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f" % a_loss.detach().item(),
                        )

        checkpoint = torch.load("ginn.pth")
        self.autoencoder.load_state_dict(checkpoint["auto_state_dict"])
        self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])
        if adversarial:
            self.critic.load_state_dict(checkpoint["crit_state_dict"])
            self.optim_crit.load_state_dict(checkpoint["optim_crit_state_dict"])
        return dur, a_losses, c_losses, r_losses, w_losses

    def fit_transform(
        self,
        epochs=3000,
        batch_size=32,
        patience=10,
        auto_lr=1e-3,
        crit_lr=1e-5,
        crit_iter=5,
        weight_decay=0,
        adversarial=True,
        fine_tune=False,
        verbose=False,
        glob_attr=True,
    ):
        """
            Trains the network, if fine_tune=True uses the previous state of the optimizer 
            instantiated before.
        """
        if not fine_tune:
            self.optim_auto = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=auto_lr,
                # betas=(0.0, 0.9),
                weight_decay=weight_decay,
            )
            if adversarial:
                self.optim_crit = torch.optim.Adam(
                    self.critic.parameters(),
                    lr=crit_lr,
                    betas=(0.0, 0.9),
                    weight_decay=weight_decay,
                )

        cat_loss_weigth = len(self.cat_cols) / (len(self.cat_cols) + len(self.num_cols))
        if cat_loss_weigth > 0.0:
            bce_criterion = nn.BCELoss().to(self.device)
        if cat_loss_weigth < 1.0:
            mse_criterion = nn.MSELoss().to(self.device)
        if glob_attr:
            global_criterion = nn.MSELoss().to(self.device)

        dur = []
        a_losses = []
        c_losses = []
        r_losses = []
        w_losses = []
        t0 = time.time()

        featT = torch.FloatTensor(self.features).to(self.device)
        maskT = torch.FloatTensor(self.mask).to(self.device)
        num_maskT = torch.ByteTensor(self.num_mask.astype(int)).to(self.device)
        cat_maskT = torch.ByteTensor(self.cat_mask.astype(int)).to(self.device)

        b_loss = 999
        patience_cnt = 0

        for epoch in range(epochs + 1):
            self.autoencoder.train()
            self.critic.train()
            # Reconstruction
            iX = self.autoencoder(featT)
            a_loss = 0
            if cat_loss_weigth < 1.0:
                num_loss = mse_criterion(iX[num_maskT], featT[num_maskT])
                a_loss += (1 - cat_loss_weigth) * num_loss
            if cat_loss_weigth > 0.0:
                cat_loss = bce_criterion(iX[cat_maskT], featT[cat_maskT])
                a_loss += cat_loss_weigth * cat_loss
            # Regularization
            if adversarial:
                if glob_attr:
                    ag_loss = a_loss + global_criterion(
                        torch.FloatTensor(np.mean(self.features, axis=0)).to(
                            self.device
                        ),
                        torch.FloatTensor(
                            np.mean(iX.cpu().detach().numpy(), axis=0)
                        ).to(self.device),
                    )
                    self.optim_auto.zero_grad()
                    ag_loss.backward()
                    self.optim_auto.step()
                else:
                    self.optim_auto.zero_grad()
                    a_loss.backward()
                    self.optim_auto.step()

                for _ in range(crit_iter):

                    b_mask = batch_mask(self.features.shape[0], batch_size)
                    batchT = torch.ByteTensor(b_mask.astype(int)).to(self.device)

                    iX = self.autoencoder(featT)

                    x_real = torch.mul(featT[batchT], maskT[batchT])
                    x_fake = torch.mul(iX[batchT], maskT[batchT])

                    x_fake = proper_onehot(
                        x_fake.detach().cpu().numpy(), self.oh_cat_cols
                    )
                    x_fake = torch.FloatTensor(x_fake).to(self.device)
                    c_real = self.critic(x_real)
                    c_fake = self.critic(x_fake)
                    gp = hard_gradient_penalty(self.critic, x_real, x_fake, self.device)
                    w_loss = c_fake.mean() - c_real.mean()
                    c_loss = w_loss + gp

                    self.optim_crit.zero_grad()
                    c_loss.backward()
                    self.optim_crit.step()

                b_mask = batch_mask(self.features.shape[0], batch_size)
                batchT = torch.ByteTensor(b_mask.astype(int)).to(self.device)

                iX = self.autoencoder(featT)

                b_featT = featT[batchT]
                b_iX = iX[batchT]
                b_maskT = maskT[batchT]
                b_num_maskT = num_maskT[batchT]
                b_cat_maskT = cat_maskT[batchT]

                ra_loss = 0
                if cat_loss_weigth < 1.0:
                    num_loss = mse_criterion(b_iX[b_num_maskT], b_featT[b_num_maskT])
                    ra_loss += (1 - cat_loss_weigth) * num_loss

                if cat_loss_weigth > 0.0:
                    cat_loss = bce_criterion(b_iX[b_cat_maskT], b_featT[b_cat_maskT])
                    ra_loss += cat_loss_weigth * cat_loss

                x_fake = torch.mul(b_iX, b_maskT)
                x_fake = torch.FloatTensor(
                    proper_onehot(x_fake.detach().cpu().numpy(), self.oh_cat_cols)
                ).to(self.device)

                rc_fake = self.critic(x_fake)
                r_loss = -rc_fake.mean() + ra_loss
                self.optim_auto.zero_grad()
                r_loss.backward()
                self.optim_auto.step()

            else:
                if glob_attr:
                    ag_loss = a_loss + global_criterion(
                        torch.FloatTensor(np.mean(self.features, axis=0)).to(
                            self.device
                        ),
                        torch.FloatTensor(
                            np.mean(iX.cpu().detach().numpy(), axis=0)
                        ).to(self.device),
                    )
                    self.optim_auto.zero_grad()
                    ag_loss.backward()
                    self.optim_auto.step()
                else:
                    self.optim_auto.zero_grad()
                    a_loss.backward()
                    self.optim_auto.step()

            if epoch % 100 == 0:
                patience_cnt += 1
                if a_loss < b_loss:
                    patience_cnt = 0
                    b_loss = a_loss
                    if adversarial:
                        torch.save(
                            {
                                "auto_state_dict": self.autoencoder.state_dict(),
                                "optim_auto_state_dict": self.optim_auto.state_dict(),
                                "crit_state_dict": self.critic.state_dict(),
                                "optim_crit_state_dict": self.optim_crit.state_dict(),
                            },
                            "ginn.pth",
                        )
                    else:
                        torch.save(
                            {
                                "auto_state_dict": self.autoencoder.state_dict(),
                                "optim_auto_state_dict": self.optim_auto.state_dict(),
                            },
                            "ginn.pth",
                        )

            if patience_cnt > patience:
                break

            if epoch % 1 == 0:
                dur.append(time.time() - t0)

                a_losses.append(a_loss.detach().item())
                if adversarial:
                    c_losses.append(c_loss.detach().item())
                    r_losses.append(r_loss.detach().item())
                    w_losses.append(w_loss.detach().item())
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f," % a_loss.detach().item(),
                            "C_LOSS: %f," % c_loss.detach().item(),
                            "R_LOSS: %f," % r_loss.detach().item(),
                            "W_LOSS: %f " % w_loss.detach().item(),
                            "= (%f" % c_fake.mean().detach().item(),
                            "-(%f))" % c_real.mean().detach().item(),
                            "GP: %f" % gp.detach().item(),
                        )
                else:
                    if verbose:
                        print(
                            "EPOCH: %05d," % epoch,
                            "A_LOSS: %f" % a_loss.detach().item(),
                        )

        checkpoint = torch.load("ginn.pth")
        self.autoencoder.load_state_dict(checkpoint["auto_state_dict"])
        self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])
        if adversarial:
            self.critic.load_state_dict(checkpoint["crit_state_dict"])
            self.optim_crit.load_state_dict(checkpoint["optim_crit_state_dict"])
        imputed_data = imputation(self.autoencoder, featT, self.oh_cat_cols)
        filled_data = np.where(self.mask, featT.cpu(), imputed_data)
        return filled_data, dur, a_losses, c_losses, r_losses, w_losses

    def transform(self, multiple=False, imputation_num=5):
        """
            Impute the missing values in the dataset
        """
        featT = torch.FloatTensor(self.features).to(self.device)
        if multiple:
            imputed_data = multiple_imputation(
                self.autoencoder, featT, self.oh_cat_cols, imputation_num
            )
        else:
            imputed_data = imputation(self.autoencoder, featT, self.oh_cat_cols)
        filled_data = np.where(self.mask, featT.cpu(), imputed_data)
        return filled_data
