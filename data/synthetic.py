# from imp import PY_FROZEN
# import os
# from os.path import join
# import gzip
# import shutil
# from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.linalg import block_diag
from scipy.special import softmax

from torch.distributions import Categorical


def get_decoder(manifold, x_dim, z_dim, rng_data_gen, dtype=torch.double):
    """Mixing function.

    Args:
        manifold (str): 'nn' (only value implemented)
        x_dim (int): 
        z_dim (int): 
        rng_data_gen (numpy.random.default_rng): random number generator

    Raises:
        NotImplementedError: if other manifold type is given as argument.

    Returns:
        tuple: (function) decoder, (float) noise_std
    """
    if manifold == "nn":
        # NOTE: injectivity requires z_dim <= h_dim <= x_dim
        h_dim = x_dim
        neg_slope = 0.2
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # sampling NN weight matrices
        W1 = rng_data_gen.normal(size=(z_dim, h_dim))
        W1 = np.linalg.qr(W1.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W1, W1.T) - np.eye(self.z_dim))))
        W1 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (z_dim + h_dim))
        W1 = torch.tensor(W1, dtype=dtype).to(device)
        W1.requires_grad = False

        W2 = rng_data_gen.normal(size=(h_dim, h_dim))
        W2 = np.linalg.qr(W2.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W2, W2.T) - np.eye(h_dim))))
        W2 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (2 * h_dim))
        W2 = torch.tensor(W2, dtype=dtype).to(device)
        W2.requires_grad = False

        W3 = rng_data_gen.normal(size=(h_dim, h_dim))
        W3 = np.linalg.qr(W3.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W3, W3.T) - np.eye(h_dim))))
        W3 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (2 * h_dim))
        W3 = torch.tensor(W3, dtype=dtype).to(device)
        W3.requires_grad = False

        W4 = rng_data_gen.normal(size=(h_dim, x_dim))
        W4 = np.linalg.qr(W4.T)[0].T
        # print("distance to identity:", np.max(np.abs(np.matmul(W4, W4.T) - np.eye(h_dim))))
        W4 *= np.sqrt(2 / (1 + neg_slope ** 2)) * np.sqrt(2. / (x_dim + h_dim))
        W4 = torch.tensor(W4, dtype=dtype).to(device)
        W4.requires_grad = False

        # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
        # since Wx is injective
        # when columns are linearly indep, which happens almost surely,
        # plus, composition of injective functions is injective.
        def decoder(z):
            """Function that maps the latent variable z into an observed variable x.

            Args:
                z (ndarray):
            """
            with torch.no_grad():
                z = torch.tensor(z, dtype=dtype).to(device)
                h1 = torch.matmul(z, W1)
                h1 = torch.maximum(neg_slope * h1, h1)  # leaky relu
                h2 = torch.matmul(h1, W2)
                h2 = torch.maximum(neg_slope * h2, h2)  # leaky relu
                h3 = torch.matmul(h2, W3)
                h3 = torch.maximum(neg_slope * h3, h3)  # leaky relu
                out = torch.matmul(h3, W4)
            return out.cpu().numpy()

        noise_std = 0.01
    else:
        raise NotImplementedError(f"The manifold {manifold} is not implemented.")

    return decoder, noise_std


class ActionToyManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, manifold, transition_model, num_samples, seed, x_dim, z_dim, no_norm=False, seed_data=265542):
        """Action dataset.

        Args:
            manifold (str): 'nn' (only value implemented)
            transition_model (str): "action_sparsity_trivial" or "action_sparsity_non_trivial" or "action_sparsity_non_trivial_no_suff_var" or "action_sparsity_non_trivial_no_graph_crit"
            num_samples (int): 
            seed (int): seed for sampling dataset
            x_dim (int): 
            z_dim (int):
            no_norm (bool, optional): Defaults to False.
            seed_data (int): seed for sampling data generation process.

        Raises:
            NotImplementedError: if other transition model is given as argument.

        Returns:
            ActionToyManifoldDataset: 
        """
        super(ActionToyManifoldDataset, self).__init__()
        self.manifold = manifold
        self.transition_model = transition_model
        self.rng = np.random.default_rng(seed)  # use for dataset sampling
        self.rng_data_gen = np.random.default_rng(seed_data)  # use for sampling actual data generating process.
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.no_norm = no_norm

        if self.transition_model == "action_sparsity_trivial":
            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.sin(c)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.eye(self.z_dim)

        elif self.transition_model == "action_sparsity_non_trivial":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.sum(gt_gc * np.sin(c[:, None, :] * mat_range + shift), 2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_no_suff_var":
            gt_gc = np.concatenate([np.eye(self.z_dim), np.eye(self.z_dim)[:, 0:1]], 1)[:, 1:] + np.eye(self.z_dim)
            A = self.rng_data_gen.normal(size=(self.z_dim, self.z_dim)) * gt_gc

            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.matmul(c, A.T)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        elif self.transition_model == "action_sparsity_non_trivial_no_graph_crit":
            assert self.z_dim % 2 == 0
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_gc = block_diag(*[np.ones((2, 2)) for _ in range(int(self.z_dim / 2))])
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(c, var_fac=0.0001):
                mu_tp1 = np.sum(gt_gc * np.sin(c[:, None, :] * mat_range + shift), 2)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_gc = torch.Tensor(gt_gc)

        else:
            raise NotImplementedError(f"The transition model {self.transition_model} is not implemented.")

        self.decoder, self.noise_std = get_decoder(self.manifold, self.x_dim, self.z_dim, self.rng_data_gen)
        self.get_mean_var = get_mean_var
        self.create_data()

    def __len__(self):
        return self.num_samples

    def sample_z_given_c(self, c):
        mu_tp1, var_tp1 = self.get_mean_var(c)
        return self.rng.normal(mu_tp1, np.sqrt(var_tp1))

    def create_data(self):
        c = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.z_dim))
        z = self.sample_z_given_c(c)
        x = self.decoder(z)

        # normalize
        if not self.no_norm:
            x = (x - x.mean(0)) / x.std(0)

        x = x + self.noise_std * self.rng.normal(0, 1, size=(self.num_samples, self.x_dim))

        self.x = torch.Tensor(x)
        self.z = torch.Tensor(z)
        self.c = torch.Tensor(c) # actions

    def __getitem__(self, item):
        obs = self.x[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        cont_c = self.c[item]
        disc_c = torch.Tensor(np.array([0.])).long()
        valid = True # mask indicating that the sample should be included in the minibatch. not useful for synthetic data, but down the pipeline
        latent = self.z[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        return obs, cont_c, disc_c, valid, latent

    


class TemporalToyManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, manifold, transition_model, num_samples, seed, x_dim, z_dim, no_norm=False, seed_data=265542):
        super(TemporalToyManifoldDataset, self).__init__()
        self.manifold = manifold
        self.transition_model = transition_model
        self.rng = np.random.default_rng(seed)  # use for dataset sampling
        self.rng_data_gen = np.random.default_rng(seed_data)   # use for sampling actual data generating process.
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.no_norm = no_norm

        if self.transition_model == "temporal_sparsity_trivial":
            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                mu_tp1 =  z_t + lr * np.sin(z_t)
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.eye(self.z_dim)

        elif self.transition_model == "temporal_sparsity_non_trivial":
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.sum(gt_g * np.sin(z_t[:, None, :] * mat_range + shift), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_no_graph_crit":
            assert self.z_dim % 2 == 0
            mat_range = np.repeat(np.arange(3, self.z_dim + 3)[:, None] / np.pi, self.z_dim, 1)
            gt_g = block_diag(np.ones((int(self.z_dim / 2),int(self.z_dim / 2))), np.ones((int(self.z_dim / 2),int(self.z_dim / 2))))
            shift = np.repeat(np.arange(0, self.z_dim)[:, None], self.z_dim, 1)

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.sum(gt_g * np.sin(z_t[:, None, :] * mat_range + shift), 2)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)

        elif self.transition_model == "temporal_sparsity_non_trivial_no_suff_var":
            gt_g = np.tril(np.ones((self.z_dim, self.z_dim)))
            A = self.rng_data_gen.normal(size=(self.z_dim, self.z_dim)) * gt_g

            def get_mean_var(z_t, lr=0.5, var_fac=0.0001):
                delta = np.matmul(z_t, A.T)
                mu_tp1 = z_t + lr * delta
                var_tp1 = var_fac * np.ones_like(mu_tp1)
                return mu_tp1, var_tp1

            self.gt_g = torch.Tensor(gt_g)
        else:
            raise NotImplementedError(f"The transition model {self.transition_model} is not implemented.")

        self.decoder, self.noise_std = get_decoder(self.manifold, self.x_dim, self.z_dim, self.rng_data_gen)
        self.get_mean_var = get_mean_var
        self.create_data()

    def __len__(self):
        return self.num_samples

    def next_z(self, z_t):
        mu_tp1, var_tp1 = self.get_mean_var(z_t)
        if not self.transition_model.startswith("laplacian"):
            return self.rng.normal(mu_tp1, np.sqrt(var_tp1))
        else:
            return self.rng.laplace(mu_tp1, np.sqrt(0.5 * var_tp1))

    def rollout(self,):
        z_init = self.rng.normal(0, 1, size=(self.num_samples, self.z_dim))

        zs = np.zeros((self.num_samples, 2, self.z_dim))
        zs[:, 0, :] = z_init
        zs[:, 1, :] = self.next_z(zs[:, 0])

        return zs

    def create_data(self):
        # rollout in latent space
        z = self.rollout()

        # decode
        x = self.decoder(z.reshape(2 * self.num_samples, self.z_dim))

        # normalize
        if not self.no_norm:
            x = (x - x.mean(0)) / x.std(0)

        x = x + self.noise_std * self.rng.normal(0, 1, size=(2 * self.num_samples, self.x_dim))

        self.x = torch.Tensor(x.reshape(self.num_samples, 2, self.x_dim))
        self.z = torch.Tensor(z)

    def __getitem__(self, item):
        obs = self.x[item]
        cont_c = torch.Tensor(np.array([0.]))
        disc_c = torch.Tensor(np.array([0.])).long()
        valid = True
        other = self.z[item]

        return obs, cont_c, disc_c, valid, other


class DiscreteActionToyManifoldDataset(torch.utils.data.Dataset):
    def __init__(self, manifold, transition_model, num_samples, seed, x_dim, z_dim, n_classes=4, no_norm=True, seed_data=265542):
        """Dataset for action sparsity and discrete variables.

        Args:
            manifold (str): 'nn' (only value implemented)
            transition_model (str): "action_sparsity_trivial" or "action_sparsity_non_trivial" or "action_sparsity_non_trivial_no_suff_var" or "action_sparsity_non_trivial_no_graph_crit"
            num_samples (int): 
            seed (int): seed for sampling dataset
            x_dim (int): number of observed variables
            z_dim (int): number of latent variables
            n_classes (int): number of classes
            no_norm (bool, optional): Defaults to False.
            seed_data (int): seed for sampling data generation process.

        Raises:
            NotImplementedError: if other transition model is given as argument.

        Returns:
            ActionToyManifoldDataset: 
        """
        super(DiscreteActionToyManifoldDataset, self).__init__()
        self.manifold = manifold
        self.transition_model = transition_model
        self.rng = np.random.default_rng(seed)  # use for dataset sampling
        self.rng_data_gen = np.random.default_rng(seed_data)  # use for sampling actual data generating process.
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.num_samples = num_samples
        self.no_norm = no_norm
        self.k = self.n_classes - 1 # dimension of minimal sufficient statistics
        self.a_dim = self.z_dim # number of groups

        if self.transition_model == "simplest_discrete":

            self.G = torch.eye(self.z_dim) # G matrix, mask of latent and actions
            ## this only works when d_z = n_classes-1 = a !!!!!!

            # def p_z_given_a():
            # this currently works
            #     # "a" is encoded in the graph
            #     # ones_k = np.ones(self.k)
            #     lambda_0 = self.rng_data_gen.normal(size=(self.z_dim, self.k))
            #     # n = self.rng_data_gen.normal(size=(self.num_samples, self.k*self.z_dim))
            #     n = self.rng_data_gen.normal(size=(self.z_dim, self.k))

            #     lambdas = np.zeros(shape=(self.z_dim, self.a_dim, self.k))
                
            #     for i in range(self.G.shape[0]):
            #         for j in range(self.G.shape[1]):
            #             lambdas[i,j] = lambda_0[i] + np.expand_dims(self.G[i,j],-1) * n[i]

            #     # for l in range(self.G.shape[1]): # loop over groups
            #     #     extended_mask = np.kron(self.G[:,l], ones_k) # this is encoding the actions
            #     #     lambda_a = n[l] * extended_mask
            #     #     lambdas[l] = lambda_0 + lambda_a
            #     # lambdas_reshaped = lambdas.reshape((self.z_dim, self.z_dim, self.k))

            #     # append a row of ones for the probability of the last class
            #     zeros_c = np.zeros(shape=(self.z_dim, self.z_dim, 1))
            #     lambdas_c = np.concatenate([lambdas, zeros_c], -1) # logits
            #     p = softmax(lambdas_c, axis=-1)
            #     return p

            self.n_children = self.G.sum(axis=1) # number of elements in each group

            def p_z_given_a():
                # "a" is encoded in the graph
                # ones_k = np.ones(self.k)
                lambda_0 = self.rng_data_gen.normal(size=(self.z_dim, self.k))
                
                lambdas = np.zeros(shape=(self.z_dim, self.a_dim, self.k))
                
                for i in range(self.z_dim):
                    for g in range(self.a_dim):
                        n_intra = int(self.n_children[g].item())
                        n_g = self.rng_data_gen.normal(size=(self.z_dim, self.k, n_intra))
                        for j in range(n_intra):
                            lambdas[i,g,j] = lambda_0[i] + np.expand_dims(self.G[i,g],-1) * n_g[i,j]

                # for l in range(self.G.shape[1]): # loop over groups
                #     extended_mask = np.kron(self.G[:,l], ones_k) # this is encoding the actions
                #     lambda_a = n[l] * extended_mask
                #     lambdas[l] = lambda_0 + lambda_a
                # lambdas_reshaped = lambdas.reshape((self.z_dim, self.z_dim, self.k))

                # append a row of ones for the probability of the last class
                zeros_c = np.zeros(shape=(self.z_dim, self.z_dim, 1))
                lambdas_c = np.concatenate([lambdas, zeros_c], -1) # logits
                p = softmax(lambdas_c, axis=-1)
                return p


            # def p_z_given_a():
            #     # "a" is encoded in the graph
            #     ones_k = np.ones(self.k)
            #     lambda_0 = self.rng_data_gen.normal(size=self.k*self.z_dim)
            #     # n = self.rng_data_gen.normal(size=(self.num_samples, self.k*self.z_dim))
            #     n = self.rng_data_gen.normal(size=(self.k*self.z_dim))

            #     lambdas = np.zeros((self.z_dim, self.k*self.z_dim)) # here, the first z_dim is the number of groups

            #     for l in range(self.G.shape[1]): # loop over groups
            #         extended_mask = np.kron(self.G[:,l], ones_k) # this is encoding the actions
            #         lambda_a = n[l] * extended_mask
            #         lambdas[l] = lambda_0 + lambda_a
            #     lambdas_reshaped = lambdas.reshape((self.z_dim, self.z_dim, self.k))

            #     # append a row of ones for the probability of the last class
            #     zeros_c = np.zeros((self.z_dim, self.z_dim, 1))
            #     lambdas_c = np.concatenate([lambdas_reshaped, zeros_c], -1) # logits
            #     p = softmax(lambdas_c, axis=-1)
            #     return p

        else:
            raise NotImplementedError(f"The transition model {self.transition_model} is not implemented.")

        # now generate a's for i in each num_samples
        # do this for all lambdas depending on the value of a

        self.decoder, self.noise_std = get_decoder(self.manifold, self.x_dim, self.z_dim, self.rng_data_gen)
        self.p_z_given_a = p_z_given_a
        self.create_data()

    def __len__(self):
        return self.num_samples

    # def sample_z_given_c(self, c):
    #     mu_tp1, var_tp1 = self.get_mean_var(c)
    #     return self.rng.normal(mu_tp1, np.sqrt(var_tp1))

    # def sample_z_given_c_by_class(self, c):
    #     d = c.shape[0]
    #     mu_tp1, var_tp1 = self.get_mean_var(c)
    #     return self.rng.normal(mu_tp1, np.sqrt(var_tp1), size=(self.n_classes, d, self.z_dim))

    def create_data(self):
        # c = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.z_dim)) # sample actions
        # c = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.n_classes, self.z_dim)) # sample actions

        #### old!
        # sample actions
        # a = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.z_dim)) 

        # # generate continuous z
        # z = self.sample_z_given_c_by_class(c)
        # z = torch.Tensor(z)
        # z = z.permute(1,0,2)

        # # discretize z
        # softmax_z = torch.nn.Softmax(dim=1)
        # self.p_z = softmax_z(z) # categorical probabilities
        # categorical_z = Categorical(self.p_z)
        # disc_z = categorical_z.sample(sample_shape=torch.Size([self.z_dim])) # categorical samples
        # disc_z = disc_z.permute(1,2,0)
        # self.disc_z = disc_z

        ## new
        # sample actions
        # a = self.rng_data_gen.uniform(-2, 2, size=(self.num_samples, self.z_dim))
        # a = np.random.normal(size=1) # in this case, we have only one action per group. we would need another for loop otherwise
        
        a = np.random.choice(self.n_classes, (self.num_samples, self.z_dim))

        prob = self.p_z_given_a()
        p_z = Categorical(torch.Tensor(prob))
        z = p_z.sample(sample_shape=torch.Size([self.num_samples])).double()

        # generate observation
        x = self.decoder(z)

        # # normalize
        # if not self.no_norm:
        #     x = (x - x.mean(0)) / x.std(0)

        # do not add noise
        # x = x + self.noise_std * self.rng.normal(0, 1, size=(self.num_samples, self.x_dim))

        self.x = torch.Tensor(x)
        # self.cont_z = z
        self.a = torch.Tensor(a) # actions


        # uncomment below for discrete data
        ## discrete data
        # softmax_x = torch.nn.Softmax(dim=-2)
        # self.p_x = softmax_x(self.x) # categorical probabilities
        # categorical_x = Categorical(self.p_x)
        # disc_x = categorical_x.sample(sample_shape=torch.Size([self.x_dim])) # categorical samples
        # disc_x = disc_x.permute(1,2,0)
        # self.disc_x = disc_x

        # this is still continuous data:
        # normalize
        # if not self.no_norm:
        #     x = (x - x.mean(0)) / x.std(0)

        # x = x + self.noise_std * self.rng.normal(0, 1, size=(self.num_samples, self.x_dim))

        self.x = torch.Tensor(x)
        self.z = z

        # just return things in the right place
        # self.disc_x = self.x
        # self.p_x = None

        # self.p_z = prob

    def __getitem__(self, item):
        obs = self.x[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        a = self.a[item]
        # disc_a = torch.Tensor(np.array([0.])).long()
        valid = True # mask indicating that the sample should be included in the minibatch. not useful for synthetic data, but down the pipeline
        # cont_z = self.cont_z[item: item + 1]  # must have a dimension for time (of size 1 since no temporal dependencies)
        # prob_z = self.p_z[item: item + 1]
        disc_z = self.z[item: item + 1]
        # prob_x = self.p_x[item: item + 1]
        x = self.x[item: item + 1]
        return obs, a, valid, disc_z, x


def get_ToyManifoldDatasets(manifold, transition_model, split=(0.7, 0.15, 0.15), z_dim=2, x_dim=10, num_samples=1e6,
                            no_norm=False, discrete=True):
    """

    Args:
        manifold (str): 'nn' (only value implemented)
        transition_model (str): "[action/temporal]_sparsity_trivial" or "[action/temporal]_sparsity_non_trivial" or "[action/temporal]_sparsity_non_trivial_no_suff_var" or "[action/temporal]_sparsity_non_trivial_no_graph_crit"
        split (tuple, optional): Proportion of data in training, validation, and test sets. Defaults to (0.7, 0.15, 0.15).
        z_dim (int, optional): latent variable dimension. Defaults to 2.
        x_dim (int, optional): observed variable dimension. Defaults to 10.
        num_samples (int, optional): number of samples. Defaults to 1e6.
        no_norm (bool, optional): no normalization. Defaults to False.

    Returns:
        tuple (length 5):   (tuple) image_shape, 
                            (int) cont_c_dim, 
                            (int) disc_c_dim, 
                            (list) disc_c_n_values, 
                            ([Action/Temporal]ToyManifoldDataset) train_dataset, 
                            ([Action/Temporal]ToyManifoldDataset) valid_dataset, 
                            ([Action/Temporal]ToyManifoldDataset) test_dataset
    """
    if discrete:
        cont_c_dim = 0
        disc_c_dim = 0
        disc_c_n_values = []
        train_dataset = DiscreteActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[0]), seed=1,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
        valid_dataset = DiscreteActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[1]), seed=2,
                                                 x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
        test_dataset = DiscreteActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[2]), seed=3,
                                                x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)

    else:
    
        if transition_model.startswith("action_sparsity"):
            cont_c_dim = z_dim
            disc_c_dim = 0
            disc_c_n_values = []
            train_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[0]), seed=1,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
            valid_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[1]), seed=2,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
            test_dataset = ActionToyManifoldDataset(manifold, transition_model, int(num_samples * split[2]), seed=3,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)

        elif transition_model.startswith("temporal_sparsity"):
            cont_c_dim = 0
            disc_c_dim = 0
            disc_c_n_values = []
            train_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[0]), seed=1,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
            valid_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[1]), seed=2,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)
            test_dataset = TemporalToyManifoldDataset(manifold, transition_model, int(num_samples * split[2]), seed=3,
                                                    x_dim=x_dim, z_dim=z_dim, no_norm=no_norm)

    

    image_shape = (x_dim,)

    return image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset

