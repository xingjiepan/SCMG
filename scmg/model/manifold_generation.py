import os
import json

from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn



def sinusoidal_embedding(n, d):
    '''Create sinusoidal embeddings.'''
    return torch.tensor([[np.sin(i / 10000 ** (2 * j / d)) 
                          if j % 2 == 0 else np.cos(i / 10000 ** (2 * j / d))
                          for j in range(d)] for i in range(n)],
                          dtype=torch.float32)

class ConditionalDiffusionModel(nn.Module):
    '''A denoise diffusion model.'''
    def __init__(self, n_feature, 
                 n_time_feature, 
                 condition_classes,
                 n_condition_feature,
                 n_steps=1000, 
                 min_beta=1e-4, 
                 max_beta=0.02,
                 n_network_blocks=3,
                 ):
        super(ConditionalDiffusionModel, self).__init__()
        self.n_feature = n_feature
        self.n_steps = n_steps
        self.betas =  nn.Parameter(torch.linspace(min_beta, max_beta, n_steps),
                                   requires_grad=False)
        self.alphas = nn.Parameter(1 - self.betas, requires_grad=False)
        self.alpha_bars = nn.Parameter(
                            torch.tensor([torch.prod(self.alphas[:i + 1])
                            for i in range(len(self.alphas))]),
                            requires_grad=False)
        
        # Define the temporal embeddings
        self.time_embeddings = nn.Parameter(
            sinusoidal_embedding(n_steps, n_time_feature), requires_grad=False)
        
        # Define the condition embeddings
        self.condition_classes = list(np.unique(condition_classes))
        self.condition_to_idx = {c: i for i, c in 
                                 enumerate(self.condition_classes)}
        self.condition_embeddings = nn.Parameter(1e-3 * 
            torch.randn(len(self.condition_classes), n_condition_feature),
            requires_grad=True)

        # Define the denoiser
        self.network = MLPDenoiser(n_feature, 
                                   n_time_feature,
                                   n_condition_feature,
                                   n_blocks=n_network_blocks,
                                   )
        
    def forward(self, x0, t, eta=None):
        '''Add noise to the data.
        x0 has shape (n_batch, n_feature) and t has shape (n_batch,).
        '''
        device = next(self.parameters()).device

        n_batch, n_feature = x0.shape
        x0 = x0.to(device)
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n_batch, n_feature).to(device)

        noisy = (a_bar.sqrt().reshape(n_batch, 1) * x0 
                 + (1 - a_bar).sqrt().reshape(n_batch, 1) * eta)

        return noisy
    
    def backward(self, x, t, x_cond):
        '''Predict the mean of distribution at t-1 given the data at t.'''
        t_emb = self.time_embeddings[t.reshape(-1)]
        return self.network(x, t_emb, x_cond)
    
    def generate(self, x_cond):
        '''Generate samples from the model.'''
        device = next(self.parameters()).device

        n_feature = self.n_feature
        n_sample = len(x_cond)

        # Start from random noise
        x = torch.randn(n_sample, n_feature).to(device)

        with torch.no_grad():
            for idx in range(self.n_steps):
                t = self.n_steps - idx - 1
                # Estimate teh noise to remove
                time_tensor = (torch.ones(n_sample,) * t).long().to(device)
                eta_theta = self.backward(x, time_tensor, x_cond)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # Partially denoise the data
                x = (1 / alpha_t.sqrt()) * (x 
                        - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                # Add noise except for the last step
                if t > 0:
                    z = torch.randn(n_sample, n_feature).to(device)

                    # Define the noise level
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Add noise
                    x = x + sigma_t * z

        return x

class RecurrentBlock(nn.Module):
    '''A MLP style recurrent block.'''
    def __init__(self, n_input, n_output, n_hidden):
        super(RecurrentBlock, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.norm = nn.LayerNorm(n_hidden)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
    
class MLPDenoiser(nn.Module):
    '''A simple MLP denoiser.'''
    def __init__(self, n_feature, n_time_feature, n_condition_feature,
                 n_hidden=2048, n_blocks=3):
        super(MLPDenoiser, self).__init__()

        self.fc1 = nn.Linear(n_feature + n_time_feature + n_condition_feature, 
                             n_hidden)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(RecurrentBlock(
                n_feature + n_time_feature + n_condition_feature + n_hidden,
                n_hidden,
                n_hidden))

        self.fc2 = nn.Linear(
            n_feature + n_time_feature + n_condition_feature + n_hidden, 
            n_feature)

    def forward(self, x, t_emb, x_cond):
        x_hidden = self.fc1(torch.cat([x, t_emb, x_cond], dim=1))

        for block in self.blocks:
            x_hidden = block(torch.cat([x, t_emb, x_cond, x_hidden], dim=1))

        x_out = self.fc2(torch.cat([x, t_emb, x_cond, x_hidden], dim=1))
        return x_out
    
def train_diffusion_model(
        model, 
        loader, 
        num_epochs, 
        output_path,
        lr=1e-4,
        loss_history=None,
    ):
    '''Train the diffusion model.'''
    os.makedirs(output_path, exist_ok=True)
    device = next(model.parameters()).device

    if loss_history is None:
        loss_history = {'loss': []}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_loss = float('inf')

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        epoch_loss = 0.0

        for batch in loader:
            # Load data
            cond_classes = batch['cell_type']
            x0 = batch['X_ce_latent'].to(device)
            n = len(x0)

            # Pick a noise and a time step for each data point
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, model.n_steps, (n,)).to(device)

            # Add noise
            noisy_x = model(x0, t, eta)

            # Predict the noise to remove
            eta_theta = model.backward(noisy_x, t, cond_classes)

            # Compute the loss
            loss = mse(eta_theta, eta)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        loss_history['loss'].append(epoch_loss)
        pbar.set_description(f"Loss: {epoch_loss}")

        with open(f'{output_path}/loss_history.json', 'w') as f:
            json.dump(loss_history, f)

        # Store the model if it is the best
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model, f'{output_path}/model.pt')
            torch.save(model.state_dict(), f'{output_path}/best_state_dict.pth')


def generate_cells(model, cond_classes, batch_size=512):
    '''Generate cells from the model.'''
    n_sample = len(cond_classes)
    n_batches = n_sample // batch_size + 1

    samples = []
    with torch.no_grad():
        for idx in tqdm(range(n_batches)):
            start_idx = idx * batch_size
            end_idx = min(n_sample, (idx + 1) * batch_size)

            cond_indices = [model.condition_to_idx[c] for c in cond_classes[start_idx:end_idx]]
            x_cond = model.condition_embeddings[cond_indices]
            batch_cells = model.generate(x_cond)
            samples.append(batch_cells.detach().cpu().numpy())

    return np.concatenate(samples, axis=0)

def generate_transition_cells(model, start_cell_type, end_cell_type, n_cells,
                              batch_size=512):
    '''Generate cells between two cell types.'''
    start_ct_emb = model.condition_embeddings[model.condition_to_idx[start_cell_type]].reshape(1, -1)
    end_ct_emb = model.condition_embeddings[model.condition_to_idx[end_cell_type]].reshape(1, -1)

    emb_weights = torch.linspace(0, 1, n_cells).reshape(-1, 1).to(start_ct_emb.device)
    n_batches = n_cells // batch_size + 1

    samples = []
    with torch.no_grad():
        for idx in tqdm(range(n_batches)):
            start_idx = idx * batch_size
            end_idx = min(n_cells, (idx + 1) * batch_size)

            batch_weights = emb_weights[start_idx:end_idx]
            batch_cells = model.generate(
                batch_weights * end_ct_emb + (1 - batch_weights) * start_ct_emb)
            samples.append(batch_cells.detach().cpu().numpy())

    cond_classes = [start_cell_type] * (n_cells // 2) + [end_cell_type] * (n_cells - (n_cells // 2))

    return np.concatenate(samples, axis=0), cond_classes

