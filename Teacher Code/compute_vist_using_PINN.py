"""
Unified PINN trainer for turbulent viscosity estimation
======================================================

This module merges the functionality of the original ``compute-vist-using-PINN-load.py``
and ``compute-vist-using-PINN-save.py`` scripts.  Both of the original programs
trained a simple physics‑informed neural network (PINN) to estimate the turbulent
eddy viscosity ``ν_t`` from one-dimensional channel flow data.  The ``save``
variant trained the model from scratch and optionally wrote a PyTorch checkpoint,
while the ``load`` variant resumed training from a previously saved checkpoint
and continued to improve the model.  The unified version below provides a
single entry point that can both start from scratch or resume from a saved
checkpoint, controlled via command‑line arguments.

Major improvements over the original scripts
-------------------------------------------

* **Encapsulated logic into functions.**  Data loading, model building,
  training, and plotting are each implemented in separate functions for
  improved readability and maintainability.
* **Command‑line interface.**  You can now specify the data directory,
  checkpoint path, output directory, maximum epochs, learning rate, and
  whether to resume from a checkpoint via ``argparse``.  This removes
  hard‑coded paths such as ``'/Users/cesar/Documents/GitHub/PINNs_Turbulence_Flow/…'``.
* **Device selection.**  The network will run on a GPU if available, falling
  back to CPU otherwise.  The original scripts only used the CPU.
* **Cleaner training loop.**  The training loop now supports resuming from
  arbitrary epochs, gracefully handles the scheduler, and offers early
  stopping once the loss falls below a specified tolerance.  Previously, the
  ``save`` version omitted scheduler updates and the ``load`` version
  redefined the model mid‑loop.
* **Robust checkpointing.**  When training from scratch, you can set
  ``--checkpoint-out`` to save intermediate progress.  When resuming, the script
  validates the checkpoint file before loading and can restore the scheduler
  state if available.
* **Comprehensive plotting.**  The plotting routines are factored into a
  helper that reproduces the diagnostic figures from the original scripts.
  Plots are saved into a user‑specified output directory.
* **Documentation and type hints.**  Each function has a docstring, and
  critical variables are type‑annotated for clarity.  Additional comments
  explain the physics and training objective.

To use this script as a module, run ``python compute_vist_using_PINN.py --help``
to view the available options.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import grad


@dataclass
class TrainingConfig:
    """Configuration parameters for training the PINN."""

    data_path: str  # Directory containing DNS and RANS files
    skip_cells: int = 5  # How many wall‑adjacent cells to skip
    max_epochs: int = 100_000  # Maximum number of training epochs
    learning_rate: float = 0.2  # Adam learning rate
    milestones: Tuple[int, ...] = (6400, 16700, 19200, 37500, 38000, 80000, 94000)
    gamma: float = 0.5  # Multiplicative factor for the LR scheduler
    early_stop_tol: float = 5e-5  # Loss tolerance for early stopping
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dns_data(config: TrainingConfig) -> Tuple[torch.Tensor, ...]:
    """
    Load DNS and RANS profiles from ASCII files and interpolate them onto
    the k‑ω grid.  Returns tensors ready for training.

    Parameters
    ----------
    config : TrainingConfig
        Configuration containing the data directory and skip information.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Tuple containing (x, y_DNS, yplus_DNS, k_DNS, Pk_DNS, diss_DNS,
        d2kdy2_DNS, dkdy_DNS, diff_DNS, diff_DNS_visc, vist_DNS, vist_kom,
        viscos_lam).
    """
    # Viscosity and path definitions
    viscos: float = 1.0 / 5200.0
    path = config.data_path

    # Load mean profiles from DNS
    DNS_mean = np.genfromtxt(os.path.join(path, "LM_Channel_5200_mean_prof.dat"), comments="%")
    y_DNS = DNS_mean[:, 0]
    yplus_DNS = DNS_mean[:, 1]
    u_DNS = DNS_mean[:, 2]
    # Derivative of mean velocity with respect to y+
    dudy_DNS = np.gradient(u_DNS, yplus_DNS)

    # Load fluctuating velocity components from DNS
    DNS_stress = np.genfromtxt(os.path.join(path, "LM_Channel_5200_vel_fluc_prof.dat"), comments="%")
    u2_DNS = DNS_stress[:, 2]
    v2_DNS = DNS_stress[:, 3]
    w2_DNS = DNS_stress[:, 4]
    uv_DNS = DNS_stress[:, 5]

    # Compute turbulent kinetic energy k and its derivatives on the DNS grid
    k_DNS_raw = 0.5 * (u2_DNS + v2_DNS + w2_DNS)
    dkdy_DNS_raw = np.gradient(k_DNS_raw, yplus_DNS, edge_order=2)
    d2kdy2_DNS_raw = np.gradient(dkdy_DNS_raw, yplus_DNS, edge_order=2)

    # Load k‑ω solution (RANS) to provide the grid and k/ω values
    kom_data = np.loadtxt(os.path.join(path, "y_u_k_om_uv_5200-RANS-half-channel.txt"))
    y_kom = kom_data[:, 0]
    k_kom = kom_data[:, 2]
    om_kom = kom_data[:, 3]
    vist_kom_raw = k_kom / om_kom / viscos

    # Optionally skip near‑wall cells to avoid numerical noise
    j = config.skip_cells
    y_kom = y_kom[j:]
    vist_kom_raw = vist_kom_raw[j:]

    # Number of points after skipping
    nj = len(y_kom)
    viscos_lam = np.ones(nj)

    # Interpolate DNS quantities onto the k‑ω grid
    k_DNS = np.interp(y_kom, y_DNS, k_DNS_raw)
    # Load k‑equation terms once, then extract columns
    RSTE = np.genfromtxt(
        os.path.join(path, "LM_Channel_5200_RSTE_k_prof.dat"), comments="%"
    )
    Pk_DNS = np.interp(y_kom, y_DNS, RSTE[:, 2])
    diss_DNS = np.interp(y_kom, y_DNS, RSTE[:, 7])
    diff_DNS = np.interp(y_kom, y_DNS, RSTE[:, 3])
    diff_DNS_visc = np.interp(y_kom, y_DNS, RSTE[:, 4])
    vist_DNS_raw = np.abs(uv_DNS / dudy_DNS)
    vist_DNS = np.interp(y_kom, y_DNS, vist_DNS_raw)
    d2kdy2_DNS = np.interp(y_kom, y_DNS, d2kdy2_DNS_raw)
    dkdy_DNS = np.interp(y_kom, y_DNS, dkdy_DNS_raw)
    yplus_interp = np.interp(y_kom, y_DNS, yplus_DNS)

    # Convert everything to tensors on the chosen device
    def to_tensor(arr: np.ndarray, req_grad: bool = False) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=config.device, requires_grad=req_grad).view(-1, 1)

    k_DNS_t = to_tensor(k_DNS)
    Pk_DNS_t = to_tensor(Pk_DNS)
    diss_DNS_t = to_tensor(diss_DNS)
    d2kdy2_DNS_t = to_tensor(d2kdy2_DNS)
    dkdy_DNS_t = to_tensor(dkdy_DNS)
    diff_DNS_t = to_tensor(diff_DNS)
    diff_DNS_visc_t = to_tensor(diff_DNS_visc)  # note: used as numpy below but storing tensor for consistency
    vist_DNS_t = to_tensor(vist_DNS)
    viscos_lam_t = to_tensor(viscos_lam)
    yplus_DNS_t = to_tensor(yplus_interp, req_grad=True)
    y_DNS_t = to_tensor(y_kom)
    vist_kom_t = to_tensor(vist_kom_raw)

    # Create input (yplus) tensor
    x = yplus_DNS_t

    return (
        x,
        y_DNS_t,
        yplus_DNS_t,
        k_DNS_t,
        Pk_DNS_t,
        diss_DNS_t,
        d2kdy2_DNS_t,
        dkdy_DNS_t,
        diff_DNS_t,
        diff_DNS_visc_t,
        vist_DNS_t,
        vist_kom_t,
        viscos_lam_t,
    )


class PINN(nn.Module):
    """Simple fully connected neural network used to represent the eddy viscosity field."""

    def __init__(self, hidden: int = 10) -> None:
        super().__init__()
        # Use three hidden layers with tanh activation.  Larger networks may
        # capture more complex relationships but require more training data.
        self.ll1 = nn.Linear(1, hidden)
        self.act = nn.Tanh()
        self.ll2 = nn.Linear(hidden, hidden)
        self.ll3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.ll1(x))
        z = self.act(self.ll2(z))
        z = self.act(self.ll3(z))
        return self.out(z)


def get_derivative(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute df/dx using automatic differentiation."""
    return grad(
        f,
        x,
        torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


def compute_losses(
    model: PINN,
    x: torch.Tensor,
    d2kdy2: torch.Tensor,
    dkdy: torch.Tensor,
    Pk: torch.Tensor,
    diss: torch.Tensor,
    viscos_lam: torch.Tensor,
    vist_DNS: torch.Tensor,
    vist_0: torch.Tensor,
    vist_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the differential equation and boundary losses for the PINN.

    The differential equation is derived from the transport equation for k,
    assuming that the diffusive term contains both a laminar (ν) and
    turbulent (ν_t) contribution.  The boundary conditions enforce that the
    predicted ν_t matches the DNS values at both ends of the domain.

    Parameters
    ----------
    model : PINN
        Neural network representing ν_t/ν.
    x : torch.Tensor
        Input coordinate y+; requires_grad must be True.
    d2kdy2 : torch.Tensor
        Second derivative of k with respect to y+.
    dkdy : torch.Tensor
        First derivative of k with respect to y+.
    Pk : torch.Tensor
        Production term from DNS.
    diss : torch.Tensor
        Dissipation term from DNS.
    viscos_lam : torch.Tensor
        Laminar viscosity (constant ones).
    vist_DNS : torch.Tensor
        Turbulent viscosity from DNS (used for initialization and reference).
    vist_0 : torch.Tensor
        DNS ν_t value at the wall (first entry).
    vist_1 : torch.Tensor
        DNS ν_t value at the far wall (last entry).

    Returns
    -------
    Tuple[torch.Tensor, ...]
        differential_equation_loss, boundary_condition_loss, total_loss, imbalance
    """
    vist_pred = model(x)
    dvist_dy = get_derivative(vist_pred, x)
    # term from the conservative formulation of the diffusion
    diff_term = (vist_pred + viscos_lam) * d2kdy2 + dkdy * dvist_dy
    # The imbalance should vanish when the PDE is satisfied
    imbalance = diff_term + (Pk - diss)
    de_loss = torch.sum(imbalance ** 2)
    bc_loss = (vist_pred[0] - vist_0) ** 2 + (vist_pred[-1] - vist_1) ** 2
    total_loss = de_loss + 1000.0 * bc_loss
    return de_loss, bc_loss, total_loss, imbalance.detach()


def plot_results(
    output_dir: str,
    epoch_range: np.ndarray,
    de_history: np.ndarray,
    bc_history: np.ndarray,
    x: torch.Tensor,
    y_DNS: torch.Tensor,
    yplus_DNS: torch.Tensor,
    dkdy_DNS: torch.Tensor,
    d2kdy2_DNS: torch.Tensor,
    diff_DNS: torch.Tensor,
    diff_DNS_visc: torch.Tensor,
    Pk_DNS: torch.Tensor,
    diss_DNS: torch.Tensor,
    vist_DNS: torch.Tensor,
    vist_pred: torch.Tensor,
    vist_kom: torch.Tensor,
    viscos: float = 1.0 / 5200.0,
) -> None:
    """
    Reproduce diagnostic plots from the original scripts.

    Parameters
    ----------
    output_dir : str
        Directory where PNG files will be saved.
    epoch_range : np.ndarray
        Array of epoch indices for plotting training history.
    de_history : np.ndarray
        Differential equation loss history.
    bc_history : np.ndarray
        Boundary condition loss history.
    x : torch.Tensor
        Input positions in y+.
    y_DNS : torch.Tensor
        Physical y coordinate (same size as x, but measured in units of δ).  Only
        used to convert to y+ = y/ν.
    yplus_DNS : torch.Tensor
        y+ positions corresponding to DNS grid.
    dkdy_DNS : torch.Tensor
        First derivative of k for diffusion plots.
    d2kdy2_DNS : torch.Tensor
        Second derivative of k for diffusion plots.
    diff_DNS : torch.Tensor
        Diffusive term from DNS.
    diff_DNS_visc : torch.Tensor
        Viscous diffusion term from DNS.
    Pk_DNS : torch.Tensor
        Production term from DNS.
    diss_DNS : torch.Tensor
        Dissipation term from DNS.
    vist_DNS : torch.Tensor
        Turbulent viscosity from DNS.
    vist_pred : torch.Tensor
        Predicted turbulent viscosity from the PINN.
    vist_kom : torch.Tensor
        Turbulent viscosity from the k‑ω model.
    viscos : float
        The inverse Reynolds number used to convert y to y+.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy for plotting
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().flatten()

    yplus = to_np(x)
    vist_dns_np = to_np(vist_DNS)
    vist_pred_np = to_np(vist_pred)
    vist_kom_np = to_np(vist_kom)
    de_hist = de_history
    bc_hist = bc_history

    # 1. Plot training losses
    fig, ax = plt.subplots()
    ax.semilogy(epoch_range, bc_hist, 'r-', label='BC error')
    ax.semilogy(epoch_range, de_hist, 'b-', label='PDE error')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training errors')
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'loss_history.png'), bbox_inches='tight')
    plt.close(fig)

    # 2. ν_t comparison (full range)
    fig, ax = plt.subplots()
    ax.plot(yplus, vist_dns_np, 'r:', linewidth=5, label='DNS')
    ax.plot(yplus, vist_pred_np, 'k-', linewidth=2, label=r'$\nu_t^{\mathrm{pred}}$')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$\nu_t/\nu$')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'vist_comparison.png'), bbox_inches='tight')
    plt.close(fig)

    # 3. ν_t comparison with k‑ω and zoomed view
    fig, ax = plt.subplots()
    ax.plot(yplus, vist_dns_np, 'r:', linewidth=5, label='DNS')
    ax.plot(yplus, vist_pred_np, 'k-', linewidth=2, label=r'$\nu_t^{\mathrm{pred}}$')
    # Convert physical y to y+ for k‑ω (y_DNS is physical coordinate δ; y+ = y/ν)
    ax.plot(to_np(y_DNS) / viscos, vist_kom_np, 'g-', linewidth=2, label=r'$\nu_{t, k-\omega}$')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel(r'$\nu_t/\nu$')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'vist_comparison_zoom.png'), bbox_inches='tight')
    plt.close(fig)

    # 4. Diffusion term comparison
    fig, ax = plt.subplots()
    dkdy_np = to_np(dkdy_DNS)
    d2kdy2_np = to_np(d2kdy2_DNS)
    y_phys = to_np(y_DNS)
    diff_dns_np = to_np(diff_DNS)
    diff_visc_np = to_np(diff_DNS_visc)
    # Compute predicted diffusion: conservative and non‑conservative forms
    term = dkdy_np * vist_pred_np
    # Finite difference gradient for the conservative term
    diff_pred_np = np.gradient(term, y_phys)
    # Non‑conservative formulation using automatic differentiation
    vist_pred_t = vist_pred.detach()
    dvist_dy_t = get_derivative(vist_pred_t, x).detach()
    diff_non_cons = vist_pred_np * d2kdy2_np + dkdy_np * to_np(dvist_dy_t)
    ax.plot(y_phys / viscos, diff_pred_np + diff_visc_np, 'k-', linewidth=2, label='Predicted (cons)')
    ax.plot(y_phys / viscos, diff_dns_np + diff_visc_np, 'b-', linewidth=2, label='DNS')
    ax.plot(y_phys / viscos, diff_non_cons, 'r-', linewidth=2, label='Predicted (non‑cons)')
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel('Diffusion')
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'diffusion_comparison.png'), bbox_inches='tight')
    plt.close(fig)

    # 5. Diffusion zoomed in
    fig, ax = plt.subplots()
    ax.plot(y_phys / viscos, diff_dns_np, 'r--', linewidth=2, label='DNS')
    ax.plot(y_phys / viscos, diff_non_cons, 'b-', linewidth=2, label='PINN')
    ax.plot(y_phys / viscos, diff_non_cons, 'bo', markersize=2)
    ax.set_xlabel(r'$y^+$')
    ax.set_ylabel('Diffusion')
    ax.set_xlim(0, 100)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'diffusion_zoom.png'), bbox_inches='tight')
    plt.close(fig)

    # 6. k‑equation balance (imbalance, Pk, dissipation, diffusive terms)
    fig, ax = plt.subplots()
    # Recompute the non‑conservative diffusion term
    vist_pred_t = vist_pred.detach()
    dvist_dy_t = get_derivative(vist_pred_t, x).detach()
    diff_non_cons = vist_pred_np * d2kdy2_np + dkdy_np * to_np(dvist_dy_t)
    # Compute imbalance = diffusion + Pk - diss
    Pk_np = to_np(Pk_DNS)
    diss_np = to_np(diss_DNS)
    imbalance_np = diff_non_cons + Pk_np - diss_np
    ax.plot(yplus, imbalance_np, 'r:', linewidth=5, label='Imbalance (PINN)')
    ax.plot(yplus, Pk_np, 'k-', linewidth=2, label=r'$P_{k, DNS}$')
    ax.plot(yplus, -diss_np, 'b-', linewidth=2, label=r'$\varepsilon_{DNS}$')
    ax.plot(yplus, diff_visc_np, 'r--', linewidth=2, label=r'$D^\nu_{DNS}$')
    ax.plot(yplus, diff_dns_np, 'b--', linewidth=2, label=r'$D^t_{DNS}$')
    ax.set_xlabel(r'$y^+$')
    ax.set_xlim(0, 100)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'k_balance_zoom.png'), bbox_inches='tight')
    plt.close(fig)


def train_pinn(
    config: TrainingConfig,
    checkpoint_in: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
    output_dir: str = ".",
) -> Dict[str, torch.Tensor]:
    """
    Train or resume training of the PINN for turbulent viscosity estimation.

    Parameters
    ----------
    config : TrainingConfig
        Hyperparameter and path configuration for training.
    checkpoint_in : Optional[str], default ``None``
        Path to a checkpoint file from which training should be resumed.  If
        ``None``, training starts from randomly initialized weights.
    checkpoint_out : Optional[str], default ``None``
        Path to which a checkpoint will be saved after the first epoch.  This
        mimics the behaviour of the original ``save`` script, which wrote a
        checkpoint early in training.  When ``None``, no checkpoint is saved.
    output_dir : str, default ``'.'``
        Directory where diagnostic figures and predicted ν_t values will be
        written.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing the trained model and selected data arrays.
    """
    # Load data
    (
        x,
        y_DNS,
        yplus_DNS,
        k_DNS,
        Pk_DNS,
        diss_DNS,
        d2kdy2_DNS,
        dkdy_DNS,
        diff_DNS,
        diff_DNS_visc,
        vist_DNS,
        vist_kom,
        viscos_lam,
    ) = load_dns_data(config)

    model = PINN().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones), gamma=config.gamma)

    start_epoch = 0
    # If resuming, load checkpoint
    if checkpoint_in:
        if not os.path.exists(checkpoint_in):
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_in}' not found.")
        ckpt = torch.load(checkpoint_in, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Restore scheduler state if available in checkpoint
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # When resuming training, ensure that x requires gradient
        x = x.detach().clone().requires_grad_(True)
        print(f"Resuming training from epoch {start_epoch}.")

    # Storage for loss history
    de_history = np.zeros(config.max_epochs)
    bc_history = np.zeros(config.max_epochs)

    vist_0 = vist_DNS[0]
    vist_1 = vist_DNS[-1]

    loss_min = np.inf
    total_epochs = config.max_epochs
    for epoch in range(start_epoch, total_epochs):
        optimizer.zero_grad()
        de_loss, bc_loss, total_loss, _ = compute_losses(
            model,
            x,
            d2kdy2_DNS,
            dkdy_DNS,
            Pk_DNS,
            diss_DNS,
            viscos_lam,
            vist_DNS,
            vist_0,
            vist_1,
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record losses
        de_history[epoch] = de_loss.item()
        bc_history[epoch] = bc_loss.item()
        loss_np = total_loss.item()
        loss_min = min(loss_np, loss_min)

        # Print progress every few thousand epochs to avoid flooding stdout
        if (epoch - start_epoch) % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}, LR: {current_lr:.4e}, loss: {loss_np:.4e}, best: {loss_min:.4e}")

        # Save a checkpoint after the first epoch if requested
        if checkpoint_out and epoch == 1:
            ckpt_out = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_np,
            }
            torch.save(ckpt_out, checkpoint_out)
            print(f"Checkpoint saved to {checkpoint_out}")

        # Early stopping
        if loss_np < config.early_stop_tol:
            print(f"Early stopping at epoch {epoch+1} with loss {loss_np:.4e} < {config.early_stop_tol:.4e}")
            break

    # Compute final predictions and produce plots
    with torch.no_grad():
        vist_pred = model(x).cpu()

    # Determine actual number of epochs used for history arrays
    epoch_indices = np.arange(start_epoch, epoch + 1)
    plot_results(
        output_dir,
        epoch_indices,
        de_history[epoch_indices],
        bc_history[epoch_indices],
        x.cpu(),
        y_DNS.cpu(),
        yplus_DNS.cpu(),
        dkdy_DNS.cpu(),
        d2kdy2_DNS.cpu(),
        diff_DNS.cpu(),
        diff_DNS_visc.cpu(),
        Pk_DNS.cpu(),
        diss_DNS.cpu(),
        vist_DNS.cpu(),
        vist_pred,
        vist_kom.cpu(),
    )

    # Save predicted ν_t to a text file for post‑processing
    np.savetxt(os.path.join(output_dir, 'vist_predicted.txt'), vist_pred.numpy().flatten())

    return {
        'model': model,
        'x': x,
        'vist_pred': vist_pred,
        'vist_dns': vist_DNS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a physics‑informed neural network to predict turbulent viscosity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='.',
        help='Directory containing DNS and RANS data files',
    )
    parser.add_argument(
        '--skip-cells',
        type=int,
        default=5,
        help='Number of near‑wall cells to skip when interpolating onto the k‑ω grid',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100000,
        help='Maximum number of training epochs',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.2,
        help='Learning rate for the Adam optimizer',
    )
    parser.add_argument(
        '--milestones',
        type=int,
        nargs='*',
        default=[6400, 16700, 19200, 37500, 38000, 80000, 94000],
        help='Epochs at which to decay the learning rate',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.5,
        help='Multiplicative factor for the learning rate decay',
    )
    parser.add_argument(
        '--early-stop-tol',
        type=float,
        default=5e-5,
        help='Stop training when the total loss drops below this threshold',
    )
    parser.add_argument(
        '--checkpoint-in',
        type=str,
        default=None,
        help='Path to a checkpoint file from which to resume training',
    )
    parser.add_argument(
        '--checkpoint-out',
        type=str,
        default=None,
        help='Path to save a checkpoint after the first epoch (optional)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory where plots and the predicted ν_t will be saved',
    )
    args = parser.parse_args()

    config = TrainingConfig(
        data_path=args.data_path,
        skip_cells=args.skip_cells,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        milestones=tuple(args.milestones),
        gamma=args.gamma,
        early_stop_tol=args.early_stop_tol,
    )

    train_pinn(
        config=config,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()