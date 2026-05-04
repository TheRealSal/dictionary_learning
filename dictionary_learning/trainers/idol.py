"""
Implements the LinearIDOL trainer for temporal + instantaneous SAE training.
"""
import torch as t
from typing import Optional
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule
from ..dictionary import LinearIDOL


def gen_window_slicing_batch(batch: t.Tensor, window_size: int, stride: int = 1) -> t.Tensor:
    """
    Slide a window over sequential activations.
    batch: [seq_len, activation_dim] -> [n_windows, activation_dim, window_size]
    """
    seq_len = batch.shape[0]
    windows = [
        batch[i : i + window_size, :].T
        for i in range(0, seq_len - window_size + 1, stride)
    ]
    if not windows:
        return t.empty(0, batch.shape[1], window_size, device=batch.device, dtype=batch.dtype)
    return t.stack(windows, dim=0)


class LinearIDOLTrainer(SAETrainer):
    """
    Trainer for LinearIDOL - learns temporal and instantaneous latent dynamics.

    update() expects sequential activation batches [seq_len, activation_dim];
    window slicing is applied internally to form [n_windows, activation_dim, tau+1].

    Loss weights:
        l_mse_Zt : Z_t reconstruction loss (0 = off by default)
        l_ind    : independence / innovation loss
        l_spB    : L1 on temporal matrices B_1..B_tau
        l_spM    : L1 on instantaneous matrix M
        l_spZ    : L1 on latent Z_t
    """

    def __init__(
        self,
        steps: int,
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        tau: int = 20,
        w: float = 0.5,
        noise_mode: str = 'lap',
        topk_sparsity: int = 0,
        mode: str = 'both',
        lr: float = 1e-2,
        wd: float = 1e-4,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        l_mse_Zt: float = 0.0,
        l_ind: float = 0.1,
        l_spB: float = 0.01,
        l_spM: float = 0.01,
        l_spZ: float = 0.01,
        seed: Optional[int] = None,
        device=None,
        wandb_name: Optional[str] = 'LinearIDOLTrainer',
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.tau = tau
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        self.l_mse_Zt = l_mse_Zt
        self.l_ind    = l_ind
        self.l_spB    = l_spB if mode in ('temporal', 'both') else 0.0
        self.l_spM    = l_spM if mode in ('instantaneous', 'both') else 0.0
        self.l_spZ    = l_spZ

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.device = device or ('cuda' if t.cuda.is_available() else 'cpu')

        self.ae = LinearIDOL(
            activation_dim=activation_dim,
            dict_size=dict_size,
            tau=tau,
            w=w,
            noise_mode=noise_mode,
            topk_sparsity=topk_sparsity,
            mode=mode,
        ).to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr, weight_decay=wd)
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.logging_parameters = [
            'loss_total', 'loss_mse_Xt', 'loss_mse_Zt',
            'loss_indep', 'loss_sparse_Bs', 'loss_sparse_M', 'loss_sparse_Zt',
        ]
        self.loss_total = self.loss_mse_Xt = self.loss_mse_Zt = 0.0
        self.loss_indep = self.loss_sparse_Bs = self.loss_sparse_M = self.loss_sparse_Zt = 0.0

    def _compute_loss(self, Xp: t.Tensor):
        (loss_mse_Xt, loss_mse_Zt, loss_indep,
         loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt) = self.ae(Xp)
        loss = (
            loss_mse_Xt
            + self.l_mse_Zt * loss_mse_Zt
            + self.l_ind    * loss_indep
            + self.l_spB    * loss_sparse_Bs
            + self.l_spM    * loss_sparse_M
            + self.l_spZ    * loss_sparse_Zt
        )
        return loss, loss_mse_Xt, loss_mse_Zt, loss_indep, loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt

    def loss(self, x: t.Tensor, step: int = 0, logging: bool = False, **kwargs):
        """x: [seq_len, activation_dim] or pre-windowed [batch, activation_dim, tau+1]."""
        x = x.to(self.device)
        Xp = gen_window_slicing_batch(x, window_size=self.tau + 1) if x.dim() == 2 else x
        loss, *terms = self._compute_loss(Xp)

        if not logging:
            return loss

        loss_mse_Xt, loss_mse_Zt, loss_indep, loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt = terms
        with t.no_grad():
            Xt, f = Xp[:, :, -1], self.ae.encode(Xp[:, :, -1])
        return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
            Xt, self.ae.decode(f), f,
            {
                'loss_total':     loss.item(),
                'loss_mse_Xt':    loss_mse_Xt.item(),
                'loss_mse_Zt':    loss_mse_Zt.item(),
                'loss_indep':     loss_indep.item(),
                'loss_sparse_Bs': loss_sparse_Bs.item(),
                'loss_sparse_M':  loss_sparse_M.item(),
                'loss_sparse_Zt': loss_sparse_Zt.item(),
            },
        )

    def update(self, step: int, activations: t.Tensor):
        """activations: [seq_len, activation_dim] sequential batch from the buffer."""
        Xp = gen_window_slicing_batch(activations.to(self.device), window_size=self.tau + 1)
        if Xp.shape[0] == 0:
            return

        self.ae.train()
        self.optimizer.zero_grad()
        loss, v0, v1, v2, v3, v4, v5 = self._compute_loss(Xp)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        (self.loss_total, self.loss_mse_Xt, self.loss_mse_Zt,
         self.loss_indep, self.loss_sparse_Bs, self.loss_sparse_M, self.loss_sparse_Zt) = (
            loss.item(), v0.item(), v1.item(), v2.item(), v3.item(), v4.item(), v5.item()
        )

    @property
    def config(self):
        ae = self.ae
        return {
            'dict_class':     'LinearIDOL',
            'trainer_class':  'LinearIDOLTrainer',
            'activation_dim': ae.activation_dim,
            'dict_size':      ae.dict_size,
            'tau':            ae.tau,
            'w':              ae.w,
            'noise_mode':     ae.noise_mode,
            'topk_sparsity':  ae.topk_sparsity,
            'mode':           ae.mode,
            'l_mse_Zt':       self.l_mse_Zt,
            'l_ind':          self.l_ind,
            'l_spB':          self.l_spB,
            'l_spM':          self.l_spM,
            'l_spZ':          self.l_spZ,
            'steps':          self.steps,
            'warmup_steps':   self.warmup_steps,
            'decay_start':    self.decay_start,
            'seed':           self.seed,
            'device':         self.device,
            'layer':          self.layer,
            'lm_name':        self.lm_name,
            'wandb_name':     self.wandb_name,
            'submodule_name': self.submodule_name,
        }