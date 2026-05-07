"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.bias.data *= scale

    def normalize_decoder(self):
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = t.norm(self.decoder.weight, dim=0)
        assert t.allclose(new_norms, t.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)


    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None, dtype=None, device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim
        self.device = device
        self.dtype = dtype

    def encode(self, x):
        if self.device is not None:
            x = x.to(self.device)
        if self.dtype is not None:
            x = x.to(self.dtype)
        return x

    def decode(self, f):
        if self.device is not None:
            f = f.to(self.device)
        if self.dtype is not None:
            f = f.to(self.dtype)
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if self.device is not None:
            x = x.to(self.device)
        if self.dtype is not None:
            x = x.to(self.dtype)
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, activation_dim, path, dtype=None, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(activation_dim, device=device, dtype=dtype)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(self, activation_dim, dict_size, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def encode(self, x: t.Tensor, return_gate:bool=False, normalize_decoder:bool=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f: t.Tensor, normalize_decoder:bool=False):
        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x:t.Tensor, output_features:bool=False, normalize_decoder:bool=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        if normalize_decoder:
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.decoder_bias.data *= scale
        self.mag_bias.data *= scale
        self.gate_bias.data *= scale

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(t.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim, device=device))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(t.ones(dict_size, device=device) * 0.001)  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = t.load(path)
            activation_dim, dict_size = state_dict["W_enc"].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)


# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:  # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class LinearIDOL(Dictionary, nn.Module):
    """
    Linear IDOL (Independent Dynamics Of Latents) - a temporal-instantaneous SAE.

    Learns latent dynamics with optional temporal (lagged) and instantaneous causal structure.

    mode:
        'temporal'      -- only lagged transition matrices B_1, ..., B_tau.
        'instantaneous' -- only instantaneous mixing matrix M.
        'both'          -- temporal + instantaneous (default).

    encode/decode operate on single-step activations [batch, activation_dim].
    forward expects windowed sequences [batch, activation_dim, tau+1] and
    returns six scalar loss terms used for training.
    """

    VALID_MODES = ('temporal', 'instantaneous', 'both')

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        tau: int = 20,
        w: float = 0.5,
        noise_mode: str = 'lap',
        topk_sparsity: int = 100,
        mode: str = 'both',
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}.")

        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.tau = tau
        self.w = w
        self.noise_mode = noise_mode
        self.topk_sparsity = topk_sparsity
        self.mode = mode

        self.register_buffer('_tau', t.tensor(tau, dtype=t.long))

        self.F_enc = nn.Parameter(t.ones(activation_dim, dict_size), requires_grad=True)
        self.F_dec = nn.Parameter(t.ones(dict_size, activation_dim), requires_grad=True)

        self.Bs = nn.ParameterList([
            nn.Parameter(t.zeros(dict_size, dict_size), requires_grad=True)
            for _ in range(tau)
        ] if self._uses_temporal() else [])

        self.M = nn.Parameter(
            t.ones(dict_size, dict_size),
            requires_grad=self._uses_instantaneous(),
        )

        self._init_params()

    def scale_biases(self, scale: float):
        pass  # IDOL has no bias terms that require activation-scale normalization

    def _uses_temporal(self) -> bool:
        return self.mode in ('temporal', 'both')

    def _uses_instantaneous(self) -> bool:
        return self.mode in ('instantaneous', 'both')

    def _init_params(self):
        init.xavier_normal_(self.F_enc.data)
        init.xavier_normal_(self.F_dec.data)
        if self._uses_instantaneous():
            init.xavier_normal_(self.M.data)

    def encode(self, x: t.Tensor) -> t.Tensor:
        """x: [batch, activation_dim] -> [batch, dict_size]."""
        return t.einsum('hd,bd->bh', self.F_enc, x)

    def decode(self, f: t.Tensor) -> t.Tensor:
        """f: [batch, dict_size] -> [batch, activation_dim]."""
        return t.einsum('dh,bh->bd', self.F_dec, f)

    def _encode_window(self, Xp: t.Tensor):
        Zp = t.einsum('hd,bdt->bht', self.F_enc.T, Xp)
        recons_Xp = t.einsum('dh,bht->bdt', self.F_dec.T, Zp)
        loss_mse_Xt = t.nn.functional.mse_loss(recons_Xp[:, :, -1], Xp[:, :, -1])
        return Zp, loss_mse_Xt

    def _temporal_contribution(self, Zp, w, device, dtype):
        B = Zp.shape[0]
        if not self._uses_temporal():
            return (t.zeros(B, self.dict_size, device=device, dtype=dtype),
                    t.zeros((), device=device, dtype=dtype))
        Zt_temp = t.zeros(B, self.dict_size, device=device, dtype=dtype)
        loss_sparse_Bs = t.zeros((), device=device, dtype=dtype)
        for lag in range(1, self.tau + 1):
            B_lag = self.Bs[lag - 1]
            loss_sparse_Bs = loss_sparse_Bs + t.nn.functional.l1_loss(B_lag, t.zeros_like(B_lag))
            Zt_temp = Zt_temp + w * t.einsum('hd,bd->bh', B_lag, Zp[:, :, self.tau - lag])
        return Zt_temp, loss_sparse_Bs

    def _instantaneous_contribution(self, Zp, w, device, dtype):
        if not self._uses_instantaneous():
            B = Zp.shape[0]
            return (t.zeros(B, self.dict_size, device=device, dtype=dtype),
                    t.zeros((), device=device, dtype=dtype))
        M_used = t.tril(self.M, diagonal=1)
        Zt_inst = w * t.einsum('hd,bd->bh', M_used, Zp[:, :, self.tau])
        loss_sparse_M = t.nn.functional.l1_loss(M_used, t.zeros_like(M_used))
        return Zt_inst, loss_sparse_M

    def _apply_topk(self, Zt, topk):
        if topk <= 0:
            return Zt
        _, topk_indices = t.topk(t.abs(Zt), topk, dim=1)
        mask = t.zeros_like(Zt)
        mask.scatter_(1, topk_indices, 1.0)
        return Zt * mask

    def _independence_loss(self, Et):
        if self.noise_mode == 'gau':
            return t.trace(t.cov(Et))
        elif self.noise_mode == 'lap':
            return t.nn.functional.l1_loss(Et, t.zeros_like(Et))
        raise NotImplementedError(f"noise_mode={self.noise_mode!r} not supported.")

    def forward(self, Xp: t.Tensor):
        """
        Xp: [batch, activation_dim, tau+1]
        Returns: (loss_mse_Xt, loss_mse_Zt, loss_indep,
                  loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt)
        """
        Xp = Xp.to(self.F_enc.dtype)
        device, dtype = Xp.device, Xp.dtype
        topk = self.topk_sparsity if self.training else 0

        Zp, loss_mse_Xt = self._encode_window(Xp)
        Zt_temp, loss_sparse_Bs = self._temporal_contribution(Zp, 1.0, device, dtype)
        Zt_inst, loss_sparse_M  = self._instantaneous_contribution(Zp, 1.0, device, dtype)

        Zt = self._apply_topk(Zt_temp + Zt_inst, topk)

        loss_mse_Zt  = t.nn.functional.mse_loss(Zt, Zp[:, :, self.tau])
        Et           = Zp[:, :, self.tau] - Zt
        loss_indep   = self._independence_loss(Et)
        loss_sparse_Zt = t.nn.functional.l1_loss(Zt, t.zeros_like(Zt))

        return (loss_mse_Xt, loss_mse_Zt, loss_indep,
                loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt)

    @classmethod
    def from_pretrained(cls, path: str, device=None, **kwargs) -> "LinearIDOL":
        """
        Load from a state dict file. activation_dim, dict_size, tau are inferred
        from the weights; pass w/noise_mode/topk_sparsity/mode as kwargs.
        """
        state_dict = t.load(path, map_location='cpu')
        activation_dim, dict_size = state_dict['F_enc'].shape
        # '_tau' buffer is authoritative; fall back to counting Bs keys for old checkpoints
        if '_tau' in state_dict:
            tau = state_dict['_tau'].item()
        else:
            tau = sum(1 for k in state_dict if k.startswith('Bs.'))
        model = cls(activation_dim=activation_dim, dict_size=dict_size, tau=tau, **kwargs)
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model