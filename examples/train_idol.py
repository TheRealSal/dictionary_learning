import torch
import argparse
from nnsight import LanguageModel

import dictionary_learning.utils as utils
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.idol import LinearIDOLTrainer


def build_run_name(args):
    def fmt_M(n):
        return '{:g}M'.format(n / 1_000_000)
    parts = [
        f'mode={args.mode}', f'tau={args.tau}', f'z={args.z_dim}',
        f'topk={args.topk}', args.noise_mode, fmt_M(args.total_tokens_int),
        f'seed={args.seed}',
    ]
    if args.mse_Zt:
        parts.append('mseZt')
    if args.normalize_activations:
        parts.append('norm')
    return '_'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',    default=123,   type=int)
    parser.add_argument('--lr',      default=0.01,  type=float)
    parser.add_argument('--wd',      default=1e-4,  type=float)
    parser.add_argument('--z-dim',   default=3072,  type=int)
    parser.add_argument('--tau',     default=20,    type=int)
    parser.add_argument('--w',       default=0.5,   type=float)
    parser.add_argument('--noise-mode', default='lap', type=str, choices=['gau', 'lap'])
    parser.add_argument('--mse-Zt',  default=False, action='store_true')
    parser.add_argument('--l-ind',   default=0.1,   type=float)
    parser.add_argument('--l-spB',   default=0.01,  type=float)
    parser.add_argument('--l-spM',   default=0.01,  type=float)
    parser.add_argument('--l-spZ',   default=0.01,  type=float)
    parser.add_argument('--normalize-activations', default=False, action='store_true')
    parser.add_argument('--topk',    default=100,   type=int, choices=[0, 25, 50, 100])
    parser.add_argument('--mode',    default='both', type=str,
                        choices=['temporal', 'instantaneous', 'both'])
    parser.add_argument('--results-dir',  required=True)
    parser.add_argument('--run-name',     default=None, type=str)
    parser.add_argument('--hgf-token',    default='',   type=str)
    parser.add_argument('--total-tokens', default='50M', type=str)
    parser.add_argument('--model-name',   default='EleutherAI/pythia-160m-deduped', type=str)
    parser.add_argument('--layer',        default=8,    type=int)
    parser.add_argument('--out-batch-ratio', default=0.1, type=float)
    parser.add_argument('--buffer-size',  default='0.1M', type=str)
    parser.add_argument('--text',         default='monology/pile-uncopyrighted', type=str)
    parser.add_argument('--wandb-project', default='coev-linearidol', type=str)
    parser.add_argument('--wandb-entity',  default=None, type=str)
    parser.add_argument('--wandb-mode',    default='online', type=str,
                        choices=['online', 'offline', 'disabled'])
    args = parser.parse_args()

    assert args.total_tokens[-1].lower() == 'm'
    assert args.buffer_size[-1].lower()  == 'm'
    args.total_tokens_int = int(float(args.total_tokens[:-1]) * 1_000_000)
    args.buffer_size_int  = int(float(args.buffer_size[:-1])  * 1_000_000)
    args.run_name = args.run_name or build_run_name(args)

    if args.hgf_token:
        from huggingface_hub import login
        login(args.hgf_token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LanguageModel(args.model_name, dispatch=True, device_map=device)
    model = model.to(dtype=torch.float32)
    submodule   = utils.get_submodule(model, args.layer)
    act_dim     = model.config.hidden_size
    out_batch_size = int(args.buffer_size_int * args.out_batch_ratio)

    activation_buffer = ActivationBuffer(
        hf_dataset_to_generator(args.text),
        model, submodule,
        n_ctxs=args.buffer_size_int // 128,
        ctx_len=128,
        out_batch_size=out_batch_size,
        io='out',
        d_submodule=act_dim,
        device=device,
    )

    steps = (args.total_tokens_int + out_batch_size - 1) // out_batch_size

    trainSAE(
        data=activation_buffer,
        trainer_configs=[{
            'trainer':        LinearIDOLTrainer,
            'steps':          steps,
            'activation_dim': act_dim,
            'dict_size':      args.z_dim,
            'layer':          args.layer,
            'lm_name':        args.model_name,
            'tau':            args.tau,
            'w':              args.w,
            'noise_mode':     args.noise_mode,
            'topk_sparsity':  args.topk,
            'mode':           args.mode,
            'lr':             args.lr,
            'wd':             args.wd,
            'l_mse_Zt':       1.0 if args.mse_Zt else 0.0,
            'l_ind':          args.l_ind,
            'l_spB':          args.l_spB,
            'l_spM':          args.l_spM,
            'l_spZ':          args.l_spZ,
            'seed':           args.seed,
            'device':         device,
            'wandb_name':     args.run_name,
            'submodule_name': f'resid_post_layer_{args.layer}',
        }],
        steps=steps,
        use_wandb=args.wandb_mode != 'disabled',
        wandb_entity=args.wandb_entity or '',
        wandb_project=args.wandb_project,
        save_steps=[int(steps * p) for p in (0.25, 0.5, 0.75)],
        save_dir=args.results_dir,
        log_steps=max(1, steps // 100),
        normalize_activations=args.normalize_activations,
        device=device,
        autocast_dtype=torch.bfloat16,
        run_cfg={'run_name': args.run_name, 'mode': args.mode},
    )


if __name__ == '__main__':
    main()