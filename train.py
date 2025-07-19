# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from utils.data_utils import get_loader
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from model.Revit import ReViT
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

# Utility class to track average metric values (e.g., loss) across iterations
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Compute regression metrics: Pearson R, MAE, RMSE
def all_accuracy(true, pred):
    true = true.reshape(-1)  # 或者 true = np.ravel(true)
    pred = pred.reshape(-1)
    r, _ = pearsonr(true, pred)
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    return r, mae, rmse

# Save model checkpoint
def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

# Build model from config and send it to device
def setup(args):
    class Config:
        class DATA:
            crop_size = 224

        class MODEL:
            num_classes = 1
            dropout_rate = 0.1
            head_act = None

        class ReViT:
            mode = "conv"
            pool_first = False
            patch_kernel = [16, 16]
            patch_stride = [16, 16]
            patch_padding = [0, 0]
            embed_dim = 768
            num_heads = 12
            mlp_ratio = 4
            qkv_bias = True
            drop_path = 0.2
            depth = 12
            dim_mul = []
            head_mul = []
            pool_qkv_kernel = []
            pool_kv_stride_adaptive = []
            pool_q_stride = []
            zero_decay_pos = False
            use_abs_pos = True
            use_rel_pos = False
            rel_pos_zero_init = False
            residual_pooling = False
            dim_mul_in_att = False
            alpha = True
            visualize = True
            cls_embed_on = False

    cfg = Config()
    model = ReViT(cfg)
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter Count: \t%2.1fM" % num_params)
    return args, model

# Count trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# Set random seeds for reproducibility
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Evaluation loop
def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_veg, all_vegt = [], []

    epoch_iterator = tqdm(test_loader, desc="Validating... (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, disable=args.local_rank not in [-1, 0])
    loss_fn = torch.nn.MSELoss()

    for _, batch in enumerate(epoch_iterator):
        batch_x, batch_position, batch_veg = batch
        batch_x, batch_position, batch_veg = batch_x.to(args.device), batch_position.to(args.device), batch_veg.to(args.device)

        with torch.no_grad():
            with autocast(device_type=args.device.type):
                veg = model(batch_x, batch_position)
                loss = loss_fn(veg, batch_veg)
                eval_losses.update(loss.item())

            all_veg.append(veg.cpu().numpy())
            all_vegt.append(batch_veg.cpu().numpy())

        epoch_iterator.set_description(f"Validating... (loss={eval_losses.val:.5f})")

    all_veg = np.concatenate(all_veg, axis=0)
    all_vegt = np.concatenate(all_vegt, axis=0)
    all_veg[all_veg < 0] = 0  # clamp negative values

    # Calculate evaluation metrics
    veg_r, veg_mae, veg_rmse = all_accuracy(all_vegt, all_veg)

    logger.info("\nValidation Results")
    logger.info(f"Global Steps: {global_step}")
    logger.info(f"Valid Loss: {eval_losses.avg:.5f}")
    logger.info(f"Valid Pearson R: {veg_r:.5f}")
    logger.info(f"Valid MAE: {veg_mae:.5f}")
    logger.info(f"Valid RMSE: {veg_rmse:.5f}")

    writer.add_scalar("test/accuracy", scalar_value=veg_r, global_step=global_step)

    return veg_r

# Training loop
def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, args.warmup_steps, args.num_steps) if args.decay_type == "cosine" else WarmupLinearSchedule(optimizer, args.warmup_steps, args.num_steps)
    scaler = GradScaler('cuda') if args.fp16 else None

    logger.info("***** Running training *****")
    logger.info(f"Total optimization steps = {args.num_steps}")
    logger.info(f"Instantaneous batch size per GPU = {args.train_batch_size}")
    logger.info(f"Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_r = 0, 0

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, disable=args.local_rank not in [-1, 0])
        loss_fn = torch.nn.MSELoss()

        for step, batch in enumerate(epoch_iterator):
            batch_x, batch_position, batch_veg = [b.to(args.device) for b in batch]

            if args.fp16:
                with autocast(device_type=args.device.type):
                    veg = model(batch_x, batch_position)
                    loss = loss_fn(veg, batch_veg)
            else:
                veg = model(batch_x, batch_position)
                loss = loss_fn(veg, batch_veg)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                with torch.no_grad():
                    losses.update(loss.item() * args.gradient_accumulation_steps)

                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                epoch_iterator.set_description(f"Training ({global_step} / {args.num_steps} Steps) (loss={losses.val:.5f})")

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", losses.val, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

                    if global_step % args.eval_every == 0:
                        r = valid(args, model, writer, test_loader, global_step)
                        if best_r < r:
                            save_model(args, model)
                            best_r = r
                        torch.cuda.empty_cache()
                        model.train()

                if global_step >= args.num_steps:
                    break

        losses.reset()
        torch.cuda.empty_cache()
        if global_step >= args.num_steps:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Pearson R: \t%f" % best_r)
    logger.info("Training Complete!")

# Entry point for training
def main():
    parser = argparse.ArgumentParser()

    # Training configuration parameters
    parser.add_argument("--name", required=True, help="Experiment name for logging and checkpointing.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz")
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_steps", default=10, type=int)
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--warmup_steps", default=5, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O2')
    parser.add_argument('--loss_scale', type=float, default=0)

    # Hard-coded argument values for direct script execution (can be replaced with CLI)
    args = parser.parse_args([
        '--name', 'GVI_Revit',
        '--pretrained_dir', 'checkpoint/Revit_checkpoint.bin',
        '--output_dir', 'output',
        '--train_batch_size', '256',
        '--eval_batch_size', '256',
        '--eval_every', '50',
        '--learning_rate', '1e-4',
        '--weight_decay', '0',
        '--num_steps', '2500',
        '--decay_type', 'cosine',
        '--warmup_steps', '500',
        '--max_grad_norm', '1.0',
        '--local_rank', '-1',
        '--seed', '42',
        '--gradient_accumulation_steps', '1',
        '--fp16',
        '--fp16_opt_level', 'O2',
        '--loss_scale', '0'
    ])

    # Setup environment and logging
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
        args.n_gpu = 1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}, distributed: {args.local_rank != -1}, fp16: {args.fp16}")

    set_seed(args)
    args, model = setup(args)
    train(args, model)

if __name__ == "__main__":
    main()
