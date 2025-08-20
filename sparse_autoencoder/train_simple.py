import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def training_loop_(
    ae, train_acts_iter, loss_fn, lr, comms, eps=6.25e-10, clip_grad=None, ema_multiplier=0.999, logger=None
):
    if logger is None:
        logger = Logger(dummy=True)

    scaler = torch.cuda.amp.GradScaler()
    autocast_ctx_manager = torch.cuda.amp.autocast()

    opt = torch.optim.Adam(ae.parameters(), lr=lr, eps=eps, fused=True)
    if ema_multiplier is not None:
        ema = EmaModel(ae, ema_multiplier=ema_multiplier)

    for i, flat_acts_train_batch in enumerate(train_acts_iter):
        flat_acts_train_batch = flat_acts_train_batch.cuda()

        with autocast_ctx_manager:
            recons, info = ae(flat_acts_train_batch)

            loss = loss_fn(ae, flat_acts_train_batch, recons, info, logger)

        print0(i, loss)

        logger.logkv("loss_scale", scaler.get_scale())

        if RANK == 0:
            wandb.log({"train_loss": loss.item()})

        loss = scaler.scale(loss)
        loss.backward()

        unit_norm_decoder_(ae)
        unit_norm_decoder_grad_adjustment_(ae)

        # allreduce gradients
        comms.dp_allreduce_(ae)

        # keep fp16 loss scale synchronized across shards
        comms.sh_allreduce_scale(scaler)

        # if you want to do anything with the gradients that depends on the absolute scale (e.g clipping, do it after the unscale_)
        scaler.unscale_(opt)

        # gradient clipping
        if clip_grad is not None:
            grad_norm = sharded_grad_norm(ae, comms)
            logger.logkv("grad_norm", grad_norm)
            grads = [x.grad for x in ae.parameters() if x.grad is not None]
            torch._foreach_mul_(grads, clip_grad / torch.clamp(grad_norm, min=clip_grad))

        if ema_multiplier is not None:
            ema.step()

        # take step with optimizer
        scaler.step(opt)
        scaler.update()
        
        logger.dumpkvs()