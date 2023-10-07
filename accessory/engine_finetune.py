import math
import sys
import os
from typing import Iterable
import contextlib

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from fairscale.nn.model_parallel import initialize as fs_init

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    model.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch_data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter):
        if len(batch_data) == 4:
            examples, labels, example_mask, imgs = batch_data
        else:
            examples, labels, example_mask = batch_data
            imgs = None
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_epoch(optimizer, data_iter_step / len(data_loader) + epoch, args)

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss = model(examples, labels, images=imgs)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        update_grad = (data_iter_step + 1) % accum_iter == 0
        grad_norm = loss_scaler(
            loss, optimizer, model,
            parameters=model.parameters(),
            update_grad=update_grad,
            clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
        )

        if update_grad:
            assert grad_norm is not None
            if torch.any(torch.isinf(grad_norm)):
                print("grad norm is inf")
            else:
                metric_logger.update(grad_norm=grad_norm)

            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if update_grad:
            grad_norm_reduce = misc.all_reduce_mean(grad_norm)
        if log_writer is not None and update_grad:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            if update_grad:
                log_writer.add_scalar('grad_norm', grad_norm_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


        # save within epoch
        n_update_per_save = 10000 // accum_iter
        if update_grad and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0:
            misc.save_checkpoint(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}