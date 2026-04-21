"""Script for a pretraining run."""

import torch
import hydra

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming
from cramming.backend.utils import prepare_pretraining_dataloader

log = logging.getLogger(__name__)


def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    train_dataset, eval_dataset = split_pretraining_dataset(dataset, cfg.validation)
    checkpoint_rendevous = os.path.join(cfg.base_dir, cfg.name, "intermediate_state.pth")
    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        try:
            metadata = torch.load(checkpoint_rendevous, map_location=torch.device("cpu"))["metadata"]
            initial_step, elapsed_time = metadata["step"], metadata["elapsed"]
        except RuntimeError:
            log.info("Checkpoint file unreadable or corrupted.")
            os.remove(checkpoint_rendevous)
            initial_step, elapsed_time = 0, 0.0
    else:
        initial_step, elapsed_time = 0, 0.0

    model_engine, _, _, dataloader = cramming.load_backend(model, train_dataset, tokenizer, cfg.train, cfg.impl, elapsed_time, setup=setup)
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = prepare_pretraining_dataloader(
            eval_dataset, tokenizer, cfg.train, cfg.impl, infinite=False, shuffle=False
        )
        log.info(
            f"Pretraining validation enabled with {len(train_dataset):,} train blocks and {len(eval_dataset):,} eval blocks. "
            f"Eval interval: {cfg.validation.interval} steps."
        )
    if cfg.impl.resume_run_after_preempt and os.path.isfile(checkpoint_rendevous):
        log.info(f"Loading intermediate checkpoint from previous run onto device {cfg.impl.local_rank}...")
        model_engine.load_training_checkpoint(checkpoint_rendevous)
    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time() - elapsed_time
    train_time = time.time()
    training_allowed, no_recovery_necessary = True, True
    loss_vals = []

    # Launch training
    for step, batch in enumerate(dataloader, initial_step + 1):

        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        loss = model_engine.step(device_batch)
        loss_vals.append(loss.detach())

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        # Collect stats and print to console and upload to wandb
        if step % cfg.impl.print_loss_every_nth_step == 0:
            loss_vals, train_time = collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg)
            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        if eval_dataloader is not None and step % cfg.validation.interval == 0:
            collect_validation_stats(step, stats, model_engine, eval_dataloader, cfg)

        # Checkpointing is triggered from stopping criteria and normal intervals
        if cfg.impl.save_intermediate_checkpoints and step % cfg.impl.save_every_nth_step == 0:
            if loss.detach().isfinite() and cramming.utils.is_main_process() and not cfg.dryrun:
                model_engine.save_training_checkpoint(checkpoint_rendevous, metadata=dict(step=step, elapsed=time.time() - wallclock_timer))

        if not loss.detach().isfinite():
            training_allowed, no_recovery_necessary = engage_troubleshooting(
                model_engine, step, training_allowed, no_recovery_necessary, cfg
            )

        communicate_flags(training_allowed, no_recovery_necessary)

        if (cfg.dryrun and step > 2) or not training_allowed:
            break

        if not no_recovery_necessary:  # synced across devices
            log.info(f"Attempting reload of checkpoint on device {cfg.impl.local_rank}.")
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            no_recovery_necessary = True

    # Save to summary:
    cramming.utils.save_summary("pretrain", cfg, stats, time.time() - local_time, setup)
    if cramming.utils.is_main_process():
        # Save final checkpoint? Might have to recover the latest checkpoint first
        if not loss.detach().isfinite() and cfg.impl.save_intermediate_checkpoints:
            model_engine.load_training_checkpoint(checkpoint_rendevous)
            loss = torch.as_tensor(16.0)  # fake value for model file name
        if loss.detach().isfinite():
            now = datetime.datetime.now()
            long_checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{loss:2.4f}"
            model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), long_checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)

            if cfg.impl.push_to_huggingface_hub:
                model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)
    metrics = dict(num_params=sum([p.numel() for p in model.parameters()]))
    return metrics


def split_pretraining_dataset(dataset, validation_cfg):
    if not validation_cfg.enabled or validation_cfg.split <= 0:
        return dataset, None
    if isinstance(dataset, torch.utils.data.IterableDataset):
        raise ValueError("Validation during pretraining requires a map-style dataset, but an IterableDataset was given.")

    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Validation split requires at least 2 pretraining blocks.")

    if validation_cfg.split >= 1:
        eval_size = int(validation_cfg.split)
    else:
        eval_size = int(round(dataset_size * validation_cfg.split))
    eval_size = min(max(1, eval_size), dataset_size - 1)

    generator = torch.Generator().manual_seed(validation_cfg.seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
    eval_indices = shuffled_indices[:eval_size]
    train_indices = shuffled_indices[eval_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    return train_dataset, eval_dataset


def check_deadline(launch_time, hour_limit):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    return True if (current_time - launch_time) / 3600 > hour_limit else False


def check_early_termination(launch_time, loss, early_termination):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        return True if (current_time - launch_time) / 3600 > early_termination.budget else False
    else:
        return False


def collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg):
    stats["step"] += [step]
    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss

    current_lr = model_engine.optimizer.param_groups[0].get("lr", float("NaN"))
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals = []
    train_time = time.time()
    return loss_vals, train_time


@torch.no_grad()
def collect_validation_stats(step, stats, model_engine, eval_dataloader, cfg):
    was_training = model_engine.training
    model_engine.eval()

    eval_start = time.time()
    loss_sum = torch.zeros(1, device=model_engine.setup["device"], dtype=torch.float32)
    batch_count = torch.zeros(1, device=model_engine.setup["device"], dtype=torch.float32)

    for batch_idx, batch in enumerate(eval_dataloader, start=1):
        device_batch = model_engine.to_device(batch)
        loss = model_engine.forward(**device_batch)["loss"]
        loss_sum += loss.detach().float()
        batch_count += 1

        if cfg.validation.max_batches is not None and batch_idx >= cfg.validation.max_batches:
            break

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(loss_sum, torch.distributed.ReduceOp.SUM, async_op=False)
        torch.distributed.all_reduce(batch_count, torch.distributed.ReduceOp.SUM, async_op=False)

    eval_loss = (loss_sum / batch_count.clamp_min(1)).item()
    stats["eval_step"] += [step]
    stats["eval_loss"] += [eval_loss]
    stats["best_eval_loss"] += [min(stats["eval_loss"])]
    stats["eval_time"] += [time.time() - eval_start]

    if was_training:
        model_engine.train(cfg.train.pretrain_in_train_mode)

    log.info(
        f"Eval loss {eval_loss:2.4f} at step {step}. "
        f"[Best: {stats['best_eval_loss'][-1]:2.4f}] "
        f"Eval time: {stats['eval_time'][-1]:2.2f}s."
    )
    cramming.utils.wandb_log(
        {
            "eval_step": [step],
            "eval_loss": [eval_loss],
            "best_eval_loss": [stats["best_eval_loss"][-1]],
            "eval_time": [stats["eval_time"][-1]],
        },
        cfg,
    )


def engage_troubleshooting(model_engine, step, training_allowed, no_recovery_necessary, cfg):
    log.info(f"Non-finite loss in step {step} on device {cfg.impl.local_rank}.")

    is_finite_grad = [torch.isfinite(p.grad).all() for p in model_engine.model.parameters() if p.grad is not None]
    has_finite_gradients = torch.stack(is_finite_grad).all() if len(is_finite_grad) > 0 else True
    if not has_finite_gradients:
        if "dump_nan_grads" in cfg.impl.troubleshoot_strategy:
            log.info(f"Non-finite gradients in step {step} on device {cfg.impl.local_rank}, dumping...")
            model_engine.optimizer.zero_grad()
        else:
            if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
                no_recovery_necessary = False
            else:
                training_allowed = False
                log.info(f"Stopping training due to non-finite grads in step {step} on device {cfg.impl.local_rank}.")

    has_finite_parameters = torch.stack([torch.isfinite(p).all() for p in model_engine.model.parameters()]).all()
    if not has_finite_parameters:
        if "recover_checkpoint" in cfg.impl.troubleshoot_strategy:
            no_recovery_necessary = False
        else:
            training_allowed = False
            log.info(f"Stopping training due to non-finite parameters in step {step} on device {cfg.impl.local_rank}.")
    return training_allowed, no_recovery_necessary


def communicate_flags(training_allowed, no_recovery_necessary):
    """A quick and dirty communication through the comm protocol. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor_allowed = torch.as_tensor([training_allowed, no_recovery_necessary])
        comm_tensor_allowed = comm_tensor_allowed.cuda() if torch.cuda.is_available() else comm_tensor_allowed.float()
        torch.distributed.all_reduce(comm_tensor_allowed, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor_allowed[0] >= 1:  # training indeed allowed on all devices
            return True, comm_tensor_allowed[1] > 0
        else:
            return False, True
    else:
        return training_allowed, no_recovery_necessary


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")


if __name__ == "__main__":
    launch()
