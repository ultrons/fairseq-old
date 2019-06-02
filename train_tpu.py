import collections
import math
import os
import random
import shutil

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq import optim

import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm



from argparse import Namespace

args=Namespace(
    activation_dropout=0.0,
    activation_fn='relu',
    adam_betas='(0.9, 0.98)',
    adam_eps=1e-08,
    adaptive_input=False,
    adaptive_softmax_cutoff=None,
    adaptive_softmax_dropout=0,
    arch='transformer_vaswani_wmt_en_de_big',
    attention_dropout=0.1,
    bucket_cap_mb=25,
    clip_norm=0.0,
    cpu=False,
    criterion='label_smoothed_cross_entropy',
    curriculum=0,
    data='/home/sivaibhav/data/wmt18_en_de_bpej32k',
    dataset_impl='cached',
    ddp_backend='c10d',
    decoder_attention_heads=16,
    decoder_embed_dim=1024,
    decoder_embed_path=None,
    decoder_ffn_embed_dim=4096,
    decoder_input_dim=1024,
    decoder_layers=6,
    decoder_learned_pos=False,
    decoder_normalize_before=False,
    decoder_output_dim=1024,
    device_id=0,
    distributed_backend='nccl',
    distributed_init_method='tcp://10.138.0.17:8085',
    distributed_no_spawn=False,
    distributed_port=8085,
    distributed_rank=0,
    distributed_world_size=1,
    dropout=0.3,
    encoder_attention_heads=16,
    encoder_embed_dim=1024,
    encoder_embed_path=None,
    encoder_ffn_embed_dim=4096,
    encoder_layers=6,
    encoder_learned_pos=False,
    encoder_normalize_before=False,
    fix_batches_to_gpus=False,
    fp16=True,
    fp16_init_scale=128,
    fp16_scale_tolerance=0.0,
    fp16_scale_window=None,
    keep_interval_updates=-1,
    keep_last_epochs=-1,
    label_smoothing=0.1,
    lazy_load=False,
    left_pad_source='True',
    left_pad_target='False',
    log_format=None,
    log_interval=100,
    lr=[0.0005],
    lr_scheduler='inverse_sqrt',
    max_epoch=0,
    max_sentences=None,
    max_sentences_valid=None,
    max_source_positions=1024,
    max_target_positions=1024,
    max_tokens=3584,
    max_update=100000,
    memory_efficient_fp16=False,
    min_loss_scale=0.0001,
    min_lr=1e-09,
    no_epoch_checkpoints=False,
    no_progress_bar=True,
    no_save=True,
    no_token_positional_embeddings=False,
    num_workers=0,
    optimizer='adam',
    optimizer_overrides='{}',
    raw_text=False,
    required_batch_size_multiple=8,
    reset_lr_scheduler=False,
    reset_optimizer=False,
    restore_file='checkpoint_last.pt',
    save_dir='checkpoints',
    save_interval=1,
    save_interval_updates=16000,
    seed=3,
    sentence_avg=False,
    share_all_embeddings=True,
    share_decoder_input_output_embed=False,
    skip_invalid_size_inputs_valid_test=False,
    source_lang='en',
    target_lang='de',
    task='translation',
    tensorboard_logdir='',
    threshold_loss_scale=None,
    train_subset='train',
    update_freq=[1],
    upsample_primary=16,
    user_dir=None,
    valid_subset='valid',
    validate_interval=1,
    warmup_init_lr=1e-07,
    warmup_updates=4000,
    weight_decay=0.0)

task = tasks.setup_task(args)
task.load_dataset(args.train_subset, combine=True, epoch=0)
for valid_sub_split in args.valid_subset.split(','):
    task.load_dataset(valid_sub_split, combine=True, epoch=0)


#devices = xm.get_xla_supported_devices(max_devices=8) # Got error for max devices argument :(
devices = xm.get_xla_supported_devices()
model_parallel = dp.DataParallel(lambda: task.build_model(args), device_ids=devices)

#max_positions = utils.resolve_max_positions(
#                task.max_positions(),
#                        model.max_positions(),
#
#        )
max_positions= (1024,1024) # Hardcoded for the moment since the computation requires
                           # model object which will be created by model_parallel __call__
                           # Re-factor in a cleaner way



# Initialize dataloader
epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.
        distributed_rank,
        num_workers=args.num_workers,
    )

update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
# Create the iterator for training
train_loader = iterators.GroupedIterator(itr, update_freq)

def build_optimizer (args, model):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    return optim.build_optimizer(args, params)


# task and args objects are referred to from global scope
# Not sure if it's ok if the function is replicated on multi thread
# Should be in the Closure
# from multi-GPU example it seems that it should not be a problem
def train_loop_fn (model, loader, device='cpu?', context=None):
    criterion = task.build_criterion(args)
    tracker = xm.RateTracker()
    optimizer=build_optimizer(args, model)
    for i, samples in loader:
        print("Processing minibatch:%d" %i)
        task.train_step(samples[0], model, criterion, optimizer,False)
        xm.optimizer_step(optimizer)

# Print some samples from train_loader
print(next(train_loader))


# Run training from one epoch
model_parallel(train_loop_fn, train_loader)




