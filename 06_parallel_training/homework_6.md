## HW-6 Log Files
```🚀 View run stilted-puddle-1 at: https://wandb.ai/bikash-kharel2019/WordPlay/runs/lrcby4bd```

```⚡ View job at https://wandb.ai/bikash-kharel2019/WordPlay/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjE1NTEyMzgzOA==/version_details/v0```

[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 2
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 1
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 3
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 4
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 5
Failed to download font: IBM Plex Sans, skipping!
Failed to download font: IBM Plex Sans Condensed, skipping!
Failed to download font: IBM Plex Serif, skipping!
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 7
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 6
[2024-04-02 01:59:23][INFO][distributed_c10d:442] - Added key: store_based_barrier_key:1 to store for rank: 0
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:23][INFO][distributed_c10d:476] - Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 8 nodes.
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=3/7][local_rank=3/3][node=1/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=2/7][local_rank=2/3][node=0/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=1/7][local_rank=1/3][node=1/1]
[2024-04-02 01:59:25][INFO][dist:239] - DistInfo={
    "DEVICE": "cuda",
    "DEVICE_ID": "cuda:0",
    "DISTRIBUTED_BACKEND": "nccl",
    "GPUS_PER_NODE": 4,
    "HOSTFILE": "/var/spool/pbs/aux/1819844.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov",
    "HOSTNAME": "x3005c0s37b1n0.hsn.cm.polaris.alcf.anl.gov",
    "HOSTS": "['x3005c0s37b1n0', 'x3005c0s7b0n0']",
    "LOCAL_RANK": 0,
    "MACHINE": "Polaris",
    "NGPUS": 8,
    "NODE_ID": 0,
    "NUM_NODES": 2,
    "RANK": 0,
    "SCHEDULER": "PBS",
    "WORLD_SIZE_IN_USE": 8,
    "WORLD_SIZE_TOTAL": 8
}
[2024-04-02 01:59:25][INFO][dist:605] - [0/8] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=4/7][local_rank=0/3][node=0/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=7/7][local_rank=3/3][node=1/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=6/7][local_rank=2/3][node=0/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=5/7][local_rank=1/3][node=1/1]
[2024-04-02 01:59:25][INFO][dist:290] - [device='cuda'][rank=0/7][local_rank=0/3][node=0/1]
[2024-04-02 01:59:25][WARNING][dist:296] - Using [8 / 8] available "cuda" devices !!
[2024-04-02 01:59:25][INFO][configs:317] - Loading val from /home/kharelbikash/wordplay/data/shakespeare_char/val.bin
[2024-04-02 01:59:25][INFO][configs:317] - Loading train from /home/kharelbikash/wordplay/data/shakespeare_char/train.bin
[2024-04-02 01:59:25][INFO][configs:442] - Tokens per iteration: 131,072
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][configs:465] - Using self.ptdtype=torch.bfloat16 on self.device_type='cuda'
[2024-04-02 01:59:26][INFO][configs:471] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][dist:751] - Setting up wandb from rank: 0
[2024-04-02 01:59:26][INFO][dist:752] - Using: WB PROJECT: WordPlay
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:2'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:3'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:1'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:1'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:0'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:2'"
[2024-04-02 01:59:26][CRITICAL][trainer:296] - "devid='cuda:3'"
wandb: Currently logged in as: bikash-kharel2019. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.16.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.15.12
wandb: Run data is saved locally in /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23/wandb/run-20240402_015927-lrcby4bd
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run stilted-puddle-1
wandb: ⭐️ View project at https://wandb.ai/bikash-kharel2019/WordPlay
wandb: 🚀 View run at https://wandb.ai/bikash-kharel2019/WordPlay/runs/lrcby4bd
[2024-04-02 01:59:38][INFO][dist:782] - W&B RUN: [stilted-puddle-1](https://wandb.ai/bikash-kharel2019/WordPlay/runs/lrcby4bd)
[2024-04-02 01:59:38][INFO][dist:810] - Running on machine='Polaris'
[2024-04-02 01:59:38][WARNING][__main__:87] - {
    "train": {
        "framework": "pytorch",
        "backend": "DDP",
        "device": null,
        "seed": null,
        "port": null,
        "ds_config_path": null,
        "precision": null,
        "ngpus": null,
        "use_wandb": true,
        "eval_interval": 250,
        "log_interval": 5,
        "eval_iters": 200,
        "eval_only": false,
        "always_save_checkpoint": false,
        "init_from": "scratch",
        "wandb_project": "WordPlay",
        "max_iters": 100,
        "warmup_iters": 100,
        "dtype": "bfloat16",
        "compile": false
    },
    "model": {
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "batch_size": 64,
        "block_size": 256,
        "activation": "gelu",
        "dropout": 0.2,
        "bias": false,
        "vocab_size": 65
    },
    "data": {
        "dataset": "shakespeare_char",
        "out_dir": "out-shakespeare-char",
        "root_path": null
    },
    "optimizer": {
        "gas": 1,
        "name": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.99,
        "grad_clip": 1.0,
        "decay_lr": true,
        "lr_decay_iters": 5000,
        "min_lr": 0.0001
    }
}
[2024-04-02 01:59:38][WARNING][__main__:88] - Output dir: /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23
[2024-04-02 01:59:38][INFO][trainer:227] - Initializing a new model from scratch
[2024-04-02 01:59:38][INFO][model:255] - number of parameters: 10.65M
[2024-04-02 01:59:38][INFO][model:445] - num decayed parameter tensors: 26, with 10,740,096 parameters
[2024-04-02 01:59:38][INFO][model:449] - num non-decayed parameter tensors: 13, with 4,992 parameters
[2024-04-02 01:59:38][INFO][model:465] - using fused AdamW: True
[2024-04-02 01:59:38][CRITICAL][trainer:296] - "devid='cuda:0'"
[2024-04-02 01:59:41][INFO][trainer:333] - • self.model=GPT(
  (transformer): ModuleDict(
    (wte): Embedding(65, 384)
    (wpe): Embedding(256, 384)
    (drop): Dropout(p=0.2, inplace=False)
    (h): ModuleList(
      (0-5): 6 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=384, out_features=1152, bias=False)
          (c_proj): Linear(in_features=384, out_features=384, bias=False)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (resid_dropout): Dropout(p=0.2, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=384, out_features=1536, bias=False)
          (act_fn): GELU(approximate='none')
          (c_proj): Linear(in_features=1536, out_features=384, bias=False)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=384, out_features=65, bias=False)
)
[2024-04-02 01:59:41][INFO][trainer:334] - • self.grad_scaler=<torch.cuda.amp.grad_scaler.GradScaler object at 0x151246372710>
[2024-04-02 01:59:41][INFO][trainer:335] - • self.model_engine=DistributedDataParallel(
  (module): GPT(
    (transformer): ModuleDict(
      (wte): Embedding(65, 384)
      (wpe): Embedding(256, 384)
      (drop): Dropout(p=0.2, inplace=False)
      (h): ModuleList(
        (0-5): 6 x Block(
          (ln_1): LayerNorm()
          (attn): CausalSelfAttention(
            (c_attn): Linear(in_features=384, out_features=1152, bias=False)
            (c_proj): Linear(in_features=384, out_features=384, bias=False)
            (attn_dropout): Dropout(p=0.2, inplace=False)
            (resid_dropout): Dropout(p=0.2, inplace=False)
          )
          (ln_2): LayerNorm()
          (mlp): MLP(
            (c_fc): Linear(in_features=384, out_features=1536, bias=False)
            (act_fn): GELU(approximate='none')
            (c_proj): Linear(in_features=1536, out_features=384, bias=False)
            (dropout): Dropout(p=0.2, inplace=False)
          )
        )
      )
      (ln_f): LayerNorm()
    )
    (lm_head): Linear(in_features=384, out_features=65, bias=False)
  )
)
[2024-04-02 01:59:41][INFO][trainer:336] - • self.optimizer=AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.99)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: True
    lr: 0.001
    maximize: False
    weight_decay: 0.1

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.99)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: True
    lr: 0.001
    maximize: False
    weight_decay: 0.0
)
  0%|          | 0/100 [00:00<?, ?it/s][2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.3313
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.5185
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.4102
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.2374
                              Training Legend
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    abbr    ┃ desc                                                        ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    step    │ Current training iteration                                  │
│    loss    │ Loss value                                                  │
│     dt     │ Elapsed time per training step (measured in **ms**)         │
│    dtf     │ Elapsed time per forward step (measured in **ms**)          │
│    dtb     │ Elapsed time per backward step (measured in **ms**)         │
│    sps     │ Samples per second                                          │
│    mtps    │ Tokens per second, measured in MEGA (1 x 10^6) tokens / sec │
│    mfu     │ Model flops utilization                                     │
│ train_loss │ Training loss value                                         │
│  val_loss  │ Validation loss value                                       │
└────────────┴─────────────────────────────────────────────────────────────┘
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.2398
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.2393
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.2400
[2024-04-02 01:59:41][INFO][trainer:769] - Startup time: 18.1922
  1%|          | 1/100 [00:04<07:04,  4.29s/it][2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
[2024-04-02 01:59:45][INFO][distributed:1140] - Reducer buckets have been rebuilt in this iteration.
  4%|▍         | 4/100 [00:04<01:12,  1.32it/s][2024-04-02 01:59:46][INFO][trainer:837] - step=5 loss=3.6606 dt=108.1093 dtf=4.6110 dtb=100.7493 sps=73.9992 mtps=1.2124 mfu=-100.0000 train_loss=4.2922 val_loss=4.2905
  9%|▉         | 9/100 [00:05<00:19,  4.61it/s][2024-04-02 01:59:46][INFO][trainer:837] - step=10 loss=3.2386 dt=85.2848 dtf=4.6529 dtb=77.9241 sps=93.8033 mtps=1.5369 mfu=4.3692 train_loss=4.2922 val_loss=4.2905
 14%|█▍        | 14/100 [00:05<00:12,  7.10it/s][2024-04-02 01:59:47][INFO][trainer:837] - step=15 loss=2.9490 dt=106.7558 dtf=4.4962 dtb=99.6278 sps=74.9374 mtps=1.2278 mfu=4.2813 train_loss=4.2922 val_loss=4.2905
 19%|█▉        | 19/100 [00:06<00:08,  9.84it/s][2024-04-02 01:59:47][INFO][trainer:837] - step=20 loss=2.7594 dt=112.4793 dtf=4.5539 dtb=104.8465 sps=71.1242 mtps=1.1653 mfu=4.1844 train_loss=4.2922 val_loss=4.2905
 24%|██▍       | 24/100 [00:06<00:08,  8.90it/s][2024-04-02 01:59:48][INFO][trainer:837] - step=25 loss=2.6489 dt=97.2465 dtf=4.5298 dtb=90.0386 sps=82.2652 mtps=1.3478 mfu=4.1492 train_loss=4.2922 val_loss=4.2905
 29%|██▉       | 29/100 [00:07<00:07,  9.69it/s][2024-04-02 01:59:48][INFO][trainer:837] - step=30 loss=2.6084 dt=73.1144 dtf=4.5159 dtb=65.8305 sps=109.4176 mtps=1.7927 mfu=4.2439 train_loss=4.2922 val_loss=4.2905
 33%|███▎      | 33/100 [00:07<00:06, 10.62it/s][2024-04-02 01:59:49][INFO][trainer:837] - step=35 loss=2.5493 dt=58.2279 dtf=4.5330 dtb=51.6754 sps=137.3912 mtps=2.2510 mfu=4.4595 train_loss=4.2922 val_loss=4.2905
 39%|███▉      | 39/100 [00:08<00:05, 10.96it/s][2024-04-02 01:59:49][INFO][trainer:837] - step=40 loss=2.5460 dt=80.2574 dtf=4.4826 dtb=73.0572 sps=99.6793 mtps=1.6331 mfu=4.4778 train_loss=4.2922 val_loss=4.2905
 43%|████▎     | 43/100 [00:08<00:05, 10.72it/s][2024-04-02 01:59:50][INFO][trainer:837] - step=45 loss=2.5063 dt=92.2348 dtf=5.1066 dtb=85.1571 sps=86.7351 mtps=1.4211 mfu=4.4340 train_loss=4.2922 val_loss=4.2905
 49%|████▉     | 49/100 [00:09<00:04, 10.57it/s][2024-04-02 01:59:50][INFO][trainer:837] - step=50 loss=2.4985 dt=84.5382 dtf=4.4968 dtb=77.4286 sps=94.6317 mtps=1.5504 mfu=4.4314 train_loss=4.2922 val_loss=4.2905
 53%|█████▎    | 53/100 [00:09<00:04, 10.59it/s][2024-04-02 01:59:51][INFO][trainer:837] - step=55 loss=2.4959 dt=98.6381 dtf=4.4309 dtb=92.2478 sps=81.1045 mtps=1.3288 mfu=4.3660 train_loss=4.2922 val_loss=4.2905
 59%|█████▉    | 59/100 [00:09<00:03, 10.82it/s][2024-04-02 01:59:51][INFO][trainer:837] - step=60 loss=2.4925 dt=72.9118 dtf=4.4733 dtb=65.6653 sps=109.7216 mtps=1.7977 mfu=4.4405 train_loss=4.2922 val_loss=4.2905
 63%|██████▎   | 63/100 [00:10<00:03, 10.24it/s][2024-04-02 01:59:52][INFO][trainer:837] - step=65 loss=2.4818 dt=115.5730 dtf=4.4727 dtb=108.4105 sps=69.2203 mtps=1.1341 mfu=4.3188 train_loss=4.2922 val_loss=4.2905
 69%|██████▉   | 69/100 [00:10<00:02, 10.84it/s][2024-04-02 01:59:52][INFO][trainer:837] - step=70 loss=2.4616 dt=64.0492 dtf=4.5093 dtb=56.8391 sps=124.9040 mtps=2.0464 mfu=4.4687 train_loss=4.2922 val_loss=4.2905
 73%|███████▎  | 73/100 [00:11<00:02, 12.54it/s][2024-04-02 01:59:52][INFO][trainer:837] - step=75 loss=2.4382 dt=69.1861 dtf=4.4955 dtb=62.6730 sps=115.6302 mtps=1.8945 mfu=4.5604 train_loss=4.2922 val_loss=4.2905
 79%|███████▉  | 79/100 [00:11<00:01, 12.47it/s][2024-04-02 01:59:53][INFO][trainer:837] - step=80 loss=2.4526 dt=95.5524 dtf=4.5071 dtb=88.3516 sps=83.7237 mtps=1.3717 mfu=4.4944 train_loss=4.2922 val_loss=4.2905
 83%|████████▎ | 83/100 [00:12<00:01, 12.81it/s][2024-04-02 01:59:53][INFO][trainer:837] - step=85 loss=2.4519 dt=98.3922 dtf=4.4717 dtb=91.8971 sps=81.3073 mtps=1.3321 mfu=4.4236 train_loss=4.2922 val_loss=4.2905
 89%|████████▉ | 89/100 [00:12<00:00, 13.70it/s][2024-04-02 01:59:54][INFO][trainer:837] - step=90 loss=2.4445 dt=66.0176 dtf=4.4979 dtb=58.8554 sps=121.1799 mtps=1.9854 mfu=4.5457 train_loss=4.2922 val_loss=4.2905
 93%|█████████▎| 93/100 [00:12<00:00, 12.58it/s][2024-04-02 01:59:54][INFO][trainer:837] - step=95 loss=2.4312 dt=93.2178 dtf=4.5200 dtb=86.7065 sps=85.8205 mtps=1.4061 mfu=4.4909 train_loss=4.2922 val_loss=4.2905
 99%|█████████▉| 99/100 [00:13<00:00, 12.82it/s][2024-04-02 01:59:54][INFO][trainer:837] - step=100 loss=2.4075 dt=55.9236 dtf=4.5000 dtb=48.7769 sps=143.0523 mtps=2.3438 mfu=4.7081 train_loss=4.2922 val_loss=4.2905
100%|██████████| 100/100 [00:13<00:00,  7.48it/s]
[2024-04-02 01:59:56][INFO][__main__:113] - ['prompt']: 'What is an LLM?'
[2024-04-02 01:59:56][INFO][__main__:114] - ['response']:

What is an LLM?

LOUS:
Therat he muritteng p mur, ait the pesof thileem
OUS:
HUSeave sir ave, breenoood fred ashanthear sess sive aveat me
Shete t my distorrerenche thas sho,
Thamo sthecer hemand se owe,
Than t dinderrere towee ar he yoous then inder fane sse po mestothe
[2024-04-02 01:59:56][INFO][trainer:735] - Saving checkpoint to: /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23
[2024-04-02 01:59:56][INFO][trainer:736] - Saving model to: /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23/model.pth
[2024-04-02 01:59:56][INFO][configs:141] - Appending /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23 to /home/kharelbikash/wordplay/src/ckpts/checkpoints.log
wandb: Waiting for W&B process to finish... (success).
wandb: | 41.197 MB of 41.197 MB uploaded (0.000 MB deduped)
wandb: Run history:
wandb:              Loss/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:             Loss/lossf █▆▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:               Loss/mfu ▁███████████████████
wandb:             Loss/train ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:               Loss/val ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:          Timing/dt_avg ▇▄▇█▆▃▁▄▅▄▆▃█▂▃▆▆▂▅▁
wandb:         Timing/dt_iter ▇▄▇█▆▃▁▄▅▄▆▃█▂▃▆▆▂▅▁
wandb:          Timing/dt_tot ▇▄▇█▆▃▁▄▅▄▆▃█▂▃▆▆▂▅▁
wandb:         Timing/dtb_avg ▇▄▇█▆▃▁▄▅▄▆▃█▂▃▆▆▂▅▁
wandb:         Timing/dtb_tot ▇▄▇█▆▃▁▄▅▄▆▃█▂▃▆▆▂▅▁
wandb:         Timing/dtf_avg ▃▃▂▂▂▂▂▂█▂▁▁▁▂▂▂▁▂▂▂
wandb:         Timing/dtf_tot ▃▃▂▂▂▂▂▂█▂▁▁▁▂▂▂▁▂▂▂
wandb:            Timing/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb: Timing/samples_per_sec ▁▃▂▁▂▅▇▄▃▃▂▅▁▆▅▂▂▆▃█
wandb:    Timing/startup_time ▁
wandb:  Timing/tokens_per_sec ▁▃▂▁▂▅▇▄▃▃▂▅▁▆▅▂▂▆▃█
wandb:          Training/iter ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:          Training/loss █▆▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:      Training/loss_tot █▆▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:            Training/lr ▁▁▂▂▂▃▃▄▄▄▅▅▅▆▆▇▇▇██
wandb:
wandb: Run summary:
wandb:              Loss/iter 100
wandb:             Loss/lossf 2.40747
wandb:               Loss/mfu 4.70809
wandb:             Loss/train 4.2922
wandb:               Loss/val 4.29047
wandb:          Timing/dt_avg 0.02664
wandb:         Timing/dt_iter 0.05592
wandb:          Timing/dt_tot 0.05328
wandb:         Timing/dtb_avg 0.04878
wandb:         Timing/dtb_tot 0.04878
wandb:         Timing/dtf_avg 0.0045
wandb:         Timing/dtf_tot 0.0045
wandb:            Timing/iter 99
wandb: Timing/samples_per_sec 143.05231
wandb:    Timing/startup_time 18.23741
wandb:  Timing/tokens_per_sec 2343769.08296
wandb:          Training/iter 99
wandb:          Training/loss 2.40747
wandb:      Training/loss_tot 2.40747
wandb:            Training/lr 0.00099
wandb:
wandb: 🚀 View run stilted-puddle-1 at: https://wandb.ai/bikash-kharel2019/WordPlay/runs/lrcby4bd
wandb: ️⚡ View job at https://wandb.ai/bikash-kharel2019/WordPlay/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjE1NTEyMzgzOA==/version_details/v0
wandb: Synced 6 W&B file(s), 0 media file(s), 25 artifact file(s) and 0 other file(s)
wandb: Find logs at: /home/kharelbikash/wordplay/src/outputs/runs/shakespeare/pytorch/DDP/2024-04-02/01-59-23/wandb/run-20240402_015927-lrcby4bd/logs
Application 1bb8e45c resources: utime=188s stime=106s maxrss=3534612KB inblock=1629772 oublock=506784 minflt=4876385 majflt=0 nvcsw=101072 nivcsw=27937
