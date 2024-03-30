# HW 7 log files with some comments
## 1. Cerebras
This is with batch size of 2048:
```2024-03-30 12:53:41,263 INFO:   Post-layout optimizations...
2024-03-30 12:53:49,248 INFO:   Allocating buffers...
2024-03-30 12:53:51,845 INFO:   Code generation...
2024-03-30 12:54:06,871 INFO:   Compiling image...
2024-03-30 12:54:06,877 INFO:   Compiling kernels
2024-03-30 12:56:06,285 INFO:   Compiling final image
2024-03-30 12:58:27,932 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_12842439843636108263
2024-03-30 12:58:27,987 INFO:   Heartbeat thread stopped for wsjob-gcvn73ljr37qrv2t7mozfe.
2024-03-30 12:58:27,990 INFO:   Compile was successful!
2024-03-30 12:58:27,996 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2024-03-30 12:58:30,684 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-30 12:58:31,041 INFO:   Initiating a new execute wsjob against the cluster server.
2024-03-30 12:58:31,185 INFO:   execute job id: wsjob-6prysnbkfzkwxptnh8vgzc, remote log path: /n1/wsjob/workdir/job-operator/wsjob-6prysnbkfzkwxptnh8vgzc
2024-03-30 12:58:41,234 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job is scheduled.
2024-03-30 12:59:01,223 INFO:   Poll ingress status: Waiting for job service readiness.
2024-03-30 12:59:11,240 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-03-30 12:59:11,407 INFO:   Preparing to execute using 1 CSX
2024-03-30 12:59:40,483 INFO:   About to send initial weights
2024-03-30 13:00:14,777 INFO:   Finished sending initial weights
2024-03-30 13:00:14,779 INFO:   Finalizing appliance staging for the run
2024-03-30 13:00:14,829 INFO:   Waiting for device programming to complete
2024-03-30 13:02:17,551 INFO:   Device programming is complete
2024-03-30 13:02:18,450 INFO:   Using network type: ROCE
2024-03-30 13:02:18,451 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2024-03-30 13:02:18,502 INFO:   Input workers have begun streaming input data
2024-03-30 13:02:35,586 INFO:   Appliance staging is complete
2024-03-30 13:02:35,591 INFO:   Beginning appliance run
2024-03-30 13:03:05,540 INFO:   | Train Device=CSX, Step=100, Loss=9.48438, Rate=6855.28 samples/sec, GlobalRate=6855.28 samples/sec
2024-03-30 13:03:36,111 INFO:   | Train Device=CSX, Step=200, Loss=8.48438, Rate=6761.58 samples/sec, GlobalRate=6776.30 samples/sec
2024-03-30 13:04:06,674 INFO:   | Train Device=CSX, Step=300, Loss=7.77344, Rate=6725.18 samples/sec, GlobalRate=6750.98 samples/sec
2024-03-30 13:04:36,976 INFO:   | Train Device=CSX, Step=400, Loss=7.64062, Rate=6745.20 samples/sec, GlobalRate=6752.87 samples/sec
2024-03-30 13:05:07,364 INFO:   | Train Device=CSX, Step=500, Loss=7.37500, Rate=6741.85 samples/sec, GlobalRate=6750.22 samples/sec
2024-03-30 13:05:38,101 INFO:   | Train Device=CSX, Step=600, Loss=7.42188, Rate=6694.44 samples/sec, GlobalRate=6735.49 samples/sec
2024-03-30 13:06:08,670 INFO:   | Train Device=CSX, Step=700, Loss=7.25000, Rate=6697.53 samples/sec, GlobalRate=6730.34 samples/sec
2024-03-30 13:06:39,101 INFO:   | Train Device=CSX, Step=800, Loss=7.12500, Rate=6717.01 samples/sec, GlobalRate=6730.30 samples/sec
2024-03-30 13:07:09,549 INFO:   | Train Device=CSX, Step=900, Loss=7.25000, Rate=6722.62 samples/sec, GlobalRate=6729.86 samples/sec
2024-03-30 13:07:40,068 INFO:   | Train Device=CSX, Step=1000, Loss=7.14844, Rate=6715.34 samples/sec, GlobalRate=6727.92 samples/sec
2024-03-30 13:07:40,069 INFO:   Saving checkpoint at step 1000
2024-03-30 13:08:14,560 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_1000.mdl
2024-03-30 13:09:10,595 INFO:   Heartbeat thread stopped for wsjob-6prysnbkfzkwxptnh8vgzc.
2024-03-30 13:09:10,602 INFO:   Training completed successfully!
2024-03-30 13:09:10,602 INFO:   Processed 2048000 sample(s) in 304.403202469 seconds.
```
## 2. Graphcore
I trained the MNIST dataset with following hyperparameters

learning_rate = 0.01

epochs = 50

batch_size = 16

test_batch_size = 32

```Epochs: 100%|██████████| 10/10 [01:48<00:00, 10.81s/it]                         Graph compilation: 100%|██████████| 100/100 [00:14<00:00]
 94%|███████�TrainingModelWithLoss(00, 14.18it/s]<00:02]
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.72%
```
