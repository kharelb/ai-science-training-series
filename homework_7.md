# HW 7 log files with some comments
## 1. Cerebras
This is with batch size of 512:
```(venv_cerebras_pt) (base) [kharelbikash@cer-login-02 bert]$ cat mytest.log
2024-03-30 14:09:43,663 INFO:   Effective batch size is 512.
2024-03-30 14:09:43,689 INFO:   Checkpoint autoloading is enabled. Looking for latest checkpoint in "model_dir_bert_large_pytorch" directory with the following naming convention: `checkpoint_(step)(_timestamp)?.mdl`.
2024-03-30 14:09:43,690 INFO:   No checkpoints were found in "model_dir_bert_large_pytorch".
2024-03-30 14:09:43,690 INFO:   No checkpoint was provided. Using randomly initialized model parameters.
2024-03-30 14:09:45,001 INFO:   Saving checkpoint at step 0
2024-03-30 14:10:12,350 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_0.mdl
2024-03-30 14:10:26,616 INFO:   Compiling the model. This may take a few minutes.
2024-03-30 14:10:26,617 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-30 14:10:27,959 INFO:   Initiating a new image build job against the cluster server.
2024-03-30 14:10:28,070 INFO:   Custom worker image build is disabled from server.
2024-03-30 14:10:28,078 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-30 14:10:28,417 INFO:   Initiating a new compile wsjob against the cluster server.
2024-03-30 14:10:28,536 INFO:   compile job id: wsjob-n3ncvfxvasxkabs2kgywrv, remote log path: /n1/wsjob/workdir/job-operator/wsjob-n3ncvfxvasxkabs2kgywrv
2024-03-30 14:10:38,581 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: current job is top of queue but likely blocked by running jobs, 1 execute job(s) running using 1 system(s), 1 compile job(s) running using 67Gi memory. For more information, please run 'csctl get jobs'.
2024-03-30 14:13:58,669 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job is scheduled.
2024-03-30 14:14:08,696 INFO:   Poll ingress status: Waiting for job service readiness.
2024-03-30 14:14:28,718 INFO:   Poll ingress status: Waiting for job ingress readiness.
2024-03-30 14:14:48,744 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-03-30 14:14:52,696 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_8939750200954608837
2024-03-30 14:14:52,701 INFO:   Heartbeat thread stopped for wsjob-n3ncvfxvasxkabs2kgywrv.
2024-03-30 14:14:52,703 INFO:   Compile was successful!
2024-03-30 14:14:52,707 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2024-03-30 14:14:54,767 INFO:   Defaulted to use the job-operator namespace as the usernode config /opt/cerebras/config_v2 only has access to that namespace.
2024-03-30 14:14:55,123 INFO:   Initiating a new execute wsjob against the cluster server.
2024-03-30 14:14:55,254 INFO:   execute job id: wsjob-awx6fid64f4fzt6gfqmvdv, remote log path: /n1/wsjob/workdir/job-operator/wsjob-awx6fid64f4fzt6gfqmvdv
2024-03-30 14:15:05,301 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: 1 execute job(s) queued before current job. For more information, please run 'csctl get jobs'.
2024-03-30 14:45:06,950 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: 1 execute job(s) queued before current job. For more information, please run 'csctl get jobs'.
2024-03-30 15:03:48,015 INFO:   Poll ingress status: Waiting for job running, current job status: Queueing, msg: job is queueing. Job queue status: current job is top of queue but likely blocked by running jobs, 1 execute job(s) running using 1 system(s). For more information, please run 'csctl get jobs'.
2024-03-30 15:11:58,462 INFO:   Poll ingress status: Waiting for job running, current job status: Scheduled, msg: job is scheduled.
2024-03-30 15:12:08,481 INFO:   Poll ingress status: Waiting for job service readiness.
2024-03-30 15:12:28,520 INFO:   Poll ingress status: Waiting for job ingress readiness.
2024-03-30 15:12:48,565 INFO:   Ingress is ready: Job ingress ready, poll ingress success.
2024-03-30 15:12:48,719 INFO:   Preparing to execute using 1 CSX
2024-03-30 15:13:18,513 INFO:   About to send initial weights
2024-03-30 15:13:52,109 INFO:   Finished sending initial weights
2024-03-30 15:13:52,112 INFO:   Finalizing appliance staging for the run
2024-03-30 15:13:52,133 INFO:   Waiting for device programming to complete
2024-03-30 15:15:58,853 INFO:   Device programming is complete
2024-03-30 15:15:59,793 INFO:   Using network type: ROCE
2024-03-30 15:15:59,794 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2024-03-30 15:15:59,816 INFO:   Input workers have begun streaming input data
2024-03-30 15:16:16,586 INFO:   Appliance staging is complete
2024-03-30 15:16:16,590 INFO:   Beginning appliance run
2024-03-30 15:16:33,873 INFO:   | Train Device=CSX, Step=100, Loss=9.39062, Rate=2983.97 samples/sec, GlobalRate=2983.97 samples/sec
2024-03-30 15:16:51,449 INFO:   | Train Device=CSX, Step=200, Loss=8.70312, Rate=2941.41 samples/sec, GlobalRate=2948.08 samples/sec
2024-03-30 15:17:08,962 INFO:   | Train Device=CSX, Step=300, Loss=7.79688, Rate=2930.67 samples/sec, GlobalRate=2939.84 samples/sec
2024-03-30 15:17:26,398 INFO:   | Train Device=CSX, Step=400, Loss=7.39062, Rate=2934.21 samples/sec, GlobalRate=2939.02 samples/sec
2024-03-30 15:17:43,810 INFO:   | Train Device=CSX, Step=500, Loss=7.80469, Rate=2937.94 samples/sec, GlobalRate=2939.30 samples/sec
2024-03-30 15:18:01,430 INFO:   | Train Device=CSX, Step=600, Loss=7.53125, Rate=2918.68 samples/sec, GlobalRate=2933.67 samples/sec
2024-03-30 15:18:18,968 INFO:   | Train Device=CSX, Step=700, Loss=7.35156, Rate=2919.05 samples/sec, GlobalRate=2931.61 samples/sec
2024-03-30 15:18:36,572 INFO:   | Train Device=CSX, Step=800, Loss=7.27344, Rate=2912.68 samples/sec, GlobalRate=2928.69 samples/sec
2024-03-30 15:18:54,039 INFO:   | Train Device=CSX, Step=900, Loss=7.35938, Rate=2923.83 samples/sec, GlobalRate=2928.98 samples/sec
2024-03-30 15:19:11,785 INFO:   | Train Device=CSX, Step=1000, Loss=7.12500, Rate=2900.63 samples/sec, GlobalRate=2924.54 samples/sec
2024-03-30 15:19:11,786 INFO:   Saving checkpoint at step 1000
2024-03-30 15:19:47,127 INFO:   Saved checkpoint model_dir_bert_large_pytorch/checkpoint_1000.mdl
2024-03-30 15:20:24,230 INFO:   Heartbeat thread stopped for wsjob-awx6fid64f4fzt6gfqmvdv.
2024-03-30 15:20:24,238 INFO:   Training completed successfully!
2024-03-30 15:20:24,238 INFO:   Processed 512000 sample(s) in 175.070497198 seconds.
```

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
__So upon increasing the batch size from 512 to 2048 the processing time went from 175 seconds to 304 seconds in the Cerebras.__
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

## 3. Groq
**I modified the bert_tiny.py to accept custom input. Below is what I got running the code**
```(groqflow) kharelbikash@groq-r01-gn-08:~/groqflow/proof_points/natural_language_processing/bert$ python bert_tiny.py --custom_input "What are the Highest and Lowest point on Earth?"
/home/kharelbikash/miniconda3/envs/groqflow/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()

Woohoo! Build "bert_tiny" (build_name auto-selected) found in cache. Loading it!
Preprocessing data.
/home/kharelbikash/.local/lib/python3.10/site-packages/datasets/load.py:1461: FutureWarning: The repository for sst contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/sst
You can avoid this message in future by passing the argument `trust_remote_code=True`.
Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
  warnings.warn(

Info: No inputs received for benchmark. Using the inputs provided during model compilation.
Running inference on GroqChip.
Running inference using PyTorch model (CPU).
100%|█████████████████████████████████████████████████████████████████████████████████████████| 2210/2210 [00:04<00:00, 449.23it/s]
+--------+----------+-------------------------+----------------+----------------------+-------------+
| Source | Accuracy | end-to-end latency (ms) | end-to-end IPS | on-chip latency (ms) | on-chip IPS |
+--------+----------+-------------------------+----------------+----------------------+-------------+
|  cpu   |  77.47%  |           2.23          |     448.91     |          --          |      --     |
|  groq  |  77.47%  |           0.05          |    19328.60    |         0.03         |   37576.72  |
+--------+----------+-------------------------+----------------+----------------------+-------------+
Proof point /home/kharelbikash/groqflow/proof_points/natural_language_processing/bert/bert_tiny.py finished!
(groqflow) kharelbikash@groq-r01-gn-08:~/groqflow/proof_points/natural_language_processing/bert$
```

## 4. Sambanova
```2024-03-30 22:32:25,359 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1574467 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])

Machine state After:
Platform: DataScale SN30-8

Physical Inventory:
Component Name                        | Serial Number       | Inventory State | Functional State
------------------------------------------------------------------------------------------------
/NODE/XRDU_0/RDU_0                    | 306004356D2D5895    | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_0/DIMM_A0    | 1F6F5C2             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_1/DIMM_B0    | 1F6F638             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_2/DIMM_E0    | 1F6F859             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_3/DIMM_F0    | 1F6F59A             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_4/DIMM_G0    | 1F6F59B             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_5/DIMM_H0    | 1F6F8E3             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_6/DIMM_C0    | 1F6F637             | Present         | Online
/NODE/XRDU_0/RDU_0/DDRCH_7/DIMM_D0    | 1F6F598             | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_0/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1                    | 104838356D2D5895    | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_0/DIMM_J0    | 1F6F878             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_1/DIMM_K0    | 1F6F6BE             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_2/DIMM_N0    | 1F6F5AD             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_3/DIMM_P0    | 1F6F615             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_4/DIMM_Q0    | 1F6F5A1             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_5/DIMM_R0    | 1F6F8A8             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_6/DIMM_L0    | 1F6F5D9             | Present         | Online
/NODE/XRDU_0/RDU_1/DDRCH_7/DIMM_M0    | 1F6F5E1             | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_0/RDU_1/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0                     | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_0              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_1              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_2              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_3              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_4              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_5              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_6              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_7              | N/A                 | Present         | Online
/NODE/XRDU_0/SW_0/PORT_8              | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0                    | 304812B16ABDB895    | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_0/DIMM_A0    | 1F5BD2F             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_1/DIMM_B0    | 1F5BBA1             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_2/DIMM_E0    | 1F5BB51             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_3/DIMM_F0    | 1F5BB89             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_4/DIMM_G0    | 1F5BCE7             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_5/DIMM_H0    | 1F5BBC9             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_6/DIMM_C0    | 1F5BB7B             | Present         | Online
/NODE/XRDU_1/RDU_0/DDRCH_7/DIMM_D0    | 1F5BB40             | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_0/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1                    | 20483AB16ABDB895    | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_0/DIMM_J0    | 1F5BD23             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_1/DIMM_K0    | 1F5BD1F             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_2/DIMM_N0    | 1F5BD13             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_3/DIMM_P0    | 1F5BC9E             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_4/DIMM_Q0    | 1F5BB3F             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_5/DIMM_R0    | 1F5BD72             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_6/DIMM_L0    | 1F5BD30             | Present         | Online
/NODE/XRDU_1/RDU_1/DDRCH_7/DIMM_M0    | 1F5BD7D             | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_1/RDU_1/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0                     | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_0              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_1              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_2              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_3              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_4              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_5              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_6              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_7              | N/A                 | Present         | Online
/NODE/XRDU_1/SW_0/PORT_8              | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0                    | 400842B16ABDB895    | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_0/DIMM_A0    | 1F5BC0A             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_1/DIMM_B0    | 1F5BCB4             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_2/DIMM_E0    | 1F5BCAF             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_3/DIMM_F0    | 1F5BC1C             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_4/DIMM_G0    | 1F5BC0B             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_5/DIMM_H0    | 1F5BBB3             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_6/DIMM_C0    | 1F5BC1B             | Present         | Online
/NODE/XRDU_2/RDU_0/DDRCH_7/DIMM_D0    | 1F5BCCE             | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_0/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1                    | 506042B16ABDB895    | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_0/DIMM_J0    | 1F5BCF4             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_1/DIMM_K0    | 1F5BBD8             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_2/DIMM_N0    | 1F5BBCA             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_3/DIMM_P0    | 1F5BBC5             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_4/DIMM_Q0    | 1F5BDBB             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_5/DIMM_R0    | 1F5BBCB             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_6/DIMM_L0    | 1F5BBE8             | Present         | Online
/NODE/XRDU_2/RDU_1/DDRCH_7/DIMM_M0    | 1F5BD0D             | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_2/RDU_1/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0                     | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_0              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_1              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_2              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_3              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_4              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_5              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_6              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_7              | N/A                 | Present         | Online
/NODE/XRDU_2/SW_0/PORT_8              | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0                    | 2050187469B35895    | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_0/DIMM_A0    | 1F5BB9D             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_1/DIMM_B0    | 1F5BC37             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_2/DIMM_E0    | 1F5BC7B             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_3/DIMM_F0    | 1F5BDBF             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_4/DIMM_G0    | 1F5BAC1             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_5/DIMM_H0    | 1F5BC01             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_6/DIMM_C0    | 1F5BC2B             | Present         | Online
/NODE/XRDU_3/RDU_0/DDRCH_7/DIMM_D0    | 1F5BCAE             | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_0/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1                    | 30600C7469B35895    | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_0/DIMM_J0    | 1F5BD67             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_1/DIMM_K0    | 1F5BBB4             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_2/DIMM_N0    | 1F5BB8A             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_3/DIMM_P0    | 1F5BB6D             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_4/DIMM_Q0    | 1F5BC32             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_5/DIMM_R0    | 1F5BC2C             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_6/DIMM_L0    | 1F5BD4B             | Present         | Online
/NODE/XRDU_3/RDU_1/DDRCH_7/DIMM_M0    | 1F5BC8A             | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_0             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_1             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_2             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_3             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_4             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/PCIE_5             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_0             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_1             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_2             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_3             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_4             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_5             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_6             | N/A                 | Present         | Online
/NODE/XRDU_3/RDU_1/TILE_7             | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0                     | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_0              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_1              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_2              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_3              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_4              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_5              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_6              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_7              | N/A                 | Present         | Online
/NODE/XRDU_3/SW_0/PORT_8              | N/A                 | Present         | Online
/NODE/HOST/HIC_0/DPORT                | N/A                 | Present         | Online
/NODE/HOST/HIC_1/DPORT                | N/A                 | Present         | Online
/NODE/HOST/HIC_2/DPORT                | N/A                 | Present         | Online
/NODE/HOST/HIC_3/DPORT                | N/A                 | Present         | Online
Duration:  486
```
**After changing the values of ntasks**
```Iteration:  13%|█▎        | 321/2552 [05:02<32:00,  1.16it/s]2024-04-06 18:32:42,002 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:322|global_step:322|average_loss:9.01663|step_loss:9.01663|step_ns_loss:0.62694|step_mlm_loss:8.38970|learning_rate:4.49e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 322/2552 [05:03<31:57,  1.16it/s]2024-04-06 18:32:42,860 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:323|global_step:323|average_loss:8.90979|step_loss:8.90979|step_ns_loss:0.61203|step_mlm_loss:8.29776|learning_rate:4.51e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 323/2552 [05:04<31:55,  1.16it/s]2024-04-06 18:32:43,722 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:324|global_step:324|average_loss:8.98399|step_loss:8.98399|step_ns_loss:0.63829|step_mlm_loss:8.34570|learning_rate:4.52e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 324/2552 [05:05<31:56,  1.16it/s]2024-04-06 18:32:44,582 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:325|global_step:325|average_loss:8.95036|step_loss:8.95036|step_ns_loss:0.60815|step_mlm_loss:8.34221|learning_rate:4.54e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 325/2552 [05:06<31:55,  1.16it/s]2024-04-06 18:32:45,442 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:326|global_step:326|average_loss:8.97083|step_loss:8.97083|step_ns_loss:0.64554|step_mlm_loss:8.32529|learning_rate:4.55e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 326/2552 [05:07<31:54,  1.16it/s]2024-04-06 18:32:46,302 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:327|global_step:327|average_loss:8.88871|step_loss:8.88871|step_ns_loss:0.61278|step_mlm_loss:8.27593|learning_rate:4.56e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 327/2552 [05:08<31:53,  1.16it/s]2024-04-06 18:32:47,163 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:328|global_step:328|average_loss:8.97002|step_loss:8.97002|step_ns_loss:0.65581|step_mlm_loss:8.31422|learning_rate:4.58e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 328/2552 [05:08<31:53,  1.16it/s]2024-04-06 18:32:48,022 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:329|global_step:329|average_loss:9.01366|step_loss:9.01366|step_ns_loss:0.65236|step_mlm_loss:8.36130|learning_rate:4.59e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 329/2552 [05:09<31:52,  1.16it/s]2024-04-06 18:32:48,885 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:330|global_step:330|average_loss:8.96154|step_loss:8.96154|step_ns_loss:0.65224|step_mlm_loss:8.30931|learning_rate:4.61e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 330/2552 [05:10<31:52,  1.16it/s]2024-04-06 18:32:49,746 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:331|global_step:331|average_loss:8.95811|step_loss:8.95811|step_ns_loss:0.67087|step_mlm_loss:8.28724|learning_rate:4.62e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 331/2552 [05:11<31:52,  1.16it/s]2024-04-06 18:32:50,606 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:332|global_step:332|average_loss:8.88261|step_loss:8.88261|step_ns_loss:0.62520|step_mlm_loss:8.25741|learning_rate:4.63e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 332/2552 [05:12<31:50,  1.16it/s]2024-04-06 18:32:51,465 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:333|global_step:333|average_loss:8.89764|step_loss:8.89764|step_ns_loss:0.63345|step_mlm_loss:8.26419|learning_rate:4.65e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 333/2552 [05:13<31:48,  1.16it/s]2024-04-06 18:32:52,325 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:334|global_step:334|average_loss:8.99843|step_loss:8.99843|step_ns_loss:0.66168|step_mlm_loss:8.33675|learning_rate:4.66e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 334/2552 [05:14<31:47,  1.16it/s]2024-04-06 18:32:53,185 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:335|global_step:335|average_loss:8.93142|step_loss:8.93142|step_ns_loss:0.63901|step_mlm_loss:8.29241|learning_rate:4.68e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 335/2552 [05:14<31:46,  1.16it/s]2024-04-06 18:32:54,043 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:336|global_step:336|average_loss:8.96375|step_loss:8.96375|step_ns_loss:0.65129|step_mlm_loss:8.31246|learning_rate:4.69e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 336/2552 [05:15<31:45,  1.16it/s]2024-04-06 18:32:54,904 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:337|global_step:337|average_loss:8.96953|step_loss:8.96953|step_ns_loss:0.64105|step_mlm_loss:8.32847|learning_rate:4.70e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 337/2552 [05:16<31:44,  1.16it/s]2024-04-06 18:32:55,765 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:338|global_step:338|average_loss:8.86808|step_loss:8.86808|step_ns_loss:0.61803|step_mlm_loss:8.25005|learning_rate:4.72e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 338/2552 [05:17<31:44,  1.16it/s]2024-04-06 18:32:56,624 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:339|global_step:339|average_loss:8.94030|step_loss:8.94030|step_ns_loss:0.64615|step_mlm_loss:8.29415|learning_rate:4.73e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 339/2552 [05:18<31:42,  1.16it/s]2024-04-06 18:32:57,485 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:340|global_step:340|average_loss:8.83698|step_loss:8.83698|step_ns_loss:0.63463|step_mlm_loss:8.20235|learning_rate:4.75e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 340/2552 [05:19<31:42,  1.16it/s]2024-04-06 18:32:58,342 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:341|global_step:341|average_loss:8.94318|step_loss:8.94318|step_ns_loss:0.66205|step_mlm_loss:8.28114|learning_rate:4.76e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 341/2552 [05:20<31:40,  1.16it/s]2024-04-06 18:32:59,202 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:342|global_step:342|average_loss:8.80947|step_loss:8.80947|step_ns_loss:0.63713|step_mlm_loss:8.17234|learning_rate:4.77e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 342/2552 [05:20<31:39,  1.16it/s]2024-04-06 18:33:00,061 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:343|global_step:343|average_loss:8.89256|step_loss:8.89256|step_ns_loss:0.59799|step_mlm_loss:8.29457|learning_rate:4.79e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 343/2552 [05:21<31:38,  1.16it/s]2024-04-06 18:33:00,921 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:344|global_step:344|average_loss:8.92776|step_loss:8.92776|step_ns_loss:0.65387|step_mlm_loss:8.27389|learning_rate:4.80e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  13%|█▎        | 344/2552 [05:22<31:37,  1.16it/s]2024-04-06 18:33:01,778 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:345|global_step:345|average_loss:8.91226|step_loss:8.91226|step_ns_loss:0.63997|step_mlm_loss:8.27229|learning_rate:4.82e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 345/2552 [05:23<31:35,  1.16it/s]2024-04-06 18:33:02,636 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:346|global_step:346|average_loss:8.96509|step_loss:8.96509|step_ns_loss:0.65387|step_mlm_loss:8.31122|learning_rate:4.83e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 346/2552 [05:24<31:34,  1.16it/s]2024-04-06 18:33:03,497 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:347|global_step:347|average_loss:8.89635|step_loss:8.89635|step_ns_loss:0.65378|step_mlm_loss:8.24257|learning_rate:4.84e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 347/2552 [05:25<31:34,  1.16it/s]2024-04-06 18:33:04,357 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:348|global_step:348|average_loss:8.88080|step_loss:8.88080|step_ns_loss:0.64833|step_mlm_loss:8.23247|learning_rate:4.86e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 348/2552 [05:26<31:33,  1.16it/s]2024-04-06 18:33:05,215 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:349|global_step:349|average_loss:8.93255|step_loss:8.93255|step_ns_loss:0.67650|step_mlm_loss:8.25605|learning_rate:4.87e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 349/2552 [05:26<31:32,  1.16it/s]2024-04-06 18:33:06,076 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:350|global_step:350|average_loss:8.82658|step_loss:8.82658|step_ns_loss:0.64186|step_mlm_loss:8.18472|learning_rate:4.89e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▎        | 350/2552 [05:27<31:33,  1.16it/s]2024-04-06 18:33:06,938 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:351|global_step:351|average_loss:8.89328|step_loss:8.89328|step_ns_loss:0.66505|step_mlm_loss:8.22822|learning_rate:4.90e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 351/2552 [05:28<31:33,  1.16it/s]2024-04-06 18:33:07,797 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:352|global_step:352|average_loss:8.82935|step_loss:8.82935|step_ns_loss:0.63321|step_mlm_loss:8.19615|learning_rate:4.91e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 352/2552 [05:29<31:32,  1.16it/s]2024-04-06 18:33:08,657 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:353|global_step:353|average_loss:8.87562|step_loss:8.87562|step_ns_loss:0.64551|step_mlm_loss:8.23011|learning_rate:4.93e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 353/2552 [05:30<31:31,  1.16it/s]2024-04-06 18:33:09,516 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:354|global_step:354|average_loss:8.82990|step_loss:8.82990|step_ns_loss:0.63815|step_mlm_loss:8.19175|learning_rate:4.94e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 354/2552 [05:31<31:29,  1.16it/s]2024-04-06 18:33:10,376 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:355|global_step:355|average_loss:8.89385|step_loss:8.89385|step_ns_loss:0.64249|step_mlm_loss:8.25137|learning_rate:4.96e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 355/2552 [05:32<31:28,  1.16it/s]2024-04-06 18:33:11,234 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:356|global_step:356|average_loss:8.83471|step_loss:8.83471|step_ns_loss:0.64027|step_mlm_loss:8.19443|learning_rate:4.97e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 356/2552 [05:32<31:26,  1.16it/s]2024-04-06 18:33:12,092 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:357|global_step:357|average_loss:8.86383|step_loss:8.86383|step_ns_loss:0.64053|step_mlm_loss:8.22329|learning_rate:4.98e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 357/2552 [05:33<31:25,  1.16it/s]2024-04-06 18:33:12,952 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:358|global_step:358|average_loss:8.87275|step_loss:8.87275|step_ns_loss:0.62823|step_mlm_loss:8.24452|learning_rate:5.00e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 358/2552 [05:34<31:24,  1.16it/s]2024-04-06 18:33:13,810 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:359|global_step:359|average_loss:8.73931|step_loss:8.73931|step_ns_loss:0.63512|step_mlm_loss:8.10419|learning_rate:5.01e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 359/2552 [05:35<31:23,  1.16it/s]2024-04-06 18:33:14,667 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:360|global_step:360|average_loss:8.79603|step_loss:8.79603|step_ns_loss:0.60649|step_mlm_loss:8.18954|learning_rate:5.03e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 360/2552 [05:36<31:21,  1.16it/s]2024-04-06 18:33:15,526 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:361|global_step:361|average_loss:8.81814|step_loss:8.81814|step_ns_loss:0.62061|step_mlm_loss:8.19754|learning_rate:5.04e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 361/2552 [05:37<31:20,  1.16it/s]2024-04-06 18:33:16,384 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:362|global_step:362|average_loss:8.76583|step_loss:8.76583|step_ns_loss:0.61470|step_mlm_loss:8.15113|learning_rate:5.05e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 362/2552 [05:38<31:19,  1.17it/s]2024-04-06 18:33:17,243 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:363|global_step:363|average_loss:8.84680|step_loss:8.84680|step_ns_loss:0.64624|step_mlm_loss:8.20056|learning_rate:5.07e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 363/2552 [05:38<31:19,  1.16it/s]2024-04-06 18:33:18,104 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:364|global_step:364|average_loss:8.75860|step_loss:8.75860|step_ns_loss:0.64453|step_mlm_loss:8.11407|learning_rate:5.08e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 364/2552 [05:39<31:20,  1.16it/s]2024-04-06 18:33:18,964 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:365|global_step:365|average_loss:8.83042|step_loss:8.83042|step_ns_loss:0.62619|step_mlm_loss:8.20423|learning_rate:5.10e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 365/2552 [05:40<31:19,  1.16it/s]2024-04-06 18:33:19,823 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:366|global_step:366|average_loss:8.77526|step_loss:8.77526|step_ns_loss:0.64058|step_mlm_loss:8.13469|learning_rate:5.11e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 366/2552 [05:41<31:19,  1.16it/s]2024-04-06 18:33:20,685 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:367|global_step:367|average_loss:8.79147|step_loss:8.79147|step_ns_loss:0.62865|step_mlm_loss:8.16283|learning_rate:5.12e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 367/2552 [05:42<31:20,  1.16it/s]2024-04-06 18:33:21,544 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:368|global_step:368|average_loss:8.85160|step_loss:8.85160|step_ns_loss:0.62820|step_mlm_loss:8.22340|learning_rate:5.14e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 368/2552 [05:43<31:18,  1.16it/s]2024-04-06 18:33:22,405 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:369|global_step:369|average_loss:8.73718|step_loss:8.73718|step_ns_loss:0.63826|step_mlm_loss:8.09892|learning_rate:5.15e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 369/2552 [05:44<31:17,  1.16it/s]2024-04-06 18:33:23,264 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:370|global_step:370|average_loss:8.85284|step_loss:8.85284|step_ns_loss:0.66983|step_mlm_loss:8.18301|learning_rate:5.17e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  14%|█▍        | 370/2552 [05:44<31:15,  1.16it/s]2024-04-06 18:33:24,123 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:371|global_step:371|average_loss:8.83007|step_loss:8.83007|step_ns_loss:0.65286|step_mlm_loss:8.17721|learning_rate:5.18e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 371/2552 [05:45<31:14,  1.16it/s]2024-04-06 18:33:24,981 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:372|global_step:372|average_loss:8.87085|step_loss:8.87085|step_ns_loss:0.67833|step_mlm_loss:8.19252|learning_rate:5.19e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 372/2552 [05:46<31:12,  1.16it/s]2024-04-06 18:33:25,839 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:373|global_step:373|average_loss:8.82363|step_loss:8.82363|step_ns_loss:0.63371|step_mlm_loss:8.18991|learning_rate:5.21e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 373/2552 [05:47<31:11,  1.16it/s]2024-04-06 18:33:26,813 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:374|global_step:374|average_loss:8.82711|step_loss:8.82711|step_ns_loss:0.65257|step_mlm_loss:8.17454|learning_rate:5.22e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 374/2552 [05:48<32:25,  1.12it/s]2024-04-06 18:33:27,674 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:375|global_step:375|average_loss:8.64158|step_loss:8.64158|step_ns_loss:0.58522|step_mlm_loss:8.05636|learning_rate:5.24e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 375/2552 [05:49<32:02,  1.13it/s]2024-04-06 18:33:28,532 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:376|global_step:376|average_loss:8.77538|step_loss:8.77538|step_ns_loss:0.63812|step_mlm_loss:8.13726|learning_rate:5.25e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 376/2552 [05:50<31:45,  1.14it/s]2024-04-06 18:33:29,390 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:377|global_step:377|average_loss:8.73461|step_loss:8.73461|step_ns_loss:0.63712|step_mlm_loss:8.09749|learning_rate:5.26e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 377/2552 [05:51<31:33,  1.15it/s]2024-04-06 18:33:30,248 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:378|global_step:378|average_loss:8.80460|step_loss:8.80460|step_ns_loss:0.64444|step_mlm_loss:8.16016|learning_rate:5.28e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 378/2552 [05:51<31:24,  1.15it/s]2024-04-06 18:33:31,107 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:379|global_step:379|average_loss:8.70606|step_loss:8.70606|step_ns_loss:0.64792|step_mlm_loss:8.05814|learning_rate:5.29e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 379/2552 [05:52<31:18,  1.16it/s]2024-04-06 18:33:31,968 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:380|global_step:380|average_loss:8.72469|step_loss:8.72469|step_ns_loss:0.63963|step_mlm_loss:8.08506|learning_rate:5.31e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 380/2552 [05:53<31:15,  1.16it/s]2024-04-06 18:33:32,827 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:381|global_step:381|average_loss:8.74677|step_loss:8.74677|step_ns_loss:0.63190|step_mlm_loss:8.11487|learning_rate:5.32e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 381/2552 [05:54<31:11,  1.16it/s]2024-04-06 18:33:33,687 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:382|global_step:382|average_loss:8.73331|step_loss:8.73331|step_ns_loss:0.66983|step_mlm_loss:8.06348|learning_rate:5.33e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▍        | 382/2552 [05:55<31:09,  1.16it/s]2024-04-06 18:33:34,545 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:383|global_step:383|average_loss:8.76630|step_loss:8.76630|step_ns_loss:0.64510|step_mlm_loss:8.12121|learning_rate:5.35e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 383/2552 [05:56<31:06,  1.16it/s]2024-04-06 18:33:35,403 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:384|global_step:384|average_loss:8.72678|step_loss:8.72678|step_ns_loss:0.64526|step_mlm_loss:8.08152|learning_rate:5.36e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 384/2552 [05:57<31:03,  1.16it/s]2024-04-06 18:33:36,264 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:385|global_step:385|average_loss:8.81939|step_loss:8.81939|step_ns_loss:0.68896|step_mlm_loss:8.13042|learning_rate:5.38e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 385/2552 [05:57<31:03,  1.16it/s]2024-04-06 18:33:37,122 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:386|global_step:386|average_loss:8.67388|step_loss:8.67388|step_ns_loss:0.62469|step_mlm_loss:8.04918|learning_rate:5.39e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 386/2552 [05:58<31:02,  1.16it/s]2024-04-06 18:33:37,981 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:387|global_step:387|average_loss:8.66852|step_loss:8.66852|step_ns_loss:0.62769|step_mlm_loss:8.04084|learning_rate:5.40e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 387/2552 [05:59<31:00,  1.16it/s]2024-04-06 18:33:38,844 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:388|global_step:388|average_loss:8.73388|step_loss:8.73388|step_ns_loss:0.64146|step_mlm_loss:8.09242|learning_rate:5.42e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 388/2552 [06:00<31:02,  1.16it/s]2024-04-06 18:33:39,704 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:389|global_step:389|average_loss:8.68816|step_loss:8.68816|step_ns_loss:0.63567|step_mlm_loss:8.05249|learning_rate:5.43e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 389/2552 [06:01<31:00,  1.16it/s]2024-04-06 18:33:40,563 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:390|global_step:390|average_loss:8.69850|step_loss:8.69850|step_ns_loss:0.64124|step_mlm_loss:8.05726|learning_rate:5.45e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 390/2552 [06:02<30:58,  1.16it/s]2024-04-06 18:33:41,423 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:391|global_step:391|average_loss:8.74665|step_loss:8.74665|step_ns_loss:0.66655|step_mlm_loss:8.08010|learning_rate:5.46e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 391/2552 [06:03<30:58,  1.16it/s]2024-04-06 18:33:42,283 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:392|global_step:392|average_loss:8.65072|step_loss:8.65072|step_ns_loss:0.64676|step_mlm_loss:8.00396|learning_rate:5.47e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 392/2552 [06:04<30:57,  1.16it/s]2024-04-06 18:33:43,142 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:393|global_step:393|average_loss:8.71492|step_loss:8.71492|step_ns_loss:0.67354|step_mlm_loss:8.04138|learning_rate:5.49e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 393/2552 [06:04<30:55,  1.16it/s]2024-04-06 18:33:44,001 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:394|global_step:394|average_loss:8.60166|step_loss:8.60166|step_ns_loss:0.63480|step_mlm_loss:7.96686|learning_rate:5.50e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 394/2552 [06:05<30:54,  1.16it/s]2024-04-06 18:33:44,859 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:395|global_step:395|average_loss:8.63407|step_loss:8.63407|step_ns_loss:0.63475|step_mlm_loss:7.99932|learning_rate:5.52e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  15%|█▌        | 395/2552 [06:06<30:53,  1.16it/s]2024-04-06 18:33:45,718 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:396|global_step:396|average_loss:8.68400|step_loss:8.68400|step_ns_loss:0.66141|step_mlm_loss:8.02259|learning_rate:5.53e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 396/2552 [06:07<30:52,  1.16it/s]2024-04-06 18:33:46,577 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:397|global_step:397|average_loss:8.73810|step_loss:8.73810|step_ns_loss:0.60225|step_mlm_loss:8.13586|learning_rate:5.54e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 397/2552 [06:08<30:51,  1.16it/s]2024-04-06 18:33:47,437 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:398|global_step:398|average_loss:8.65745|step_loss:8.65745|step_ns_loss:0.63708|step_mlm_loss:8.02036|learning_rate:5.56e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 398/2552 [06:09<30:50,  1.16it/s]2024-04-06 18:33:48,470 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:399|global_step:399|average_loss:8.62863|step_loss:8.62863|step_ns_loss:0.61472|step_mlm_loss:8.01392|learning_rate:5.57e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 399/2552 [06:10<32:43,  1.10it/s]2024-04-06 18:33:49,330 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:400|global_step:400|average_loss:8.64783|step_loss:8.64783|step_ns_loss:0.62619|step_mlm_loss:8.02164|learning_rate:5.59e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 400/2552 [06:11<32:07,  1.12it/s]2024-04-06 18:33:50,189 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:401|global_step:401|average_loss:8.72779|step_loss:8.72779|step_ns_loss:0.66527|step_mlm_loss:8.06253|learning_rate:5.60e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 401/2552 [06:11<31:43,  1.13it/s]2024-04-06 18:33:51,050 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:402|global_step:402|average_loss:8.62342|step_loss:8.62342|step_ns_loss:0.64159|step_mlm_loss:7.98183|learning_rate:5.61e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 402/2552 [06:12<31:27,  1.14it/s]2024-04-06 18:33:52,332 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:403|global_step:403|average_loss:8.66224|step_loss:8.66224|step_ns_loss:0.64801|step_mlm_loss:8.01423|learning_rate:5.63e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 403/2552 [06:14<35:46,  1.00it/s]2024-04-06 18:33:53,191 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:404|global_step:404|average_loss:8.64269|step_loss:8.64269|step_ns_loss:0.65016|step_mlm_loss:7.99253|learning_rate:5.64e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 404/2552 [06:14<34:15,  1.04it/s]2024-04-06 18:33:54,054 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:405|global_step:405|average_loss:8.65104|step_loss:8.65104|step_ns_loss:0.65776|step_mlm_loss:7.99328|learning_rate:5.66e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 405/2552 [06:15<33:14,  1.08it/s]2024-04-06 18:33:54,915 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:406|global_step:406|average_loss:8.56855|step_loss:8.56855|step_ns_loss:0.62535|step_mlm_loss:7.94320|learning_rate:5.67e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 406/2552 [06:16<32:29,  1.10it/s]2024-04-06 18:33:55,776 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:407|global_step:407|average_loss:8.65140|step_loss:8.65140|step_ns_loss:0.61798|step_mlm_loss:8.03342|learning_rate:5.68e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 407/2552 [06:17<31:57,  1.12it/s]2024-04-06 18:33:56,637 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:408|global_step:408|average_loss:8.59736|step_loss:8.59736|step_ns_loss:0.60033|step_mlm_loss:7.99704|learning_rate:5.70e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 408/2552 [06:18<31:35,  1.13it/s]2024-04-06 18:33:57,498 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:409|global_step:409|average_loss:8.62664|step_loss:8.62664|step_ns_loss:0.66125|step_mlm_loss:7.96539|learning_rate:5.71e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 409/2552 [06:19<31:19,  1.14it/s]2024-04-06 18:33:58,358 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:410|global_step:410|average_loss:8.66265|step_loss:8.66265|step_ns_loss:0.59502|step_mlm_loss:8.06764|learning_rate:5.73e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 410/2552 [06:20<31:08,  1.15it/s]2024-04-06 18:33:59,223 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:411|global_step:411|average_loss:8.61987|step_loss:8.61987|step_ns_loss:0.65678|step_mlm_loss:7.96309|learning_rate:5.74e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 411/2552 [06:20<31:02,  1.15it/s]2024-04-06 18:34:00,083 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:412|global_step:412|average_loss:8.65427|step_loss:8.65427|step_ns_loss:0.65627|step_mlm_loss:7.99801|learning_rate:5.75e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 412/2552 [06:21<30:55,  1.15it/s]2024-04-06 18:34:00,942 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:413|global_step:413|average_loss:8.56672|step_loss:8.56672|step_ns_loss:0.60406|step_mlm_loss:7.96266|learning_rate:5.77e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 413/2552 [06:22<30:49,  1.16it/s]2024-04-06 18:34:01,802 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:414|global_step:414|average_loss:8.67527|step_loss:8.67527|step_ns_loss:0.64101|step_mlm_loss:8.03427|learning_rate:5.78e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▌        | 414/2552 [06:23<30:45,  1.16it/s]2024-04-06 18:34:02,661 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:415|global_step:415|average_loss:8.62011|step_loss:8.62011|step_ns_loss:0.64442|step_mlm_loss:7.97569|learning_rate:5.80e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 415/2552 [06:24<30:41,  1.16it/s]2024-04-06 18:34:03,521 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:416|global_step:416|average_loss:8.61527|step_loss:8.61527|step_ns_loss:0.63881|step_mlm_loss:7.97646|learning_rate:5.81e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 416/2552 [06:25<30:39,  1.16it/s]2024-04-06 18:34:04,381 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:417|global_step:417|average_loss:8.50062|step_loss:8.50062|step_ns_loss:0.63248|step_mlm_loss:7.86815|learning_rate:5.82e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 417/2552 [06:26<30:37,  1.16it/s]2024-04-06 18:34:05,241 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:418|global_step:418|average_loss:8.66304|step_loss:8.66304|step_ns_loss:0.60310|step_mlm_loss:8.05994|learning_rate:5.84e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 418/2552 [06:26<30:36,  1.16it/s]2024-04-06 18:34:06,100 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:419|global_step:419|average_loss:8.54454|step_loss:8.54454|step_ns_loss:0.62549|step_mlm_loss:7.91905|learning_rate:5.85e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 419/2552 [06:27<30:35,  1.16it/s]2024-04-06 18:34:06,959 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:420|global_step:420|average_loss:8.55123|step_loss:8.55123|step_ns_loss:0.62070|step_mlm_loss:7.93053|learning_rate:5.87e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 420/2552 [06:28<30:33,  1.16it/s]2024-04-06 18:34:07,817 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:421|global_step:421|average_loss:8.56890|step_loss:8.56890|step_ns_loss:0.59612|step_mlm_loss:7.97278|learning_rate:5.88e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  16%|█▋        | 421/2552 [06:29<30:31,  1.16it/s]2024-04-06 18:34:08,675 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:422|global_step:422|average_loss:8.56537|step_loss:8.56537|step_ns_loss:0.60062|step_mlm_loss:7.96475|learning_rate:5.89e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 422/2552 [06:30<30:29,  1.16it/s]2024-04-06 18:34:09,535 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:423|global_step:423|average_loss:8.57348|step_loss:8.57348|step_ns_loss:0.62970|step_mlm_loss:7.94378|learning_rate:5.91e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 423/2552 [06:31<30:29,  1.16it/s]2024-04-06 18:34:10,395 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:424|global_step:424|average_loss:8.61730|step_loss:8.61730|step_ns_loss:0.65730|step_mlm_loss:7.96000|learning_rate:5.92e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 424/2552 [06:32<30:28,  1.16it/s]2024-04-06 18:34:11,254 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:425|global_step:425|average_loss:8.61095|step_loss:8.61095|step_ns_loss:0.64702|step_mlm_loss:7.96393|learning_rate:5.94e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 425/2552 [06:32<30:28,  1.16it/s]2024-04-06 18:34:12,112 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:426|global_step:426|average_loss:8.54852|step_loss:8.54852|step_ns_loss:0.65491|step_mlm_loss:7.89362|learning_rate:5.95e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 426/2552 [06:33<30:26,  1.16it/s]2024-04-06 18:34:12,971 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:427|global_step:427|average_loss:8.54466|step_loss:8.54466|step_ns_loss:0.62570|step_mlm_loss:7.91896|learning_rate:5.96e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 427/2552 [06:34<30:25,  1.16it/s]2024-04-06 18:34:13,831 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:428|global_step:428|average_loss:8.58427|step_loss:8.58427|step_ns_loss:0.57715|step_mlm_loss:8.00712|learning_rate:5.98e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 428/2552 [06:35<30:25,  1.16it/s]2024-04-06 18:34:14,692 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:429|global_step:429|average_loss:8.52650|step_loss:8.52650|step_ns_loss:0.63501|step_mlm_loss:7.89149|learning_rate:5.99e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 429/2552 [06:36<30:25,  1.16it/s]2024-04-06 18:34:15,553 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:430|global_step:430|average_loss:8.47665|step_loss:8.47665|step_ns_loss:0.61322|step_mlm_loss:7.86343|learning_rate:6.01e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 430/2552 [06:37<30:25,  1.16it/s]2024-04-06 18:34:16,572 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:431|global_step:431|average_loss:8.49069|step_loss:8.49069|step_ns_loss:0.62003|step_mlm_loss:7.87066|learning_rate:6.02e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 431/2552 [06:38<32:05,  1.10it/s]2024-04-06 18:34:17,430 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:432|global_step:432|average_loss:8.61077|step_loss:8.61077|step_ns_loss:0.65179|step_mlm_loss:7.95898|learning_rate:6.03e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 432/2552 [06:39<31:32,  1.12it/s]2024-04-06 18:34:18,289 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:433|global_step:433|average_loss:8.55148|step_loss:8.55148|step_ns_loss:0.63629|step_mlm_loss:7.91519|learning_rate:6.05e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 433/2552 [06:40<31:10,  1.13it/s]2024-04-06 18:34:19,149 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:434|global_step:434|average_loss:8.49604|step_loss:8.49604|step_ns_loss:0.60654|step_mlm_loss:7.88951|learning_rate:6.06e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 434/2552 [06:40<30:55,  1.14it/s]2024-04-06 18:34:20,010 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:435|global_step:435|average_loss:8.56047|step_loss:8.56047|step_ns_loss:0.62434|step_mlm_loss:7.93612|learning_rate:6.08e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 435/2552 [06:41<30:44,  1.15it/s]2024-04-06 18:34:20,869 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:436|global_step:436|average_loss:8.47728|step_loss:8.47728|step_ns_loss:0.61649|step_mlm_loss:7.86079|learning_rate:6.09e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 436/2552 [06:42<30:35,  1.15it/s]2024-04-06 18:34:21,728 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:437|global_step:437|average_loss:8.46177|step_loss:8.46177|step_ns_loss:0.63770|step_mlm_loss:7.82407|learning_rate:6.10e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 437/2552 [06:43<30:29,  1.16it/s]2024-04-06 18:34:22,587 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:438|global_step:438|average_loss:8.57538|step_loss:8.57538|step_ns_loss:0.64413|step_mlm_loss:7.93124|learning_rate:6.12e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 438/2552 [06:44<30:24,  1.16it/s]2024-04-06 18:34:23,445 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:439|global_step:439|average_loss:8.56848|step_loss:8.56848|step_ns_loss:0.67003|step_mlm_loss:7.89845|learning_rate:6.13e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 439/2552 [06:45<30:21,  1.16it/s]2024-04-06 18:34:24,304 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:440|global_step:440|average_loss:8.50361|step_loss:8.50361|step_ns_loss:0.63802|step_mlm_loss:7.86560|learning_rate:6.15e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 440/2552 [06:46<30:18,  1.16it/s]2024-04-06 18:34:25,164 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:441|global_step:441|average_loss:8.48429|step_loss:8.48429|step_ns_loss:0.67332|step_mlm_loss:7.81097|learning_rate:6.16e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 441/2552 [06:46<30:16,  1.16it/s]2024-04-06 18:34:26,023 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:442|global_step:442|average_loss:8.46427|step_loss:8.46427|step_ns_loss:0.63237|step_mlm_loss:7.83190|learning_rate:6.17e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 442/2552 [06:47<30:15,  1.16it/s]2024-04-06 18:34:26,883 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:443|global_step:443|average_loss:8.51296|step_loss:8.51296|step_ns_loss:0.67242|step_mlm_loss:7.84053|learning_rate:6.19e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 443/2552 [06:48<30:13,  1.16it/s]2024-04-06 18:34:27,741 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:444|global_step:444|average_loss:8.47863|step_loss:8.47863|step_ns_loss:0.62744|step_mlm_loss:7.85118|learning_rate:6.20e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 444/2552 [06:49<30:11,  1.16it/s]2024-04-06 18:34:28,599 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:445|global_step:445|average_loss:8.51140|step_loss:8.51140|step_ns_loss:0.63202|step_mlm_loss:7.87938|learning_rate:6.22e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 445/2552 [06:50<30:10,  1.16it/s]2024-04-06 18:34:29,457 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:446|global_step:446|average_loss:8.54797|step_loss:8.54797|step_ns_loss:0.64349|step_mlm_loss:7.90448|learning_rate:6.23e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  17%|█▋        | 446/2552 [06:51<30:08,  1.16it/s]2024-04-06 18:34:30,317 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:447|global_step:447|average_loss:8.56537|step_loss:8.56537|step_ns_loss:0.66770|step_mlm_loss:7.89767|learning_rate:6.24e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 447/2552 [06:52<30:08,  1.16it/s]2024-04-06 18:34:31,176 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:448|global_step:448|average_loss:8.56857|step_loss:8.56857|step_ns_loss:0.65660|step_mlm_loss:7.91197|learning_rate:6.26e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 448/2552 [06:52<30:07,  1.16it/s]2024-04-06 18:34:32,036 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:449|global_step:449|average_loss:8.39751|step_loss:8.39751|step_ns_loss:0.62007|step_mlm_loss:7.77744|learning_rate:6.27e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 449/2552 [06:53<30:07,  1.16it/s]2024-04-06 18:34:32,895 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:450|global_step:450|average_loss:8.50851|step_loss:8.50851|step_ns_loss:0.66621|step_mlm_loss:7.84230|learning_rate:6.29e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 450/2552 [06:54<30:06,  1.16it/s]2024-04-06 18:34:33,757 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:451|global_step:451|average_loss:8.50039|step_loss:8.50039|step_ns_loss:0.63544|step_mlm_loss:7.86495|learning_rate:6.30e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 451/2552 [06:55<30:06,  1.16it/s]2024-04-06 18:34:34,619 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:452|global_step:452|average_loss:8.35512|step_loss:8.35512|step_ns_loss:0.62625|step_mlm_loss:7.72887|learning_rate:6.31e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 452/2552 [06:56<30:06,  1.16it/s]2024-04-06 18:34:35,483 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:453|global_step:453|average_loss:8.41097|step_loss:8.41097|step_ns_loss:0.62909|step_mlm_loss:7.78188|learning_rate:6.33e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 453/2552 [06:57<30:08,  1.16it/s]2024-04-06 18:34:36,343 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:454|global_step:454|average_loss:8.49192|step_loss:8.49192|step_ns_loss:0.63147|step_mlm_loss:7.86045|learning_rate:6.34e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 454/2552 [06:58<30:07,  1.16it/s]2024-04-06 18:34:37,202 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:455|global_step:455|average_loss:8.46231|step_loss:8.46231|step_ns_loss:0.61217|step_mlm_loss:7.85014|learning_rate:6.36e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 455/2552 [06:58<30:05,  1.16it/s]2024-04-06 18:34:38,061 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:456|global_step:456|average_loss:8.41525|step_loss:8.41525|step_ns_loss:0.64006|step_mlm_loss:7.77519|learning_rate:6.37e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 456/2552 [06:59<30:03,  1.16it/s]2024-04-06 18:34:38,921 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:457|global_step:457|average_loss:8.37108|step_loss:8.37108|step_ns_loss:0.60169|step_mlm_loss:7.76939|learning_rate:6.38e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 457/2552 [07:00<30:01,  1.16it/s]2024-04-06 18:34:39,781 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:458|global_step:458|average_loss:8.40627|step_loss:8.40627|step_ns_loss:0.64888|step_mlm_loss:7.75739|learning_rate:6.40e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 458/2552 [07:01<30:00,  1.16it/s]2024-04-06 18:34:40,640 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:459|global_step:459|average_loss:8.42521|step_loss:8.42521|step_ns_loss:0.61536|step_mlm_loss:7.80985|learning_rate:6.41e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 459/2552 [07:02<29:58,  1.16it/s]2024-04-06 18:34:41,499 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:460|global_step:460|average_loss:8.37743|step_loss:8.37743|step_ns_loss:0.63582|step_mlm_loss:7.74161|learning_rate:6.43e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 460/2552 [07:03<29:57,  1.16it/s]2024-04-06 18:34:42,357 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:461|global_step:461|average_loss:8.47471|step_loss:8.47471|step_ns_loss:0.62650|step_mlm_loss:7.84821|learning_rate:6.44e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 461/2552 [07:04<29:56,  1.16it/s]2024-04-06 18:34:43,215 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:462|global_step:462|average_loss:8.44705|step_loss:8.44705|step_ns_loss:0.64352|step_mlm_loss:7.80352|learning_rate:6.45e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 462/2552 [07:04<29:54,  1.16it/s]2024-04-06 18:34:44,074 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:463|global_step:463|average_loss:8.48024|step_loss:8.48024|step_ns_loss:0.64061|step_mlm_loss:7.83963|learning_rate:6.47e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 463/2552 [07:05<29:53,  1.16it/s]2024-04-06 18:34:44,933 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:464|global_step:464|average_loss:8.37382|step_loss:8.37382|step_ns_loss:0.63400|step_mlm_loss:7.73982|learning_rate:6.48e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 464/2552 [07:06<29:53,  1.16it/s]2024-04-06 18:34:45,793 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:465|global_step:465|average_loss:8.38564|step_loss:8.38564|step_ns_loss:0.62640|step_mlm_loss:7.75924|learning_rate:6.50e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 465/2552 [07:07<29:53,  1.16it/s]2024-04-06 18:34:46,651 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:466|global_step:466|average_loss:8.43528|step_loss:8.43528|step_ns_loss:0.63676|step_mlm_loss:7.79851|learning_rate:6.51e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 466/2552 [07:08<29:51,  1.16it/s]2024-04-06 18:34:47,511 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:467|global_step:467|average_loss:8.42895|step_loss:8.42895|step_ns_loss:0.59740|step_mlm_loss:7.83156|learning_rate:6.52e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 467/2552 [07:09<29:51,  1.16it/s]2024-04-06 18:34:48,369 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:468|global_step:468|average_loss:8.37291|step_loss:8.37291|step_ns_loss:0.62448|step_mlm_loss:7.74844|learning_rate:6.54e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 468/2552 [07:10<29:49,  1.16it/s]2024-04-06 18:34:49,227 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:469|global_step:469|average_loss:8.30273|step_loss:8.30273|step_ns_loss:0.62823|step_mlm_loss:7.67449|learning_rate:6.55e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 469/2552 [07:10<29:48,  1.16it/s]2024-04-06 18:34:50,086 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:470|global_step:470|average_loss:8.31412|step_loss:8.31412|step_ns_loss:0.61214|step_mlm_loss:7.70198|learning_rate:6.57e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 470/2552 [07:11<29:47,  1.16it/s]2024-04-06 18:34:50,944 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:471|global_step:471|average_loss:8.41021|step_loss:8.41021|step_ns_loss:0.64195|step_mlm_loss:7.76826|learning_rate:6.58e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 471/2552 [07:12<29:46,  1.16it/s]2024-04-06 18:34:51,803 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:472|global_step:472|average_loss:8.35471|step_loss:8.35471|step_ns_loss:0.61627|step_mlm_loss:7.73844|learning_rate:6.59e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  18%|█▊        | 472/2552 [07:13<29:45,  1.16it/s]2024-04-06 18:34:52,663 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:473|global_step:473|average_loss:8.39816|step_loss:8.39816|step_ns_loss:0.67558|step_mlm_loss:7.72258|learning_rate:6.61e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 473/2552 [07:14<29:46,  1.16it/s]2024-04-06 18:34:53,525 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:474|global_step:474|average_loss:8.37581|step_loss:8.37581|step_ns_loss:0.61836|step_mlm_loss:7.75744|learning_rate:6.62e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 474/2552 [07:15<29:47,  1.16it/s]2024-04-06 18:34:54,385 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:475|global_step:475|average_loss:8.34118|step_loss:8.34118|step_ns_loss:0.61722|step_mlm_loss:7.72396|learning_rate:6.64e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 475/2552 [07:16<29:46,  1.16it/s]2024-04-06 18:34:55,245 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:476|global_step:476|average_loss:8.31331|step_loss:8.31331|step_ns_loss:0.58920|step_mlm_loss:7.72410|learning_rate:6.65e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 476/2552 [07:16<29:44,  1.16it/s]2024-04-06 18:34:56,105 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:477|global_step:477|average_loss:8.36907|step_loss:8.36907|step_ns_loss:0.61557|step_mlm_loss:7.75350|learning_rate:6.66e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 477/2552 [07:17<29:44,  1.16it/s]2024-04-06 18:34:56,965 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:478|global_step:478|average_loss:8.38878|step_loss:8.38878|step_ns_loss:0.61699|step_mlm_loss:7.77180|learning_rate:6.68e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▊        | 478/2552 [07:18<29:43,  1.16it/s]2024-04-06 18:34:57,825 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:479|global_step:479|average_loss:8.27197|step_loss:8.27197|step_ns_loss:0.60020|step_mlm_loss:7.67177|learning_rate:6.69e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 479/2552 [07:19<29:42,  1.16it/s]2024-04-06 18:34:58,685 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:480|global_step:480|average_loss:8.38707|step_loss:8.38707|step_ns_loss:0.64919|step_mlm_loss:7.73788|learning_rate:6.71e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 480/2552 [07:20<29:41,  1.16it/s]2024-04-06 18:34:59,545 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:481|global_step:481|average_loss:8.40836|step_loss:8.40836|step_ns_loss:0.64561|step_mlm_loss:7.76274|learning_rate:6.72e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 481/2552 [07:21<29:41,  1.16it/s]2024-04-06 18:35:00,404 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:482|global_step:482|average_loss:8.34668|step_loss:8.34668|step_ns_loss:0.62451|step_mlm_loss:7.72217|learning_rate:6.73e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 482/2552 [07:22<29:39,  1.16it/s]2024-04-06 18:35:01,262 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:483|global_step:483|average_loss:8.26552|step_loss:8.26552|step_ns_loss:0.61392|step_mlm_loss:7.65160|learning_rate:6.75e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 483/2552 [07:22<29:38,  1.16it/s]2024-04-06 18:35:02,125 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:484|global_step:484|average_loss:8.26737|step_loss:8.26737|step_ns_loss:0.60345|step_mlm_loss:7.66392|learning_rate:6.76e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 484/2552 [07:23<29:39,  1.16it/s]2024-04-06 18:35:02,985 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:485|global_step:485|average_loss:8.37246|step_loss:8.37246|step_ns_loss:0.63588|step_mlm_loss:7.73658|learning_rate:6.78e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 485/2552 [07:24<29:38,  1.16it/s]2024-04-06 18:35:03,847 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:486|global_step:486|average_loss:8.33774|step_loss:8.33774|step_ns_loss:0.62830|step_mlm_loss:7.70944|learning_rate:6.79e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 486/2552 [07:25<29:38,  1.16it/s]2024-04-06 18:35:04,707 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:487|global_step:487|average_loss:8.30614|step_loss:8.30614|step_ns_loss:0.57315|step_mlm_loss:7.73299|learning_rate:6.80e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 487/2552 [07:26<29:36,  1.16it/s]2024-04-06 18:35:05,569 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:488|global_step:488|average_loss:8.24042|step_loss:8.24042|step_ns_loss:0.58891|step_mlm_loss:7.65151|learning_rate:6.82e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 488/2552 [07:27<29:37,  1.16it/s]2024-04-06 18:35:06,470 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:489|global_step:489|average_loss:8.27037|step_loss:8.27037|step_ns_loss:0.59297|step_mlm_loss:7.67740|learning_rate:6.83e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 489/2552 [07:28<30:01,  1.15it/s]2024-04-06 18:35:07,332 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:490|global_step:490|average_loss:8.33993|step_loss:8.33993|step_ns_loss:0.64125|step_mlm_loss:7.69868|learning_rate:6.85e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 490/2552 [07:29<29:53,  1.15it/s]2024-04-06 18:35:08,191 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:491|global_step:491|average_loss:8.32002|step_loss:8.32002|step_ns_loss:0.64514|step_mlm_loss:7.67488|learning_rate:6.86e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 491/2552 [07:29<29:45,  1.15it/s]2024-04-06 18:35:09,051 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:492|global_step:492|average_loss:8.31311|step_loss:8.31311|step_ns_loss:0.65018|step_mlm_loss:7.66293|learning_rate:6.87e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 492/2552 [07:30<29:40,  1.16it/s]2024-04-06 18:35:09,910 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:493|global_step:493|average_loss:8.23359|step_loss:8.23359|step_ns_loss:0.61235|step_mlm_loss:7.62124|learning_rate:6.89e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 493/2552 [07:31<29:36,  1.16it/s]2024-04-06 18:35:10,770 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:494|global_step:494|average_loss:8.28898|step_loss:8.28898|step_ns_loss:0.62118|step_mlm_loss:7.66780|learning_rate:6.90e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 494/2552 [07:32<29:34,  1.16it/s]2024-04-06 18:35:11,629 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:495|global_step:495|average_loss:8.29043|step_loss:8.29043|step_ns_loss:0.64514|step_mlm_loss:7.64530|learning_rate:6.92e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 495/2552 [07:33<29:31,  1.16it/s]2024-04-06 18:35:12,488 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:496|global_step:496|average_loss:8.24496|step_loss:8.24496|step_ns_loss:0.59695|step_mlm_loss:7.64801|learning_rate:6.93e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 496/2552 [07:34<29:28,  1.16it/s]2024-04-06 18:35:13,347 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:497|global_step:497|average_loss:8.28161|step_loss:8.28161|step_ns_loss:0.61470|step_mlm_loss:7.66691|learning_rate:6.94e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  19%|█▉        | 497/2552 [07:35<29:27,  1.16it/s]2024-04-06 18:35:14,205 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:498|global_step:498|average_loss:8.33857|step_loss:8.33857|step_ns_loss:0.64075|step_mlm_loss:7.69782|learning_rate:6.96e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 498/2552 [07:35<29:25,  1.16it/s]2024-04-06 18:35:15,064 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:499|global_step:499|average_loss:8.28565|step_loss:8.28565|step_ns_loss:0.63020|step_mlm_loss:7.65544|learning_rate:6.97e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 499/2552 [07:36<29:24,  1.16it/s]2024-04-06 18:35:15,923 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:500|global_step:500|average_loss:8.25099|step_loss:8.25099|step_ns_loss:0.61343|step_mlm_loss:7.63756|learning_rate:6.99e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 500/2552 [07:37<29:22,  1.16it/s]2024-04-06 18:35:16,781 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:501|global_step:501|average_loss:8.24255|step_loss:8.24255|step_ns_loss:0.58813|step_mlm_loss:7.65442|learning_rate:7.00e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 501/2552 [07:38<29:21,  1.16it/s]2024-04-06 18:35:17,639 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:502|global_step:502|average_loss:8.34939|step_loss:8.34939|step_ns_loss:0.64207|step_mlm_loss:7.70732|learning_rate:7.01e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 502/2552 [07:39<29:20,  1.16it/s]2024-04-06 18:35:18,498 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:503|global_step:503|average_loss:8.26233|step_loss:8.26233|step_ns_loss:0.64823|step_mlm_loss:7.61409|learning_rate:7.03e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 503/2552 [07:40<29:19,  1.16it/s]2024-04-06 18:35:19,358 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:504|global_step:504|average_loss:8.20443|step_loss:8.20443|step_ns_loss:0.62620|step_mlm_loss:7.57824|learning_rate:7.04e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
Iteration:  20%|█▉        | 504/2552 [07:41<29:19,  1.16it/s]2024-04-06 18:35:20,218 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - epoch:0|local_step:505|global_step:505|average_loss:8.28232|step_loss:8.28232|step_ns_loss:0.63679|step_mlm_loss:7.64553|learning_rate:7.06e-06|eval_step:0.00000|validation_average_loss:0.00000|validation_total_loss:0.00000|validation_mlm_loss:0.00000|validation_ns_loss:0.00000
2024-04-06 18:35:20,226 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - final_loss 8.261721
2024-04-06 18:35:20,241 - apps.nlp.transformers_on_rdu.tasks.lm_tasks.bert_mlperf_trainer - Process ID 1130816 - info     - {'e2e_train_time': 465.37109112739563, 'training_sequences_per_second': 497817.2568449901, 'final_loss': 8.261720657348633, 'training_samples_per_second': 3889.197319101485}
2024-04-06 18:35:20,380 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130817 - info     - NLP app finished
2024-04-06 18:35:20,382 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130830 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
2024-04-06 18:35:20,388 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130827 - info     - NLP app finished
2024-04-06 18:35:20,388 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130834 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
2024-04-06 18:35:20,390 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130824 - info     - NLP app finished
2024-04-06 18:35:20,391 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130828 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130825 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130826 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130829 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130831 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130832 - info     - NLP app finished
2024-04-06 18:35:20,393 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130835 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
2024-04-06 18:35:20,394 - apps.nlp.transformers_on_rdu.transformers_hook - Process ID 1130833 - info     - NLP app finished
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
import blocksparse tasks from _NamespacePath(['/opt/sambaflow/apps/nlp/transformers_on_rdu/blocksparse/common/tasks'])
Traceback (most recent call last):
  File "sambaflow/samba/abexit.py", line 90, in sambaflow.samba.abexit.SambaAtexit.run_with_abexit
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py", line 540, in main
    task_module.do_training(args=args,
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/tasks/lm_tasks/bert_mlperf_lm.py", line 484, in do_training
    mlperf_pretrain.do_training(args, model, optims, model_config,
  File "/opt/sambaflow/apps/nlp/transformers_on_rdu/tasks/lm_tasks/bert_mlperf_trainer.py", line 576, in do_training
    assert training_perf >= args.min_throughput, \
AssertionError: Expected throughput to be at least 560000.0, instead found 497817.2568449901
srun: error: sn30-r1-h1: task 0: Exited with exit code 1
srun: Terminating job step 30672.0
slurmstepd: error: *** STEP 30672.0 ON sn30-r1-h1 CANCELLED AT 2024-04-06T18:35:22 ***
srun: error: sn30-r1-h1: tasks 8,10: Terminated
srun: error: sn30-r1-h1: tasks 1-7,9,11-13: Killed
srun: Force Terminated job step 30672.0
```
