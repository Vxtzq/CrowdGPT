##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[ Back to arXiv ](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2407.07852v1 "Back to abstract page") [ Download PDF](/pdf/2407.07852v1 "Download PDF") [ ](javascript:toggleNavTOC\(\); "Toggle navigation") [ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

  1. Abstract
  2. 1 Introduction
  3. 2 Implementation
     1. General Implementation Details
     2. Implementation with torch.distributed
     3. Hivemind Implementation
  4. 3 Experiments
     1. Replication Experiment Setup
     2. Main Results
     3. Number of Worker and FLOP Efficiency Ablation
     4. Practical Usage
     5. All-Reduce in FP16
     6. Scaling DiLoCo to Billion Parameter Models
     7. Globally Distributed Training Setting
  5. 4 Conclusion
  6. References
  7. A Model Configuration

[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2407.07852v1 [cs.LG] 10 Jul 2024

# OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training

Sami Jaghouar  Affiliation: Prime Intellect, Inc.  Jack Min Ong  Affiliation: Prime Intellect, Inc.  Johannes Hagemann  Affiliation: Prime Intellect, Inc.  Correspondence to: [johannes@primeintellect.ai](mailto:johannes@primeintellect.ai)

###### Abstract

OpenDiLoCo is an open-source implementation and replication of the Distributed Low-Communication (DiLoCo) training method for large language models. We provide a reproducible implementation of the DiLoCo experiments, offering it within a scalable, decentralized training framework using the Hivemind library. We demonstrate its effectiveness by training a model across two continents and three countries, while maintaining 90-95% compute utilization. Additionally, we conduct ablations studies focusing on the algorithm’s compute efficiency, scalability in the number of workers and show that its gradients can be all-reduced using FP16 without any performance degradation. Furthermore, we scale OpenDiLoCo to 3×3\times the size of the original work, demonstrating its effectiveness for billion parameter models.

###### Keywords: 

Distributed Training, Decentralized AI, Local-SGD, DiLoCo 

## 1 Introduction

Large language models (LLMs) have revolutionized numerous applications of machine learning, yet training these models requires substantial computational resources typically concentrated in a single, well-connected cluster to efficiently parallelize workloads for distributed model training (Hagemann et al., 2023). Novel approaches, such as DiLoCo by Douillard et al., address these challenges by enabling efficient training across multiple, poorly connected devices. Their approach dramatically reduces the need for frequent communication, making it feasible to train LLMs on a global scale.   
We reproduce DiLoCo’s results in an open manner and implement them in a real-world setting using the Hivemind library (team, 2020), showcasing its applications and analyzing its compute efficiency. In summary, the contributions of our work are as follows:

  * •

Reproduction and Scaling of DiLoCo Experiments: We replicate the DiLoCo experiments for a large language model pre-training and validate their results in a reproducible manner. We also successfully extend the DiLoCo experiments to the billion-parameter model scale.

  * •

Open-Source Implementation: We provide implementations of DiLoCo built on top of the Hivemind library alongside a concise 180-line PyTorch version, significantly lowering the barrier for performing decentralized training. Our framework enables single DiLoCo workers to scale to hundreds of machines through our integration with PyTorch FSDP.

  * •

Global Decentralized Training: We demonstrate our approach in a real-world decentralized training setting executed across two continents and three countries, achieving 90-95% compute utilization.

  * •

Analytical Insights and Ablations: We conduct an ablation study of DiLoCo, focusing on the algorithm’s scalability in the number of workers and compute efficiency. We also demonstrate that DiLoCo pseudo gradients can be effectively all-reduced using FP16 without any performance degradation.

We publish the full data of our experiments, the Hivemind as well as the PyTorch distributed training code implementation of OpenDiLoCo on GitHub at [github.com/PrimeIntellect-ai/OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo).

## 2 Implementation

DiLoCo is a local SGD algorithm (Stich, 2019) that leverages two distinct optimization processes: an inner optimizer and an outer optimizer. The inner optimizer, AdamW (Loshchilov & Hutter, 2017), performs local updates on individual workers, while the outer optimizer, SGD with Nesterov momentum (Nesterov, 1983), synchronizes the workers using pseudo-gradients calculated by subtracting the locally updated weight θ⁡(t+h)\theta(t+h) from the original weight θ⁡(t)\theta(t).

This local SGD approach significantly reduces the frequency of communication (up to 500 times), thus lowering the bandwidth requirements for distributed training.

### General Implementation Details

Our implementation of DiLoCo instantiates the two optimizers (inner and outer) and creates two copies of the model: the main model θ⁡(t+h)\theta(t+h), which will be updated by the inner optimizer, and a copy of the original weights, θ⁡(t)\theta(t), which is needed to compute the pseudo-gradient. The inner optimizer is called at the end of each step, while the outer optimizer is called periodically. Both of our implementations compute the pseudo-gradients manually and store them in FP32 inside the PyTorch gradient buffer (within p​a​r​a​m.g​r​a​dparam.grad) of the model. Further experiments show that the pseudo gradient can be stored and all-reduced in FP16 without noticeable performance hit. See Figure 5.

In mixed precision training (Micikevicius et al., 2017) with FP16, a gradient scaler is used to improve the dynamic range of the gradients while avoiding underflow and overflow. The gradient scaler should be called during the inner optimization step but not during the outer one because the pseudo-gradients are calculated manually in FP32.

We offer two open-source DiLoCo implementations, one reference implementation using torch.distributed and an implementation built using the Hivemind library for a more practical decentralized training setting.

### Implementation with torch.distributed

The following details our PyTorch implementation, utilizing the torch.distributed package with NCCL for the communication backend.

[⬇](data:text/plain;base64,Zm9yIGJhdGNoLCBzdGVwIGluIGVudW1lcmF0ZSh0cmFpbl9sb2FkZXIpOgogICAgLi4uICMgbG9zcyBjYWxjdWxhdGlvbgogICAgaW5uZXJfb3B0aW1pemVyLnN0ZXAoKQogICAgaWYgcmVhbF9zdGVwICUgbG9jYWxfc3RlcHMgPT0gMDoKICAgICAgICBmb3Igb2xkX3BhcmFtLCBwYXJhbSBpbiBcCiAgICAgICAgICAgICAgICB6aXAob3JpZ2luYWxfcGFyYW1zLCBtb2RlbC5wYXJhbWV0ZXJzKCkpOgoKICAgICAgICAgICAgcGFyYW0uZ3JhZCA9IG9sZF9wYXJhbSAtIHBhcmFtLmRhdGEKICAgICAgICAgICAgZGlzdC5hbGxfcmVkdWNlKAogICAgICAgICAgICAgICAgdGVuc29yPXBhcmFtLmdyYWQsCiAgICAgICAgICAgICAgICBvcD1kaXN0LlJlZHVjZU9wLkFWRwogICAgICAgICAgICApCiAgICAgICAgICAgIHBhcmFtLmRhdGEgPSBvbGRfcGFyYW0KICAgICAgICBvdXRlcl9vcHRpbWl6ZXIuc3RlcCgpCiAgICBvcmlnaW5hbF9wYXJhbXMgPSBbCiAgICAgICAgcC5kZXRhY2goKS5jbG9uZSgpIGZvciBwIGluIG1vZGVsLnBhcmFtZXRlcnMoKQogICAgXQ==)

for batch, step in enumerate(train_loader): 

… # loss calculation

inner_optimizer.step() 

if real_step % local_steps == 0: 

for old_param, param in \ 

zip(original_params, model.parameters()): 

param.grad = old_param - param.data

dist.all_reduce( 

tensor=param.grad, 

op=dist.ReduceOp.AVG

) 

param.data = old_param

outer_optimizer.step() 

original_params = [ 

p.detach().clone() for p in model.parameters() 

] 

Figure 1: Pseudo-Code for Outer Optimizer in OpenDiLoCo.

We highlight the most important outer optimization part in Figure 1.

Due to the use of a dual optimizer setup and the calculation of pseudo-gradients, this implementation requires custom training code, making it incompatible out of the box with popular training scripts from Hugging Face or PyTorch Lightning. The communication in this implementation also uses the NCCL backend which cannot communicate across networks using NAT, preventing its use over the internet. Our second implementation using Hivemind alleviates both of these issues.

### Hivemind Implementation

The following implementation is built on top of the Hivemind framework 11 1 [github.com/learning-at-home/hivemind](https://github.com/learning-at-home/hivemind). Instead of using torch.distributed for the worker communication, Hivemind utilizes a distributed hash table (DHT) spread across each worker to communicate metadata and synchronize them. This DHT is implemented using the open-source libp2p project 22 2 [github.com/libp2p/libp2p](https://github.com/libp2p/libp2p). Hivemind provides an optimized all-reduce algorithm designed for execution on a pool of poorly connected workers.

[⬇](data:text/plain;base64,ZnJvbSBoaXZlbWluZC5kaHQuZGh0IGltcG9ydCBESFQKZnJvbSBvcGVuX2RpbG9jbyBpbXBvcnQgRGlMb0NvT3B0aW1pemVyCgpvcHRpbWl6ZXIgPSBEaUxvQ29PcHRpbWl6ZXIoCiAgICBicywgIyBiYXRjaCBzaXplCiAgICBscywgIyBsZWFybmluZyByYXRlIHNjaGVkdWxlcgogICAgREhUKCksICMgZGlzdHJpYnV0ZWQgaGFzaCB0YWJsZSBmb3IgY29vcmRpbmF0aW9uCiAgICBpX29wdCwgIyBpbm5lciBvcHRpbWl6ZXIKICAgIG9fb3B0LCAjIG91dGVyIG9wdGltaXplcgogICAgbS5wYXJhbXMoKSAjIG1vZGVsIHBhcmFtZXRlcnMKKQoKZm9yIGJhdGNoIGluIHRyYWluX2RhdGFsb2FkZXI6CgogICAgbW9kZWwoYmF0Y2gpLmxvc3MuYmFja3dhcmQoKQogICAgb3B0aW1pemVyLnN0ZXAoKQogICAgIyB0aGUgb3V0ZXIgc3RlcCwgaW5jbHVkaW5nIHBlZXIgc3luY2hyb25pemF0aW9uCiAgICAjIGFuZCBjb21tdW5pY2F0aW9uLCBpcyB0cmlnZ2VyZWQgYXV0b21hdGljYWxseQogICAgIyBhZnRlciBhbGwgbG9jYWwgc3RlcHMKICAgIG9wdGltaXplci56ZXJvX2dyYWQoKQ==)

from hivemind.dht.dht import DHT

from open_diloco import DiLoCoOptimizer

optimizer = DiLoCoOptimizer( 

bs, # batch size

ls, # learning rate scheduler

DHT(), # distributed hash table for coordination

i_opt, # inner optimizer

o_opt, # outer optimizer

m.params() # model parameters

) 

for batch in train_dataloader: 

model(batch).loss.backward() 

optimizer.step() 

# the outer step, including peer synchronization

# and communication, is triggered automatically

# after all local steps

optimizer.zero_grad() 

Figure 2: OpenDiLoCo - Hivemind API.

Our integration with Hivemind enables a real-world decentralized training setup for DiLoCo, making many of its inherent properties usable, such as:

  * •

On/Off ramping of resources: The amount of available compute may not be constant, with new devices and clusters coming and going.

  * •

Fault tolerance: For decentralized training, some devices may be less reliable than others. Through Hivemind’s fault-tolerant training, a device could become unavailable at any time without stopping the training process.

  * •

Peer-to-Peer: There is no master node. All communication is done in a peer-to-peer fashion.

Unlike the torch.distributed implementation, our Hivemind implementation wraps both optimizers into a single optimizer class, making it compatible with popular training codebases that assume a single optimizer, such as the Hugging Face Trainer. This allows for the use of OpenDiLoCo via a simple Hivemind-compatible API by instantiating a customizable DiLoCoOptimizer, as shown in Figure 2.

Additionally, our custom implementation allows to combine both Hivemind and PyTorch FSDP (Zhao et al., 2023), enabling to scale single DiLoCo workers to multiple nodes or whole clusters.

## 3 Experiments

### Replication Experiment Setup

Our OpenDiLoCo replication experiment setup largely follows the main experiments from Douillard et al.. We conduct various experiments using a model with 150 million parameters on a language modeling task using the C4 dataset (Raffel et al., 2019). The hyperparameters are consistent with DiLoCo across experiments: an inner learning rate of 4​e−44e^{-4}, 1,0001{,}000 warm-up steps, 0.10.1 weight decay, a batch size of 512512, a sequence length of 1,0241{,}024, a learning rate for the Nesterov outer optimizer of 0.70.7, and Nesterov momentum of 0.90.9. Similarly, we run the experiments for a total of 88,00088{,}000 steps.   
The one difference in our experiment setup is that we choose the Llama (Touvron et al., 2023) model architecture for our experiments, due to its recent popularity, while the original DiLoCo authors used the Chinchilla architecture (Hoffmann et al., 2022). These two architectures are generally quite similar but have slight differences. For instance, Llama uses the SwiGLU activation function (Shazeer, 2020) for the MLP and has a dimension of 23​4​d\frac{2}{3}4d instead of 4​d4d. For more details about the model configuration, see Appendix A.

In addition to the DiLoCo experiments, we conduct experiments with a varying number of workers to analyze if diminishing returns occur before reaching the 8 workers reported in the DiLoCo work and to generally measure the FLOP efficiency of the algorithm.

We also run experiments in a real-world decentralized training setup, training across workers from three different countries simultaneously.

Model | Communication | Time | Compute & Data | Perplexity  
---|---|---|---|---  
Baseline, no replica, from scratch | 0 | 1×1\times | 1×1\times | 16.17  
Baseline, 8×8\times batch size with DP | 8×N8\times N | 1×1\times | 8×8\times | 13.68  
DiLoCo, 8 replicas, 500 local steps | 8×NH8\times\frac{N}{H} | 1×1\times | 8×8\times | 13.73  
  
Table 1: Final Evaluation Perplexity Comparison: We compare our two baselines vs DiLoCo with 8 replicas for a 150 million parameter model pre-training across their communication cost, time spent, compute & data used and final perplexity after 88,00088{,}000 steps, similar to Douillard et al.. For the same time and amount of compute, we can compare the second baseline and DiLoCo. The former communicates gradients at each time step (NN total steps), while DiLoCo communicates H=500H=500 times less.

Our baselines also follow a similar setup as Douillard et al.. We use two baselines: the first is a weak baseline that runs without DiLoCo and without replicas for 88,00088{,}000 steps. The second is a stronger baseline, which uses an 8×8\times larger batch size with data parallelism, maintaining a similar compute budget as our DiLoCo experiment but with significantly larger communication requirements.

### Main Results

Figure 3: Main result: 150 million parameter Llama model pre-training with 8 DiLoCo workers yields significantly lower perplexity than the baseline without DiLoCo, and even compared to the baseline using 8 times larger batch size with the same compute budget, while communicating 500 times less. 

Figure 3shows our main experimental results. It demonstrates that DiLoCo with 8 replicas significantly outperforms the baseline without any replicas and matches the performance of the stronger baseline with 500×500\times larger communication requirements, as indicated by the final perplexity results in Table 1. These findings are consistent with the main experimental results of Douillard et al.. One noticeable difference is that in Douillard et al.’s experiments, the DiLoCo run is already approaching and surpassing the stronger baseline at around 64,00064{,}000 steps, while our DiLoCo training run only starts to exactly match the performance of the strong baseline at the end of the training at 88,00088{,}000 steps. This difference might be due to the fact that their experiments start from a checkpoint with 24,00024{,}000 pre-training steps, while ours start from scratch.

### Number of Worker and FLOP Efficiency Ablation

To determine the compute efficiency of DiLoCo, we conduct an ablation study on the number of workers, as shown in Figure 4. These experiments are set up identically to our main experiment, with the only difference being a reduction in the local step size from 500 to 50.

Our results demonstrate a steady improvement in perplexity as the number of workers in DiLoCo increases.

Furthermore, Figure 5 presents the same ablation as Figure 4, but with the x-axis representing global steps instead of local steps. This provides a more accurate approximation of DiLoCo’s FLOP efficiency by comparing the total compute spent on the model. These results reinforce our previous observation: DiLoCo with more than one worker is initially not as compute efficient as the same number of global steps on a single machine or when using Distributed Data Parallel training. DiLoCo may only achieve comparable FLOP efficiency after a large number of steps due to slower initial convergence, as shown in our main experiment in Figure 3.

Figure 4: Ablation Study on the Number of Workers in DiLoCo: Performance comparison of DiLoCo with different numbers of workers and 50 local steps against the baseline without DiLoCo. Due to compute constraints, these ablation experiments were not extended to 88,00088{,}000 steps like the other experiments.

### Practical Usage

According to our main experimental results in Figure 3, eight DiLoCo workers yield a final perplexity comparable to that of DDP after 88,00088{,}000 steps. However, training for only 44,00044{,}000 steps with eight workers results in a significantly worse performing model than DDP with the same number of global steps, making four DiLoCo workers a more efficient choice in this case. Our interpretation suggests that while training with eight DiLoCo workers ultimately results in a stronger model, increasing the number of workers does not accelerate the initial convergence phase as data parallelism would.

Figure 5: Ablation Study on FLOP Efficiency Relative to Number of Workers in DiLoCo: This figure compares the performance of DiLoCo with different numbers of workers and 50 local steps against the baseline without DiLoCo. The x-axis shows the global steps instead of local steps, providing a better approximation of DiLoCo’s FLOP efficiency by comparing the total amount of compute spent on the model.

### All-Reduce in FP16

Our main experiments perform the all-reduce operation of the pseudo gradients in FP32, following the original methodology outlined in the DiLoCo paper. We repeated the DiLoCo experiment, this time using FP16 for the pseudo gradient.

Figure 6: FP16 vs FP32 All-Reduce Ablation: The first group is 4 workers and 50 local steps, the second group is 8 workers and 500 local steps.

Figure 6shows there is no noticeable impact on performance both with 88 workers and 500500 local steps and 44 workers and 5050 local steps, indicating that FP16 all-reduce is effective for use with DiLoCo and can halve the communication time required for the all-reduce operation.

### Scaling DiLoCo to Billion Parameter Models

The original DiLoCo paper demonstrated the efficacy of the method up to model sizes of 400 million parameters. We expand on this and test the scalability of DiLoCo to larger models sizes by pre-training a 1.1 billion parameter model.

We adopt the same hyperparameters as TinyLlama (Zhang et al., 2024), employing a model with 1.1B parameters, a learning rate of 4​e−44e^{-4} and a batch size of 20482048. We conduct our experiment with four workers and an outer learning rate of 0.70.7. We experiment with two DiLoCo runs for this model size: the first with 500 local steps, as in our experiments in Figure 3, and the second with 125 local steps. Since the batch size of this training run is 4 times larger than the batch size of our main experiment, the second run with 125 local steps effectively has the same number of tokens per outer step on each DiLoCo worker as the main experiment with 500 local steps.

We compare our results against two baselines: a weak baseline without DiLoCo and without replicas, and a stronger baseline using a 4×4\times larger batch size with data parallelism, maintaining a similar compute budget as the DiLoCo experiment.

Our baselines are trained using PyTorch FSDP (Zhao et al., 2023) with the hybrid sharding strategy on two co-located nodes, each equipped with eight H100 GPUs. For our DiLoCo experiments, each of the four DiLoCo workers operates on individual nodes with eight H100 GPUs. Intra-node communication is handled by FSDP using the NCCL backend to leverage fast interconnect speeds, while Hivemind manages inter-node, low-bandwidth communication. We use the no-shard strategy in FSDP to avoid incompatibility between the Hivemind state averager and FSDP Sharded DTensors.

Figure 7: 1.1B Scaling Experiment. Comparing a 1.1B training with OpenDiLoCo with 4 workers syncing every 500 local steps and every 125 local steps against the two baselines.

As depicted in Figure 7, both DiLoCo experiments significantly outperform the weaker baseline. However, only the OpenDiLoCo run with 125 local steps nearly matches the performance of the stronger baseline with the same compute budget, while communicating 125 times less. The final perplexity difference between the 4 worker DiLoCo run with 125 local steps and the stronger baseline is 0.240.24 as show in Table 2

Model | Communication | Time | Compute & Data | Perplexity  
---|---|---|---|---  
Baseline, no replica, from scratch | 0 | 1×1\times | 1×1\times | 11.85  
Baseline, 4×4\times batch size with DP | 4×N4\times N | 1×1\times | 4×4\times | 10.52  
DiLoCo, 4 replicas, 125 local steps | 8×N/H8\times\nicefrac{{N}}{{H}} | 1×1\times | 4×4\times | 10.76  
DiLoCo, 4 replicas, 500 local steps | 8×N/H8\times\nicefrac{{N}}{{H}} | 1×1\times | 4×4\times | 11.14  
  
Table 2: Final Perplexity Comparison: We compare our two baselines vs DiLoCo with 4 replicas for a 1.1B parameter model pre-training across their communication cost, time spent, compute & data used and final perplexity after 44,00044{,}000 steps. For the same time and amount of compute, we can compare the second baseline and DiLoCo. The former communicates gradients at each time step (NN total steps), while DiLoCo communicates HH times less (with H=125H=125 or H=500H=500) 

We propose a hypothesis for why the OpenDiLoCo run with 500 local steps underperforms compared to the stronger baseline:   
In our initial experiment with the 150m model, we run for a total of 88,00088,000 steps. For the scaled-up 1.1B parameter experiment, we limit it to 44,00044,000 steps because of the 4×4\times larger batch size. This means that for the same number of training tokens, the DiLoCo synchronization happens only a quarter of the time as often in the 1.1B experiment compared to the 150m experiment. This makes the well-performing 125 local step experiment a better comparison. However, even in the 500 local steps DiLoCo run, we observe faster convergence in the later stages of training, gradually catching up to the stronger baseline.

While we demonstrate that DiLoCo works at the billion parameter scale, we believe that further work is needed to make it effective with even larger batch sizes and more local steps.

### Globally Distributed Training Setting

To showcase the functionality of decentralized training with OpenDiLoCo executed across different continents, we utilize our Hivemind implementation. We use four DiLoCo workers, each with eight H100 GPUs, located in Canada, Finland, and two different states within the United States. footnote 4 shows the network bandwidth between the workers, which varies between 127 to 935 Mbit/s. We train our 1.1B parameter model with 500 local steps, as in our scaling experiment. The gradients are all-reduced in FP16.

Through the large number of local steps, the four workers run independently for around 67.5 minutes before communicating for gradient averaging. For the outer optimizer step, our experiment shows an average all-reduce time between the workers of 300 seconds.

Additionally, we observe variations in the training speed between our different cloud instances. Although all workers have the same GPU type, we could not control for configuration variables such as the number of CPU cores and the amount of RAM, which led to slightly different training times for the 500 inner steps.   
Nevertheless, due to the significant reduction in communication time, the all-reduce bottleneck only accounts for 6.9% of the training time, minimally impacting the overall training speed. Additional training time is spent idling by the fastest worker in our scenario. In future work, we will address this issue by exploring DiLoCo in an asynchronous setting, as done by Liu et al..

Figure 8: Network Bandwidth between Workers: Average bidirectional network bandwidth between the four different workers in our decentralized training setup (in Mbit/s). The GPUs are located in three different countries and hosted by different cloud providers: Canada (Hyperstack); Finland (DataCrunch); United States, Texas (Voltage Park); and United States, Delaware (Runpod). Measured using the iperf package 44 4 iperf package: <https://packages.ubuntu.com/jammy/iperf3>.

## 4 Conclusion

We successfully reproduce the main experiment results of DiLoCo, scale the method to 3×3\times the parameter size of the original work and demonstrate its application in a real-world decentralized training setting. We train a large language model using our OpenDiLoCo implementation across 2 continents and 3 countries and achieve 90-95% compute utilization through the low-communication training approach.

We show that DiLoCo exhibits strong performance with two or four replicas, opening up practical applications. However, while scaling DiLoCo to more than eight workers is a promising research direction for enabling effective, low-communication training across globally distributed GPUs, our ablation study shows using eight workers does not yet match the computational efficiency of Distributed Data Parallel (DDP) training when running for a shorter amount of steps.

For future work, more compute-efficient methods need to be developed for decentralized training, which also improve the scalability to support a significantly larger number of distributed workers. More sophisticated model merging techniques could be used to improve stability and convergence speed. On top of that, compute idle time could be reduced by implementing methods that perform the weight averaging communication asynchronously, interleaving them with the computation for the next outer optimizer update.   
Additional efforts will be directed towards scaling OpenDiLoCo to test the algorithm’s scaling behavior on even larger model sizes, further enhancing its applicability and efficiency in real-world scenarios.

## Acknowledgements

We want to thank Max Ryabinin for his guidance and help with the Hivemind library. His insights have been very helpful for our project.

We would also like to thank Arthur Douillard for his work on DiLoCo and for helping us figure out the details of reproducing the original experiments.

## References

  * Douillard et al. (2023) Douillard, A., Feng, Q., Rusu, A. A., Chhaparia, R., Donchev, Y., Kuncoro, A., Ranzato, M., Szlam, A., and Shen, J.  Diloco: Distributed low-communication training of language models, 2023. 
  * Hagemann et al. (2023) Hagemann, J., Weinbach, S., Dobler, K., Schall, M., and de Melo, G.  Efficient parallelization layouts for large-scale distributed model training, 2023. 
  * Hoffmann et al. (2022) Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L.  Training compute-optimal large language models, 2022. 
  * Liu et al. (2024) Liu, B., Chhaparia, R., Douillard, A., Kale, S., Rusu, A. A., Shen, J., Szlam, A., and Ranzato, M.  Asynchronous local-sgd training for language modeling, 2024. 
  * Loshchilov & Hutter (2017) Loshchilov, I. and Hutter, F.  Fixing weight decay regularization in adam.  _CoRR_ , abs/1711.05101, 2017.  URL <http://arxiv.org/abs/1711.05101>. 
  * Micikevicius et al. (2017) Micikevicius, P., Narang, S., Alben, J., Diamos, G. F., Elsen, E., García, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., and Wu, H.  Mixed precision training.  _CoRR_ , abs/1710.03740, 2017.  URL <http://arxiv.org/abs/1710.03740>. 
  * Nesterov (1983) Nesterov, Y.  A method for solving the convex programming problem with convergence rate o⁡(1/k2)o(1/k^{2}).  _Proceedings of the USSR Academy of Sciences_ , 269:543–547, 1983.  URL <https://api.semanticscholar.org/CorpusID:145918791>. 
  * Raffel et al. (2019) Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J.  Exploring the limits of transfer learning with a unified text-to-text transformer.  _J. Mach. Learn. Res._ , 21:140:1–140:67, 2019.  URL <https://api.semanticscholar.org/CorpusID:204838007>. 
  * Shazeer (2020) Shazeer, N.  Glu variants improve transformer, 2020. 
  * Stich (2019) Stich, S. U.  Local sgd converges fast and communicates little, 2019. 
  * team (2020) team, L.  Hivemind: a Library for Decentralized Deep Learning.  <https://github.com/learning-at-home/hivemind>, 2020. 
  * Touvron et al. (2023) Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G.  Llama: Open and efficient foundation language models, 2023. 
  * Zhang et al. (2024) Zhang, P., Zeng, G., Wang, T., and Lu, W.  Tinyllama: An open-source small language model, 2024.  URL <https://arxiv.org/abs/2401.02385>. 
  * Zhao et al. (2023) Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., and Li, S.  Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023.  URL <https://arxiv.org/abs/2304.11277>. 

## Appendix A Model Configuration

Model Parameters | 150M | 1.1B  
---|---|---  
Number of layers | 12 | 22  
Hidden dim | 1,0241{,}024 | 2,0482{,}048  
Number of heads | 16 | 32  
K/V size | 64 | 64  
Vocab size | 32,00032{,}000  
Inner learning rate (AdamW) | 4​e−44e^{-4}  
Number of warmup steps | 1,0001{,}000  
Weight decay | 0.10.1  
Batch Size | 512512 | 2,0482{,}048  
Sequence length | 1,0241{,}024  
Outer Nesterov learning rate | 0.70.7  
Outer Nesterov momentum | 0.90.9  
Table 3: Model Configuration for the DiLoCo experiments. The models are based on the Llama architecture (Touvron et al., 2023).

Experimental support, please [view the build logs](./2407.07852v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our **major funders** , [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org)

Major funding support from

[ ](https://www.simonsfoundation.org/) [ ](https://www.sfi.org.bm/) [ ](https://www.schmidtsciences.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
