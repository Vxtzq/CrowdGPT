Skip to main content

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

[](https://arxiv.org/IgnoreMe) [ ](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2506.21263v1 "Back to abstract page") [ Download PDF ](/pdf/2506.21263v1 "Download PDF") Contents Reading mode Colour theme

  1. Abstract
  2. 1 Introduction
  3. 2 Method
     1. 2.1 Problem Formulation
     2. 2.2 Pipeline Parallelism with Dual Optimizer Policy
     3. 2.3 One-Step-Delay Overlap of Communication and Local Training
     4. 2.4 Design of Adaptive Gradient Compression Algorithm
        1. 2.4.1 Analysis of Communication Overhead
        2. 2.4.2 Design of Effective Compress Algorithm
        3. 2.4.3 Adaptive Gradient Compress Algorithm
  4. 3 Theoretical Analysis
  5. 4 Experiments
     1. 4.1 Experimental Setup
        1. 4.1.1 Dataset And Models
        2. 4.1.2 infrastructure
        3. 4.1.3 Baselines And Parameters
     2. 4.2 Main Results
        1. 4.2.1 Convergence Result
        2. 4.2.2 End-to-End Runtime
     3. 4.3 Ablation Study
  6. 5 Conclusion
  7. References
  8. A  _Theoretical Analysis of Convergence and Extensive_
     1. A.1 _Proof of Lemma_
     2. A.2 _Proof of Lemma_
     3. A.3 _Proof of the DiLoCoX Convergence Analysis( Corollary)_

[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2506.21263v1 [cs.LG] 26 Jun 2025

# DiLoCoX: A Low-Communication Large-Scale Training Framework for Decentralized Cluster

Ji Qi  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Correspondence to: [qiji@cmss.chinamobile.com](mailto:qiji@cmss.chinamobile.com) WenPeng Zhu  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Li Li  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Ming Wu  Affiliation: Zero Gravity Labs  YingJun Wu  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Wu He  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Xun Gao  Affiliation: China Mobile(Suzhou) Software Technology, JiangSu, China  Jason Zeng  Affiliation: Zero Gravity Labs  Michael Heinrich  Affiliation: Zero Gravity Labs 

###### Abstract

The distributed training of foundation models, particularly large language models (LLMs), demands a high level of communication. Consequently, it is highly dependent on a centralized cluster with fast and reliable interconnects. Can we conduct training on slow networks and thereby unleash the power of decentralized clusters when dealing with models exceeding 100 billion parameters? In this paper, we propose DiLoCoX, a low-communication large-scale decentralized cluster training framework. It combines Pipeline Parallelism with Dual Optimizer Policy, One-Step-Delay Overlap of Communication and Local Training, and an Adaptive Gradient Compression Scheme. This combination significantly improves the scale of parameters and the speed of model pre-training. We justify the benefits of one-step-delay overlap of communication and local training, as well as the adaptive gradient compression scheme, through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence. To the best of our knowledge, this is the first decentralized training framework successfully applied to models with over 100 billion parameters.

###### Keywords: 

Machine Learning, ICML 

## 1 Introduction

LLMs have quickly become dominant in the field of AI due to their exceptional capabilities. Their effectiveness in areas such as automated dialogue generation, machine translation, content summarization, and recommendation systems clearly demonstrates their superior advantages (Zhao et al., 2023a; Wang et al., 2022; Bommasani et al., 2021). However, as efforts to improve the accuracy of these models continue, both their complexity and scale have grown exponentially. In recent years, the number of parameters has increased from billions (Devlin, 2018) to trillions (Fedus et al., 2022; Du et al., 2022), introducing significant challenges in model training. For example, pre-training a model with 175B (Brown, 2020) parameters can require 12,000 GPUs operating for 118 days (Narayanan et al., 2021).

In distributed foundation model training, communication is the key bottleneck. As an example, fine-tuning GPT-J-6B over 10B tokens with 262K batch size across 4 machines (2 A100 GPUs each) demands 915.5 TB data communication in total training(Wang et al., 2023). As a result, the high-speed centralized cluster is currently the dominant solution for training foundation model(Rendle et al., 2016).

Decentralized clusters usually have a significant amount of computing resources than centralized cluster, but the bandwidth between clusters is relatively slower, with only hundreds of Mbps to 1 Gbps in common. When using a training framework such as Megatron-LM(Shoeybi et al., 2019) for distributed cluster training, the bandwidth between clusters often becomes a training bottleneck(Dai et al., 2024; Zhang et al., 2022), resulting in the inability to fully utilize the computing resources of decentralized clusters for model training.

Recently, there has been an exciting collection of work focusing on the decentralized training of foundation models. By introducing outer Nesterov momentum optimizers, DiLoCo (Douillard et al., 2023) can train a model with up to 400M parameters using only data parallelism and achieve even better performance than a fully synchronous model. OpenDiLoCo(Jaghouar et al., 2024) is an open-source implementation of DiLoCo. With FP16 gradient compression and Hivemind optimized collective communication operators, it can train the 1.1B model over poorly connected networks with a bandwidth of hundreds of Mbps and a negligible degradation in convergence. CocktailSGD (Wang et al., 2023) combines three distinct compression techniques, achieving 117× aggressive compression in fine-tuning LLMs with up to 20B parameters over a 500Mbps network.

However, despite these recent efforts, the scale of model parameters and communication bottleneck is still a challenge when pretraining over 100B parameter models. DiLoCo and OpenDiLoCo (Douillard et al., 2023; Jaghouar et al., 2024) have comparable model convergence with the fully synchronous model. However, they only use data parallelism and do not support FSDP (Zhao et al., 2023b) or deepspeed (Rajbhandari et al., 2020), thus the VRAM of GPU capacity limits the scale of model parameters. Furthermore, DiLoCo and OpenDiloCo (Douillard et al., 2023; Jaghouar et al., 2024) employ a pseudo-gradients synchronous mechanism. During the synchronization of pseudo-gradients, local training is in an idle state. When the model scale is large and network bandwidth is limited, the idle time for synchronizing pseudo-gradients becomes unacceptable. CocktailSGD (Wang et al., 2023) uses pipeline parallelism to support 20B model training, but the compression ratio of data parallelism is aggressive up to 117 times and does not use local training like DiLoCo or OpenDiLoCo in decentralized clusters, which has a potential impact on model convergence.

In order to train models with a scale of more than 100B parameters on low-bandwidth decentralized clusters while having comparative model convergence, we have identified the following key challenges: 1\. Introduce model parallelism to address the limitation of VRAM which has to accommodate the whole model parameters. 2\. The overlap between the synchronization of pseudo-gradients and local training to avoid the idleness of computing resources. 3\. Design an efficient gradient compression algorithm and balance it with the number of local training steps to ensure the convergence of model training.

To address the above challenges, we propose a low communication large-scale model training framework DiLoCoX for decentralized cluster. Experiments demonstrate that DiLoCoX can pre-train a 107B model and significantly hide communication overhead while ensuring model convergence on decentralized clusters with only 1Gbps network bandwidth. To the best of our knowledge, this is currently the largest-scale model for effective decentralized cluster training.

The main contributions are as follows.

  * •

Pipeline Parallelism with Dual Optimizer Policy To address the limitation of model parameter scale caused by the lack of support for model parallelism in DiLoCo-like framework, we propose Pipeline Parallelism with Dual Optimizer Policy into the framework and successfully expand the model parameters over 100B.

  * •

One-Step-Delay Overlap of Communication and Local Training We propose a one-step-delay overlap mechanism between the synchronization of pseudo-gradients and local training to avoid the idleness of computing resources, which significantly improves the efficiency of model training.

  * •

Design of Adaptive Gradient Compression Algorithm: By thoroughly analyze different compression scheme, we have designed an efficient AllReduce-compatible communication compression scheme. In order to ensure the convergence of the model and full overlap with local training, we propose an adaptive gradient compression algorithm based on this compress scheme to trade-off compress ratio and the number of local training steps to avoid aggressive compress which cause potential degradation on model convergence.

  * •

Theoretical Analysis of Convergence and Extensive Experiments: We justify the benefit of the one-step-delay overlap of communication and local training, adaptive gradient compression scheme through a theoretical analysis of convergence. Empirically, we demonstrate that DiLoCoX is capable of pre-training a 107B foundation model over a 1Gbps network. Compared to vanilla AllReduce, DiLoCoX can achieve a 357x speedup in distributed training while maintaining negligible degradation in model convergence.

## 2 Method

### 2.1 Problem Formulation

In this paper, the primary emphasis is laid on the parallel configuration comprising data parallelism DD and model parallelism MM. A total of N=D×MN=D\times M workers are involved. Each w​o​r​k​e​ri,jworker_{i,j} sustains a local data source 𝒟i\mathcal{D}_{i}, where ii and jj are the DP and MP indices, respectively. In this local data source, a local loss function fif_{i} is defined. All parallel workers cooperate in order to minimize the objective function f:ℝd→ℝf:\mathbb{R}^{d}\to\mathbb{R}, that is, to determine the parameters of the target model θ^∈ℝd\hat{\theta}\in\mathbb{R}^{d} such that θ^=argminθ∈ℝd[f(θ)=1D∑i=1D𝔼ζ∼𝒟ifi(θ;ζ)]\hat{\mathbf{\theta}}=\mathrm{argmin}_{\mathbf{\theta}\in\mathbb{R}^{d}}\left[f(\mathbf{\theta})=\frac{1}{D}\sum_{i=1}^{D}\mathbb{E}_{\zeta\sim\mathcal{D}_{i}}f_{i}(\mathbf{\theta};\zeta)\right]. Here, ζ\zeta represents the data sampled from each local data source. At iteration tt, each w​o​r​k​e​ri,jworker_{i,j} holds a fraction of the local model replica denoted as θt(i,j)\theta^{(i,j)}_{t} and computes its local gradient over the data sample ζt(i)\zeta^{(i)}_{t}. Subsequently, all workers communicate and compute the average gradient. Then, w​o​r​k​e​ri,jworker_{i,j} updates its fraction of the local model as follows: θt+1(i,j)=θt(i,j)−η​∑i=1D∇fi​(θt(i,j),ζt(i))\theta_{t+1}^{(i,j)}=\theta_{t}^{(i,j)}-\eta\sum_{i=1}^{D}\nabla f_{i}(\theta_{t}^{(i,j)};\zeta_{t}^{(i)}) where η\eta is the learning rate.

### 2.2 Pipeline Parallelism with Dual Optimizer Policy

DiLoCo (Douillard et al., 2023) and OpenDiLoCo (Jaghouar et al., 2024) establish a standard for synchronous LocalSGD in language modeling with outer Nesterov momentum optimizers. However, both of them do not support model parallelism. Consequently, the model parallelism MM is regarded as 1.

For the OpenDiLoCo framework (Jaghouar et al., 2024), each worker ii performs HH local updates utilizing an inner optimizer on their corresponding data shard 𝒟i\mathcal{D}_{i}. Subsequently, the first worker on the node that also holds the outer optimizer computes the parameter change (pseudo gradient) δt(i)=θt−1−θti\delta^{(i)}_{t}=\theta_{t-1}-\theta^{i}_{t} and averages the pseudo-gradient across clusters. Finally, the first worker of the node’s outer optimizer performs a step operation to update local parameters and broadcasts the updated parameters to the remaining workers of the node.

The OpenDiLoCo framework requires GPU VRAM that is capable of accommodating the complete model parameters and inner optimizer states. Additionally, the first worker on the node is also required to store the state of the outer optimizer. This leads to unbalanced VRAM usage and consequently restricts the further expansion of the scale of model parameters.

Based on the fact that pipeline parallelism is effective and has the least communication volume among parallel methods in the 3D parallel strategy (Narayanan et al., 2021), as shown in Figure 1, we propose Pipeline Parallelism with Dual Optimizer Policy. According to the pipeline parallelism approach, we divide the model into MM stages by layers and partition NN workers in decentralized clusters into DD data parallel groups, with the relationship N=M​×​DN=M\texttimes D. Each w​o​r​k​e​ri,jworker_{i,j} holds two distributed optimizers (inner and outer) and a fraction of the model parameters, and performs HH local updates using the inner optimizer on its corresponding data shard 𝒟i\mathcal{D}_{i}. Then, each worker computes its own parameter change (pseudo-gradient) and averages it with those of other workers in the same DP group. Finally, the distributed outer optimizer in each PP group performs a step operation to update the local parameters.

Figure 1: Pipeline Parallelism with Dual Optimizer. The example infrastructure comprises 32 workers distributed among 2 decentralized clusters, with 16 workers in each cluster. Each worker maintains distributed inner and outer optimizers’ state and a fraction of model parameters. Two clusters are trained independently for HH steps respectively. The parallel strategy is PP = 8, DP = 2 for each cluster, and the parameters are updated by using their respective inner optimizers. Finally, the outer distributed optimizer performs a step operation to update local parameters.

The advantage of Pipeline Parallelism with Dual Optimizer Policy resides in the fact that each worker merely stores a fraction of the model parameters, overcoming the limitation that the VRAM of a GPU must hold a complete model. Additionally, each worker incorporates a part of the state of the distributed outer optimizer, leading to a more balanced utilization of VRAM. Based on this scheme, we have implemented DiLoCoX training of models exceeding 100B parameters on decentralized clusters of NVIDIA 40G A800.

### 2.3 One-Step-Delay Overlap of Communication and Local Training

Through introducing the Pipeline Parallelism with Dual Optimizer Policy in section 2.2, we can train the foundation model over 100B on decentralized clusters. However, DiLoCo and OpenDiLoCo (Douillard et al., 2023; Jaghouar et al., 2024) adopt a synchronous approach between local training and pseudo-gradient averaging, which synchronizes the gradient of the outer optimizer after training HH steps. During the synchronization process, the computing resources are in idle state.

We propose one-step-delay overlap of communication and local training illustrated in Figure 2 with the following steps:

  * •

After finishing the first global HH-step, calculate current pseudo-gradients and start performing pseudo-gradients averaging asynchronously.

  * •

During the training of global steps between HH and 2​H2H, we average the last pseudo-gradients simultaneously using AllReduce.

  * •

After finishing the 2​H2H step, we calculate current pseudo-gradients and start performing asynchronous averaging of current pseudo-gradients. Then we update the model parameter by the delayed last averaged pseudo-gradients.

Figure 2: One-Step-Delay Overlap of Communication and Local Training.

In summary, the model parameters of outer step tt are obtained by θt−1\theta^{t-1} and △t−1\triangle^{t-1}. It is denoted as θ(t)←O​u​t​e​r​O​p​t​(θ(t−1),△(t−1))\theta^{(t)}\leftarrow OuterOpt(\theta^{(t-1)},\triangle^{(t-1)}). We assume that the pseudo-gradients will not change significantly between two consecutive outer steps. Therefore, the overlap of communication and local training can greatly improve the model efficiency and will not have a significant impact on the convergence of the model. We verify this in subsequent experiments.

### 2.4 Design of Adaptive Gradient Compression Algorithm

#### 2.4.1 Analysis of Communication Overhead

After HH steps of local training in CC clusters, we need to perform a pseudo-gradient update of decentralized clusters. Suppose that the number of model parameters is θ\theta, so the total number of parameters that must be communicated by all workers is θ×D\theta\times D, where DD is the degree of data parallelism.

In this paper, we solely take into account the communication overhead between clusters. Assuming the utilization of Ring AllReduce (Baidu, 2017) for pseudo-gradient updates, the total number of parameters that require communication between two clusters is 2×(C−1)×θ/C2\times(C-1)\times\theta/C.

Assuming that the pseudo-gradient is stored in FP32, the model parameter is 100B and there are three decentralized clusters, the communication overhead between the three clusters is approximately 533.3 GB for every HH step. If the bandwidth between decentralized clusters is 1 Gbps, transmitting 533.3 GB data would take 1.18 hours. Assuming that the local training step HH is 500 and the duration of every local step is 1 second, thus the total time of local training would take 0.13 hours. Even if we use section 2.2 Pipeline Parallelism with Dual Optimizer Policy, the idle time of computing is approximately 1.04 hours, which is unacceptable for decentralized cluster training. For models with a scale of more than 100B, we need to design an effective compression algorithm to compress more than 10x, ensuring that the communication time of decentralized clusters is within a reasonable range.

#### 2.4.2 Design of Effective Compress Algorithm

Currently, the major compression methods include sparsification (Strom, 2015; Wangni et al., 2017; Alistarh et al., 2018), quantization(Alistarh et al., 2017) and Low-Rank(Vogels et al., 2020), etc. These methods can significantly reduce the communication overhead of distributed training. However, none of these methods achieves a compression ratio larger than 10× without hurting the convergence(Wang et al., 2023).

CocktailSGD (Wang et al., 2023) compresses communication aggressively using a combination of Top-K sparsification, Quantization, and Random sparsification and achieves up to 117× compression in fine-tuning LLMs up to 20B parameters. Aggressive compression leads to significant differences between the parameters of the local model and the global model, resulting in a degradation of the convergence of the global model, but the approach of combined different compression has inspired us to design an efficient compression algorithm.

Firstly, we analyze four common compression algorithms: Random Sparsification, Top-K Compression, Quantization, and Low-Rank.

Random Sparsification randomly selects a portion of the gradients for update according to the seed. By sending only a random seed, the sparsity pattern can be fully recovered. However, given the same sparsity ratio, random sparsification introduces more errors since it does not necessarily keep the values of the largest norm (Wang et al., 2023).

Top-K Compression selects the top-k elements with the largest values for communication and it has fewer compression errors (in l2 norm) compared to random sparsification. However, with dd numbers as input, top-k compression requires K​log2​dK\log_{2}d bits (as index list) or dd bits (as bitmap). When dd is relatively large, it will lead to very large communication costs. In addition, Top-K compression is not AllReduce compatible and requires a parameter server and double compression, which is less efficient than AllReduce compatible compression algorithms which are essential for the 100B model (Wang et al., 2023).

Algorithm 1 Compressor 𝒞⁡[δ]\mathcal{C}[\delta]

Input: Pseudo-gradient δ\delta, low-rank rr, quantization qq-bits. 

(1) Low-Rank rr Approximation: 

δ1←LOWRANK​(δ,r)\delta_{1}\leftarrow\text{LOWRANK}(\delta,r). 

(2) Quantize values to qq bits: 

δ2←QUANTIZE​(δ1)\delta_{2}\leftarrow\text{QUANTIZE}(\delta_{1}). 

Output: Compressed pseudo-gradient δ2\delta_{2}. 

Quantization is efficient and compatible with AllReduce. However, if the original values are stored in FP16, it cannot achieve a compression ratio of more than 16 times. Furthermore, a linear decrease in the number of bits for quantization often results in an exponential increase in error. It is possible to apply quantization in conjunction with other compression algorithms that are compatible with AllReduce.

Low-Rank argues that modern deep networks are over-parameterized and can use a Low-Rank update to update the gradient (Vogels et al., 2020). It is sometimes possible to converge to the optimum using Low-Rank approximations of the gradients without the need for error feedback, and it can be implemented using the AllReduce, which is very efficient.

Algorithm 2 DiLoCoX Framework

Input: Initial Model θ0\theta^{0}, number of workers NN, Data Parallelism DD, Pipeline Parallelism MM, Data shards {𝒟1,⋯,𝒟D}\\{\mathcal{D}_{1},\cdots,\mathcal{D}_{D}\\}, error buffer ee, distributed Optimizers InnerOpt and OuterOpt, combined compress algorithm 𝒞\mathcal{C}, gradient quantization qq, gradient window cc, initial local training step H1H_{1} and initial Low-Rank compression rank r1r_{1}, adaptive gradient compress algorithm AdaGradCmp. 

for outer step t=1⋯Tt=1\cdots T do

for i=1⋯Di=1\cdots D do

for j=1⋯Mj=1\cdots M do

// Start Local Training Thread 

θi,jt←θi,jt−1\theta^{t}_{i,j}\leftarrow\theta^{t-1}_{i,j}

for inner step h=1⋯Hih=1\cdots H_{i} do

x∼𝒟ix\sim\mathcal{D}_{i}

ℒ←f⁡(x,θi,jt)\mathcal{L}\leftarrow f(x,\theta^{t}_{i,j})

θi,jt←InnerOpt​(θi,jt,∇ℒ)\theta^{t}_{i,j}\leftarrow\text{InnerOpt}(\theta^{t}_{i,j},\nabla_{\mathcal{L}})

end for

// Start Compress and Communicate Thread 

if tt >1 then

Δjt−1←AllReduce(𝒞(δi,jt−1|q,ri))i∈{1⋯D}\Delta^{t-1}_{j}\leftarrow\text{AllReduce}(\mathcal{C}(\delta^{t-1}_{i,j}|q,r_{i}))_{i\in\\{1\cdots D\\}}

ei,jt←δi,jt−1−Δjt−1e^{t}_{i,j}\leftarrow\delta^{t-1}_{i,j}-\Delta^{t-1}_{j}

end if

// Waiting all threads to finish 

ri+1,Hi+1←AdaGradCmp​(c,ri,Hi,Δjt−1)r_{i+1},H_{i+1}\leftarrow\text{AdaGradCmp}(c,r_{i},H_{i},\Delta^{t-1}_{j})

δi,jt←(θj(t−1)−θi,jt)+ei,jt\delta^{t}_{i,j}\leftarrow(\theta^{(t-1)}_{j}-\theta^{t}_{i,j})+e^{t}_{i,j} // pseudo-gradients and error compensation 

if tt >1 then

θjt←OuterOpt​(θi,jt−1,Δjt−1)\theta^{t}_{j}\leftarrow\text{OuterOpt}(\theta^{t-1}_{i,j},\Delta^{t-1}_{j})

end if

end for

end for

end for

DiLoCoX is a variant of LocalSGD and DiLoCo. When updating the pseudo-gradient after local training for HH steps, we need to synchronize all gradient information as efficiently as possible to reduce the gradient differences between different nodes caused by compression. Therefore, we choose Quantization and Low-Rank compression methods that support AllReduce. To maximize the compression ratio, inspired by the combined different compression of CocktailSGD, we adopt a combined compression method of Quantization and Low-Rank which is illustrated in Algorithm 1.

#### 2.4.3 Adaptive Gradient Compress Algorithm

How to coordinate the number of steps of local training and two combined compression algorithms to balance the training efficiency and the convergence of the model when pre-training over a 100B model?

###### Theorem 2.1 (Principle of Rank Diminishing (Feng et al., 2022)).

Suppose that each layer fi,i=1,…,Lf_{i},i=1,\dots,L of network FF is almost everywhere smooth and data domain 𝒳\mathcal{X} is a manifold, then both the rank of sub-networks and intrinsic dimension of feature manifolds decrease monotonically by depth:

| Rank⁡(f1)\displaystyle\operatorname{Rank}(f_{1}) | ≥Rank⁡(f2∘f1)≥⋯\displaystyle\geq\operatorname{Rank}(f_{2}\circ f_{1})\geq\cdots |   
---|---|---|---  
|  | ≥Rank(fL−1∘⋯∘f1)≥Rank(FL),\displaystyle\geq\operatorname{Rank}(f_{L-1}\circ\cdots\circ f_{1})\geq\operatorname{Rank}(F_{L}), |   
| dim(𝒳)\displaystyle\dim(\mathcal{X}) | ≥dim(ℱ1)≥dim(ℱ2)≥⋯≥dim(ℱL).\displaystyle\geq\dim(\mathcal{F}_{1})\geq\dim(\mathcal{F}_{2})\geq\cdots\geq\dim(\mathcal{F}_{L}). |   
  
This principle (Feng et al., 2022) describes the behavior of generic neural networks with almost everywhere smooth components, which exhibits the monotonic decreasing (but not strictly) of network ranks and intrinsic dimensionality of feature manifolds. This property will affect the gradient rank of the parameter space through back-propagation. The direction of parameter updates in the network gradually focuses on a few principal components, increasing the redundancy of the gradient matrix and causing a natural decrease in rank.

This implies that as training progresses, the effective information of gradients gradually concentrates within a low-rank space. Therefore, during the model training process, as iterations deepen, we can progressively utilize lower ranks to further compress communication data.

Algorithm 3 Adaptive Gradient Compress Algorithm

Input: Set the gradient rank window cc, initialize the rank for practical compression r1r_{1} and corresponding local training step H1H_{1}. Here tt denotes the iteration of outer optimizer, rank reduction ratio αr′\alpha_{r^{\prime}}. 

// After each outer optimizer completes the AllReduce operation. 

// Calculate rt′r^{\prime}_{t} with the globally averaged gradient. 

if t<ct<c then

rt=r1r_{t}=r_{1}

αr′=1\alpha_{r^{\prime}}=1

Ht=H1H_{t}=H_{1}

else

rt=1c​∑t−c+1tri′r_{t}=\frac{1}{c}\sum_{t-c+1}^{t}{r^{\prime}_{i}}

αr′=r1−rtr1\alpha_{r^{\prime}}=\frac{r_{1}-r_{t}}{r_{1}}

Ht=H1⋅αr′H_{t}=H_{1}\cdot\alpha_{r^{\prime}}

end if

Output: Adaptive rtr_{t} and HtH_{t}. 

On the other hand, Local SGD and quantization also provide some data compression capabilities. The overall compression effectiveness depends on their combined usage.

When the rank rr is dynamically and adaptively adjusted during the training process, the hyperparameter HH in Local SGD can also be continuously tuned accordingly to maximize the overlap between local training and gradient updates while ensuring model convergence.

We propose Adaptive Gradient Compress Algorithm illustrated in Alg 3. During the initial and intermediate phases of model training, gradients descend rapidly, leading to a sharp decrease in rt′r^{\prime}_{t}. Our rtr_{t} dynamically responds to the changes in gradient ranks, enabling more efficient low-rank compression. And HtH_{t} will also be adaptively adjusted accordingly by computing αr′\alpha_{r^{\prime}}. In the final phase, gradient descent slows significantly, and the low-rank parameters evolve at a reduced pace. Consequently, this leads to a stable rtr_{t} and HtH_{t}, which means a stable low-rank compression.

## 3 Theoretical Analysis

###### Assumption 3.1.

(Smoothness) For any i∈{0,⋯,N−1}i\in\\{0,\cdots,N-1\\}, the loss function fif_{i} is L-smooth. There exists a constant L>0L>0 such that for any θ1,θ2∈ℝd\theta_{1},\theta_{2}\in\mathbb{R}^{d}, the following holds:

| ‖∇f~i​(θ1)−∇f~i​(θ2)‖≤L⁡‖θ1−θ2‖.\displaystyle\left\|\nabla\tilde{f}_{i}(\theta_{1})-\nabla\tilde{f}_{i}(\theta_{2})\right\|\leq L\left\|\theta_{1}-\theta_{2}\right\|. |   
---|---|---  
  
###### Assumption 3.2.

(Data Sampling) The stochastic gradient computed is an unbiased estimation for the full gradient with bounded variance, i.e, there exists a constant σ>0\sigma>0 such that for any local dataset 𝒟i\mathcal{D}_{i}, it holds that for any θ∈ℝd\theta\in\mathbb{R}^{d}:

| 𝔼ξ∼𝒟i​[∇fi​(θ,ξ)]=∇fi​(θ),\displaystyle\mathbb{E}_{\xi\sim\mathcal{D}_{i}}[\nabla f_{i}(\theta;\xi)]=\nabla f_{i}(\theta), |   
---|---|---  
  
and

| 𝔼ξ∼𝒟i​‖∇fi​(θ1,ξ)−∇fi​(θ1)‖2≤σ2.\displaystyle\mathbb{E}_{\xi\sim\mathcal{D}_{i}}\|\nabla f_{i}(\theta_{1};\xi)-\nabla f_{i}(\theta_{1})\|^{2}\leq\sigma^{2}. |   
---|---|---  
  
###### Assumption 3.3.

(Data Heterogeneity) There exists ξ>0\xi>0 such that for any θ∈ℝd\theta\in\mathbb{R}^{d}, it holds that

| 1N​∑i=0N−1‖∇fi​(θ)−f⁡(θ)‖2≤ξ2\displaystyle\frac{1}{N}\sum_{i=0}^{N-1}\left\|\nabla f_{i}(\theta)-f(\theta)\right\|^{2}\leq\xi^{2} |   
---|---|---  
  
###### Lemma 3.4.

(Local Update Stability) For any θ∈ℝd\theta\in\mathbb{R}^{d}, after HH steps of AdamW local training, the parameter deviation satisfies:

| 𝔼​‖θi,jt−θt−1‖2≤η2​H2​σ2\displaystyle\mathbb{E}\|\theta_{i,j}^{t}-\theta^{t-1}\|^{2}\leq\eta^{2}H^{2}\sigma^{2} |   
---|---|---  
  
where η\eta is a constant related to the AdamW learning rate.

###### Assumption 3.5.

(Compression Error) The end-to-end compression procedure 𝒞\mathcal{C} in Algorithm 1 has bounded error such that for any θ∈ℝd\theta\in\mathbb{R}^{d}, there exists a constant 0≤ω<10\leq\omega<1

| 𝔼​‖𝒞⁡(θ)−θ‖2≤ω2​‖θ‖2\displaystyle\mathbb{E}\left\|\mathcal{C}(\theta)-\theta\right\|^{2}\leq\omega^{2}\left\|\theta\right\|^{2} |   
---|---|---  
  
###### Lemma 3.6.

CL,CQC_{L},C_{Q} to be LOW-RANK and QUANTIZE respectively. Then it holds that the end to end compression in DiLoCoX: C=CQ∘CLC=C_{Q}\circ C_{L} fulfills the Assumption 3.3

| ω2=1−rd⋅2−q\displaystyle\omega^{2}=1-\frac{r}{d}\cdot 2^{-q} |   
---|---|---  
  
where θ∈ℝd\theta\in\mathbb{R}^{d}, dd is the dimension of θ\theta, rr is the low rank and qq denotes the precision level given in the  QUANTIZE function (Alistarh et al., 2017).

###### Lemma 3.7.

Assuming the number of training steps for the outer optimizer is TT, the number of training steps for the inner optimizer is HH, data parallelism is DD, and the convergence of the local SGD algorithm is O⁡(1D​T​H)O(\frac{1}{\sqrt{DTH}}).

###### Corollary 3.8.

(Convergence Rate) Under assumption 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, and 3.7 if we use Algorithm 1 as 𝒞\mathcal{C}, under the learning rate γ=O⁡(1L​D​H​T+ω2/3L​T1/3)\gamma=O\left(\frac{1}{L\sqrt{DHT}}+\frac{\omega^{2/3}}{LT^{1/3}}\right) the convergence rate of Algorithm 2 is:

| 1T​∑t=1T𝔼​‖∇f​(θt)‖2≤O⁡(L⁡(f⁡(θ0)−f∗)D​H​TCLOSEOPEN+L4/3​(σ2+ξ2+η2​H2​σ2)1/3​ω2/3T2/3)\displaystyle\begin{split}\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\|\nabla f(\theta^{t})\|^{2}&\leq O\biggl(\frac{L(f(\theta^{0})-f^{*})}{\sqrt{DHT}}\\\ &\quad+\frac{L^{4/3}(\sigma^{2}+\xi^{2}+\eta^{2}H^{2}\sigma^{2})^{1/3}\omega^{2/3}}{T^{2/3}}\biggr)\end{split} |   
---|---|---  
  
where DD is the data parallelism degree, HH is the number of local steps, TT is the total outer steps, and f∗=infθf⁡(θ)f^{*}=\inf_{\theta}f(\theta).

We observe Corollary 3.8 that the parameter HH related to local SGD acts on the first term when the compression ratio is not very aggressive and the first term is the leading term. The second term of compression is almost introduced by gradient compression and accumulated bias from local multi-step updates. Under Assumption 3.6, when ω2→0\omega^{2}\rightarrow 0, adaptive compression that combines Low-Rank and Quantization has almost no impact on convergence.

The accumulated bias from local multi-step updates appears as a higher-order term in the convergence rate and slightly degrades the convergence speed when the number of local training steps HH is excessively large. By adjusting the parameters r,q,Hr,q,H, an optimal trade-off can be achieved among communication cost, computational efficiency, and model accuracy.

## 4 Experiments

In this section, we design several experiments to demonstrate the effectiveness of our proposed DiLoCoX.

### 4.1 Experimental Setup

#### 4.1.1 Dataset And Models

We pretrain OPT-1.3B and a modified Qwen1.5-107B model (reducing the total number of layers from 80 to 78 for GPU memory optimization) on the WikiText-103 dataset (Merity et al., 2016; Wang et al., 2023).

#### 4.1.2 infrastructure

We conduct large-scale model training on NVIDIA A800-40G GPUs. To emulate decentralized slow-network environments, we apply Linux traffic control (tc) to limit inter-worker communication bandwidth to 1 Gbps for data parallelism. For the OPT-1.3B model, we deploy 2 nodes with 8 A800 GPUs each (16 GPUs total). For the modified Qwen1.5-107B model, 20 nodes are allocated, each containing 8 A800 GPUs, resulting in 160 GPUs in total.

#### 4.1.3 Baselines And Parameters

We compare with three baselines:

AllReduce without local training and gradient compression as the first baseline because the AllReduce method is equivalent to centralized distributed training.

OpenDiLoCo as the second baseline for its performance and application in a real-world decentralized local training setting(Jaghouar et al., 2024).

CocktailSGD as the third baseline for its aggressive compression which can achieve up to 117× without hurting the convergence (Wang et al., 2023).

Hyperparameter Tuning For a fair comparison, we adjust the compression ratios of different algorithms to the same level through hyperparameter tuning.

For the OPT-1.3B model, OpenDiLoCo sets the local training step HH to 500. CocktailSGD random sparsification ratio is set to 0.1, the top-k ratio is 0.08, and quantization is Int4. DiLoCoX sets the local training step H1H_{1} to 125 and quantizes to Int4. The communication compression ratio of all algorithms is set to 500x.

For the customized Qwen1.5-107B model, CocktailSGD random sparsification ratio is set to 0.1, the top-k ratio is 0.04, and quantize to Int4. DiLoCoX set the local training step H1H_{1} to 125, Low-Rank r1r_{1} to 2,048 (approximately 2x compression) and quantize to Int4. The communication compression ratio of all algorithms is set to 1,000x.

(a) Loss of AllReduce, DiLoCoX, OpenDiLoCo and CocktailSGD on OPT-1.3B model training

(b) Loss of AllReduce, DiLoCoX and CocktailSGD on Qwen1.5-107B model training

Figure 3: Training loss comparison across different distributed optimization methods.

The total training step of all experiments is set to a fixed 4,000 steps, and the gradient rank window cc is set to 5.

### 4.2 Main Results

#### 4.2.1 Convergence Result

We conducted comparative experiments evaluating AllReduce, DiLoCoX, OpenDiLoCo, and CocktailSGD on the OPT-1.3B and Qwen1.5-107B models, respectively. For the OPT-1.3B model, the experimental results are shown in Figure 3(a). After 4,000 steps, the losses of AllReduce, DiLoCoX, OpenDiLoCo, and CocktailSGD reach 4.06, 4.27, 5.37, and 5.79, respectively. The loss of DiLoCoX is negligible compared to AllReduce and significantly outperforms OpenDiLoCo and CocktailSGD by a large margin. We believe that the primary reasons for the degraded convergence performance lie in OpenDiLoCo’s excessively large HH causing gradient staleness and CocktailSGD’s overly aggressive compression strategy, whereas DiLoCoX achieves superior convergence by adopting a more balanced compression strategy. For the OPT-1.3B model, we did not use the adaptive gradient compression algorithm because Int4 quantization and 125-step local training can overlap well and achieve good results.

When training the Qwen1.5-107B model, OpenDiLoCo encounters out-of-memory (OOM) errors due to GPU memory constraints. Consequently, we evaluate DiLoCoX’s convergence performance against AllReduce and CocktailSGD under this setting. The experimental results demonstrate that after 4,000 training steps, the losses of AllReduce, DiLoCoX, and CocktailSGD reach 3.90, 4.20, and 5.23, respectively. As shown in Figure 3(b), DiLoCoX achieves consistently superior convergence compared to CocktailSGD while maintaining competitive performance relative to AllReduce.

Figure 4: Throughput Comparison between AllReduce, OpenDiLoCo, CocktailSGD and DiLoCoX. 

#### 4.2.2 End-to-End Runtime

The experimental results of the throughput of AllReduce, OpenDiLoCo, CocktailSGD, and DiLoCoX at different model scales are shown in Figure 4. Under a 1 Gbps low-bandwidth environment, the throughputs of AllReduce, CocktailSGD, and DiLoCoX reach 745 tokens/s, 16,161 tokens/s, and 23,880 tokens/s respectively on the OPT-1.3B model. DiLoCoX achieves the highest throughput - 32× higher than AllReduce. When scaled to the Qwen1.5-107B model, the throughputs of AllReduce, CocktailSGD, and DiLoCoX reach 10.4 tokens/s, 2,427 tokens/s, and 3,728 tokens/s, where DiLoCoX demonstrates 1.35× and 357× throughput advantages over CocktailSGD and AllReduce respectively.

### 4.3 Ablation Study

To validate the effectiveness of key components in DiLoCoX, we conduct ablation experiments that focusing on two core innovations that affect model convergence and throughput: One-Step-Delay Overlap, Adaptive Gradient Compression. All experiments are performed on Qwen1.5-107B models under 1Gbps bandwidth constraints.

Table 1 shows the results for the Qwen1.5-107B model. The full DiLoCoX configuration attains a loss of 4.20 and a throughput of 3,728 tokens/s. Without the One-Step-Delay Overlap, the loss reduces to 4.15, yet the throughput drops substantially to 2,197 tokens/s. When the Adaptive Gradient Compression is removed, the loss is 4.02, and the throughput further decreases to 1,168 tokens/s. The AllReduce method has the lowest loss of 3.90 among all configurations but an extremely low throughput of only 10.4 tokens/s, highlighting the inefficiency of this traditional approach compared to DiLoCoX.

Table 1: Loss and Throughput of Qwen1.5-107B Configuration | Qwen1.5-107B Loss | Throughput  
---|---|---  
Full DiLoCoX | 4.20 | 3,728  
w/o Overlap | 4.15 | 2,197  
w/o Compression | 4.02 | 1,168  
AllReduce | 3.90 | 10.4  
  
## 5 Conclusion

For conducting training models exceeding 100B on low-communication decentralized clusters, this paper proposes DiLoCoX, a low-communication large-scale decentralized cluster training framework which significantly improves the speed of model pre-training, expands the scale of model parameters, and provides theoretical analysis of the convergence. Empirically, we show that DiLoCoX can pre-train a 107B foundation model over a 1Gbps network. Compared with a centralized cluster, DiLoCoX can significantly achieve 357x distributed training speed with negligible model convergence degradation. To the best of our knowledge, this is currently the largest-scale model for decentralized clusters training. This breakthrough provides new possibilities for fully utilizing the comprehensive computing power of decentralized clusters in the future to achieve the goal of training larger-scale model training.

## References

  * Alistarh et al. (2017) Alistarh, D., Grubic, D., Li, J., Tomioka, R., and Vojnovic, M.  Qsgd: Communication-efficient sgd via gradient quantization and encoding, 2017.  URL <https://arxiv.org/abs/1610.02132>. 
  * Alistarh et al. (2018) Alistarh, D., Hoefler, T., Johansson, M., Khirirat, S., Konstantinov, N., and Renggli, C.  The convergence of sparsified gradient methods, 2018.  URL <https://arxiv.org/abs/1809.10505>. 
  * Baidu (2017) Baidu.  Bringing hpc techniques to deep learning, 2017. 
  * Bommasani et al. (2021) Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al.  On the opportunities and risks of foundation models.  _arXiv preprint arXiv:2108.07258_ , 2021. 
  * Brown (2020) Brown, T. B.  Language models are few-shot learners.  _arXiv preprint arXiv:2005.14165_ , 2020. 
  * Dai et al. (2024) Dai, L., Qi, H., Chen, W., and Lu, X.  High-speed data communication with advanced networks in large language model training.  _IEEE Micro_ , 2024. 
  * Devlin (2018) Devlin, J.  Bert: Pre-training of deep bidirectional transformers for language understanding.  _arXiv preprint arXiv:1810.04805_ , 2018. 
  * Douillard et al. (2023) Douillard, A., Feng, Q., Rusu, A. A., Chhaparia, R., Donchev, Y., Kuncoro, A., Ranzato, M., Szlam, A., and Shen, J.  Diloco: Distributed low-communication training of language models.  _arXiv preprint arXiv:2311.08105_ , 2023. 
  * Du et al. (2022) Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al.  Glam: Efficient scaling of language models with mixture-of-experts.  In _International Conference on Machine Learning_ , pp. 5547–5569. PMLR, 2022. 
  * Fedus et al. (2022) Fedus, W., Zoph, B., and Shazeer, N.  Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.  _Journal of Machine Learning Research_ , 23(120):1–39, 2022. 
  * Feng et al. (2022) Feng, R., Zheng, K., Huang, Y., Zhao, D., Jordan, M., and Zha, Z.-J.  Rank diminishing in deep neural networks.  _Advances in Neural Information Processing Systems_ , 35:33054–33065, 2022. 
  * Jaghouar et al. (2024) Jaghouar, S., Ong, J. M., and Hagemann, J.  Opendiloco: An open-source framework for globally distributed low-communication training.  _arXiv preprint arXiv:2407.07852_ , 2024. 
  * Langley (2000) Langley, P.  Crafting papers on machine learning.  In Langley, P. (ed.), _Proceedings of the 17th International Conference on Machine Learning (ICML 2000)_ , pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann. 
  * Merity et al. (2016) Merity, S., Xiong, C., Bradbury, J., and Socher, R.  Pointer sentinel mixture models.  _arXiv preprint arXiv:1609.07843_ , 2016. 
  * Narayanan et al. (2021) Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., et al.  Efficient large-scale language model training on gpu clusters using megatron-lm.  In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ , pp. 1–15, 2021. 
  * Rajbhandari et al. (2020) Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y.  Zero: Memory optimizations toward training trillion parameter models, 2020.  URL <https://arxiv.org/abs/1910.02054>. 
  * Rendle et al. (2016) Rendle, S., Fetterly, D., Shekita, E. J., and Su, B.-y.  Robust large-scale machine learning in the cloud.  In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , pp. 1125–1134, 2016. 
  * Shoeybi et al. (2019) Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B.  Megatron-lm: Training multi-billion parameter language models using model parallelism.  _arXiv preprint arXiv:1909.08053_ , 2019. 
  * Strom (2015) Strom, N.  Scalable distributed dnn training using commodity gpu cloud computing.  In _Interspeech_ , 2015.  URL <https://api.semanticscholar.org/CorpusID:9338808>. 
  * Vogels et al. (2020) Vogels, T., Karimireddy, S. P., and Jaggi, M.  Powersgd: Practical low-rank gradient compression for distributed optimization, 2020.  URL <https://arxiv.org/abs/1905.13727>. 
  * Wang et al. (2022) Wang, H., Li, J., Wu, H., Hovy, E., and Sun, Y.  Pre-trained language models and their applications.  _Engineering_ , 2022. 
  * Wang et al. (2023) Wang, J., Lu, Y., Yuan, B., Chen, B., Liang, P., De Sa, C., Re, C., and Zhang, C.  Cocktailsgd: fine-tuning foundation models over 500mbps networks.  2023\. 
  * Wangni et al. (2017) Wangni, J., Wang, J., Liu, J., and Zhang, T.  Gradient sparsification for communication-efficient distributed optimization, 2017.  URL <https://arxiv.org/abs/1710.09854>. 
  * Zhang et al. (2022) Zhang, Z., Zheng, S., Wang, Y., Chiu, J., Karypis, G., Chilimbi, T., Li, M., and Jin, X.  Mics: near-linear scaling for training gigantic model on public cloud.  _arXiv preprint arXiv:2205.00119_ , 2022. 
  * Zhao et al. (2023a) Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al.  A survey of large language models.  _arXiv preprint arXiv:2303.18223_ , 2023a. 
  * Zhao et al. (2023b) Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., Desmaison, A., Balioglu, C., Damania, P., Nguyen, B., Chauhan, G., Hao, Y., Mathews, A., and Li, S.  Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023b.  URL <https://arxiv.org/abs/2304.11277>. 

## Appendix A _Theoretical Analysis of Convergence and Extensive_

### A.1 _Proof of Lemma 3.4_

Consider the global parameter θt−1\theta^{t-1} at iteration t−1t-1. After HH steps of AdamW local training, the updated local parameter is θi,jt\theta_{i,j}^{t}. Define the parameter deviation at step hh as:

| Δh=θ(h)−θ(h−1)(h=1,2,…,H),\Delta_{h}=\theta^{(h)}-\theta^{(h-1)}\quad(h=1,2,\dots,H), |   
---|---|---  
  
where θ(0)=θt−1\theta^{(0)}=\theta^{t-1} and θ(H)=θi,jt\theta^{(H)}=\theta_{i,j}^{t}. The total parameter deviation is:

| θi,jt−θt−1=∑h=1HΔh.\theta_{i,j}^{t}-\theta^{t-1}=\sum_{h=1}^{H}\Delta_{h}. |   
---|---|---  
  
The AdamW update rule at step hh is:

| Δh=−η⋅mhvh+ϵ,\Delta_{h}=-\eta\cdot\frac{m_{h}}{\sqrt{v_{h}}+\epsilon}, |   
---|---|---  
  
where mh=β1​mh−1+(1−β1)​ghm_{h}=\beta_{1}m_{h-1}+(1-\beta_{1})g_{h}, vh=β2​vh−1+(1−β2)​gh2v_{h}=\beta_{2}v_{h-1}+(1-\beta_{2})g_{h}^{2}, ghg_{h} is the stochastic gradient at step hh, η\eta is the learning rate, β1,β2∈[0,1)\beta_{1},\beta_{2}\in[0,1) are decay rates, and ϵ>0\epsilon>0 ensures numerical stability.

The stochastic gradient ghg_{h} satisfies:

| 𝔼⁡[gh]=∇f​(θ(h−1)),𝔼​‖gh−∇f​(θ(h−1))‖2≤σ2.\mathbb{E}[g_{h}]=\nabla f(\theta^{(h-1)}),\quad\mathbb{E}\|g_{h}-\nabla f(\theta^{(h-1)})\|^{2}\leq\sigma^{2}. |   
---|---|---  
  
Under steady-state conditions (vh≈𝔼⁡[gh2]v_{h}\approx\mathbb{E}[g_{h}^{2}] and ϵ≪1\epsilon\ll 1), the update simplifies to:

| Δh≈−η⋅gh𝔼⁡[gh2].\Delta_{h}\approx-\eta\cdot\frac{g_{h}}{\sqrt{\mathbb{E}[g_{h}^{2}]}}. |   
---|---|---  
  
Using the gradient noise assumption 𝔼​‖gh‖2≤‖∇f‖2+σ2\mathbb{E}\|g_{h}\|^{2}\leq\|\nabla f\|^{2}+\sigma^{2}, we derive:

| 𝔼​‖Δh‖2≤η2⋅𝔼​‖gh‖2𝔼⁡[gh2]≤η2​σ2.\mathbb{E}\|\Delta_{h}\|^{2}\leq\eta^{2}\cdot\frac{\mathbb{E}\|g_{h}\|^{2}}{\mathbb{E}[g_{h}^{2}]}\leq\eta^{2}\sigma^{2}. |   
---|---|---  
  
The total deviation squared norm expectation is:

| 𝔼​‖∑h=1HΔh‖2=∑h=1H𝔼​‖Δh‖2+2​∑i<j𝔼⁡[Δi⋅Δj].\mathbb{E}\left\|\sum_{h=1}^{H}\Delta_{h}\right\|^{2}=\sum_{h=1}^{H}\mathbb{E}\|\Delta_{h}\|^{2}+2\sum_{i<j}\mathbb{E}\left[\Delta_{i}\cdot\Delta_{j}\right]. |   
---|---|---  
  
Assume updates are fully positively correlated:

| 𝔼⁡[Δi⋅Δj]=𝔼​‖Δi‖⋅𝔼​‖Δj‖≤η2​σ2.\mathbb{E}\left[\Delta_{i}\cdot\Delta_{j}\right]=\mathbb{E}\|\Delta_{i}\|\cdot\mathbb{E}\|\Delta_{j}\|\leq\eta^{2}\sigma^{2}. |   
---|---|---  
  
Then, we get

| 𝔼​‖∑h=1HΔh‖2≤H​η2​σ2+H⁡(H−1)​η2​σ2=H2​η2​σ2.\mathbb{E}\left\|\sum_{h=1}^{H}\Delta_{h}\right\|^{2}\leq H\eta^{2}\sigma^{2}+H(H-1)\eta^{2}\sigma^{2}=H^{2}\eta^{2}\sigma^{2}. |   
---|---|---  
  
That completes the proof of Lemma 3.4.

### A.2 _Proof of Lemma 3.6_

For an input gradient δ∈ℝd\delta\in\mathbb{R}^{d}, after low-rank compression, we obtain δ1=LOWRANK​(δ,r)\delta_{1}=\text{LOWRANK}(\delta,r), where the rank rr satisfies:

| ‖δ1−δ‖F2≤(1−rd)​‖δ‖F2.\|\delta_{1}-\delta\|_{F}^{2}\leq\left(1-\frac{r}{d}\right)\|\delta\|_{F}^{2}. |   
---|---|---  
  
Further quantizing δ1\delta_{1} to qq-bits, the quantization error satisfies:

| ‖δ2−δ1‖F2≤2−q​‖δ1‖F2.\|\delta_{2}-\delta_{1}\|_{F}^{2}\leq 2^{-q}\|\delta_{1}\|_{F}^{2}. |   
---|---|---  
  
The total compression error is bounded by:

| 𝔼​‖δ2−δ‖2≤(1−rd⋅2−q)​‖δ‖2.\mathbb{E}\|\delta_{2}-\delta\|^{2}\leq\left(1-\frac{r}{d}\cdot 2^{-q}\right)\|\delta\|^{2}. |   
---|---|---  
  
Thus, we get it,

| ω2=1−rd⋅2−q.\omega^{2}=1-\frac{r}{d}\cdot 2^{-q}. |   
---|---|---  
  
### A.3 _Proof of the DiLoCoX Convergence Analysis( Corollary 3.8)_

For DiLoCoX model training, inner optimizer use AdamW while outer optimizer use Nesterov Momentum. After HH local training steps of the inner optimizer, the parameter deviation is θi,jt−θt−1\theta_{i,j}^{t}-\theta^{t-1}. The pseudo gradient θi,jt\theta_{i,j}^{t} with error compensation is:

| δi,jt=(θt−1−θi,jt)+ei,jt,\delta_{i,j}^{t}=(\theta^{t-1}-\theta_{i,j}^{t})+e_{i,j}^{t}, |   
---|---|---  
  
where ei,jte_{i,j}^{t} is the compression error buffer. The compressed global gradient is:

| Δjt−1=1D​∑i=1D𝒞⁡(δi,jt−1).\Delta_{j}^{t-1}=\frac{1}{D}\sum_{i=1}^{D}\mathcal{C}(\delta_{i,j}^{t-1}). |   
---|---|---  
  
The compression error satisfies:

| 𝔼​‖Δjt−1−1D​∑i=1Dδi,jt−1‖2≤ω2D​∑i=1D𝔼​‖δi,jt−1‖2.\mathbb{E}\left\|\Delta_{j}^{t-1}-\frac{1}{D}\sum_{i=1}^{D}\delta_{i,j}^{t-1}\right\|^{2}\leq\frac{\omega^{2}}{D}\sum_{i=1}^{D}\mathbb{E}\|\delta_{i,j}^{t-1}\|^{2}. |   
---|---|---  
  
The inner optimizer updates parameters as θi,jt=θt−1−η​∑h=1Hmhvh+ϵ\theta_{i,j}^{t}=\theta^{t-1}-\eta\sum_{h=1}^{H}\frac{m_{h}}{\sqrt{v_{h}}+\epsilon}, where mhm_{h} and vhv_{h} are momentum terms. By the bounded gradient variance 𝔼​‖∇fi​(θ,ξ)‖2≤σ2\mathbb{E}\|\nabla f_{i}(\theta;\xi)\|^{2}\leq\sigma^{2}, we derive:

| 𝔼​‖mhvh+ϵ‖2≤η2​σ2\mathbb{E}\left\|\frac{m_{h}}{\sqrt{v_{h}}+\epsilon}\right\|^{2}\leq\eta^{2}\sigma^{2} |   
---|---|---  
  
And we can get

| 𝔼​‖θi,jt−θt−1‖2≤η2​H2​σ2.\mathbb{E}\|\theta_{i,j}^{t}-\theta^{t-1}\|^{2}\leq\eta^{2}H^{2}\sigma^{2}. |   
---|---|---  
  
Using Nesterov Momentum for the outer optimizer updates θt=θt−1+γ​Δt−1+β⁡(θt−1−θt−2)\theta^{t}=\theta^{t-1}+\gamma\Delta^{t-1}+\beta(\theta^{t-1}-\theta^{t-2}), the objective function satisfies:

| f⁡(θt)≤f⁡(θt−1)−γ⁡⟨∇f​(θt−1),Δt−1⟩+γ2​L2​‖Δt−1‖2+β​L​‖θt−1−θt−2‖2.f(\theta^{t})\leq f(\theta^{t-1})-\gamma\langle\nabla f(\theta^{t-1}),\Delta^{t-1}\rangle+\frac{\gamma^{2}L}{2}\|\Delta^{t-1}\|^{2}+\beta L\|\theta^{t-1}-\theta^{t-2}\|^{2}. |   
---|---|---  
  
Taking expectation and rearranging terms:

| 𝔼⁡[f⁡(θt)]≤𝔼⁡[f⁡(θt−1)]−γ​𝔼​‖∇f​(θt−1)‖2+γ2​L​𝔼​‖Δt−1‖2+β​L​𝔼​‖θt−1−θt−2‖2+γ​L​𝔼​‖Δt−1−∇f​(θt−1)‖2.\mathbb{E}[f(\theta^{t})]\leq\mathbb{E}[f(\theta^{t-1})]-\gamma\mathbb{E}\|\nabla f(\theta^{t-1})\|^{2}+\gamma^{2}L\mathbb{E}\|\Delta^{t-1}\|^{2}+\beta L\mathbb{E}\|\theta^{t-1}-\theta^{t-2}\|^{2}+\gamma L\mathbb{E}\|\Delta^{t-1}-\nabla f(\theta^{t-1})\|^{2}. |   
---|---|---  
  
The global gradient error is decomposed into three parts:

| 𝔼​‖Δt−1−∇f​(θt−1)‖2≤𝔼​‖Δt−1−1D​∑i=1Dδi,jt−1‖2⏟Compression Error+𝔼​‖1D​∑i=1D(δi,jt−1−∇f​(θt−1))‖2⏟Local Deviation+𝔼​‖1D​∑i=1D(∇f​(θt−1,x)−∇f​(θt−1))‖2⏟Stochastic Noise.\mathbb{E}\|\Delta^{t-1}-\nabla f(\theta^{t-1})\|^{2}\leq\underbrace{\mathbb{E}\left\|\Delta^{t-1}-\frac{1}{D}\sum_{i=1}^{D}\delta_{i,j}^{t-1}\right\|^{2}}_{\text{Compression Error}}+\underbrace{\mathbb{E}\left\|\frac{1}{D}\sum_{i=1}^{D}(\delta_{i,j}^{t-1}-\nabla f(\theta^{t-1}))\right\|^{2}}_{\text{Local Deviation}}+\underbrace{\mathbb{E}\left\|\frac{1}{D}\sum_{i=1}^{D}(\nabla f(\theta^{t-1};x)-\nabla f(\theta^{t-1}))\right\|^{2}}_{\text{Stochastic Noise}}. |   
---|---|---  
  
By Assumption 3.5, we can infer to the term of the compression error:

| 𝔼​‖Δt−1−1D​∑i=1Dδi,jt−1‖2≤ω2D​∑i=1D𝔼​‖δi,jt−1‖2.\mathbb{E}\left\|\Delta^{t-1}-\frac{1}{D}\sum_{i=1}^{D}\delta_{i,j}^{t-1}\right\|^{2}\leq\frac{\omega^{2}}{D}\sum_{i=1}^{D}\mathbb{E}\|\delta_{i,j}^{t-1}\|^{2}. |   
---|---|---  
  
By Lemma 3.1 and Lemma 3.4, we can infer to the term of the local deviation:

| 𝔼​‖δi,jt−1−∇f​(θt−1)‖2≤L2​η2​H2​σ2.\mathbb{E}\|\delta_{i,j}^{t-1}-\nabla f(\theta^{t-1})\|^{2}\leq L^{2}\eta^{2}H^{2}\sigma^{2}. |   
---|---|---  
  
By Lemma 3.2 and Lemma 3.3, we can infer to the term of the stochastic noise:

| 𝔼​‖1D​∑i=1D(∇f​(θt−1,ξ)−∇f​(θt−1))‖2≤σ2D+ξ2.\mathbb{E}\left\|\frac{1}{D}\sum_{i=1}^{D}(\nabla f(\theta^{t-1};\xi)-\nabla f(\theta^{t-1}))\right\|^{2}\leq\frac{\sigma^{2}}{D}+\xi^{2}. |   
---|---|---  
  
Integrating the three aspects discussed above, summing over t=1,…,Tt=1,\dots,T, optimizing γ\gamma, and bounding terms:

| 1T​∑t=1T𝔼​‖∇f​(θt)‖2≤2​(f⁡(θ0)−f∗)γ​T+γ​L​(σ2D+ξ2+L2​η2​H2​σ2)+γ2​L​ω2​𝔼​‖δ‖2.\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\|\nabla f(\theta^{t})\|^{2}\leq\frac{2(f(\theta^{0})-f^{*})}{\gamma T}+\gamma L\left(\frac{\sigma^{2}}{D}+\xi^{2}+L^{2}\eta^{2}H^{2}\sigma^{2}\right)+\gamma^{2}L\omega^{2}\mathbb{E}\|\delta\|^{2}. |   
---|---|---  
  
while choosing the learning rate as γ=O⁡(1L​D​H​T+ω2/3L​T1/3)\gamma=O\left(\frac{1}{L\sqrt{DHT}}+\frac{\omega^{2/3}}{LT^{1/3}}\right), we obtain the final convergence rate. Thus,

| 1T​∑t=1T𝔼​‖∇f​(θt)‖2≤O⁡(L⁡(f⁡(θ0)−f∗)D​H​T+L4/3​(σ2+ξ2+η2​H2​σ2)1/3​ω2/3T2/3)\displaystyle\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\|\nabla f(\theta^{t})\|^{2}\leq O\biggl(\frac{L(f(\theta^{0})-f^{*})}{\sqrt{DHT}}+\frac{L^{4/3}(\sigma^{2}+\xi^{2}+\eta^{2}H^{2}\sigma^{2})^{1/3}\omega^{2/3}}{T^{2/3}}\biggr) |   
---|---|---  
  
That completes the proof.

13

Experimental support, please [view the build logs](./2506.21263v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our [**major funders**](https://info.arxiv.org/about/major_funders.html), [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe.html) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
