Skip to main content

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

[](https://arxiv.org/IgnoreMe) [ ](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2401.09135v1 "Back to abstract page") [ Download PDF ](/pdf/2401.09135v1 "Download PDF") Contents Reading mode Colour theme

  1. Abstract
  2. 1 Introduction
     1. 1\. Framework (Section ).
     2. 2\. Optimization Challenge (Section ).
     3. 3\. Proposed Solutions (Section ).
  3. 2 Background
  4. 3 Async. Local-SGD Framework
     1. Data Shard Sampling
     2. Learning Rate Scheduling
     3. Grace Period for Model Synchronization
     4. Asynchronous Task Scheduling
  5. 4 Optimization Challenge
     1. Effect of InnerOpt \+ OuterOpt
     2. Momentum in the OuterOpt
     3. Is Staleness the Cause?
     4. Baselines
     5. Existing Fixes
  6. 5 Proposed Solutions
     1. Delayed Nesterov Update
     2. Dynamic Local Updates
  7. 6 A Minimal Toy Example
  8. 7 Experiments
     1. Evaluating Delayed Nesterov (DN) and Dynamic Local Updates (DyLU)
     2. Assessing Different Levels of Worker Heterogeneity
     3. Ablation with Different Numbers of Workers
     4. Ablation with Different Model Sizes
     5. Ablation with Different cc
  9. 8 Related Work
     1. Local-SGD and Distributed Optimization
     2. Asynchronous Training
     3. Local-SGD for Language Modeling
  10. 9 Limitations
  11. 10 Conclusion
  12. References
  13. 10.1 Implementation Details
     1. Network Architecture
     2. Training Dataset
     3. Hyperparameters
     4. Inner Optimizer States
  14. 10.2 Aync. Training Pseudocode


[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2401.09135v1 [cs.LG] 17 Jan 2024

\reportnumber

# Asynchronous Local-SGD Training for Language Modeling

Bo Liu  Rachita Chhaparia  Affiliation: Google DeepMind  Arthur Douillard  Affiliation: Google DeepMind  Satyen Kale  Affiliation: Google Research  Andrei A. Rusu  Affiliation: Google DeepMind  Jiajun Shen  Affiliation: Google DeepMind  Arthur Szlam  Affiliation: Google DeepMind  Marc’Aurelio Ranzato  Affiliation: Google DeepMind  Affiliation: Work done as an intern at Google DeepMind  Affiliation: The University of Texas at Austin 

###### Abstract

Local stochastic gradient descent (Local-SGD), also referred to as federated averaging, is an approach to distributed optimization where each device performs more than one SGD update per communication. This work presents an empirical study of asynchronous Local-SGD for training language models; that is, each worker updates the global parameters as soon as it has finished its SGD steps. We conduct a comprehensive investigation by examining how worker hardware heterogeneity, model size, number of workers, and optimizer could impact the learning performance. We find that with naive implementations, asynchronous Local-SGD takes more iterations to converge than its synchronous counterpart despite updating the (global) model parameters more frequently. We identify momentum acceleration on the global parameters when worker gradients are stale as a key challenge. We propose a novel method that utilizes a delayed Nesterov momentum update and adjusts the workers’ local training steps based on their computation speed. This approach, evaluated with models up to 150M parameters on the C4 dataset, matches the performance of synchronous Local-SGD in terms of perplexity per update step, and significantly surpasses it in terms of wall clock time.

###### keywords

asynchronous training, language modeling, large-scale distributed learning 

## 1 Introduction

Large language models (LLMs) have revolutionized many applications, transforming the way machines interact with human language. The cornerstone of this revolution is training these models at massive scale. To manage such large-scale training in reasonable amounts of time, it has been necessary to distribute computations across multiple devices. However, the standard approaches to this distributed training uses co-located devices with fast interconnects.

One might hope to be able to effectively harness a broader range of computational resources, perhaps geographically distant from each other, in order to build even more powerful large models. However, utilizing numerous distant devices faces a significant hurdle: communication latency. When devices focus solely on computing gradients before sending them back to a central server, the communication time can overshadow the computation time, creating a bottleneck in efficiency.

Figure 1: Illustration of async. v.s. sync. training with 2 workers (in blue and red). Sync. training suffers from the straggler effect, while async. training reduces the idling time of the fast worker. Figure 2: Comparative evaluation of language models using sync. and async. Local-SGD methods with 4 heterogeneous workers on a 20M parameter model. The state-of-the-art sync. Local-SGD method, DiLoCo (Douillard et al., 2023), employs AdamW and Nesterov momentum as the worker-side and server-side optimizers, respectively. This optimizer combination remains the strongest for async. Local-SGD training (See Figure 5), yet underperforms DiLoCo significantly. By integrating Delayed Nesterov (DN) (Algorithm 3) for outer optimization and Dynamic Local Updates (DyLU) (Section 5), we significantly bridge the performance gap in terms of perplexity versus updates between sync. and async. training in language modeling. Moreover, the proposed method significantly surpasses DiLoCo in terms of perplexity versus wall clock time.

Local Stochastic Gradient Descent (Local-SGD) is a collection of optimization methods that can reduce communication bottlenecks.11 1 The term Local-SGD, sometimes also known as Federated Average (FedAvg), is used here to emphasize its roots in distributed optimization, where users have control over data allocation to different workers. These methods involve each device performing multiple local gradient steps before syncing their parameter updates with a parameter server. While Local-SGD enhances training efficiency by reducing communication frequency, it can suffer from the _straggler effect_ caused by heterogeneous devices. For instance, faster devices are idle waiting for slower ones to catch up, undermining the overall efficiency of the system. Moreover, all devices are forced to communicate at the same time requiring high bandwidth connection with the parameter server. Asynchronous Local-SGD presents a more viable solution (illustrated in Figure 1), as it allows the server to update the model as soon as the updates of a worker are available, thereby enhancing computational utilization and minimizing communication bandwidth requirements.

In this study, we explore the viability of asynchronously training LLMs using Local-SGD. We expand upon previous works that have attempted to alternate steps on subsets of workers or randomly drop certain subset of workers during synchronous Local-SGD (Ryabinin et al., 2021; Douillard et al., 2023). The main content is structured into three parts:

#### 1\. Framework (Section 3).

The first part introduces our high-level design for the asynchronous training framework. We discuss how each worker determines which data shard to train on, for how many steps, with what learning rates, and how the server updates models asynchronously.

#### 2\. Optimization Challenge (Section 4).

In the second part, we conduct an empirical study of various existing optimization strategies suitable for asynchronous Local-SGD. This includes both worker-side optimization (inner optimization) and server-side optimization (outer optimization). We uncover a key challenge in utilizing momentum effectively. Notably, while adaptive momentum methods generally accelerate convergence of both inner and outer optimizations, their efficacy in asynchronous Local-SGD is comparatively reduced when both optimizations employ momentum techniques, especially when contrasted with the synchronous implementation.

#### 3\. Proposed Solutions (Section 5).

We introduce two simple and effective techniques: the Delayed Nesterov momentum update (DN) and Dynamic Local Updates (DyLU). These techniques, when combined and evaluated on training language model, allow asynchronous Local-SGD to approach synchronous Local-SGD in terms of perplexity versus the total number of local updates, and further improve asynchronous Local-SGD vs. synchronous Local-SGD in terms of perplexity versus wall-clock, as detailed in Figure 2.

## 2 Background

In this study, we focus on the distributed optimization of shared model parameters θ\theta across kk data shards, denoted as 𝒟={𝒟1,…,𝒟k}\mathcal{D}=\\{\mathcal{D}_{1},\dots,\mathcal{D}_{k}\\}, with kk workers.22 2 We assume the number of workers (kk) equals the number of data shards, though our methods are also applicable when there are fewer workers than data shards. The primary goal is described by the following equation:

| min⁡∑i=1kθ⁡|𝒟i|∑j|𝒟j|​𝔼x∼𝒟i​[ℓ⁡(x,θ)],\min_{\theta}\sum_{i=1}^{k}\frac{|\mathcal{D}_{i}|}{\sum_{j}|\mathcal{D}_{j}|}\mathbb{E}_{x\sim\mathcal{D}_{i}}\big[\ell(x;\theta)\big], |  | (1)  
---|---|---|---  
  
where ℓ⁡(⋅,θ)\ell(\cdot;\theta) represents the loss function (for instance, cross entropy loss for next token prediction in language modeling), and |⋅||\cdot| indicates the set size.

Algorithm 1 DiLoCo Algorithm (synchronous)

1: Initial pretrained model θ(0)\theta^{(0)}

2: kk workers 

3: Data shards {𝒟1,…,𝒟k}\\{\mathcal{D}_{1},\dots,\mathcal{D}_{k}\\}

4: Optimizers InnerOpt and OuterOpt

5: for outer step t=1​…​Tt=1\ldots T do

6: parallel for worker i=1​…​ki=1\ldots k do

7: θi(t)←θ(t−1)\theta_{i}^{(t)}\leftarrow\theta^{(t-1)}

8: for inner step h=1​…​Hh=1\ldots H do

9: x∼𝒟ix\sim\mathcal{D}_{i}

10: ℒ←f⁡(x,θi(t))\mathcal{L}\leftarrow f(x,\theta_{i}^{(t)})

11: θi(t)←InnerOpt​(θi(t),∇ℒ)\theta_{i}^{(t)}\leftarrow\texttt{InnerOpt}(\theta_{i}^{(t)},\nabla_{\mathcal{L}})

12: end for

13: δi(t)=θ(t−1)−θi(t)\delta_{i}^{(t)}=\theta^{(t-1)}-\theta_{i}^{(t)}

14: end parallel for

15: Δ(t)←1k​∑i=1kδi(t)\Delta^{(t)}\leftarrow\frac{1}{k}\sum_{i=1}^{k}\delta_{i}^{(t)}⊳\triangleright outer gradient

16: θ(t)←OuterOpt​(θ(t−1),Δ(t))\theta^{(t)}\leftarrow\texttt{OuterOpt}(\theta^{(t-1)},\Delta^{(t)})

17: end for

We extend the definition of Local-SGD in this work to include not just the original Local-SGD method, but also its variants that incorporate advanced optimization techniques. We particularly focus on DiLoCo (Douillard et al., 2023), which sets the standard for synchronous Local-SGD in language modeling. DiLoCo’s methodology is detailed in Algorithm 1. Each worker ii performs HH local updates using an _inner optimizer_ on their data shard 𝒟i\mathcal{D}_{i} before sending the parameter change (pseudo-gradient) δi(t)=θ(t−1)−θi(t)\delta^{(t)}_{i}=\theta^{(t-1)}-\theta^{(t)}_{i} back to the server. The server then computes the aggregated outer gradient Δ(t)=1k​∑i=1kδi(t)\Delta^{(t)}=\frac{1}{k}\sum_{i=1}^{k}\delta^{(t)}_{i}, and applies an _outer optimizer_ with Δ(t)\Delta^{(t)} to update θ\theta. A key insight from DiLoCo is the optimal use of AdamW and Nesterov Momentum as the best inner and outer optimizers, respectively.

## 3 Async. Local-SGD Framework

This section outlines the asynchronous Local-SGD pipeline design, where we assume a central server controls all workers and asynchronously aggregates their updates.

#### Data Shard Sampling

Unlike in the federated learning setting where each device is attached to its own data, in distributed optimization, the user has the right to choose which data shard is assigned to which worker, even dynamically. To balance the learning progress on different data shards (as workers are heterogeneous), whenever a worker is ready to start a new local optimization round, we sample a data shard inversely proportional to its “learning progress". Specifically, define nin_{i} as the number of learned data points in 𝒟i\mathcal{D}_{i}, then we sample a shard isampledi_{\text{sampled}} according to:

| isampled∼p,where​pi∝max​(|𝒟i|∑j|𝒟j|−ni∑jnj,0).\begin{split}i_{\text{sampled}}&\sim p,\\\ \text{where}\penalty\ p_{i}&\propto\text{max}(\frac{|\mathcal{D}_{i}|}{\sum_{j}|\mathcal{D}_{j}|}-\frac{n_{i}}{\sum_{j}n_{j}},0).\end{split} |  | (2)  
---|---|---|---  
  
In other words, we sample a data shard only when it is “under-sampled" (i.e., ni∑jnj≤|𝒟i|∑j|𝒟j|\frac{n_{i}}{\sum_{j}n_{j}}\leq\frac{|\mathcal{D}_{i}|}{\sum_{j}|\mathcal{D}_{j}|}). The degree to which a shard is under-sampled determines its sampling rate. By doing so, we ensure that the data shard with slower progress is more likely to be sampled for training, therefore balancing the learning progress across shards.

#### Learning Rate Scheduling

In contrast to synchronous training methods like DiLoCo, asynchronous training can lead to uneven progress across different data shards, especially when workers are allowed varying numbers of training steps. This raises the question of how to effectively schedule learning rates. In our approach we assign each data shard its own learning rate schedule. Specifically, we implement a linear warmup combined with a cosine learning rate decay, where TT represents the target total training iterations for each data shard:

| ηt={t​ηmax/twarmupt<twarmupηmin+0.5​(ηmax−ηmin)(1+cos⁡(t−twarmupT−twarmup​π))t≥twarmup.\eta_{t}=\begin{cases}t\eta_{\text{max}}/t_{\text{warmup}}&t<t_{\text{warmup}}\\\ \eta_{\text{min}}+0.5(\eta_{\text{max}}-\eta_{\text{min}})\\\ \penalty\ \penalty\ \penalty\ \penalty\ \big(1+\cos\big(\frac{t-t_{\text{warmup}}}{T-t_{\text{warmup}}}\pi\big)\big)&t\geq t_{\text{warmup}}.\end{cases} |  | (3)  
---|---|---|---  
  
In practice, asynchronous training may conclude with different final iteration counts (tendt_{\text{end}}) for each data shard. Since we cannot predetermine tendt_{\text{end}} due to the unpredictability of asynchrony, we set the minimum learning rate (ηmin\eta_{\text{min}}) to a small positive value. This ensures continued progress even if tt exceeds TT. Additionally, we adjust T−twarmupT-t_{\text{warmup}} to be non-negative and ensure that the ratio t−twarmupT−twarmup\frac{t-t_{\text{warmup}}}{T-t_{\text{warmup}}} remains within the range of [0,1][0,1]. This helps maintain effective learning rate adjustments throughout the training process.

#### Grace Period for Model Synchronization

In asynchronous training, the completion time of each worker’s tasks can vary. For example, if worker B completes training shortly after worker A, it might be beneficial for A to wait briefly until the server processes updates from both workers before receiving the updated model for its next training task. However, this waiting period should be minimal and occur only when necessary. Specifically, if no other worker completes its task within the grace period while worker A is synchronizing with the server’s model, A should promptly commence its new training task using the server’s current model. For a visual representation of this process, please refer to Figure 3.

Figure 3: We consecutively synchronize the update from B after we synchronize A because B finishes its training after A but before the end of the grace period. A and B will therefore use the same server model to start the new training jobs, while C will start its own grace period. 

#### Asynchronous Task Scheduling

In Algorithm 2, we present the asynchronous task scheduling pipeline. Throughout the algorithm, we use τ\tau to denote the actual wall clock time and tt to denote model updates. In line 1-4, we initialize the model, total local updates tlocalt_{\text{local}}, and the list of workers 𝒲\mathcal{W} and the completed workers 𝒲completed\mathcal{W}_{\text{completed}}. In line 5, we start the first training job for all workers with the initial model parameter θ(0)\theta^{(0)}. Note that the train() function implements the data sampling technique and performs the learning rate scheduling mentioned before. In line 6, we reset the starting time of the grace period τsync\tau_{\text{sync}} to ∞\infty. This is because we want to synchronize with a worker only when its completion time is within τsync+τgrace\tau_{\text{sync}}+\tau_{\text{grace}}. The main asynchronous Local-SGD training loop is provided in line 6-19. Within the loop, we first attempt to get a completed worker ww (line 7). We retrieve the earliest completed worker that we have not yet processed yet, as long as its completion time is still within the grace period (e.g., ww.completed_time ≤τsync+τgrace\leq\tau_{\text{sync}}+\tau_{\text{grace}}). If no such workers exist, get_worker() will return null. In line 10-15 where such a worker ww is found, we synchronize its update with the server model θ\theta. In line 17-20 when no such workers are found, we assign new training jobs for all completed workers and empty the list of completed workers. For the detailed pseudocode of the train() and get_worker() functions, please refer to Appendix 10.2. In practice, for the sake of reproducibility of research, we implement a _determininistic_ version of Algorithm 2 with faked training time based on real-world device statistics. We validate the correctness of the training pipeline by simulating synchronous updates using the asynchronous framework.

Algorithm 2 Async. Local-SGD Task Scheduling.

1: Initial pretrained model θ(0)\theta^{(0)}

2: kk workers 

3: Grace period τgrace\tau_{\text{grace}}

4: Total local updates tmaxt_{\text{max}}

5: tlocal=0t_{\text{local}}=0

6: θ←θ(0)\theta\leftarrow\theta^{(0)}

7: 𝒲\mathcal{W} = [init_worker() for ii in [kk]]

8: 𝒲completed\mathcal{W}_{\text{completed}} = []

9: train(𝒲\mathcal{W}, θ\theta)

10: τsync=∞\tau_{\text{sync}}=\infty ⊳\triangleright start of the grace period

11: while tlocal<tmaxt_{\text{local}}<t_{\text{max}} do

12: ww = get_worker(𝒲,τgrace,τsync\mathcal{W},\tau_{\text{grace}},\tau_{\text{sync}})

13: ⊳\triangleright get a completed worker

14: if ww exists then

15: ⊳\triangleright synchronize the update with server

16: τsync\tau_{\text{sync}} = min(τsync\tau_{\text{sync}}, ww.completed_time) 

17: θ←\theta\leftarrow sync(θ\theta, ww.update)

18: 𝒲completed\mathcal{W}_{\text{completed}}.add(ww)

19: tlocalt_{\text{local}} += ww.local_updates

20: else

21: ⊳\triangleright assign jobs for completed workers

22: τsync=∞\tau_{\text{sync}}=\infty

23: train(𝒲completed\mathcal{W}_{\text{completed}}, θ\theta)

24: 𝒲completed\mathcal{W}_{\text{completed}} = []

25: end if

26: end while

## 4 Optimization Challenge

#### Effect of InnerOpt \+ OuterOpt

To study how optimization affects the language modeling performance in asynchronous Local-SGD, we first experiment with different combinations of the inner and outer optimizers (we use A+B to denote A and B as the inner and outer optimizer, respectively): SGD+Nesterov, SGD+Adam, AdamW+SGD, AdamW+SGD Momentum, AdamW+Adam, AdamW+Nesterov. The hyperparameters for each combination are tuned separately, for AdamW as InnerOpt we kept the default values. We assume there are k=4k=4 workers, whose device speed is shown in Figure 4. Then we apply asynchronous Local-SGD finetuning on a 20M-parameter language model for 64,00064{,}000 steps per worker (256,000256{,}000 local steps in total), where the initial model checkpoint was pretrained for 24,00024{,}000 steps with Adam without distributed training. We choose finetuning with Local-SGD as it has been observed that Local-SGD style methods work well in finetuning but is less efficient from scratch (Lin et al., 2018), though others have also observed that Local-SGD works well even for training from scratch (Douillard et al., 2023). The learning rate scheduling and task scheduling follow the procedures described in Section 3. We use inner steps = 5050 across all workers in all experiments by default. The result is shown in Figure 5.

Figure 4: Steps per second for each device. Figure 5: Performance of using different combinations of inner and outer optimizers for asynchronous Local-SGD training on a 20M language model with 4 workers.

Observation The analysis reveals that combining AdamW as the inner optimizer with Nesterov momentum as the outer optimizer yields the best results, aligning with findings from synchronous training, like the DiLoCo method. Notably, using AdamW as the outer optimizer is less effective. This may be because AdamW, derived from Adam, introduces a normalization effect, which can be counterproductive in Local-SGD where pseudo-gradients tend to be larger than true gradients, potentially slowing convergence. When AdamW is used in the inner optimization, SGD, SGD Momentum, and Nesterov show comparable performance. However, Nesterov not only stabilizes the learning curve but also slightly improves final performance. This can be attributed to its update mechanism (here we abuse the notation and let tt denote tservert_{\text{server}}):

| mt+1=β​mt+gtθt+1=θt−ϵ⁡(β2​mt+(1+β)​gt),\begin{split}m_{t+1}&=\beta m_{t}+g_{t}\\\ \theta_{t+1}&=\theta_{t}-\epsilon\big(\beta^{2}m_{t}+(1+\beta)g_{t}\big),\end{split} |  | (4)  
---|---|---|---  
  
where ϵ\epsilon is the learning rate, mtm_{t} is the momentum, gtg_{t} the gradient at time tt, and β∈(0,1)\beta\in(0,1) the decay factor. The key difference between Nesterov and SGD Momentum is in how Nesterov adjusts the weightings, reducing the momentum component (β2\beta^{2} instead of β\beta) and increasing the gradient component (1+β1+\beta instead of 11). This suggests that momentum plays a crucial yet intricate role in Local-SGD.

#### Momentum in the OuterOpt

To delve deeper into the momentum term’s impact on the outer optimizer, we conducted comparative analyses between AdamW+SGD and AdamW+Nesterov under both synchronous and asynchronous training settings. These experiments were carried out under identical conditions as described earlier. The results are reported in Figure 6.

Figure 6: Comparison of AdamW+SGD and AdamW+Nesterov in both synchronous and asynchronous Local-SGD training.

Observation The figure clearly shows that in asynchronous Local-SGD, AdamW+SGD, which lacks a momentum term, leads to better final perplexity and learning efficiency than its synchronous counterpart. However, incorporating Nesterov momentum into the OuterOpt significantly boosts the performance of synchronous Local-SGD, outperforming the asynchronous version. It’s noteworthy that asynchronous AdamW+Nesterov remains the best performer across all tested combinations of inner and outer optimizers (as seen in Figure 5). This observation indicates that while momentum is beneficial in asynchronous Local-SGD for language modeling, its effect is more pronounced in synchronous settings.

#### Is Staleness the Cause?

We further apply the asynchronous DiLoCo algorithm with homogeneous devices. By doing so, we maximally diminish the staled gradient problem in Local-SGD, which refers to the fact that we are using an outdated outer gradient to update the server model. In particular, this means if we have kk workers, all of them will return the computed outer gradient back to the server at the same time. Therefore, the only staleness comes from the fact that we are sequentially applying the individual updates instead of aggregating them together and apply it once. Results are summarized in Figure 7.

Figure 7: Async. DiLoCo with heterogeneous devices.

Observation Figure 7 reveals a notable finding: even with homogeneity among workers, asynchronous DiLoCo significantly lags behind its synchronous counterpart. This suggests that the _inherent staleness_ from sequentially applying simultaneous updates leads to considerable performance drops. To elucidate this effect, let’s consider a scenario with k=4k=4 workers providing identical outer gradients (denoted as gg). The standard Nesterov momentum update is outlined in Equation (4). In a sequential application of pseudo-gradients:

| mt+1=β4​mt+(1+β+β2+β3)​gθt+1=θt−ϵ⁡((4+4​β+3​β2+2​β3+β4)​gCLOSEOPEN+β2​(1+β+β2+β3)​mt).\begin{split}m_{t+1}&=\beta^{4}m_{t}+(1+\beta+\beta^{2}+\beta^{3})g\\\ \theta_{t+1}&=\theta_{t}-\epsilon\big((4+4\beta+3\beta^{2}+2\beta^{3}+\beta^{4})g\\\ &+\beta^{2}(1+\beta+\beta^{2}+\beta^{3})m_{t}\big).\end{split} |  | (5)  
---|---|---|---  
  
From this, we observe that sequential application results in a more rapidly decaying momentum term but amplifies the actual change in parameter θ\theta. Consequently, a higher β\beta maintains more recent momentum but may lead to greater changes in parameters, and vice versa. Importantly, this imbalance cannot be simply rectified by reducing the learning rate.

#### Baselines

We consider several synchronous baselines from the literature, and their naive application to an asynchronous setting: 1) Finetune 1 worker (4xbatch): This involves finetuning a single worker with a larger batch size, equating to synchronous SGD. 2) DiLoCo (Douillard et al., 2023): This synchronous Local-SGD method combines AdamW with Nesterov. 3) Async. DiLoCo: The asynchronous version of DiLoCo.

#### Existing Fixes

We investigated potential fixes from the asynchronous Local-SGD literature to address observed challenges. The following methods were considered: 1) Async. DiLoCo + Poly (Xie et al., 2019): Extends Async. DiLoCo by downweighting the pseudo-gradient with g←(1+staleness)−0.5​gg\leftarrow(1+\text{staleness})^{-0.5}g. 2) Async. DiLoCo + PolyThres: Adds a threshold to discard updates with staleness beyond 10. 3) Async. DiLoCo + Delay Comp. (Zheng et al., 2017): Introduces delay compensation (Delay Comp.) to approximate true pseudo-gradients. Denote the gradient function at θ\theta as g⁡(θ)g(\theta), then the main idea of delay compensation is to approximate the true gradient g⁡(θt)g(\theta_{t}) by a stale gradient g⁡(θt′)g(\theta_{t^{\prime}}) (t′<tt^{\prime}<t) with the first-order Taylor approximation, e.g., g⁡(θt)≈g⁡(θt′)+∇g​(θt′)​(θt−θt′)g(\theta_{t})\approx g(\theta_{t^{\prime}})+\nabla g(\theta_{t^{\prime}})(\theta_{t}-\theta_{t^{\prime}}). In practice, the Hessian ∇g​(θt′)\nabla g(\theta_{t^{\prime}}) is approximated by diagonalized gradient outer product, e.g., ∇g​(θt′)≈λ​g​(θt′)⊙g⁡(θt′)\nabla g(\theta_{t^{\prime}})\approx\lambda g(\theta_{t^{\prime}})\odot g(\theta_{t^{\prime}}), where λ∈ℝ+\lambda\in\mathbb{R}^{+} is a hyperparameter. In our setting, we apply the delay compensation technique to pseudogradients instead of gradients. 4) Async. Buffer: Accumulates and averages all gradients in a First-In, First-Out fashion before applying Nesterov updates; a variation of the original FedBuff algorithm (Nguyen et al., 2022), using AdamW+Nesterov. The results are provided in Figure 8.

Figure 8: Comparison of different asynchronous Local-SGD approaches. Poly, PolyThres, and Delay Comp. barely improve the async. Local-SGD performance. Async. Buffer significantly closes the gap between sync. and async. training, while introducing instability in early stage of training.

Observation Polynomial discounting of the pseudo-gradient shows marginal benefits. Thresholding and delay compensation techniques don’t offer much improvements. Again, the fact that delay compensation is not working well points out the difference between asynchronous SGD and asynchronous Local-SGD. The Async. Buffer method excels at convergence but exhibits instability early in training. Crucially, _none_ of the methods match the performance of the synchronous DiLoCo baseline.

## 5 Proposed Solutions

In addressing the optimization challenges outlined in Section 4, we developed two strategies.

#### Delayed Nesterov Update

Notably, the Async. Buffer method demonstrated promising performance (as shown in Figure 8). Additionally, our analysis revealed that asynchronous training with AdamW+SGD, sans outer momentum, outperforms synchronous methods (Figure 5). Based on these insights, we propose the _Delayed Nesterov_ (DN) strategy, which represents the sync() function in Algorithm 2. This approach involves using the Nesterov update intermittently—every NN server updates. Between Nesterov updates, we aggregate pseudo-gradients in a buffer Δ\Delta and update the model parameters using gradient descent (or gradient descent plus a small fraction of the old momentum). To balance gradient and momentum-based descent, we introduce a parameter c∈[0,1/N]c\in[0,1/N]. A cc value of 0 indicates pure gradient descent between Nesterov updates, while cc equal to 1 evenly distributes the momentum term over NN updates. The specifics of this algorithm are detailed in Algorithm 3. Unlike the Async. Buffer (Nguyen et al., 2022), which updates model parameters only once in NN periods, the Delayed Nesterov continuously updates using gradients, incorporating a fraction of the old momentum, and updates the momentum term once every NN server updates.

Algorithm 3 Delayed Nesterov Update.

Initial model parameter θ0\theta_{0}

Momentum decay β∈(0,1)\beta\in(0,1)

Momentum activation c∈[0,1/N]c\in[0,1/N]

⊳\triangleright default to c=0c=0

Buffer size NN

t=0t=0

m0=0m_{0}=0 ⊳\triangleright momentum

Δ=0\Delta=0 ⊳\triangleright aggregated gradient

while not finished do

Receive the pseudo-gradient gtg_{t}

⊳\triangleright sync. step in Alg. 2. 

Δ←Δ+gt\Delta\leftarrow\Delta+g_{t}

if (t+1)%​N==0(t+1)\penalty\ \%\penalty\ N==0 then

mt+1←β​mt+Δ/Nm_{t+1}\leftarrow\beta m_{t}+\Delta/N

θt+1←θt−ϵ⁡((1−c​N+c)​β​mt+1+gt/N)\theta_{t+1}\leftarrow\theta_{t}-\epsilon\big((1-cN+c)\beta m_{t+1}+g_{t}/N\big)

Δ=0\Delta=0

else

mt+1←mtm_{t+1}\leftarrow m_{t} ⊳\triangleright delay momentum update

θt+1←θt−ϵ⁡(c​β​mt+1+gt/N)\theta_{t+1}\leftarrow\theta_{t}-\epsilon\big(c\beta m_{t+1}+g_{t}/N\big)

end if

t←t+1t\leftarrow t+1

end while

#### Dynamic Local Updates

The Delayed Nesterov approach addresses the momentum challenge in the OuterOpt by buffering pseudo-gradients and strategically delaying momentum updates. An alternative perspective considers synchronous training as a solution, where pseudo-gradients from all workers are synchronized. However, the diversity in device capabilities often hinders simultaneous pseudo-gradient returns, if each worker executes the same number of local training steps. A viable workaround involves customizing local training steps (e.g., ww.steps) based on the processing speed of each device. In particular, denote v⁡(w)v(w) as the training speed (in terms of steps per second) for worker ww, we can compute a worker’s desired training steps as:

| w.step=⌊v⁡(w)maxw′∈𝒲⁡v⁡(w′)​H⌋,w.\text{step}=\bigg\lfloor\frac{v(w)}{\max_{w^{\prime}\in\mathcal{W}}v(w^{\prime})}H\bigg\rfloor, |  | (6)  
---|---|---|---  
  
where HH denotes the number of local training steps the fastest worker runs and ⌊x⌋\lfloor x\rfloor denotes the largest integer not greater than xx.33 3 Here, we implicitly assumes the device speeds are known a priori. If this is not the case, it is straightforward to estimate the device speed based on empirical observations. We name this approach the Dynamic Local Updates (DyLU). This adjustment allows slower workers to execute fewer steps, aligning the completion times across different workers. Incorporating a grace period for model synchronization in this setup further reduces the impact of stale gradients, improving overall performance.

## 6 A Minimal Toy Example

For the convenience of future research and quick prototyping of new ideas, we present a minimal toy example that replicates the observed optimization challenge in asynchronous Local-SGD (See Figure 9).44 4 Please check the Colab at <https://github.com/google-deepmind/asyncdiloco> The task is to perform classification on a mixture of mixtures of Gaussian data.

Figure 9: Replicating the optimization challenge on the toy example. Left: the dataset consists of a mixture of mixtures of Gaussians. Right: Async. Local-SGD performs worse/better than sync. Local-SGD when using AdamW+Nesterov/AdamW+SGD.

Observation Comparing Figure 9 to Figure 6, we observe that the toy example demonstrate the same optimization challenge.

## 7 Experiments

This section details experiments conducted to assess the efficacy of our two proposed methods, Delayed Nesterov (DN) and Dynamic Local Updates (DyLU). Additionally, ablation studies explore the effectiveness of these methods as we vary the number of workers and model sizes.

#### Evaluating Delayed Nesterov (DN) and Dynamic Local Updates (DyLU)

Figure 2 compares asynchronous Local-SGD with DN and DyLU against baselines such as single worker finetuning and DiLoCo, using the same setup as in Figure 8.

Observation The results demonstrate that DN combined with DyLU significantly reduces perplexity, surpassing the synchronous DiLoCo’s performance over updates. Additionally, DN+DyLU outperforms in terms of time efficiency, avoiding delays from slower workers.

#### Assessing Different Levels of Worker Heterogeneity

We examine how both the proposed DN+DyLU method and vanilla asynchronous DiLoCo fare under varying degrees of worker device heterogeneity, as shown in Figure 10 (perplexity curve) and Table 1 (final perplexity).

Level of heterogeneity | no | slight | moderate | very  
---|---|---|---|---  
Pretrained (24K) | 61.64 | 61.64 | 61.64 | 61.64  
Finetune (4×4\times batch size) | 42.47 | 42.47 | 42.47 | 42.47  
DiLoCo (Douillard et al., 2023) | 41.35 | 41.35 | 41.35 | 41.35  
Async. DiLoCo | 44.27 | 44.38 | 44.29 | 44.27  
Async. DN + DyLU (ours) | 41.27 | 41.27 | 41.09 | 41.13  
  
Table 1: Varying the level of worker heterogeneity (top-left, top-right, bottom-left, and bottom-right of Figure 10 correspond to no, slight, moderate, and very, respectively).

Observation DN+DyLU consistently excels across all heterogeneity levels.55 5 We notice that Async. DN+DyLU performs slightly better than DiLoCo when there is no heterogeneity, this is due to the numerical error, as the two methods reduce to the same and the training curves match almost perfectly. Interestingly, even with homogeneous devices, vanilla asynchronous DiLoCo struggles, suggesting that the issue partly lies in the sequential application of pseudogradients. This underscores the importance of delayed momentum updates. Additionally, a periodic oscillation in performance is observed in certain device groupings, further highlighting the lack of robustness of the original asynchronous algorithm.

#### Ablation with Different Numbers of Workers

We apply DN+DyLU while varying the number of workers (4, 8, 16) using a 20M model, with results summarized in Figure 11 (perplexity curve) and Table 2 (final perplexity).

Number of workers kk | 4 | 8 | 16  
---|---|---|---  
Pretrained (24K) | 61.64 | 61.64 | 61.64  
Finetune (k×k\times batch size) | 42.47 | 41.28 | 40.60  
DiLoCo (Douillard et al., 2023) | 41.35 | 41.23 | 41.25  
Async. DiLoCo | 44.27 | 44.23 | 44.23  
Async. DN + DyLU (ours) | 41.13 | 41.02 | 40.98  
  
Table 2: Varying the number of workers.

Observation As the number of workers increases, the benefit of Local-SGD training diminishes. Notably, with 16 workers, single worker finetuning (16x batch size) shows the best performance over updates. Yet, DN+DyLU closely aligns with synchronous DiLoCo in performance, demonstrating its potential as a DiLoCo alternative in heterogeneous settings.

#### Ablation with Different Model Sizes

Lastly, we apply DN+DyLU to models of varying sizes (20M, 60M, 150M), with results summarized in Figure 12 (perplexity curve) and Table 3 (final perplexity).

Model size | 20M | 60M | 150M  
---|---|---|---  
Pretrained (24K) | 61.64 | 30.19 | 22.80  
Finetune (4x batch size) | 42.47 | 24.80 | 17.47  
DiLoCo (Douillard et al., 2023) | 41.35 | 24.55 | 17.23  
Async. DiLoCo | 44.27 | 25.64 | 18.08  
Async. DN + DyLU (ours) | 41.13 | 24.53 | 17.26  
  
Table 3: Varying the model sizes.

Observation Both synchronous and asynchronous Local-SGD methods outperform the approach of finetuning a single worker with an increased batch size. Notably, this advantage becomes more pronounced during the later stages of convergence, aligning with findings from previous research that highlight Local-SGD’s superior generalization capabilities (Gu et al., 2023). Additionally, our proposed DN+DyLU method demonstrates consistent efficacy across various model sizes. It’s important to note that the performance disparity between synchronous and asynchronous DiLoCo does not diminish even as the model size increases.

#### Ablation with Different cc

We apply c∈{0,0.1}c\in\\{0,0.1\\} in Async. DN+DyLU with varying kk (4, 8, 16) and model sizes (20M, 60M, 150M), with the 4 “very" heterogeneous workers. This is because when the level of heterogeneity is small, using different cc will have smaller difference (e.g., when there is no heterogeneity, any cc results in the same algorithm). Results are summarized in Table 4.

Number of workers kk | 4 | 8 | 16  
---|---|---|---  
Async. DN + DyLU (c=0c=0) | 41.13 | 41.02 | 40.98  
Async. DN + DyLU (c=0.1c=0.1) | 41.16 | 40.93 | 41.04  
Model size | 20M | 60M | 150M  
Async. DN + DyLU (c=0c=0) | 41.13 | 24.53 | 17.26  
Async. DN + DyLU (c=0.1c=0.1) | 41.16 | 24.69 | 17.27  
  
Table 4: Varying the c∈{0,0.1}c\in\\{0,0.1\\} in Algorithm 3.

Observation Empirically, we observe no significant difference between c=0c=0 and c=0.1c=0.1, indicating that adding slight momentum at intermediate steps does not help too much. As a result, we set c=0c=0 as the default value in Algorithm 3, which corresponds to performing SGD updates between two consecutive Nesterov updates. Note that setting the value of cc does not introduce any overhead to the overall algorithm.

Figure 10: Varying the heterogeneity in devices. Figure 11: Varying the number of workers. Figure 12: Varying the model size.

## 8 Related Work

This section provides a concise overview of the literature on federated learning and local-SGD style distributed optimization, particularly focusing on their applications in asynchronous settings.

#### Local-SGD and Distributed Optimization

Local-SGD is a specific distributed optimization technique designed to reduce communication frequency (Stich, 2018; Zhang et al., 2016; Bijral et al., 2016; McDonald et al., 2010; Coppola, 2015; Zinkevich et al., 2010). The core principle of Local-SGD is to let each worker execute several local training iterations prior to engaging in global synchronization. This technique was later applied to the federated learning setting, leading to the development of the FedAvg method (McMahan et al., 2017), which aims to reduce communication overhead. Unlike Local-SGD, federated learning also addresses user privacy issues and typically involves heterogeneous devices. To further minimize communication overhead, FedOpt integrates adaptive optimization methods like SGD momentum and Adam (Reddi et al., 2020). However, as client/worker heterogeneity increases, the convergence rate often deteriorates. Methods like SCAFFOLD (Karimireddy et al., 2020) and MIME (Karimireddy et al., 2021) have been introduced to adapt these optimization methods for heterogeneous environments.

#### Asynchronous Training

Asynchronous training was developed to mitigate the “straggler effect" observed in synchronous distributed optimization, where learning efficiency is bottlenecked by the slowest worker (Koh et al., 2006; Recht et al., 2011; Dean et al., 2012; Lian et al., 2015; Lian et al., 2018; Diskin et al., 2021b). A significant challenge in asynchronous optimization is the staled gradient problem, which occurs when an outdated gradient is applied to a recently updated model. Asynchronous SGD with delay compensation (Zheng et al., 2017) addresses this issue by approximating the true gradient using the old gradient. Asynchronous methods have also been explored in federated learning contexts (Xie et al., 2019). Despite the challenge, asynchronous training has demonstrated success for language modeling as well (Diskin et al., 2021b), by using heterogeneous devices across the world.

#### Local-SGD for Language Modeling

The concept of local-SGD (or FedAvg) has previously been applied in the realm of language modeling. Cross-device federated learning, for instance, has been utilized to pretrain and fine-tune language models (Hilmkil et al., 2021; Ro et al., 2022; Ryabinin et al., 2021; Diskin et al., 2021a; Presser, 2020; Borzunov et al., 2022). More recently, DiLoCo has extended the local-SGD methodology to encompass larger language models, specifically proposing the use of AdamW + Nesterov momentum as the InnerOpt \+ OuterOpt pairing. In asynchronous settings, the FedBuff (Nguyen et al., 2022) algorithm buffers pseudogradients from clients, updating the server model only after accumulating a sufficient number of pseudogradients. TimelyFL (Zhang et al., 2023) aims to reduce asynchrony by allowing slower devices to train only parts of the model.

## 9 Limitations

This study, while comprehensive, has several limitations. First, we identify a significant optimization challenge linked to momentum updates in the OuterOpt, but the precise cause of this issue remains unclear. Understanding this challenge with robust theoretical backing presents an intriguing avenue for future research. Second, our empirical observations suggest that the advantages of the Local-SGD method diminish with an increasing number of workers, a phenomenon whose underlying reasons are yet to be understood. This issue currently hinders the scalability of asynchronous Local-SGD. Finally, although our proposed method DN+DyLU shows improved empirical performance, it lacks formal theoretical convergence guarantees, an aspect that merits further investigation.

## 10 Conclusion

This study presents a thorough examination of asynchronous Local-SGD in language modeling. Our central finding is that while momentum in the outer optimization loop is crucial, it may be less effective in asynchronous scenarios compared to synchronous ones when implemented naively. To bridge this gap, we introduce a novel approach, focusing on sporadic momentum updates using buffered pseudogradients, combined with continuous stochastic pseudogradient updates. Furthermore, our research reveals that tailoring local training steps to each worker’s computational speed is not only a straightforward but also an efficient strategy to enhance performance.

However, there is much work to be done. In the standard (as opposed to the “local”) gradient descent setting, the optimal batch size in terms of decreasing loss as quickly as possible in terms of number of weight updates is not usually “as large as possible”. In our view, similarly, there is hope for asynchronous Local-SGD methods that give better results per local update than synchronous Local-SGD.

## Acknowledgements

We would like to thank Adam Fisch for his valuable feedback.

\nobibliography

*

## References

  * Bijral et al. (2016) Avleen S Bijral, Anand D Sarwate, and Nathan Srebro.  On data dependence in distributed stochastic optimization.  _arXiv preprint arXiv:1603.04379_ , 2016. 
  * Borzunov et al. (2022) Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel.  Petals: Collaborative inference and fine-tuning of large models.  _arXiv preprint library_ , 2022. 
  * Coppola (2015) Gregory Francis Coppola.  Iterative parameter mixing for distributed large-margin training of structured predictors for natural language processing.  2015\. 
  * Dean et al. (2012) Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, et al.  Large scale distributed deep networks.  _Advances in neural information processing systems_ , 25, 2012. 
  * Diskin et al. (2021a) Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitry Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf, and Gennady Pekhimenko.  Distributed deep learning in open collaborations.  _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021a. 
  * Diskin et al. (2021b) Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Anton Sinitsin, Dmitry Popov, Dmitry V Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, et al.  Distributed deep learning in open collaborations.  _Advances in Neural Information Processing Systems_ , 34:7879–7897, 2021b. 
  * Douillard et al. (2023) Arthur Douillard, Qixuan Feng, Andrei A Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc’Aurelio Ranzato, Arthur Szlam, and Jiajun Shen.  Diloco: Distributed low-communication training of language models.  _arXiv preprint arXiv:2311.08105_ , 2023. 
  * Gu et al. (2023) Xinran Gu, Kaifeng Lyu, Longbo Huang, and Sanjeev Arora.  Why (and when) does local sgd generalize better than sgd?  _arXiv preprint arXiv:2303.01215_ , 2023. 
  * Hilmkil et al. (2021) Agrin Hilmkil, Sebastian Callh, Matteo Barbieri, Leon René Sütfeld, Edvin Listo Zec, and Olof Mogren.  Scaling federated learning for fine-tuning of large language models.  In _International Conference on Applications of Natural Language to Information Systems_ , pages 15–23. Springer, 2021. 
  * Hoffmann et al. (2022) Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre.  Training compute-optimal large language models.  _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 
  * Karimireddy et al. (2020) Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh.  Scaffold: Stochastic controlled averaging for federated learning.  In _International conference on machine learning_ , pages 5132–5143. PMLR, 2020. 
  * Karimireddy et al. (2021) Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian U Stich, and Ananda Theertha Suresh.  Breaking the centralized barrier for cross-device federated learning.  _Advances in Neural Information Processing Systems_ , 34:28663–28676, 2021. 
  * Koh et al. (2006) Byung-Il Koh, Alan D George, Raphael T Haftka, and Benjamin J Fregly.  Parallel asynchronous particle swarm optimization.  _International journal for numerical methods in engineering_ , 67(4):578–595, 2006. 
  * Lian et al. (2015) Xiangru Lian, Yijun Huang, Yuncheng Li, and Ji Liu.  Asynchronous parallel stochastic gradient for nonconvex optimization.  _Advances in neural information processing systems_ , 28, 2015. 
  * Lian et al. (2018) Xiangru Lian, Wei Zhang, Ce Zhang, and Ji Liu.  Asynchronous decentralized parallel stochastic gradient descent.  In _International Conference on Machine Learning_ , pages 3043–3052. PMLR, 2018. 
  * Lin et al. (2018) Tao Lin, Sebastian U Stich, Kumar Kshitij Patel, and Martin Jaggi.  Don’t use large mini-batches, use local sgd.  _arXiv preprint arXiv:1808.07217_ , 2018. 
  * Lin et al. (2020) Tao Lin, Sebastian U. Stich, Kumar Kshitij Patel, and Martin Jaggi.  Don’t use large mini-batches, use local sgd.  _Proceedings of the International Conference on Learning Representations (ICLR)_ , 2020. 
  * McDonald et al. (2010) Ryan McDonald, Keith Hall, and Gideon Mann.  Distributed training strategies for the structured perceptron.  In _Human language technologies: The 2010 annual conference of the North American chapter of the association for computational linguistics_ , pages 456–464, 2010. 
  * McMahan et al. (2017) Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.  Communication-efficient learning of deep networks from decentralized data.  In _Artificial intelligence and statistics_ , pages 1273–1282. PMLR, 2017. 
  * Nguyen et al. (2022) John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek, and Dzmitry Huba.  Federated learning with buffered asynchronous aggregation.  In _International Conference on Artificial Intelligence and Statistics_ , pages 3581–3607. PMLR, 2022. 
  * Presser (2020) Shawn Presser.  Swarm training, 2020.  URL <https://battle.shawwn.com/swarm-training-v01a.pdf>. 
  * Raffel et al. (2020) Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.  Exploring the limits of transfer learning with a unified text-to-text transformer.  _Journal of Machine Learning Research_ , 2020. 
  * Recht et al. (2011) Benjamin Recht, Christopher Re, Stephen Wright, and Feng Niu.  Hogwild!: A lock-free approach to parallelizing stochastic gradient descent.  _Advances in neural information processing systems_ , 24, 2011. 
  * Reddi et al. (2020) Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečnỳ, Sanjiv Kumar, and H Brendan McMahan.  Adaptive federated optimization.  _arXiv preprint arXiv:2003.00295_ , 2020. 
  * Ro et al. (2022) Jae Hun Ro, Theresa Breiner, Lara McConnaughey, Mingqing Chen, Ananda Theertha Suresh, Shankar Kumar, and Rajiv Mathews.  Scaling language model size in cross-device federated learning.  _arXiv preprint arXiv:2204.09715_ , 2022. 
  * Ryabinin et al. (2021) Max Ryabinin, Eduard Gorbunov, Vsevolod Plokhotnyuk, and Gennady Pekhimenko.  Moshpit sgd: Communication-efficient decentralized training on heterogeneous unreliable devices.  _Advances in Neural Information Processing Systems_ , 34:18195–18211, 2021. 
  * Stich (2018) Sebastian U Stich.  Local sgd converges fast and communicates little.  _arXiv preprint arXiv:1805.09767_ , 2018. 
  * Xie et al. (2019) Cong Xie, Sanmi Koyejo, and Indranil Gupta.  Asynchronous federated optimization.  _arXiv preprint arXiv:1903.03934_ , 2019. 
  * Zhang et al. (2016) Jian Zhang, Christopher De Sa, Ioannis Mitliagkas, and Christopher Ré.  Parallel sgd: When does averaging help?  _arXiv preprint arXiv:1606.07365_ , 2016. 
  * Zhang et al. (2023) Tuo Zhang, Lei Gao, Sunwoo Lee, Mi Zhang, and Salman Avestimehr.  Timelyfl: Heterogeneity-aware asynchronous federated learning with adaptive partial training.  In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 5063–5072, 2023. 
  * Zheng et al. (2017) Shuxin Zheng, Qi Meng, Taifeng Wang, Wei Chen, Nenghai Yu, Zhi-Ming Ma, and Tie-Yan Liu.  Asynchronous stochastic gradient descent with delay compensation.  In _International Conference on Machine Learning_ , pages 4120–4129. PMLR, 2017. 
  * Zinkevich et al. (2010) Martin Zinkevich, Markus Weimer, Lihong Li, and Alex Smola.  Parallelized stochastic gradient descent.  _Advances in neural information processing systems_ , 23, 2010. 


## Supplementary Materials

### 10.1 Implementation Details

Hyperparameter | Value  
---|---  
Inner learning rate | 0.10.1  
Final inner learning rate | 0.00.0, 0.000001\mathbf{0.000001}, 0.00020.0002  
Number of warmup steps | 00, 1,000\mathbf{1{,}000}  
Weight decay | 0.10.1  
Batch Size | 128, 512  
Sequence length | 256256  
Outer Optimizer | SGD, SGDM, Nesterov, Adam, delayed momentum SGD  
Inner Optimizer | SGD, AdamW  
Outer learning rate | 0.030.03, 0.30.3, 0.1\mathbf{0.1}, 0.7\mathbf{0.7}  
Async soup weight | 0.1250.125, 0.250.25, 0.50.5, 1.0\mathbf{1.0}  
Async soup method | constant, polynomial, svrg  
Delay period | 4, 8, 16  
Communication frequency HH | 50, 100, 150  
Number of pretraining steps | 24,00024,000  
  
Table 5: Optimization Hyperparameters evaluated during in this work. Chosen values for main experiments are highlighted in bold. Table 6: Model Configuration for the three evaluated sizes. All are based on the transformer architecture, chinchilla-style (Hoffmann et al., 2022). Hyperparameter | 20M | 60M | 150M  
---|---|---|---  
Number of layers | 6 | 3 | 12  
Hidden dim | 256 | 896 | 896  
Number of heads | 4 | 16 | 16  
K/V size | 64 | 64 | 64  
Vocab size | 32,00032{,}000  
  
#### Network Architecture

We displayed in Table 6 the architectural difference between the 20M, 60M, and 150M models. They are all transformer decoder-only, based on the Chinchilla family (Hoffmann et al., 2022).

#### Training Dataset

We consider a language modeling task on the C4 dataset, a dataset derived from Common Crawl (Raffel et al., 2020). The total number of steps is set to 88,00088{,}000 for all models, with 24,00024{,}000 steps of pre-training done without any federated learning methods, akin to post Local-SGD (Lin et al., 2020).

#### Hyperparameters

In Table 5, we outline the optimization hyperparameters considered for this study.

#### Inner Optimizer States

Following Douillard et al. (2023), in all experiments, when worker B picks up the data shard worker A just finishes training on, we reset the AdamW’s optimizer state. In other words, each local worker-side training is an independent training process with a new optimizer, and only the learning rate is adjusted according as described in Section 3.

### 10.2 Aync. Training Pseudocode

In this section, we provide the pseudocode for the train() and get_worker() functions in Algorithm 2.

1: Available workers 𝒲\mathcal{W}

2: Current server model θ\theta

3: for w∈𝒲w\in\mathcal{W} do

4: Sample shard 𝒟′\mathcal{D}^{\prime} for ww (Eq. 2). 

5: ww.local_updates = DyLU(𝒟′\mathcal{D}^{\prime}) (Eq. 6). 

6: Decide lr schedule (ww.lr) (Eq. 3). 

7: ww.update = train_worker(ww, 𝒟′\mathcal{D}^{\prime}, θ\theta). 

8: end for

Algorithm 4 train() in Algorithm 2.

1: Workers 𝒲\mathcal{W}

2: Grace period τgrace\tau_{\text{grace}}

3: Start of the grace period τsync\tau_{\text{sync}}. 

4: if all workers in 𝒲\mathcal{W} are not done then

5: return null 

6: else

7: ww = earliest completed worker in 𝒲\mathcal{W}. 

8: if w.completed_time−τsync≤τgracew.\text{completed\\_time}-\tau_{\text{sync}}\leq\tau_{\text{grace}} then

9: return ww

10: else

11: return null 

12: end if

13: end if

Algorithm 5 get_worker() in Algorithm 2.

Experimental support, please [view the build logs](./2401.09135v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.


**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our [**major funders**](https://info.arxiv.org/about/major_funders.html), [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe.html) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
