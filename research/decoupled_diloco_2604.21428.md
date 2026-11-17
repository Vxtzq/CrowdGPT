##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[ Back to arXiv ](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2604.21428v1 "Back to abstract page") [ Download PDF](/pdf/2604.21428v1 "Download PDF") [ ](javascript:toggleNavTOC\(\); "Toggle navigation") [ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

  1. Abstract
  2. 1 Introduction
     1. Contributions
  3. 2 Preliminaries
  4. 3 Decoupled DiLoCo
     1. 3.1 The Learner: Local Optimization
     2. 3.2 The Syncer: Global Aggregation
        1. Adaptive Quorum Window.
     3. 3.3 Chaos engineering for LLMs
  5. 4 System design
     1. Learner workers.
     2. Syncer worker.
     3. State coordination.
     4. Summary of benefits.
  6. 5 Experiments
     1. 5.1 Experimental Setup
     2. 5.2 Resilience to hardware failures
     3. 5.3 Scavenging
     4. 5.4 Heterogeneous hardware
     5. 5.5 Scalability
  7. 6 Conclusion
  8. References
  9. A Related Work
     1. Distributed training at scale.
     2. DiLoCo and related methods.
     3. Asynchrony and fault-tolerance at LLM scales.
  10. B Post-training
  11. C Fragmentation strategies
     1. Layer fragmentation
     2. Tensor fragmentation
     3. Balanced tensor fragmentation
     4. Sub-tensor fragmentation
     5. C.1 Fragment Size Comparison
     6. C.2 Evaluation Results
  12. D Algorithmic Ablations
     1. D.1 Mixing learner and syncer params
     2. D.2 Merging methods
        1. Direct averaging
        2. Radial-directional averaging.
        3. Other options
        4. Results
     3. D.3 Outer gradients compression
     4. D.4 Number of learners impact on asynchronous resiliency
     5. D.5 Token budget impact
  13. E Infrastructure Details
     1. E.1 Deterministic Replay via Event Tapes
        1. Non-Determinism and Event Logging.
        2. Deterministic Replay.
        3. Synthetic Tape Generation.
     2. E.2 Consistent distributed checkpointing
     3. E.3 Distributed Learner Recovery
     4. E.4 Bandwidth profile
        1. Simulated bandwidth profile
        2. Non-collocated Learners
  14. F Evaluation Benchmarks
     1. F.1 Text Benchmarks
     2. F.2 Vision Benchmarks
     3. F.3 Full results
     4. F.4 Scavenging full results


[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2604.21428v1 [cs.CL] 23 Apr 2026

\uselogo

# Decoupled DiLoCo for Resilient Distributed Pre-training

Decoupled DiLoCo Team  Affiliation: Google DeepMind  Affiliation: Google Research 

###### Abstract

Modern large-scale language model pre-training relies heavily on the single program multiple data (SPMD) paradigm, which requires tight coupling across accelerators. Due to this coupling, transient slowdowns, hardware failures, and synchronization overhead stall the entire computation, wasting significant compute time at scale. While recent distributed methods like DiLoCo reduced communication bandwidth, they remained fundamentally synchronous and vulnerable to these system stalls. To address this, we introduce Decoupled DiLoCo, an evolution of the DiLoCo framework designed to break the lock-step synchronization barrier and go beyond SPMD to maximize training goodput. Decoupled DiLoCo partitions compute across multiple independent “learners” that execute local inner optimization steps. These learners asynchronously communicate parameter fragments to a central synchronizer, which circumvents failed or straggling learners by aggregating updates using a minimum quorum, an adaptive grace window, and dynamic token-weighted merging. Inspired by “chaos engineering”, we achieve significantly improved training efficiency in failure-prone environments with millions of simulated chips with strictly zero global downtime, while maintaining competitive model performance across text and vision tasks, for both dense and mixture-of-expert architectures.

## 1 Introduction

Modern LLM pre-training relies on the tightly coupled single program, multiple data (SPMD) paradigm (e.g., data, tensor, and sequence parallelism) that requires global synchronization at every step. This “monolithic” approach creates a major reliability bottleneck: a single hardware failure or straggler can stall the entire system. As compute scales, the sheer number of components transforms rare hardware failures into routine occurrences. This is exacerbated by the long duration of pre-training regimes, where frequent interruptions lead to significant downtime and wasted compute.

Framing this analogously to the CAP theorem (Brewer, 2000), we argue the primary bottleneck of modern pre-training is a rigid adherence to parameter consistency. To reason about this trade-off, we define an analog of the “CAP properties” for the pre-training setting as follows:

  * •

Consistency (C): Every accelerator maintains a view of a globally synchronized set of model weights.

  * •

Availability (A): Training continues in the presence of hardware failures.

  * •

Partition Tolerance (P): Training continues despite interconnect instability or communication delays.


Under this view, SPMD pre-training prioritizes consistency over all else: it sacrifices availability and partition tolerance to ensure every accelerator maintains a view of the global model state. While recent slice-granular “elastic” methods (Gemini Team, Google, 2025) can reconfigure the SPMD computation to run on a smaller subset of healthy accelerators, the global overhead of fault detection and cluster-wide resizing still incurs significant downtime, as illustrated in Figure 1 (top).

Recent distributed alternatives like DiLoCo (Douillard et al., 2024) and its streaming variant (Douillard et al., 2025) have successfully addressed communication bottlenecks by reducing bandwidth requirements through intermittent synchronization. However, because previous iterations of DiLoCo remained fundamentally synchronous, they still enforced strict consistency across the cluster, leaving the system just as vulnerable to localized hardware failures and straggler penalties.

Figure 1: Slice-granularity elasticity vs decoupling: the system continues training with fewer “slices” of TPU chips when there is a localized failure. When a failure occurs affecting 1 of the MM replicas, the decoupling approach allows the other M−1/M\nicefrac{{M-1}}{{M}} replicas in the system to continue stepping. 

Inspired by the Pathways vision (Barham et al., 2022), we argue that pre-training should prioritize availability and partition tolerance over consistency by moving beyond the present tightly-coupled paradigm. We propose Decoupled DiLoCo, a distributed training framework that decomposes a global cluster into independent, asynchronous “learners”. By evolving DiLoCo’s intermittent synchronization into a fully asynchronous communication protocol, we limit the “blast radius” of a hardware failure to a single learner. This allows the majority of the system to continue training uninterrupted during local failures or reconfigurations (Figure 1). Through careful system and algorithm co-design, we demonstrate that maximum training goodput can be achieved in chaotic environments without sacrificing final model quality.

#### Contributions

Our core contributions are:

  * •

Decoupled DiLoCo: We introduce a distributed training framework that evolves previous bandwidth-focused methods by decomposing monolithic SPMD clusters into independent, asynchronous learners. This prioritizes availability and partition tolerance over strict consistency.

  * •

Scalable System Architecture: We design a system architecture featuring a central synchronizer to facilitate asynchronous parameter reconciliation, enabling the framework to operate efficiently at pre-training scales.

  * •

Empirical Validation: We provide extensive empirical evidence that Decoupled DiLoCo achieves comparable downstream performance to standard data-parallel training across various model and compute scales for both dense and Mixture-of-Experts (MoE) architectures on text and multi-modal evaluations. Moreover we show that pre-training with our asynchronous framework doesn’t hinder later post-training capabilities.

  * •

Robustness via Chaos Engineering: We apply chaos engineering (Basiri et al., 2016) principles to LLM pre-training, demonstrating that our framework maintains high availability and model quality, even under aggressive, continuous hardware failures.


## 2 Preliminaries

Let θ(t)\theta^{(t)} denote the model parameters at step tt. We wish to train a model across MM compute clusters (called _learners_ for brevity), each of which has its own model replica θm(t)\theta_{m}^{(t)}, for m∈[M]m\in[M]. In SPMD data-parallel training, each learner computes a batched gradient, and a global gradient is aggregated across learners. This requires synchronization across the learners at every step, incurring potentially high amounts of communication and bandwidth. It also means that a slowdown in a single learner delays the entire computation, and a failure requires, in principle, a replay of the partially-completed step.

In DiLoCo (Douillard et al., 2024), each of the MM learners trains in parallel using an _inner optimizer_ (e.g. AdamW) on distinct data shards, and only synchronize weights every HH steps. Rather than averaging the weights, DiLoCo applies an _outer optimization_ step (Reddi et al., 2021) when tmodH=0t\bmod H=0 by treating differences in parameter space as gradient estimates:

| Δ(t)\displaystyle\Delta^{(t)} | =1M​∑m=1MΔm(t)=1M​∑m=1M(θm(t−H)−θm(t))\displaystyle=\dfrac{1}{M}\sum_{m=1}^{M}\Delta_{m}^{(t)}=\dfrac{1}{M}\sum_{m=1}^{M}\left(\theta_{m}^{(t-H)}-\theta_{m}^{(t)}\right) |  | (1)  
---|---|---|---|---  
| θm(t)\displaystyle\theta^{(t)}_{m} | =OuterOpt​(θm(t−H),Δ(t))\displaystyle=\texttt{OuterOpt}(\theta^{(t-H)}_{m},\Delta^{(t)}) |   
  
We refer to Δm(t)\Delta_{m}^{(t)} as an _outer gradient_. Douillard et al. (2024) found that letting OuterOpt be SGD with Nesterov momentum (Sutskever et al., 2013) greatly improves model quality compared to weight averaging (which corresponds to an outer optimizer of SGD with learning rate 1).

While DiLoCo reduces total bandwidth usage by a factor of HH, it does not reduce peak bandwidth. This can be achieved by Streaming DiLoCo (Douillard et al., 2025). Partition the model weights θ\theta into PP sets (called _fragments_), and let θm,p(t)\theta_{m,p}^{(t)} denote the pp-th fragment held by learner mm at step tt. Let t1,t2,…,tP∈{0,…,H−1}t_{1},t_{2},\dots,t_{P}\in\\{0,\dots,H-1\\} be distinct offsets. Then, at step tt we only synchronize a fragment pp if tmodH=tpt\bmod H=t_{p}.

Since (1) requires no actual gradient, the outer gradients can be computed fragment-wise. For tmodH=tpt\bmod H=t_{p}, we compute

| Δp(t)\displaystyle\Delta^{(t)}_{p} | =1M​∑m=1MΔm,p(t)=1M​∑m=1M(θm,p(t−H)−θm,p(t))\displaystyle=\dfrac{1}{M}\sum_{m=1}^{M}\Delta_{m,p}^{(t)}=\dfrac{1}{M}\sum_{m=1}^{M}\left(\theta_{m,p}^{(t-H)}-\theta_{m,p}^{(t)}\right) |  | (2)  
---|---|---|---|---  
| θm,p(t)\displaystyle\theta^{(t)}_{m,p} | =OuterOpt​(θm,p(t−H),Δp(t))\displaystyle=\texttt{OuterOpt}(\theta^{(t-H)}_{m,p},\Delta^{(t)}_{p}) |   
  
As written, (2) is blocking - the learners can’t continue training until they send Δm,p(t)\Delta_{m,p}^{(t)} and receive Δp(t)\Delta_{p}^{(t)}. However, Douillard et al. (2025) show that this communication and training can be overlapped - the learners can send Δm,p(t)\Delta_{m,p}^{(t)} and receive Δp(t)\Delta_{p}^{(t)} τ\tau steps later with little change in model quality for small values of τ\tau.

Streaming DiLoCo offers a number of benefits, including significant reductions in bandwidth usage (total and peak) and the ability to make communication across steps asynchronous. However, Streaming DiLoCo still requires lock-step training across learners. Thus, when using Streaming DiLoCo with all-reduce, issues such as stragglers and learner failures can significantly slow down or halt training entirely.

## 3 Decoupled DiLoCo

To break this lock-step barrier, we completely decouple the learners from one another. We now describe Decoupled DiLoCo in full. We train across MM learners, each with their own model copy θm\theta_{m}. As in Streaming DiLoCo, we partition the model across non-overlapping fragments {θm,p}p∈𝒫\\{\theta_{m,p}\\}_{p\in\mathcal{P}}, where P=|𝒫|P=|\mathcal{P}|. We employ the same fragment-wise outer optimization as in (2). However, instead of a blocking global exchange, this optimization is performed by a central synchronizer (the _syncer_), which asynchronously receives and sends updates to learners. We discuss the learners and syncer in detail below.

Figure 2: Decoupled DiLoCo. For illustrative purposes, a simple example with M=2M=2 learners, P=3P=3 fragments, synchronized at every step (H=3H=3), and overlapped over τ=2\tau=2 steps. The second learner stalls for three steps, but the overall training never stops. All missed updates are applied to the faulty learner’s state once it continues training.

### 3.1 The Learner: Local Optimization

In Algorithm 1, each learner mm operates independently on its own data shard 𝒟m\mathcal{D}_{m} (L8), without waiting for its peers. A learner continuously executes inner optimization steps (L11) using its inner optimizer (e.g., AdamW). Because learners are decoupled, they operate at varying speeds and may encounter transient failures. To account for this, we track the local progress by maintaining local counters for the number of steps tmt_{m} taken by the learner, the number of steps cm,pstepsc^{\text{steps}}_{m,p} taken since it received an update to fragment pp, and the number of tokens cm,ptokensc^{\text{tokens}}_{m,p} processed since it received an update to fragment pp.

At each step, the learner sends its metadata (tm,{cm,psteps}p∈𝒫,{cm,ptokens}p∈𝒫)(t_{m},\\{c^{\text{steps}}_{m,p}\\}_{p\in\mathcal{P}},\\{c^{\text{tokens}}_{m,p}\\}_{p\in\mathcal{P}}) to the syncer (L13). The syncer uses this to drive the merging of fragments across learners, pulling fragments as needed. Crucially, this communication occurs in the background, while the learner continues to perform optimization steps. By never waiting for the syncer or its peers, the learner maintains high local goodput and completely isolates the blast radius of any localized hardware failure. Concurrently, the learner listens for updates from the syncer. Upon receiving an updated fragment Θp\Theta_{p} (L15), the learner overwrites its local weights for that fragment and resets cm,pstepsc^{\text{steps}}_{m,p} and cm,ptokensc^{\text{tokens}}_{m,p} (L16), partially resynchronizing with the global trajectory without halting its progress. To determine when training is complete, the learner uses a global step count tt that is updated by the syncer (L17). This is necessary because learners may proceed at variable speeds, so by the time the desired number of global steps is reached, the learners will have completed different numbers of local steps.

### 3.2 The Syncer: Global Aggregation

The syncer is responsible for reconciling the divergent states of the asynchronous learners via an outer optimization step. Crucially, this is where our approach diverges from standard data-parallel or synchronous DiLoCo methods: to guarantee high availability and maximize global goodput, the syncer does not wait for a synchronized consensus from all MM learners.

Algorithm 1 Decoupled DiLoCo: Learner

1: Fragmented initial weights {Θp(0)}p∈𝒫\\{\Theta_{p}^{(0)}\\}_{p\in\mathcal{P}}, Data shard 𝒟m\mathcal{D}_{m}

2: Per-fragment sync interval HH, Offsets tpt_{p}

3: Optimizer InnerOpt

4: ∀p:θm,p(0)←Θp(0)\forall p:\theta_{m,p}^{(0)}\leftarrow\Theta_{p}^{(0)}

5: ∀p:cm,psteps←0,cm,ptokens←0\forall p:c^{\text{steps}}_{m,p}\leftarrow 0,c^{\text{tokens}}_{m,p}\leftarrow 0

6: tm=0t_{m}=0

7: t=0t=0

8: while step t≤Tt\leq T do

9:

10: ⊳\triangleright 1\. Inner optimization

11: tm=tm+1t_{m}=t_{m}+1

12: x∼𝒟mx\sim\mathcal{D}_{m}

13: ∀p:cm,psteps←cm,psteps+1,cm,ptokens←cm,ptokens+|x|\forall p:c^{\text{steps}}_{m,p}\leftarrow c^{\text{steps}}_{m,p}+1,\;c^{\text{tokens}}_{m,p}\leftarrow c^{\text{tokens}}_{m,p}+|x|

14: ℒ←f⁡(x,θm(tm−1))\mathcal{L}\leftarrow f(x,\theta_{m}^{(t_{m}-1)})

15: θm(tm)←InnerOpt​(θm(tm−1),∇ℒ)\theta_{m}^{(t_{m})}\leftarrow\texttt{InnerOpt}(\theta_{m}^{(t_{m}-1)},\nabla_{\mathcal{L}})

16:

17: ⊳\triangleright 2\. Metadata learner →\rightarrow syncer

18: Send​(Syncer,tm,{cm,psteps}p∈𝒫,{cm,ptokens}p∈𝒫)\texttt{Send}(\text{Syncer},t_{m},\\{c^{\text{steps}}_{m,p}\\}_{p\in\mathcal{P}},\\{c^{\text{tokens}}_{m,p}\\}_{p\in\mathcal{P}})

19:

20: ⊳\triangleright 3\. Communication syncer →\rightarrow learner

21: for (Θp,ts)∈RecvAllPending​(Syncer)(\Theta_{p},t_{s})\in\texttt{RecvAllPending}(\text{Syncer}) do

22: θm,p(t)←Θp\theta_{m,p}^{(t)}\leftarrow\Theta_{p}

23: cm,psteps←0,cm,ptokens←0c^{\text{steps}}_{m,p}\leftarrow 0,\;c^{\text{tokens}}_{m,p}\leftarrow 0

24: t=tst=t_{s}

25: end for

26: end while

Algorithm 2 Decoupled DiLoCo: Syncer

1: Fragmented initial weights {Θp(0)}p∈𝒫\\{\Theta_{p}^{(0)}\\}_{p\in\mathcal{P}}, MM learners 

2: Per-fragment sync interval HH

3: Minimum learners KK

4: Optimizer OuterOpt

5: for step t=1​…​Tt=1\dots T do

6: if ∃p\exists p s.t. tmodH=tpt\bmod H=t_{p} then

7:

8: ⊳\triangleright 1\. Communication learner →\rightarrow syncer

9: 𝒰t←Recv​(Learners,at least ​K)\mathcal{U}_{t}\leftarrow\texttt{Recv}(\text{Learners},\text{at least }K)

10: {tm,cm,psteps,cm,ptokens}m∈ℳt←𝒰t\\{t_{m},c^{\text{steps}}_{m,p},c^{\text{tokens}}_{m,p}\\}_{m\in\mathcal{M}_{t}}\leftarrow\mathcal{U}_{t}

11: for m∈ℳtm\in\mathcal{M}_{t} do

12: θm,p(t)=Pull​(Learnerm,tm,p)\theta^{(t)}_{m,p}=\texttt{Pull}(\text{Learner}_{m},t_{m},p)

13:

14: end for

15: wm,p←Weight​({cm,psteps,cm,ptokens}m∈ℳt)w_{m,p}\leftarrow\texttt{Weight}(\\{c^{\text{steps}}_{m,p},c^{\text{tokens}}_{m,p}\\}_{m\in\mathcal{M}_{t}})

16:

17: ⊳\triangleright 2\. All-reduce across syncer shards

18: Δp(t)←Merge​({θm,p(t),wm,p}m∈ℳt,Θp(t−H))\Delta_{p}^{(t)}\leftarrow\texttt{Merge}\left(\\{\theta_{m,p}^{(t)},w_{m,p}\\}_{m\in\mathcal{M}_{t}},\Theta_{p}^{(t-H)}\right)

19:

20: ⊳\triangleright 3\. Outer optimization

21: Θp(t)←OuterOpt​(Θp(t−H),Δp(t))\Theta_{p}^{(t)}\leftarrow\texttt{OuterOpt}(\Theta_{p}^{(t-H)},\Delta_{p}^{(t)})

22:

23: ⊳\triangleright 4\. Communication syncer →\rightarrow learner

24: Send​(Learners,Θp(t),t)\texttt{Send}(\text{Learners},\Theta_{p}^{(t)},t)

25: end if

26: end for

We outline the syncer’s operations in Algorithm 2. The syncer maintains a global step count tt. As in (2), when tt matches a predetermined offset tpmodHt_{p}\bmod H for fragment pp (L2), it will aggregate this fragment across learners and apply outer optimization. Rather than waiting for all MM learners, the syncer only waits for a minimum threshold K≤MK\leq M learners that successfully send their metadata (L4). Let ℳt\mathcal{M}_{t} denote the subset of learners that successfully reported. The syncer then pulls the corresponding model fragment from all learners in ℳt\mathcal{M}_{t}. Any offline or straggling learners are excluded from the aggregation for that step, ensuring that localized hardware failures do not stall global training progress.

In a fully asynchronous system, fast learners naturally process more data between syncs than slow or recovering learners. To prevent the outer optimization from being disproportionately skewed by slower learners, the syncer calculates a dynamic weight wm,pw_{m,p} for each learner using a weighting function Weight, derived from the reported per-learner cstepsc^{\text{steps}} and ctokensc^{\text{tokens}}. Throughout, we use the following weighting function

| Weight​(cm,ptokens,cm,psteps)=cm,ptokens×(cm,ptokenscm,psteps).\displaystyle\texttt{Weight}(c^{\text{tokens}}_{m,p},c^{\text{steps}}_{m,p})=c^{\text{tokens}}_{m,p}\times\left(\frac{c^{\text{tokens}}_{m,p}}{c^{\text{steps}}_{m,p}}\right)\,. |   
---|---|---  
  
Intuitively, this corresponds to wm,p=(quantity)×(quality)w_{m,p}=\texttt{(quantity)}\times\texttt{(quality)}, where a learner’s contribution is of higher quality if it is amortized over fewer steps. The syncer then computes the outer gradient Δp(t)\Delta_{p}^{(t)} using a merge function (mergeFn, L11). This function determines the aggregated shift by comparing the previous global state Θp(t−H)\Theta_{p}^{(t-H)} against the weighted combination of the newly received learner fragments {θm,p(t)}m∈ℳt\\{\theta_{m,p}^{(t)}\\}_{m\in\mathcal{M}_{t}}. Finally, the syncer applies the outer optimizer (e.g., SGD with Nesterov momentum, L13) to derive an updated global fragment Θp(t)\Theta_{p}^{(t)} which is asynchronously broadcasted back to the active learners (L15).

#### Adaptive Quorum Window.

When syncer operations outpace learner compute steps, overlapping communication and computation over τ\tau steps creates a natural _slack_. Instead of proceeding immediately once the minimum quorum KK is met, we use this idle time to introduce an adaptive _grace window_ , ξgrace\xi_{\text{grace}}. This purposefully trades available network slack for improved sample efficiency by incorporating more learners into the global update, all without stalling the system or degrading goodput.

In our setting, setting H=PH=P and τ=2\tau=2 effectively double-buffers the training process. Our primary constraint is that gathering a quorum, the grace window, and synchronizing a fragment must fit within τ\tau compute steps. The available slack is defined as:

| ξslack=τ×ξstep−(ξquorum+ξsync)\xi_{\text{slack}}=\tau\times\xi_{\text{step}}-(\xi_{\text{quorum}}+\xi_{\text{sync}})\, |  | (3)  
---|---|---|---  
  
where ξstep\xi_{\text{step}}, ξquorum\xi_{\text{quorum}}, and ξsync\xi_{\text{sync}} are the durations for a compute step, reaching quorum, and synchronization, respectively. To prevent bottlenecks, the grace window ξgrace≤γ⋅ξslack\xi_{\text{grace}}\leq\gamma\cdot\xi_{\text{slack}} is bounded by a safety margin γ<1\gamma<1. In practice, ξstep\xi_{\text{step}} and ξquorum\xi_{\text{quorum}} can be tracked via exponential moving averages to adapt to changing speeds and bandwidth.

Step T=0T=0 ξstep\xi_{\text{step}}Step T=1T=1 ξstep\xi_{\text{step}}Step T=2T=2 ξstep\xi_{\text{step}}ξquorum\xi_{\text{quorum}}KK readyξgrace\xi_{\text{grace}}+KΔ+K_{\Delta} readyξsync\xi_{\text{sync}}SlackFrag 1 Receivedξquorum\xi_{\text{quorum}}KK readyξgrace\xi_{\text{grace}}+KΔ+K_{\Delta} readyξsync\xi_{\text{sync}}SlackFrag 2 Receivedξquorum\xi_{\text{quorum}}KK readyξgrace\xi_{\text{grace}}+KΔ+K_{\Delta} readyLearnersSyncer (Frag 1)Syncer (Frag 2)Syncer (Frag 3)

Figure 3: Illustration of the adaptive quorum grace window fitting within the available slack.

Illustrated in Figure 3, this brief wait trades excess bandwidth for sample efficiency by incorporating more learners. Combined with token-based weighting wm,pw_{m,p}, slightly delayed updates can still contribute to the global update, reducing outer gradient variance while naturally handling hardware speed discrepancies. We also found that in practice, when learner steps vary in speed or are not aligned, an adaptive grace window works better than setting higher KK for maintaining a consistent sync frequency without introducing gaps in utilization.

### 3.3 Chaos engineering for LLMs

Goodput | Number of chips NchipN_{\text{chip}}  
---|---  
# Learners MM | 150k | 300k | 600k | 1.2m | 2.4m  
(no elasticity) 1 | 72% | 57% | 38% | 27% | 18%  
1 | 88% | 80% | 69% | 58% | 40%  
2 | 94% | 88% | 80% | 70% | 54%  
4 | 96% | 93% | 89% | 82% | 73%  
8 | 98% | 96% | 94% | 88% | 80%  
16 | 99% | 98% | 96% | 93% | 86%  
  
(a) Goodput: % of the cluster doing useful work.

System Uptime | Number of chips NchipN_{\text{chip}}  
---|---  
# Learners MM | 150k | 300k | 600k | 1.2m | 2.4m  
(no elasticity) 1 | 72% | 57% | 38% | 27% | 18%  
1 | 88% | 81% | 69% | 59% | 42%  
2 | 100% | 97% | 96% | 90% | 80%  
4 | 100% | 100% | 99% | 99% | 99%  
8 | 100% | 100% | 100% | 100% | 99%  
16 | 100% | 100% | 100% | 100% | 99%  
  
(b) System Uptime: % of the time a cluster is stepping.

Table 1: Modeled goodput and system uptime for hypothetical configurations across number of decoupled learners MM and number of simulated chips NchipN_{\text{chip}} for a fixed MTBIchip=1​year\texttt{MTBI}_{\text{chip}}=1\,\text{year} level of simulated hardware failures. Here we assume all chips have constant speed. Note that M=1M=1 learner is equivalent to data-parallel. 

Chaos engineering (Basiri et al., 2016) tests system resilience by simulating failures, a practice originally designed for general infrastructure. We apply this principle to LLM training, anchoring in the observation that determinism and reproducibility are critical for enhancing system resilience, debugging, and algorithmic scaling properties.

We distill the wide range of possible hardware failures during LLM training (Llama Team, Meta, 2024) into five key parameters: the mean time between interruptions (MTBI) per chip (MTBIchip\texttt{MTBI}_{\text{chip}}), the total number of simulated chips (NchipN_{\text{chip}}), the processing speed variance per chip, the duration required for elastic downscaling and upscaling, and the time needed for a failed chip to return online. Crucially, for a fixed interruption rate per chip, the mean time between failures (MTBF) of the entire cluster decreases proportionally as the number of chips increases:

| MTBFcluster=MTBIchipNchip.\texttt{MTBF}_{\text{cluster}}=\frac{\texttt{MTBI}_{\text{chip}}}{N_{\text{chip}}}\,. |  | (4)  
---|---|---|---  
  
We experiment with strategies for resilience by generating tapes of events that would be logged during a system execution that encountered the failures of interest (detailed in Section E.1). The number of chips failing at any given step is sampled from a Poisson distribution parameterized by the inverse of MTBFcluster\text{MTBF}_{\text{cluster}}. When a chip fails, its entire slice is temporarily removed and only returns after a delay sampled from an exponentiated Weibull distribution. The subsequent elastic downscaling and upscaling operations consume a user-configurable, fixed amount of time.

This simulation yields a goodput metric (Wongpanich et al., 2025): the percentage of allocated cluster time actually spent executing useful steps; a training run that executes some of the steps with less slices available than expected will have lower goodput. While we omit Model Flops Utilization (MFU) for simplicity, our decoupled framework introduces no MFU regression.

Table 1reports goodput across M∈{1,2,4,8,16}M\in\\{1,2,4,8,16\\} learners (where M=1M=1 is standard data parallelism) under various amount of simulated chips. Assuming an MTBIchip\texttt{MTBI}_{\text{chip}} of 1 year and a number of chips NchipN_{\text{chip}} chosen arbitrarily for this paper and reconfiguration times of tens of seconds, standard data-parallel training without elasticity responds poorly to failures. Even with elasticity, goodput plummets to 40% for Nchip=2.4​mN_{\text{chip}}=2.4\text{m} chips. Conversely, decoupling training across multiple learners (M>1M>1) maintains consistently high goodput. We also report in 1(b), the system uptime (e.g. how often the system is stepping): notably, with sufficiently large MM, the uptime can be 100%, with not a single downtime.

Goodput Rule of Thumb: An MM-learner system matches the goodput of a single-learner system using MM times fewer chips. goodput​(M,Nchip)≈goodput​(k×M,k×Nchip)\texttt{goodput}(M,N_{\text{chip}})\approx\texttt{goodput}(k\times M,k\times N_{\text{chip}})

For simulation purposes, we simulate a significantly larger number of chips (𝒪⁡(1​m)\mathcal{O}(1\text{m})) than we physically deploy (𝒪⁡(1​k)\mathcal{O}(1\text{k})). When a simulated chip fails, its corresponding slice is discarded, reducing the effective batch size, similar to the slice-granularity elasticity described by Gemini Team, Google (2025). During our ML experiments, we proportionally scale down our real batch size to match what would occur at the larger simulated scale.

## 4 System design

Figure 4: Overview of the Decoupled DiLoCo system architecture. Each learner worker runs an independent data-parallel training loop on a partition of the accelerator mesh. Learner workers communicate with the syncer by sending parameter fragments and receiving outer-optimized updates over the data-center network (DCN). The syncer is MM-way sharded across CPU-only replicas and performs the outer optimization step. Learner workers may be temporarily absent (here learner #2), but their corresponding syncer shards persist, enabling dynamic adjustment of the number of active learners. After each step, the updated learner model is copied to RAM of the host CPUs colocated with learner TPUs. This allows the syncer to select the fragment to transmit over the data-center network while the learner continues performing inner optimization steps, without requiring extra model copies on HBM.

Current model pretraining relies on a globally synchronized execution model. While devices continuously execute computation and communication primitives, they are bound by strict synchronization barriers: a failure or slowdown in any single device stalls the entire cluster. The outermost parallelism level in standard pretraining is distributed data parallelism (DP), where multiple model replicas compute gradients with respect to the shared parameters, which are then averaged via collective communication. A critical constraint of this paradigm is that all replicas must succeed and participate on every step. Our system design removes this tight coupling by allowing replicas to take steps independently of one another. This naturally leads to a parameter server architecture, inspired by (Dean et al., 2012).

All workers are orchestrated by Pathways (Barham et al., 2022), which manages the resource allocation, device mesh construction, and inter-worker dataflow. Each of the workers described in the following section is driven by a separate Pathways client, in order to prevent a single multi-threaded client from becoming a bottleneck.

#### Learner workers.

The system consists of MM _learner workers_ , each of which runs an independent training loop. A learner worker can be viewed as a scaled-down version of a conventional DP job: if the full cluster provides capacity for RR data-parallel replicas, each learner worker operates with R/MR/M replicas. Each learner worker independently compiles and initializes its model, loads training data from its assigned shard, and executes inner optimization steps (e.g., AdamW) without coordination with the other learners. Crucially, the learner workers are _isolated from one another_ —they share no accelerator resources and do not communicate directly except in the special case of recovery, the process by which one learner obtains up-to-date state from another learner in order to rejoin after a long failure (see Section E.3). This isolation is the foundation for both decoupling and failure domain containment: a hardware failure or straggler machine in one learner has no effect on the others.

#### Syncer worker.

The learner workers are connected to a syncer worker, which instantiates the role described in Section 3.2. The syncer worker maintains the global model parameters and outer optimizer state, and is responsible for aggregating the learner updates and broadcasting the outer-optimized parameters back to the learners. Since it only holds parameters and optimizer state (no activations) and performs relatively simple element-wise operations, its computational and memory footprint is significantly smaller than that of any learner worker. Accordingly, the syncer worker runs on CPU-only resources, partitioned into MM shards—one corresponding to each learner worker.

As illustrated in Figure 4, a learner worker may be temporarily absent (e.g., due to a hardware failure), yet its corresponding syncer shard persists. Thus, when a learner is unavailable, the syncer worker can execute the same Merge computation as usual, with that learner’s weight set to 0. This design enables the system to dynamically adjust the number of active learners without modifying the syncer worker’s configuration or requiring a restart.

#### State coordination.

Each worker locally maintains a vector clock (Mattern, 1989) keeping track of its own step as well as its latest knowledge of the step for each of the other workers in the system. Learner and syncer workers communicate training progress via message passing over FIFO channels where the sender attaches its current vector clock to each message, as shown in Figure 11. These vector clocks are the basis for many operations critical to reliable system operation, including watermarking of channels to garbage-collect old state, creating consistent global checkpoints via the Chandy-Lamport distributed snapshotting algorithm (Chandy and Lamport, 1985) as discussed in Section E.2, and logging of a nondeterministic training run to enable deterministic replay as discussed in Section E.1.

#### Summary of benefits.

This architecture yields several advantages over monolithic data-parallel training. First, it ensures a reduced synchronous footprint with isolated failure domains: by partitioning the cluster into MM smaller accelerator groups, the MTBF of each group improves proportionally (see Table 1), and the blast radius of any hardware failure is strictly contained, preventing it from propagating to the rest of the system. Second, it enables resilient coordination by running the syncer worker on highly stable, CPU-only resources with a minimal failure surface. While not the primary objective of this design, the system inherently retains the massive bandwidth reduction properties of its predecessor, Streaming DiLoCo (Douillard et al., 2025), due to the fact that learners and syncer exchange _parameter fragments_ over the data-center network, and at each outer optimization step the syncer shards perform an all-reduce over only a single fragment rather than the whole model (see Section E.4). Finally, training across heterogeneous resources (e.g. chips of different generations) becomes straightforward in this setting: the decoupled learners are free to choose their own hardware, and the loose nature of the synchronization mitigates speed differences inherent from attempts to balance workload across heterogeneous compute. A combination of all those advantages also enables us to scavenge pre-emptible heterogeneous compute resources distributed across far apart locations.

Figure 5: Hardware failures resilience of Decoupled DiLoCo vs elastic data-parallel with a dense 5B model trained on 1T tokens. Compared to elastic data-parallel, Decoupled DiLoCo requires two orders of magnitude less bandwidth, observes significantly superior goodput under a heavy rate of hardware failures, and yet performs equivalently on pure ML performance.

| Elastic DP | Decoupled M=8M=8  
---|---|---  
# chips | – | 150k | 1.2m | – | 150k | 1.2m  
MTBIchip\texttt{MTBI}_{\text{chip}} | +∞+\infty | 1y | 1y | +∞+\infty | 1y | 1y  
Goodput | 100% | 88% | 58% | 100% | 98% | 88%  
Text (Avg) | 70.1 | 70.1 | 69.7 | 69.8 | 70.1 | 69.8  
Arc-Chall. | 61.2 | 60.4 | 60.4 | 59.0 | 60.3 | 59.2  
Arc-Easy | 81.9 | 81.2 | 80.9 | 81.2 | 81.4 | 81.6  
BoolQ | 76.8 | 78.7 | 78.4 | 76.8 | 79.4 | 78.1  
HellaSwag | 74.4 | 74.40 | 73.7 | 74.2 | 74.5 | 74.2  
Piqa | 79.8 | 79.5 | 79.2 | 79.9 | 79.5 | 79.1  
Siqa | 50.3 | 50.7 | 49.6 | 49.9 | 49.9 | 49.5  
Winogrande | 66.6 | 65.6 | 65.5 | 67.7 | 65.6 | 67.2  
Vision (Avg) | 58.7 | 58.5 | 59.4 | 58.8 | 59.4 | 58.6  
MMMU | 33.9 | 32.4 | 36.2 | 33.4 | 32.4 | 33.2  
COCO-Cap. | 90.1 | 89.9 | 92.2 | 88.2 | 90.8 | 89.5  
ChartQA | 25.1 | 25.0 | 26.6 | 24.8 | 24.8 | 27.4  
DocVQA | 79.6 | 78.3 | 77.8 | 80.9 | 80.0 | 76.7  
DocVQA-Hard | 56.6 | 56.9 | 56.9 | 59.7 | 59.8 | 55.7  
InfographicVQA | 48.3 | 48.5 | 47.7 | 48.1 | 50.4 | 48.9  
TextVQA | 77.0 | 78.4 | 78.3 | 76.7 | 77.6 | 78.4  
  
Table 2: Simulated hardware failure impact on ML performance of a 5B dense with 1T tokens. Given a fixed MTBI per chip, we increase the number of simulated chips, lowering the cluster MTBF and leading to more frequent hardware failures. Decoupled DiLoCo’s ML performance is not degraded by the algorithmic changes to achieve significantly higher goodput.

| Elastic DP | Decoupled M=8M=8  
---|---|---  
# chips | – | – | 1.2m  
MTBIchip\texttt{MTBI}_{\text{chip}} | +∞+\infty | +∞+\infty | 1y  
Goodput | 100% | 100% | 88%  
Text (Avg) | 68.4 | 67.9 | 68.4  
Arc-Chall. | 59.1 | 55.5 | 59.4  
Arc-Easy | 81.7 | 80.6 | 81.4  
BoolQ | 71.1 | 72.5 | 71.8  
HellaSwag | 73.5 | 73.4 | 73.1  
Piqa | 79.8 | 79.3 | 79.8  
Siqa | 49.1 | 50.3 | 49.3  
Winogrande | 64.6 | 63.7 | 64.2  
Vision (Avg) | 54.9 | 55.1 | 54.3  
MMMU | 30.4 | 29.8 | 31.1  
COCO-Cap. | 89.0 | 84.5 | 84.9  
ChartQA | 20.9 | 21.9 | 20.6  
DocVQA | 74.7 | 76.9 | 74.0  
DocVQA-Hard | 51.1 | 51.6 | 51.3  
InfographicVQA | 43.4 | 46.9 | 43.1  
TextVQA | 74.8 | 74.2 | 75.0  
  
Table 3: Simulated hardware failures impact on ML performance of a 2.8B activated Mixture-of-Experts model trained on 170B tokens, using the same setup as in Table 2.

## 5 Experiments

We now detail the experiments we conducted to validate our framework. For all experiments, we use Gemma 4 (Google DeepMind, 2026) customized for a lighter training footprint, and train on a mixture of text and vision data.

### 5.1 Experimental Setup

In all experiments, we use P=24P=24 fragments. Each fragment is synchronized once every H=24H=24 steps. The synchronization is overlapped over τ=2\tau=2 steps. As a consequence, at every step a fragment is sent and received, and there are two fragments in flight. Unless stated otherwise, we use the minimal quorum size K=1K=1 and use a grace window (see Section 3.2) that can extend up to a step.

We form the fragments via a greedy bin-packing algorithm, applied to individual tensors in the model. This results in approximately balanced fragments. We found that this strategy, which we refer to as _balanced tensor fragmentation_ , maintained model quality compared to layer-based fragmentation strategies (Douillard et al., 2025), while significantly reducing peak bandwidth. See Section C for details.

We do not just use direct averaging for the Merge operation in Algorithm 2. We propose a modified merging operation, _Radial-Directional Averaging_ (RDA) in which we separately average the norms and directions of outer gradients. This leads to greater hyperparameter stability in the outer optimizer and boosts performance when scaling the number of learners MM. For details, see Section D.2.

### 5.2 Resilience to hardware failures

In order to test the ML performance of Decoupled DiLoCo under heavy rates of failure, we simulate different scenarios. We choose to set an aggressive MTBIchip=1​year\texttt{MTBI}_{\text{chip}}=1\,\text{year} and we vary the amount of chips from Nchip=150​kN_{\text{chip}}=150\text{k} (1×)1\times) to Nchip=1.2​mN_{\text{chip}}=1.2\text{m} (8×8\times). The latter leads to a MTBFcluster\texttt{MTBF}_{\text{cluster}} of less than a minute. We display in Figure 5 the downstream performance of both Data-Parallel (DP) and Decoupled DiLoCo M=8M=8 on text and vision tasks alongside the observed goodput at various hardware failures rate. We also detailed the individual evaluation results in Table 2.

Note that in the elastic data-parallel baseline, ML performance remains invariant to the hardware failure rate, but goodput degrades drastically, resulting in significantly longer wall-clock training times. The impact of hardware failures on training dynamics is twofold: (1) for both data-parallel and Decoupled DiLoCo, the loss of chips and their corresponding slices reduces the effective batch size, and (2) specific to Decoupled DiLoCo, recovering learners rejoin the system with stale parameters and optimizer states.

Notice that with M=8M=8, the goodput degrades gracefully as the number of chips is increased (and thus the cluster MTBF is lowered) up to 88% while data-parallel, even with elasticity, reaches 58% goodput. The downstream evaluations on both text and vision remain competitive, showing both the resiliency to hardware failures and ML robustness of our framework.

We also validate that the downstream performance, in the event of frequent hardware failures, has similar impacts on mixture-of-experts models. In Table 3, we compare downstream performance of Data-Parallel and Decoupled DiLoCo M=8M=8 with no hardware failures to that of Decoupled DiLoCo with MTBIchip=1​year\texttt{MTBI}_{\text{chip}}=1\,\text{year} and Nchip=1.2​mN_{\text{chip}}=1.2\text{m} (8×8\times). We find that the evaluation results are all essentially comparable, with minor differences likely attributable to an ambient noise floor versus any algorithmic or system issue.

Model | Scavenging | Text (Avg) | Vision (Avg) | Train Time  
---|---|---|---|---  
DP | +0%+0\% | 60.160.1 | 46.446.4 | 1.00×1.00\times  
+25%+25\% | 60.560.5 | 46.646.6 | 1.07×1.07\times  
+50%+50\% | 60.060.0 | 45.945.9 | 1.09×1.09\times  
+100%+100\% | 60.160.1 | 47.047.0 | 1.02×1.02\times  
+300%+300\% | 60.460.4 | 46.446.4 | 0.80×0.80\times  
DiLoCo | +0%+0\% | 60.660.6 | 46.246.2 | 1.00×1.00\times  
+25%+25\% | 60.860.8 | 47.2\mathbf{47.2} | 0.90×0.90\times  
+50%+50\% | 61.3\mathbf{61.3} | 45.245.2 | 0.83×0.83\times  
+100%+100\% | 60.760.7 | 46.446.4 | 0.75×0.75\times  
+300%+300\% | 60.760.7 | 46.146.1 | 0.62×\mathbf{0.62\times}  
  
Table 4: Iso-FLOPs scavenging experiments with a 2B dense trained on 128B tokens as more TPUs are scavenged across distributed locations for Decoupled DiLoCo and data-parallel. All runs share the same flops & token budget, thus scavenging allows finishing earlier. Scavenged compute is available for half the FLOPS in each run, so the lower bound on relative training time is 0.5×0.5\times. Figure 6: XLA operations when learners have varying step times for quorum sizes of K=8K=8, K=1K=1, and K=1K=1 with an adaptive grace window. Notice that despite variable bandwidth and hardware heterogeneity there is always a learner stepping, achieving maximum availability.

### 5.3 Scavenging

We exploit our decoupled framework to dynamically add compute resources, “ _scavenge_ ”, on the fly during training. While slice-granular elasticity (Gemini Team, Google, 2025) allows for scaling up the slice count, it incurs a non-negligible goodput penalty during system reconfiguration. Decoupled DiLoCo, however, seamlessly integrates any temporarily available extra FLOPS by dynamically acquiring (and releasing) extra learners using the learner recovery procedure described in Section E.3 .

For the duration that more learners are available the system runs with a temporarily higher MM than baseline training. In these experiments we use M=4M=4 learners as a base level of always available compute, and increase the number of learners to M=5M=5, 66, 88 or 1616 intermittently during training to simulate a day-night cycle of increased availability. For a point of comparison we simulate Data-Parallel training being used in identical scavenging patterns, by increasing the batch size during periods of high MM to match Decoupled DiLoCo’s step-by-step compute and token usage. This gives a strong baseline for an ML performance target but incurs the practical downsides of using distributed DP training for scavenging, such as the time taken to transfer the current model state to the new learners during batch upsize, and the increased step time when all-reducing gradients over increasingly geo-separated compute that has no pre-allocated bandwidth due to the opportunistic nature of scavenging.

We run our experiments in an _iso-FLOPs_ regime where total compute usage is fixed; the goal of this style of scavenging is therefore to accelerate training by reducing steps taken and wall-clock time. Table 4 shows Decoupled DiLoCo is able to reap the benefits of ad hoc training acceleration without degrading ML performance, and does so more time-efficiently than the DP baseline which needs to more than double the number of learners to see any benefit. Section F.4 contains a full breakdown of the downstream performance metrics.

### 5.4 Heterogeneous hardware

Quorum (KK) | 8 | 1 | 1  
---|---|---|---  
Grace Window (ξgrace\xi_{\text{grace}}) | Off | Off | On  
Goodput | 84% | 100% | 100%  
Text (avg) | 54.4 | 51.4 | 55.3  
Arc-Challenge | 36.3 | 31.6 | 35.6  
Arc-Easy | 64.0 | 60.2 | 65.3  
BoolQ | 55.8 | 59.8 | 61.3  
HellaSwag | 51.4 | 42.6 | 51.8  
Piqa | 72.5 | 69.3 | 72.0  
Siqa | 46.5 | 44.7 | 46.0  
Winogrande | 54.1 | 51.9 | 55.2  
Vision (avg) | 37.6 | 25.7 | 37.0  
COCO-Captions | 62.9 | 43.4 | 62.6  
ChartQA | 13.3 | 8.4 | 13.6  
DocVQA | 48.1 | 33.0 | 50.6  
DocVQA-Hard | 30.8 | 18.5 | 32.6  
InfographicVQA | 24.4 | 19.4 | 24.4  
MMMU | 19.3 | 9.4 | 12.2  
TextVQA | 64.5 | 47.9 | 63.2  
  
Table 5: Heterogeneous hardware ML performance of a 2B dense model with 26B tokens using Decoupled DiLoCo (M=8M=8) with a mix of TPUv6e and TPUv5p.

We evaluate Decoupled DiLoCo on a heterogeneous cluster comprising four learners on TPUv5e and four on TPUv5p, adjusting the device count per learner to account for differences in HBM size. Due to these hardware and scaling differences, the slowest learners natively trailed the fastest by 18%18\%. To rigorously stress-test the system, we further artificially injected an additional 10%10\% speed variance across all learners. By employing a minimal quorum size (K=1K=1) alongside the adaptive grace window, we maximized compute utilization while matching the machine learning performance of a fully blocking, synchronous training setup (K=M=8K=M=8).

Table 5compares the downstream performance of this asynchronous, low-quorum approach (K=1K=1) against the fully synchronous baseline (K=8K=8). We demonstrate that the adaptive quorum and its grace window effectively mask the speed discrepancies between the two TPU architectures. Consequently, the system achieves the same ML performance as a synchronous baseline without being bottlenecked by the speed of its slowest chips. Moreover, we display the XProf (OpenXLA Contributors, 2026) of the XLA operations of training on learners of heterogeneous speeds in Figure 6. On the top row, notice that with a quorum of K=M=8K=M=8 the decoupled algorithm becomes blocking, resulting in gap where learners are idle. On the middle row, with a minimal quorum of K=1K=1, there is no gap, but more often than not, the syncer only collects a quorum of 1 learner at a time. This results in the syncer performing much more frequent synchronizations as each learner finishes its step (bearing similarity with Liu et al. (2024)’s setting). On the bottom row, we use the minimal quorum K=1K=1 but with the adaptive grace window, which allows the syncer to fetch a significant number of learners on each synchronization round despite learners running at different speeds.

Overall, by seamlessly integrating heterogeneous hardware into a single training run, Decoupled DiLoCo empowers practitioners to scavenge older chip generations for extra compute and enables smoother transitions during new hardware rollouts.

### 5.5 Scalability

(a) Dense: 2B, 5B, and 9B parameters models using 26B, 72B, and 141B tokens.

(b) Mixture-of-Experts: 2.8B and 3.8B activated parameters models, using 170B and 233B tokens.

Figure 7: Downstream performances for Dense (a) and Mixture-of-Experts (b) architectures.

Building on Bergsma et al. (2025); Qiu et al. (2025), we tune to ensure predictable scaling of Decoupled DiLoCo across model scales. Crucially, mirroring the observations of Charles et al. (2025), we observe that Decoupled DiLoCo exhibits predictable, improved performance as model size, batch size, and token budgets increase. Aligning with the “bitter lesson” (Sutton, 2019), that methods designed to exploit massive compute will prevail, our algorithm is ideally suited for large-scale pre-training environments where robust fault tolerance is a critical necessity.

7(a)reports the downstream performance of a standard data-parallel (DP) baseline versus Decoupled DiLoCo (M=8M=8 learners) at scales of 2B, 5B, and 9B parameters dense models, trained for 26B, 72B, and 141B tokens, respectively. At every scale, Decoupled DiLoCo matches the performance of the centralized monolithic DP baseline.

7(b)extends this evaluation to a Mixture-of-Experts (MoE) architecture (Shazeer et al., 2017; Fedus et al., 2022) with 2.8B and 3.8B active parameters trained on 170B and 233B tokens, respectively, once again demonstrating comparable downstream performance. We note that the success in this setting is potentially surprising - the learners use a load-balancing loss (Shazeer et al., 2017), but can only optimize this loss locally, not globally. Despite this, with sufficient scale we see that Decoupled DiLoCo with M=8M=8 performs comparably to SPMD data-parallel training.

## 6 Conclusion

Decoupled DiLoCo allows us to decompose monolithic SPMD pre-training into independent asynchronous learners coordinated by a lightweight synchronizer. By using minimum-quorum aggregation, token-weighted merging via Radial-Directional Averaging, balanced tensor fragmentation, and an adaptive grace window, the system prioritizes availability and partition tolerance without sacrificing downstream model quality. Our extensive experiments above show that Decoupled DiLoCo matches data-parallel performance on text and vision benchmarks across dense and MoE architectures at scales up to 9B parameters, while maintaining 88% goodput under aggressive simulated failures (versus 58% for elastic data-parallel) and perfect uptime. The framework further enables seamless scavenging of opportunistic compute and integration of heterogeneous hardware, unlocking the next frontier of cluster, distributed across geo-locations and chip generations.

We believe that Decoupled DiLoCo points at a new direction that upends the SPMD pre-training paradigm. If we relax model synchronization requirements, we can extract more useful computation from failure-prone hardware. This vision for non-SPMD pre-training mirrors that presented by Pathways (Barham et al., 2022), and in fact uses the capabilities of Pathways to make it a reality.

Decoupled DiLoCo performs better with scale relative to data-parallel training in both model quality and goodput. This observation is crucial: the pre-training settings that most benefit from Decoupled DiLoCo for system reasons (bandwidth and hardware reliability, exacerbated by the number of chips involved) are exactly the settings in which model quality is most comparable to SPMD training.

As pre-training expands to geo-distributed clusters and environments where both bandwidth and hardware reliability are severely constrained (Agüera y Arcas et al., 2025), we believe availability-first training will shift from advantageous to necessary.

## References

  * Acker et al. (2025) A. Acker, S. Becker, S. Nedelkoski, D. Scheinert, O. Kao, and P. Wiesner What happens when nanochat meets diloco?.  External Links: [Link](https://arxiv.org/abs/2511.13761) Cited by: Appendix B. 
  * Agüera y Arcas et al. (2025) B. Agüera y Arcas, T. Beals, M. Biggs, J. V. Bloom, T. Fischbacher, K. Gromov, U. Köster, R. Pravahan, and J. Manyika Towards a future space-based, highly scalable ai infrastructure system design.  External Links: 2511.19468, [Link](https://arxiv.org/abs/2511.19468) Cited by: §6. 
  * Ahn and Myung (2025) K. Ahn and B. Myung Dion: a communication-efficient optimizer for large models.  arXiv preprint arXiv:2504.05295.  External Links: [Link](https://arxiv.org/abs/2504.05295) Cited by: Appendix A. 
  * Ajanthan et al. (2025) T. Ajanthan, S. Ramasinghe, G. Avraham, Y. Zuo, and A. Long Momentum look-ahead for asynchronous distributed low-communication training.  In ICLR 2025 Workshop on Modularity for Collaborative, Decentralized, and Continual Deep Learning,  External Links: [Link](https://openreview.net/forum?id=4O8nzTkHPI) Cited by: Appendix A. 
  * Alistarh et al. (2017) D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic QSGD: communication-efficient sgd via gradient quantization and encoding.  Neural information processing systems (NeurIPS).  External Links: [Link](https://arxiv.org/abs/1610.02132) Cited by: Appendix A. 
  * Barham et al. (2022) P. Barham, A. Chowdhery, J. Dean, S. Ghemawat, S. Hand, D. Hurt, M. Isard, H. Lim, R. Pang, S. Roy, et al. Pathways: asynchronous distributed dataflow for ml.  Proceedings of Machine Learning and Systems.  External Links: [Link](https://arxiv.org/abs/2203.12533) Cited by: §1, §4, §6. 
  * Basiri et al. (2016) A. Basiri, N. Behnam, R. de Rooij, L. Hochstein, L. Kosewski, J. Reynolds, and C. Rosenthal Chaos engineering.  IEEE Software.  External Links: [Link](https://arxiv.org/abs/1702.05843) Cited by: 4th item, §3.3. 
  * Bergsma et al. (2025) S. Bergsma, B. C. Zhang, N. Dey, S. Muhammad, G. Gosal, and J. Hestness Scaling with collapse: efficient and predictable training of llm families.  External Links: 2509.25087, [Link](https://arxiv.org/abs/2509.25087) Cited by: §5.5. 
  * Beton et al. (2025) M. Beton, S. Howes, A. Cheema, and M. Baioumy Improving the efficiency of distributed training using sparse parameter averaging.  In ICLR 2025 Workshop on Modularity for Collaborative, Decentralized, and Continual Deep Learning,  External Links: [Link](https://openreview.net/forum?id=stFPf3gzq1) Cited by: Appendix A. 
  * Bisk et al. (2020) Y. Bisk, R. Zellers, J. Gao, and Y. Choi Piqa: reasoning about physical commonsense in natural language.  In Proceedings of the AAAI conference on artificial intelligence,  External Links: [Link](https://arxiv.org/abs/1911.11641) Cited by: 4th item. 
  * Bonawitz et al. (2019) K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon, J. Konečnỳ, S. Mazzocchi, B. McMahan, et al. Towards federated learning at scale: system design.  Proceedings of machine learning and systems.  External Links: [Link](https://arxiv.org/abs/1902.01046) Cited by: Appendix A. 
  * Borzunov et al. (2023) A. Borzunov, M. Ryabinin, A. Chumachenko, D. Baranchuk, T. Dettmers, Y. Belkada, P. Samygin, and C. A. Raffel Distributed inference and fine-tuning of large language models over the internet.  In Advances in Neural Information Processing Systems,  External Links: [Link](https://arxiv.org/abs/2312.08361) Cited by: Appendix A. 
  * Brewer (2000) E. A. Brewer Towards robust distributed systems.  In PODC,  External Links: [Link](https://dl.acm.org/doi/10.1145/343477.343502) Cited by: §1. 
  * Buss and Fillmore (2001) S. R. Buss and J. P. Fillmore Spherical averages and applications to spherical splines and interpolation.  ACM Transactions on Graphics (TOG) 20 (2), pp. 95–126.  External Links: [Link](https://dl.acm.org/doi/10.1145/502122.502124) Cited by: §D.2. 
  * Chandy and Lamport (1985) K. M. Chandy and L. Lamport Distributed snapshots: determining global states of a distributed system.  ACM Transactions on Computer Systems 3 (1), pp. 63–75.  External Links: [Link](https://dl.acm.org/doi/10.1145/214451.214456) Cited by: §E.2, §4. 
  * Charles et al. (2025) Z. Charles, G. Teston, L. Dery, K. Rush, N. Fallen, Z. Garrett, A. Szlam, and A. Douillard Communication-efficient language model training scales reliably and robustly: scaling laws for diloco.  In NeurIPS,  External Links: [Link](https://arxiv.org/abs/2503.09799) Cited by: Appendix A, §5.5. 
  * Chen et al. (2021) M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba Evaluating large language models trained on code.  External Links: [Link](https://arxiv.org/abs/2107.03374) Cited by: Appendix B. 
  * Chen et al. (2015) X. Chen, H. Fang, T. Lin, R. Vedantam, S. Gupta, P. Dollár, and C. L. Zitnick Microsoft coco captions: data collection and evaluation server.  arXiv preprint arXiv:1504.00325.  External Links: [Link](https://arxiv.org/abs/1504.00325) Cited by: 2nd item. 
  * Clark et al. (2019) C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova BoolQ: exploring the surprising difficulty of natural yes/no questions.  In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers),  External Links: [Link](https://arxiv.org/abs/1905.10044) Cited by: 2nd item. 
  * Clark et al. (2018) P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord Think you have solved question answering? try arc, the ai2 reasoning challenge.  arXiv preprint arXiv:1803.05457.  External Links: [Link](https://arxiv.org/abs/1803.05457) Cited by: 1st item. 
  * Cobbe et al. (2021) K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman Training verifiers to solve math word problems.  arXiv preprint arXiv:2110.14168.  External Links: [Link](https://arxiv.org/abs/2110.14168) Cited by: Appendix B. 
  * Dean et al. (2012) J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, A. Senior, P. Tucker, K. Yang, Q. V. Le, and A. Y. Ng Large scale distributed deep networks.  In Advances in Neural Information Processing Systems,  External Links: [Link](https://dl.acm.org/doi/10.5555/2999134.2999271) Cited by: Appendix A, Appendix A, §4. 
  * Douillard et al. (2025) A. Douillard, Y. Donchev, K. Rush, S. Kale, Z. Charles, Z. Garrett, G. Teston, D. Lacey, R. McIlroy, J. Shen, A. Ramé, A. Szlam, M. Ranzato, and P. Barham Streaming diloco with overlapping communication: towards a distributed free lunch.  In COLM,  External Links: [Link](https://arxiv.org/abs/2501.18512) Cited by: Appendix A, Appendix A, Appendix C, Appendix C, §D.1, §D.3, §E.4, §1, §2, §2, §4, §5.1. 
  * Douillard et al. (2023) A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, and J. Shen DiLoCo: distributed low-communication training of language models.  arXiv preprint arXiv:2311.08105.  External Links: [Link](https://arxiv.org/abs/2311.08105) Cited by: Appendix A. 
  * Douillard et al. (2024) A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, and J. Shen DiLoCo: distributed low-communication training of language models.  In ICML workshop,  External Links: [Link](https://arxiv.org/abs/2311.08105) Cited by: §D.2, §1, §2, §2. 
  * Elnozahy et al. (2002) E. Elnozahy, L. Alvisi, Y. Wang, and D. Johnson A survey of rollback-recovery protocols in message-passing systems.  ACM Computing Surveys 34, pp. .  External Links: [Document](https://dx.doi.org/10.1145/568522.568525), [Link](https://dl.acm.org/doi/10.1145/568522.568525) Cited by: §E.2. 
  * FairScale authors (2021) FairScale authors FairScale: a general purpose modular pytorch library for high performance and large scale training.  Note: <https://github.com/facebookresearch/fairscale> Cited by: Appendix A. 
  * Fan and Zhang (2025) S. Fan and Z. Zhang Pier: efficient large language model pretraining with relaxed global communication.  arXiv preprint arXiv:2511.17849.  External Links: [Link](https://arxiv.org/abs/2511.17849) Cited by: Appendix A. 
  * Fedus et al. (2022) W. Fedus, B. Zoph, and N. Shazeer Switch transformers: scaling to trillion parameter models with simple and efficient sparsity.  Journal of Machine Learning Research.  External Links: [Link](https://arxiv.org/abs/2101.03961) Cited by: §5.5. 
  * Gemini Team, Google (2025) Gemini Team, Google Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities.  External Links: [Link](https://arxiv.org/abs/2507.06261) Cited by: §1, §3.3, §5.3. 
  * Google DeepMind (2026) Google DeepMind Gemma 4.  Note: Accessed: 2026-04-21 External Links: [Link](https://deepmind.google/models/gemma/gemma-4/) Cited by: §5. 
  * Graham (1969) R. L. Graham Bounds on multiprocessing timing anomalies.  SIAM journal on Applied Mathematics.  External Links: [Link](https://www.jstor.org/stable/2099572) Cited by: Appendix C. 
  * Hoffmann et al. (2022) J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre Training compute-optimal large language models.  External Links: [Link](https://arxiv.org/abs/2203.15556) Cited by: §E.4. 
  * Hsu et al. (2019) T. H. Hsu, H. Qi, and M. Brown Measuring the effects of non-identical data distribution for federated visual classification.  arXiv preprint arXiv:1909.06335.  External Links: [Link](https://arxiv.org/abs/1909.06335) Cited by: Appendix A. 
  * Huang et al. (2019) Y. Huang, Y. Cheng, A. Bapna, O. Firat, M. X. Chen, D. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, et al. GPipe: easy scaling with micro-batch pipeline parallelism.  proceeding of Computer Science Computer Vision and Pattern Recognition.  External Links: [Link](https://arxiv.org/abs/1811.06965) Cited by: Appendix A. 
  * Huba et al. (2022) D. Huba, J. Nguyen, K. Malik, R. Zhu, M. Rabbat, A. Yousefpour, C. Wu, H. Zhan, P. Ustinov, H. Srinivas, et al. Papaya: practical, private, and scalable federated learning.  Proceedings of Machine Learning and Systems.  External Links: [Link](https://arxiv.org/abs/2111.04877) Cited by: Appendix A. 
  * Jang et al. (2023) I. Jang, Z. Yang, Z. Zhang, X. Jin, and M. Chowdhury Oobleck: resilient distributed training of large models using pipeline templates.  In Proceedings of the 29th Symposium on Operating Systems Principles,  External Links: [Link](https://arxiv.org/abs/2309.08125) Cited by: Appendix A. 
  * Jovanović et al. (2026) A. Jovanović, A. Iacob, M. Safaryan, I. Modoranu, L. Sani, W. F. Shen, X. Qiu, D. Alistarh, and N. D. Lane LoRDO: distributed low-rank optimization with infrequent communication.  arXiv preprint arXiv:2602.04396.  External Links: [Link](https://arxiv.org/abs/2602.04396) Cited by: Appendix A. 
  * Kale et al. (2025) S. Kale, A. Douillard, and Y. Donchev Eager updates for overlapped communication and computation in diloco.  arXiv preprint arXiv:2502.12996.  External Links: [Link](https://arxiv.org/abs/2502.12996) Cited by: Appendix A. 
  * Kim et al. (2025) G. Kim, J. Li, S. Gandham, O. Baldonado, A. Gangidi, P. Balaji, Z. Wang, and A. Akella HALoS: hierarchical asynchronous local sgd over slow networks for geo-distributed large language model training.  arXiv preprint arXiv:2506.04531.  External Links: [Link](https://arxiv.org/abs/2506.04531) Cited by: Appendix A, Appendix A. 
  * Kingma and Ba (2015) D. P. Kingma and J. Ba Adam: a method for stochastic optimization.  International Conference on Learning Representations (ICLR).  External Links: [Link](https://arxiv.org/abs/1412.6980) Cited by: Appendix A. 
  * Kolehmainen et al. (2025) J. Kolehmainen, N. Blagoev, J. Donaghy, O. Ersoy, and C. Nies NoLoCo: no-all-reduce low communication training method for large models.  arXiv preprint arXiv:2506.10911.  External Links: [Link](https://arxiv.org/abs/2506.10911) Cited by: Appendix A. 
  * Korthikanti et al. (2023) V. A. Korthikanti, J. Casper, S. Lym, L. McAfee, M. Andersch, M. Shoeybi, and B. Catanzaro Reducing activation recomputation in large transformer models.  Proceedings of Machine Learning and Systems.  External Links: [Link](https://arxiv.org/abs/2205.05198) Cited by: Appendix A. 
  * Lepikhin et al. (2021) D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen GShard: scaling giant models with conditional computation and automatic sharding.  In International Conference on Learning Representations,  External Links: [Link](https://arxiv.org/abs/2006.16668) Cited by: Appendix A. 
  * Li et al. (2020) S. Li, Y. Zhao, R. Varma, O. Salpekar, P. Noordhuis, T. Li, A. Paszke, J. Smith, B. Vaughan, P. Damania, et al. PyTorch distributed: experiences on accelerating data parallel training.  Very Large Data Base (VLDB).  External Links: [Link](https://www.vldb.org/pvldb/vol13/p3005-li.pdf) Cited by: Appendix A. 
  * Lidin et al. (2026) J. Lidin, A. Sarfi, E. Miahi, Q. Anthony, S. Chauhan, E. Pappas, B. Thérien, E. Belilovsky, and S. Dare Covenant-72b: pre-training a 72b llm with trustless peers over-the-internet.  External Links: 2603.08163, [Link](https://arxiv.org/abs/2603.08163) Cited by: Appendix A. 
  * Liu et al. (2024) B. Liu, R. Chhaparia, A. Douillard, S. Kale, A. A. Rusu, J. Shen, A. Szlam, and M. Ranzato Asynchronous local-sgd training for language modeling.  In ICML 2024 Workshop,  External Links: [Link](https://arxiv.org/abs/2401.09135) Cited by: Appendix A, Appendix A, §5.4. 
  * Llama Team, Meta (2024) Llama Team, Meta The llama 3 herd of models.  arXiv.  External Links: [Link](https://arxiv.org/abs/2407.21783) Cited by: §3.3. 
  * Mangasarian and Solodov (1993) O. L. Mangasarian and M. V. Solodov Backpropagation convergence via deterministic nonmonotone perturbed minimization.  Advances in Neural Information Processing Systems.  External Links: [Link](https://dl.acm.org/doi/10.5555/2987189.2987238) Cited by: Appendix A. 
  * Marczak et al. (2025) D. Marczak, S. Magistri, S. Cygert, B. Twardowski, A. D. Bagdanov, and J. van de Weijer No task left behind: isotropic model merging with common and task-specific subspaces.  International Conference on Machine Learning.  External Links: [Link](https://arxiv.org/abs/2502.04959) Cited by: §D.2. 
  * Masry et al. (2022) A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque ChartQA: a benchmark for question answering about charts with visual and logical reasoning.  In Findings of the Association for Computational Linguistics: ACL 2022,  External Links: [Link](https://arxiv.org/abs/2203.10244) Cited by: 1st item. 
  * Mathew et al. (2022) M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar InfographicVQA.  In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision,  External Links: [Link](https://arxiv.org/abs/2104.12756) Cited by: 4th item. 
  * Mathew et al. (2021) M. Mathew, D. Karatzas, and C. Jawahar Docvqa: a dataset for vqa on document images.  In Proceedings of the IEEE/CVF winter conference on applications of computer vision,  External Links: [Link](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf) Cited by: 3rd item. 
  * Mattern (1989) F. Mattern Virtual time and global states of distributed systems.  In Parallel and Distributed Algorithms,  External Links: [Link](https://homes.cs.washington.edu/~arvind/cs425/doc/mattern89virtual.pdf) Cited by: §E.1, §E.2, §4. 
  * McMahan et al. (2017) B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas Communication-efficient learning of deep networks from decentralized data.  In Artificial intelligence and statistics,  External Links: [Link](https://arxiv.org/abs/1602.05629) Cited by: Appendix A. 
  * Narayanan et al. (2019) D. Narayanan, A. Harlap, A. Phanishayee, V. Seshadri, N. R. Devanur, G. R. Ganger, P. B. Gibbons, and M. Zaharia PipeDream: generalized pipeline parallelism for DNN training.  In Proceedings of the 27th ACM symposium on operating systems principles,  External Links: [Link](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf) Cited by: Appendix A. 
  * Narayanan et al. (2021) D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm.  In Proceedings of the international conference for high performance computing, networking, storage and analysis,  External Links: [Link](https://arxiv.org/abs/2104.04473) Cited by: Appendix A. 
  * Niu et al. (2011) F. Niu, B. Recht, C. Re, and S. J. Wright HOGWILD​: a lock-free approach to parallelizing stochastic gradient descent.  External Links: [Link](https://arxiv.org/abs/1106.5730) Cited by: Appendix A. 
  * OpenXLA Contributors (2026) OpenXLA Contributors XProf: a profiling and performance analysis tool for machine learning.  GitHub.  Note: <https://github.com/openxla/xprof> Cited by: §E.4, §5.4. 
  * Petrowski et al. (1993) A. Petrowski, G. Dreyfus, and C. Girault Performance analysis of a pipelined backpropagation parallel algorithm.  IEEE Transactions on Neural Networks.  External Links: [Link](https://ieeexplore.ieee.org/document/286892/) Cited by: Appendix A. 
  * Qi et al. (2025) J. Qi, W. Zhu, L. Li, M. Wu, Y. Wu, W. He, X. Gao, J. Zeng, and M. Heinrich DiLoCoX: a low-communication large-scale training framework for decentralized cluster.  arXiv preprint arXiv:2506.21263.  External Links: [Link](https://arxiv.org/abs/2506.21263) Cited by: Appendix A. 
  * Qiu et al. (2025) S. Qiu, L. Xiao, A. G. Wilson, J. Pennington, and A. Agarwala Scaling collapse reveals universal dynamics in compute-optimally trained neural networks.  External Links: [Link](https://arxiv.org/abs/2507.02119) Cited by: §5.5. 
  * Rajbhandari et al. (2020) S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He Zero: memory optimizations toward training trillion parameter models.  In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis,  External Links: [Link](https://dl.acm.org/doi/10.5555/3433701.3433727) Cited by: Appendix A. 
  * Ramé et al. (2024) A. Ramé, J. Ferret, N. Vieillard, R. Dadashi, L. Hussenot, P. Cedoz, P. G. Sessa, S. Girgin, A. Douillard, and O. Bachem WARP: on the benefits of weight averaged rewarded policies.  External Links: 2406.16768, [Link](https://arxiv.org/abs/2406.16768) Cited by: §D.2. 
  * Reddi et al. (2021) S. J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konečn’y, S. Kumar, and H. B. McMahan Adaptive federated optimization.  In International Conference on Learning Representations,  External Links: [Link](https://openreview.net/forum?id=LkFG3lB13U5) Cited by: Appendix A, Appendix A, §2. 
  * Ren et al. (2021) J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li, and Y. He {\\{zero-Offload}\\}: democratizing {\\{billion-scale}\\} model training.  In 2021 USENIX Annual Technical Conference (USENIX ATC 21),  External Links: [Link](https://arxiv.org/abs/2101.06840) Cited by: Appendix A. 
  * Ryabinin et al. (2023) M. Ryabinin, T. Dettmers, M. Diskin, and A. Borzunov Swarm parallelism: training large models can be surprisingly communication-efficient.  In International Conference on Machine Learning,  pp. 29416–29440.  External Links: [Link](https://arxiv.org/abs/2301.11913) Cited by: Appendix A. 
  * Ryabinin et al. (2021) M. Ryabinin, E. Gorbunov, V. Plokhotnyuk, and G. Pekhimenko Moshpit sgd: communication-efficient decentralized training on heterogeneous unreliable devices.  Advances in Neural Information Processing Systems.  External Links: [Link](https://arxiv.org/abs/2103.03239) Cited by: Appendix A. 
  * Sakaguchi et al. (2020) K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi Winogrande: an adversarial winograd schema challenge at scale.  In Proceedings of the AAAI conference on artificial intelligence,  External Links: [Link](https://arxiv.org/abs/1907.10641) Cited by: 6th item. 
  * Salpekar et al. (2026) O. Salpekar, R. Varma, K. Yu, V. Ivanov, Y. Wang, A. Sharif, M. Si, S. Xu, F. Tian, S. Zheng, et al. Training llms with fault tolerant hsdp on 100,000 gpus.  arXiv preprint arXiv:2602.00277.  External Links: [Link](https://arxiv.org/abs/2602.00277) Cited by: Appendix A. 
  * Sani et al. (2024) L. Sani, A. Iacob, Z. Cao, R. Lee, B. Marino, Y. Gao, D. Cai, Z. Li, W. Zhao, X. Qiu, et al. Photon: federated LLM pre-training.  arXiv preprint arXiv:2411.02908.  External Links: [Link](https://arxiv.org/abs/2411.02908) Cited by: Appendix A. 
  * Sap et al. (2019) M. Sap, H. Rashkin, D. Chen, R. Le Bras, and Y. Choi Social iqa: commonsense reasoning about social interactions.  In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),  External Links: [Link](https://arxiv.org/abs/1904.09728) Cited by: 5th item. 
  * Sarfi et al. (2025) A. Sarfi, B. Thérien, J. Lidin, and E. Belilovsky Communication efficient llm pre-training with sparseloco.  arXiv preprint arXiv:2508.15706.  External Links: [Link](https://arxiv.org/abs/2508.15706) Cited by: Appendix A. 
  * Shazeer et al. (2018) N. Shazeer, Y. Cheng, N. Parmar, D. Tran, A. Vaswani, P. Koanantakool, P. Hawkins, H. Lee, M. Hong, C. Young, et al. Mesh-TensorFlow: deep learning for supercomputers.  Advances in neural information processing systems 31.  External Links: [Link](https://arxiv.org/abs/1811.02084) Cited by: Appendix A. 
  * Shazeer et al. (2017) N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean Outrageously large neural networks: the sparsely-gated mixture-of-experts layer.  External Links: [Link](https://openreview.net/forum?id=B1ckMDqlg) Cited by: §5.5. 
  * Shoemake (1985) K. Shoemake Animating rotation with quaternion curves.  In Proceedings of the 12th annual conference on Computer graphics and interactive techniques,  pp. 245–254.  External Links: [Link](https://dl.acm.org/doi/10.1145/325165.325242) Cited by: §D.2. 
  * Singh et al. (2019) A. Singh, R. Hu, V. Goswami, G. Couprie, W. Galuba, M. Rohrbach, and D. Kiela Towards vqa models that can read.  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  External Links: [Link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf) Cited by: 6th item. 
  * Singh et al. (2024) S. Singh, A. Romanou, C. Fourrier, D. I. Adelani, J. G. Ngui, D. Vila-Suero, P. Limkonchotiwat, K. Marchisio, W. Q. Leong, Y. Susanto, R. Ng, S. Longpre, W. Ko, M. Smith, A. Bosselut, A. Oh, A. F. T. Martins, L. Choshen, D. Ippolito, E. Ferrante, M. Fadaee, B. Ermis, and S. Hooker Global mmlu: understanding and addressing cultural and linguistic biases in multilingual evaluation.  External Links: [Link](https://arxiv.org/abs/2412.03304) Cited by: Appendix B. 
  * Stich (2018) S. U. Stich Local SGD converges fast and communicates little.  International Conference on Learning Representations (ICLR).  External Links: [Link](https://arxiv.org/abs/1805.09767) Cited by: Appendix A. 
  * Strati et al. (2025) F. Strati, Z. Zhang, G. Manos, I. S. Périz, Q. Hu, T. Chen, B. Buzcu, S. Han, P. Delgado, and A. Klimovic Sailor: automating distributed training over dynamic, heterogeneous, and geo-distributed clusters.  External Links: [Link](https://arxiv.org/abs/2504.17096) Cited by: Appendix A. 
  * Sutskever et al. (2013) I. Sutskever, J. Martens, G. Dahl, and G. Hinton On the importance of initialization and momentum in deep learning.  International Conference on Machine Learning (ICML).  External Links: [Link](https://proceedings.mlr.press/v28/sutskever13.html) Cited by: §2. 
  * Sutton (2019) R. S. Sutton The bitter lesson.  Note: <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>Incomplete Ideas (blog) Cited by: §5.5. 
  * Thérien et al. (2025) B. Thérien, X. Huang, I. Rish, and E. Belilovsky MuLoCo: muon is a practical inner optimizer for diloco.  arXiv preprint arXiv:2505.23725.  External Links: [Link](https://arxiv.org/abs/2505.23725) Cited by: Appendix A, §D.3. 
  * Wang et al. (2021) J. Wang, Z. Charles, Z. Xu, G. Joshi, H. B. McMahan, M. Al-Shedivat, G. Andrew, S. Avestimehr, K. Daly, D. Data, et al. A field guide to federated optimization.  arXiv preprint arXiv:2107.06917.  External Links: [Link](https://arxiv.org/abs/2107.06917) Cited by: Appendix A. 
  * Wang et al. (2020) J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor Tackling the objective inconsistency problem in heterogeneous federated optimization.  Advances in neural information processing systems.  External Links: [Link](https://arxiv.org/abs/2007.07481) Cited by: Appendix A. 
  * Wang et al. (2023) J. Wang, Y. Lu, B. Yuan, B. Chen, P. Liang, C. De Sa, C. Re, and C. Zhang CocktailSGD: fine-tuning foundation models over 500mbps networks.  International Conference on Machine Learning (ICML).  External Links: [Link](https://openreview.net/forum?id=w2Vrl0zlzA) Cited by: Appendix A. 
  * Wang et al. (2024) Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen MMLU-pro: a more robust and challenging multi-task language understanding benchmark.  External Links: [Link](https://arxiv.org/abs/2406.01574) Cited by: Appendix B. 
  * Wongpanich et al. (2025) A. Wongpanich, T. Oguntebi, J. B. Paredes, Y. E. Wang, P. M. Phothilimthana, R. Mitra, Z. Zhou, N. Kumar, and V. J. Reddi Machine learning fleet efficiency: analyzing and optimizing large-scale google tpu systems with ml productivity goodput.  External Links: 2502.06982, [Link](https://arxiv.org/abs/2502.06982) Cited by: §3.3. 
  * Wortsman et al. (2022) M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith, and L. Schmidt Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.  International Conference on Machine Learning.  External Links: [Link](https://arxiv.org/abs/2203.05482) Cited by: §D.2. 
  * Yadav et al. (2023) P. Yadav, D. Tam, L. Choshen, C. A. Raffel, and M. Bansal Ties-merging: resolving interference when merging models.  Advances in neural information processing systems 36, pp. 7093–7115.  External Links: [Link](https://openreview.net/forum?id=xtaX3WyCj1) Cited by: §D.2. 
  * Yue et al. (2024) X. Yue, Y. Ni, K. Zhang, T. Zheng, L. Ruoqi, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.  In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,  External Links: [Link](https://arxiv.org/abs/2410.19168) Cited by: 5th item. 
  * Zellers et al. (2019) R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi HellaSwag: can a machine really finish your sentence?.  In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics,  External Links: [Link](https://arxiv.org/abs/1905.07830) Cited by: 3rd item. 
  * Zhang et al. (2016) W. Zhang, S. Gupta, X. Lian, and J. Liu Staleness-aware async-sgd for distributed deep learning.  In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence,  IJCAI’16, pp. 2350–2356.  External Links: ISBN 9781577357704 Cited by: Appendix A. 
  * Zhao et al. (2023) Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. bit. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, et al. Pytorch FSDP: experiences on scaling fully sharded data parallel.  Very Large Data Base (VLDB).  External Links: [Link](https://arxiv.org/abs/2304.11277) Cited by: Appendix A. 
  * Zinkevich et al. (2010) M. Zinkevich, M. Weimer, L. Li, and A. Smola Parallelized stochastic gradient descent.  Advances in neural information processing systems.  External Links: [Link](https://dl.acm.org/doi/10.5555/2997046.2997185) Cited by: Appendix A. 


## Appendix A Related Work

#### Distributed training at scale.

The scale of LLM pre-training has necessitated advancements in distributed training methods. One such line of work focuses on more efficient ways to shard data-parallel training across large numbers of accelerators. This includes work on distributed data-parallel training (Shazeer et al., 2018; Li et al., 2020), ZeRO and fully-sharded data parallelism (Rajbhandari et al., 2020; Ren et al., 2021; FairScale authors, 2021; Zhao et al., 2023), pipeline parallelism (Petrowski et al., 1993; Huang et al., 2019; Narayanan et al., 2019), model parallelism (Dean et al., 2012), tensor parallelism (Narayanan et al., 2021; Korthikanti et al., 2023), sequence parallelism (Korthikanti et al., 2023), and (for mixture-of-experts models) expert parallelism (Lepikhin et al., 2021). To adapt these synchronous methods to heterogeneous hardware, systems like Sailor (Strati et al., 2025) automate the search for load-balanced parallelization configurations.

#### DiLoCo and related methods.

An alternative approach to pre-training involves using periodic synchronization across accelerators, reducing network bottlenecks in training. This idea has existed for decades (Mangasarian and Solodov, 1993), and has been periodically re-purposed (or re-discovered) in a variety of settings, including federated learning (Zinkevich et al., 2010; McMahan et al., 2017; Stich, 2018). The use of inner and outer optimizers was first proposed by Hsu et al. (2019) and Reddi et al. (2021) in the context of federated learning, focusing on SGD as the inner optimizer and SGDM or Adam (Kingma and Ba, 2015) as the outer optimizer. This type of approach was used by Douillard et al. (2023) for language model pre-training, where it was adapted specifically for this setting by using Adam as the inner optimizer and SGDM with Nesterov momentum in the outer. Since then, there has been a large amount of work building on this, including versions of DiLoCo with better system properties (Liu et al., 2024; Kale et al., 2025; Qi et al., 2025; Kim et al., 2025), variants of DiLoCo that further reduce bandwidth consumption (Douillard et al., 2025; Beton et al., 2025; Sarfi et al., 2025), empirical scaling-oriented analyses of DiLoCo (Charles et al., 2025; Thérien et al., 2025), and other algorithmic variants of DiLoCo (Sani et al., 2024; Thérien et al., 2025; Kolehmainen et al., 2025; Sarfi et al., 2025; Fan and Zhang, 2025; Lidin et al., 2026).

As discussed above, DiLoCo shares a conceptually similar underpinning to many optimization algorithms in federated learning, especially the FedOpt framework (Reddi et al., 2021). While we defer to (Wang et al., 2021) for an overview of federated optimization, we highlight a number of connections particularly relevant to Decoupled DiLoCo. First, we note that our token-based weighting strategy to account for variable speeds is similar to the step-weighting schema proposed by Wang et al. (2020). Second, we note that the idea of minimum quorum is analogous to over-provisioning client sampling in federated learning, something used in production settings by Bonawitz et al. (2019). Last, we note that many of the asynchrony-aware techniques used by Decoupled DiLoCo are conceptually similar to the Papaya framework (Huba et al., 2022) for federated learning.

DiLoCo shares many similarities with work on communication-efficient training and optimization methods for LLMs. These include techniques like quantization (Alistarh et al., 2017), sparsification (Wang et al., 2023), and communication-efficient optimizers (Ahn and Myung, 2025; Jovanović et al., 2026) though these approaches are typically orthogonal and can be combined with DiLoCo (Douillard et al., 2025).

#### Asynchrony and fault-tolerance at LLM scales.

Parallel to DiLoCo, there are a number of works that that provide robustness to hardware failures and stragglers at LLM scales by employing forms of asynchrony and fault tolerance. There are largely two axes to parallelize over when employing such techniques: the layer/pipeline axis, or the data axis. For the former, Borzunov et al. (2023) and Ryabinin et al. (2023) train LLMs over unreliable networks by dynamically re-routing computations through the model in the event of stragglers and failures. Similarly, Jang et al. (2023) provide resilience by reconfiguring pipeline templates in after node failures. Such approaches have also sparked research on designing optimizers for asynchronous pipeline parallelism (Ajanthan et al., 2025).

On the data axis, much of the work (including our own) is similar in spirit to Hogwild! (Niu et al., 2011). Building on the parameter server based formulation of asynchronous stochastic gradient descent in Dean et al. (2012), Zhang et al. (2016) adapt learning rates based on staleness metrics. Unlike variants of asynchronous SGD, the Decoupled DiLoCo approach to asynchrony never requires updating the model using a stale gradient computed on a previous version of the parameters. Other works introduce fault-tolerant implementations of all-reduce. Ryabinin et al. (2021) utilize a butterfly all-reduce to handle unreliable devices, and more recently, Salpekar et al. (2026) use fault-tolerant sharded data-parallelism for resilience in large-scale distributed jobs. The use of asynchrony in tandem with DiLoCo was first proposed by Liu et al. (2024), and was extended to reduce inter-region communication by Kim et al. (2025).

## Appendix B Post-training

To ensure that the benefits of Decoupled DiLoCo during pre-training do not come at the cost of downstream capabilities, we evaluate the impact of our framework on the post-training phase. We take the 5B parameter models trained on 1T tokens from our experiments with simulated hardware failures (see subsection 5.2) and subject them to an identical post-training pipeline, which is a light version of Gemma 4 recipe.

Specifically, we compare three pre-training setups: a standard Data-Parallel (DP) baseline in an ideal environment with no failures, Decoupled DiLoCo in an ideal environment, and Decoupled DiLoCo trained under heavy simulated hardware failures (MTBIchip=1\texttt{MTBI}_{\text{chip}}=1 year, Nchip=1.2​mN_{\text{chip}}=1.2\text{m}). We apply the same light version of the Gemma 4 post-training recipe to all three pre-trained checkpoints to observe how they adapt.

The results, presented in Table 6, demonstrate that Decoupled DiLoCo maintains highly competitive post-training performance. We note that this finding is contrary to that of Acker et al. (2025), where their iteration on DiLoCo was worse than data-parallel training even in the pre-training phase. We believe that this indicates the importance of making DiLoCo work as well as data-parallel in pre-training phases, and that post-training does not close any gaps in model quality at this phase.

Across evaluations spanning knowledge (MMLU-Pro (Wang et al., 2024)), multi-lingual understanding (GMMLU-lite (Singh et al., 2024)), mathematical reasoning (GSM8K (Cobbe et al., 2021)), and coding (HumanEval (Chen et al., 2021)), the models pre-trained with Decoupled DiLoCo perform comparably to, and in several cases, noticeably exceed, the monolithic DP baseline. Crucially, this parity holds even for the model pre-trained under chaotic, asynchronous hardware failures, confirming that the relaxed synchronization and failure recovery mechanisms of Decoupled DiLoCo do not compromise the model’s capacity for post-training.

| Simulated | MMLU-Pro | GMMLU | GSM8K | HumanEval  
---|---|---|---|---|---  
Pre-training Method | HW Fails? | knowledge | multi-lingual | math | code  
Data-Parallel | No | 21.4 | 42.3 | 23.8 | 22.0  
Decoupled DiLoCo | No | 18.6 | 43.3 | 35.4 | 22.6  
Decoupled DiLoCo | Yes | 18.3 | 42.4 | 28.7 | 22.0  
Table 6: Post-training results for Gemma 5B models, pretrained on 1T tokens. All models were pre-trained under their respective setups and subsequently fine-tuned using the same light Gemma 4 post-training recipe. Results indicate that Decoupled DiLoCo pre-training does not degrade post-training performance, even when subjected to hardware failures.

## Appendix C Fragmentation strategies

Recall the fragmented nature of Algorithm 1 and Algorithm 2. Communication to and from the syncer operates on fragments of the model. Thus, the fragmentation strategy (how we partition the model weights into PP fragments) has two distinct potential forms of impact on the performance of these algorithms.

  1. 1.

The fragmentation strategy can impact the model performance (measured via various downstream tasks).

  2. 2.

The fragmentation strategy can impact the system performance, (by changing the bandwidth requirement of different communication steps).


Perhaps surprisingly, we find that downstream model performance is quite robust to fragmentation strategy, and therefore we opt for fragmentation strategies that have maximally beneficial bandwidth usage profiles. To illustrate this, we describe the various fragmentation strategies we tried.

#### Layer fragmentation

The first strategy is that proposed by Douillard et al. (2025). In this, we split a model into fragments according to its layer structure. First, we have a single fragment for all non-transformer layers. Let LL denote the number of transformer layers. For P−1≤LP-1\leq L, we form the remaining P−1P-1 fragments by strided selection. For example, if L=6L=6 and P−1=3P-1=3, then transformer layers 1 and 4 form a fragment, as do transformer layers 2 and 5, and 3 and 6.

While this strategy yields good training performance (Douillard et al., 2025), layer fragmentation has system drawbacks. If the number of transformer layers QQ is less than HH, then there will necessarily be some steps at which we do not communicate with the syncer. This means that the communication incurred can be quite bursty. See 8(a) for a simple example of this.

#### Tensor fragmentation

To remedy this, we can instead partition model at the level of individual tensors (as opposed to transformer layers, which typically contain multiple tensors). In this strategy, we put all non-transformer tensors in the first fragment. The remaining tensors (say there are SS of them) are bucketed into P−1P-1 fragments in a strided manner by their index in some ordering of the tensors. For example, if S=9S=9 and P−1=3P-1=3, then (in addition to the fragment with non-transformer tensors), the first fragment will have tensors 1,4,71,4,7, the second will have tensors 2,5,82,5,8, and the third will have tensors 3,6,93,6,9.

While this allows us to communicate smaller fragments more often (avoiding the extremely bursty communication patterns of layer fragmentation), the fragments may still be of very different sizes, meaning that bandwidth usage is still uneven. See 8(b) for an example.

#### Balanced tensor fragmentation

The last strategy we examine attempts to make the fragments as evenly balanced, in terms of total size, as possible. To do so we use a greedy bin-packing algorithm to pack the QQ tensors in the model into PP buckets. Specifically, we sort the tensors by their total size (measured in bits), and pack them into bins according via greedy number partitioning, where we iterate over the tensors in descending order of size, and put each successive tensor in the fragment whose total size is the smallest.

This has many beneficial system properties. First, as long as Q≥HQ\geq H, we are always guaranteed to have P=HP=H, so that we communicate a fragment at every step. Moreover, the greedy algorithm is well-known to be no more than 4/34/3 times worse than the optimal packing strategy, measured in terms of the maximum size of any fragment (Graham, 1969). This means that the peak bandwidth usage of balanced tensor fragmentation is at most 4/34/3 of the minimum possible peak bandwidth of any fragmentation strategy. See 8(c) for an example.

(a) Layer Fragmentation. Because HH is larger than the number of layers, when t=2,4t=2,4, we do not send any fragment to the syncer, resulting in bursty communication with potentially high peak bandwidth usage.

(b) Tensor Fragmentation. We send a fragment to the syncer at every step, but the fragments have drastically different sizes, so the peak bandwidth usage might still be high.

(c) Balanced Tensor Fragmentation. This sends a fragment to the syncer at every step, with approximately equal sizes, reducing peak bandwidth and avoiding bursty communication.

Figure 8: Layer, tensor, and balanced tensor fragmentation applied to a model with an embedding layer and 2 transformer layers, each containing 6 tensors. We set H=5H=5, and visualize which fragment is sent from the learner to the syncer at every step.

#### Sub-tensor fragmentation

In preliminary investigations, we also evaluated a _sub-tensor_ fragmentation strategy, where we would split individual tensors across fragments. These were essentially split across the output of the tensor (e.g. the columns of the embedding layer), to guarantee that inputs were not effectively partitioned across fragments. By construction, this yields fragments of even more balanced size than the balanced tensor strategy above. While we found it competitive with other schemes, we opted not to use it in final experiments due to implementation complexity and to aid in debugging.

### C.1 Fragment Size Comparison

We now compare the size of fragments created by applying the above strategies to various models. Here we set P=24P=24 for all three methods, and record the total size (in bytes) of each fragment. This is simply the sum of the size of each tensor that is grouped under the same fragment. We show the results for the 5B and 9B dense models in Figure 9.

(a) Gemma 5B.

(b) Gemma 9B.

Figure 9: Comparison of fragment sizes on dense 5B and 9B models. Here we omit the embedding layer for visual clarity, as in all three cases it is in its own fragment that is significantly larger.

(a) Gemma 2.8B MoE.

(b) Gemma 3.8B MoE.

Figure 10: Comparison of fragment sizes on Mixture-of-Experts models with 2.8B and 3.8B activated parameters.

### C.2 Evaluation Results

We compare the three strategies on a 2B parameter model, setting H=P=24H=P=24, and M=8M=8. See Table 7 for the results. We find that the methods all perform quite comparably on downstream evaluations. This is potentially surprising - tensor and balanced tensor fragmentation do not respect the layer structure of the model, but this does not reduce model quality. Thus, the only factor that is important then is their system performance. Because balanced tensor fragmentation has the lowest peak bandwidth and most consistent bandwidth usage across fragments, we use this strategy for all other experiments.

| Text Benchmarks | Vision Benchmarks  
---|---|---  
Method | Avg | Arc-C | Arc-E | BoolQ | Hella | Piqa | Siqa | Wino | Avg | COCO | Chart | Doc | Doc-H | Info | MMMU | TextVQA  
Balanced | 54.4 | 36.3 | 64.0 | 55.8 | 51.4 | 72.5 | 46.5 | 54.1 | 37.6 | 62.9 | 13.3 | 48.1 | 30.8 | 24.4 | 19.3 | 64.5  
Layer | 55.3 | 36.5 | 64.6 | 59.1 | 52.0 | 71.5 | 46.5 | 56.7 | 35.5 | 59.5 | 12.4 | 46.7 | 28.7 | 24.9 | 14.9 | 61.1  
Tensor | 55.4 | 35.8 | 65.8 | 61.6 | 51.8 | 72.0 | 46.5 | 54.5 | 35.8 | 57.1 | 15.0 | 48.8 | 30.6 | 24.1 | 13.9 | 61.2  
  
Table 7: Downstream evaluations of Gemma 2B trained on 26B tokens with Decoupled DiLoCo using M = 8 learners and various fragmentation strategies.

## Appendix D Algorithmic Ablations

Below we briefly detail various design choices we made when concretizing Algorithm 1 and Algorithm 2.

### D.1 Mixing learner and syncer params

In Algorithm 1, upon receiving Θp(t−τ)\Theta_{p}^{(t-\tau)} from the syncer, a learner simply sets their local fragment θp(t)\theta_{p}^{(t)} to this value. We could instead interpolate via

| θp(t)←α​θp(t)+(1−α)​Θp(t).\theta_{p}^{(t)}\leftarrow\alpha\theta_{p}^{(t)}+(1-\alpha)\Theta_{p}^{(t)}\,. |  | (5)  
---|---|---|---  
  
While Douillard et al. (2025) showed that α=0.5\alpha=0.5 was useful for M=2M=2 and τ≥10\tau\geq 10, we found that at larger MM α=0\alpha=0 is optimal. Intuitively, with more learners MM, each learner contribution θm,p(t)\theta_{m,p}^{(t)} is less informative, and thus the global update Θp(t)\Theta_{p}^{(t)} yields better performance.

### D.2 Merging methods

In Algorithm 2, the syncer is responsible for merging the per-fragment learner updates. Here we discuss two methods for doing this merging, and how they compare.

#### Direct averaging

One option for the Merge function in Algorithm 2 is to mirror (Douillard et al., 2024) and directly average the per-fragment outer gradients Δm,p(t)\Delta_{m,p}^{(t)} from (2). Ignoring weighting, we would therefore define

|  | Merge​({θm,p(t)}i∈ℳt,Θp(t−H))\displaystyle\texttt{Merge}\left(\\{\theta_{m,p}^{(t)}\\}_{i\in\mathcal{M}_{t}},\Theta_{p}^{(t-H)}\right) |   
---|---|---|---  
|  | =1|ℳt|​∑i∈ℳtΔm,p(t)=1|ℳt|​∑i∈ℳt(Θp(t−H)−θm,p(t))\displaystyle=\dfrac{1}{|\mathcal{M}_{t}|}\sum_{i\in\mathcal{M}_{t}}\Delta_{m,p}^{(t)}=\dfrac{1}{|\mathcal{M}_{t}|}\sum_{i\in\mathcal{M}_{t}}\left(\Theta_{p}^{(t-H)}-\theta_{m,p}^{(t)}\right) |   
  
The weighted version is defined analogously via

|  | Merge​({θm,p(t),wm,p}i∈ℳt,Θp(t−H))\displaystyle\texttt{Merge}\left(\\{\theta_{m,p}^{(t)},w_{m,p}\\}_{i\in\mathcal{M}_{t}},\Theta_{p}^{(t-H)}\right) |   
---|---|---|---  
|  | =∑i∈ℳtwm,p​(Θp(t−H)−θm,p(t))∑i∈ℳtwm,p\displaystyle=\dfrac{\sum_{i\in\mathcal{M}_{t}}w_{m,p}\left(\Theta_{p}^{(t-H)}-\theta_{m,p}^{(t)}\right)}{\sum_{i\in\mathcal{M}_{t}}w_{m,p}} |   
  
#### Radial-directional averaging.

Empirically, the per-learner outer gradients Δm,p(t)\Delta_{m,p}^{(t)} are almost pairwise orthogonal. Thus, if they are all of norm RR, their average has norm of approximately R/MR/\sqrt{M}. Hence, we generally need to re-tune the outer optimizer when we merge via averaging.

To mitigate this, we can instead attempt to merge in a way that the output norm is essentially invariant to MM. We propose our own lightweight merging method, Radial-Directional Averaging (RDA), which separately averages the norm and direction of the vectors. Formally, RDA operates as follows. Define

| ϕ:ℝn\{0}→ℝn,x↦x‖x‖2.\phi:\mathbb{R}^{n}\backslash\\{0\\}\to\mathbb{R}^{n},x\mapsto\dfrac{x}{\|x\|_{2}}. |   
---|---|---  
  
For a set of non-zero vectors v1,…,vMv_{1},\dots,v_{M} we define the function RDA\rda via:

| RDA⁡(v1,…,vM)=(1M​∑i=1M‖vi‖2)​ϕ​(1M​∑i=1Mϕ⁡(vi)).\rda(v_{1},\dots,v_{M})=\left(\dfrac{1}{M}\sum_{i=1}^{M}\|v_{i}\|_{2}\right)\phi\left(\dfrac{1}{M}\sum_{i=1}^{M}\phi(v_{i})\right). |   
---|---|---  
  
That is, we average the norm and direction of the vectors separately, projecting the average direction to be a unit vector, and multiply them. Note that if ‖vi‖=R\|v_{i}\|=R for all ii, then ‖RDA⁡(v1,…,vM)‖=R\|\rda(v_{1},\dots,v_{M})\|=R. We define a weighted version of RDA\rda completely analogously. Let w1,…,wMw_{1},\dots,w_{M} denote the weights for each vector. Then

|  | RDA⁡({(vi,wi)}i=1M)=(∑i=1Mwi​vi∑i=1Mwi)​ϕ​(∑i=1Mwi​ϕ​(vi)∑i=1Mwi)\displaystyle\rda\left(\\{(v_{i},w_{i})\\}_{i=1}^{M}\right)=\left(\dfrac{\sum_{i=1}^{M}w_{i}v_{i}}{\sum_{i=1}^{M}w_{i}}\right)\phi\left(\dfrac{\sum_{i=1}^{M}w_{i}\phi(v_{i})}{\sum_{i=1}^{M}w_{i}}\right) |   
---|---|---|---  
  
To specify the corresponding Merge function in 2, we apply RDA\rda to the (weighted) per-learner outer gradients, as when we do direct averaging.

Note that at first glance, RDA requires two all-reduces while direct averaging only one. However, the magnitude tensors are less than 0.05%0.05\% of the size of the direction tensors, making the overhead negligible.

#### Other options

For M=2M=2 one can use SLERP (Shoemake, 1985), but it is unclear how best to generalize this to M>2M>2. While (Buss and Fillmore, 2001) develop a generalization for M>2M>2, it involves least-squares minimization, and as such incurs more computational overhead than methods like direct averaging. Ramé et al. (2024) propose an iterative version of SLERP to merge M>2M>2 tensors, but we found that RDA performed comparably if not better, with reduced complexity. We also evaluated other merging schemes, such as TIES-MERGING (Yadav et al., 2023), Iso-C (Marczak et al., 2025), and sparsity-aware merging schemes, but found that direct averaging and RDA outperformed such methods in our setting. This is potentially due to the fact that much of the work on such merging methods has focused on merging after large numbers of steps (Wortsman et al., 2022) on non-IID distributions, while we do partial model merging essentially every step.

#### Results

We compare direct averaging (Avg) and RDA on a 2B parameter model. Qualitatively, we find that the outer gradients corresponding to the embedding component of the model often did not have the “near-orthogonality” property described above. Therefore, we compared the use of Avg and RDA separately on the embedding component and on the rest of the model. We compare this for M=8M=8 learners (Table 8) and M=16M=16 learners (Table 9).

For M=8M=8, there is no entirely consistent method that outperforms the others. However, we find that on nearly all tasks, applying RDA to the model and Avg to the embedding performs comparably or better to the other methods. For M=16M=16, we find that applying RDA to the non-embedding component of the model drastically improves performance. Moreover, applying Avg to the embedding generally (though not uniformly) boosts performance slightly over RDA. Therefore, for all of our primary experiments we apply Avg to the embedding component, and RDA to the remaining parts of the model.

Merge Method | Text Benchmarks | Vision Benchmarks  
---|---|---  
Model | Embed | Avg | Arc-C | Arc-E | BoolQ | Hella | Piqa | Siqa | Wino | Avg | COCO | Chart | Doc | Doc-H | Info | MMMU | TextVQA  
Avg | Avg | 54.1 | 35.8 | 63.1 | 58.0 | 50.5 | 71.1 | 46.3 | 54.2 | 34.2 | 58.8 | 11.9 | 46.1 | 27.2 | 22.9 | 13.8 | 58.7  
Avg | RDA | 53.3 | 34.6 | 59.7 | 59.8 | 48.3 | 71.2 | 45.2 | 54.5 | 32.9 | 53.7 | 12.1 | 40.7 | 24.8 | 20.7 | 19.2 | 59.0  
RDA | Avg | 54.4 | 36.3 | 64.0 | 55.8 | 51.4 | 72.5 | 46.5 | 54.1 | 37.6 | 62.9 | 13.3 | 48.1 | 30.8 | 24.4 | 19.3 | 64.5  
RDA | RDA | 55.1 | 34.7 | 63.6 | 62.8 | 50.9 | 72.3 | 46.2 | 55.1 | 35.6 | 59.4 | 12.6 | 46.0 | 28.4 | 24.6 | 16.1 | 62.2  
Table 8: Downstream evaluations of Gemma 2B trained on 26B tokens with Decoupled DiLoCo using M=8M=8 learners and various merging methods. Merge Method | Text Benchmarks | Vision Benchmarks  
---|---|---  
Model | Embed | Avg | Arc-C | Arc-E | BoolQ | Hella | Piqa | Siqa | Wino | Avg | COCO | Chart | Doc | Doc-H | Info | MMMU | TextVQA  
Avg | Avg | 52.7 | 35.1 | 60.9 | 56.5 | 47.7 | 70.5 | 45.4 | 53.0 | 28.2 | 54.1 | 10.3 | 36.6 | 21.0 | 19.6 | 5.4 | 50.7  
Avg | RDA | 49.3 | 29.7 | 52.3 | 56.6 | 40.6 | 68.7 | 45.3 | 51.8 | 22.4 | 31.6 | 9.7 | 26.4 | 15.6 | 15.2 | 14.0 | 44.4  
RDA | Avg | 54.5 | 35.0 | 63.1 | 59.5 | 50.2 | 72.0 | 45.7 | 55.6 | 33.6 | 58.4 | 13.4 | 43.2 | 26.5 | 21.3 | 11.8 | 60.9  
RDA | RDA | 52.7 | 35.6 | 60.9 | 53.3 | 48.6 | 71.4 | 46.3 | 52.8 | 33.9 | 56.6 | 12.2 | 43.3 | 26.2 | 24.2 | 15.3 | 59.2  
Table 9: Downstream evaluations of Gemma 2B trained on 26B tokens with Decoupled DiLoCo using M=16M=16 learners and various merging methods.

### D.3 Outer gradients compression

Similarly to Streaming DiLoCo (Douillard et al., 2025), we can compress outer gradients from bf16 (16 bits) to int4 (4 bits) to minimize bandwidth constraint when doing the all-reduce synchronization. We display in Table 10 the performance of a 2B dense model trained with high level of simulated hardware failures (MTBIchip=1\texttt{MTBI}_{\text{chip}}=1 year, and using Nchip=1.2​mN_{\text{chip}}=1.2\text{m} simulated chips). We found that compressing the outer gradients before their all-reduce (in the Merge, L11 of Algorithm 2) to int4 yields comparable performance to bf16, even in the asynchronous case with heavy rate of hardware failures. We also noticed that using 2 bits or 1 bit resulted in unacceptable performance regression. More exploration could be done to exploit more advanced compression scheme such as MuLoCo (Thérien et al., 2025).

# bits / value | 16 bits | 4 bits | 2 bits | 1 bit  
---|---|---|---|---  
Relative val loss | — | +0.09% | +32% | +83%  
Text (avg) | 55.4 | 55.3 | 42.6 | 37.3  
Arc-C | 36.5 | 35.7 | 22.9 | 20.6  
Arc-Easy | 64.2 | 64.9 | 35.9 | 32.5  
BoolQ | 62.0 | 62.2 | 61.8 | 39.0  
HellaSwag | 51.8 | 51.5 | 27.2 | 25.4  
Piqa | 71.6 | 71.4 | 57.8 | 56.4  
Siqa | 45.8 | 46.6 | 40.7 | 37.1  
Winogrande | 55.6 | 54.8 | 51.7 | 50.4  
Vision (avg) | 36.6 | 34.4 | 5.3 | 0.0  
COCO-Captions | 61.5 | 60.4 | 3.5 | 0.0  
ChartQA | 12.7 | 11.8 | 2.8 | 0.1  
DocVQA | 47.7 | 45.3 | 2.2 | 0.0  
DocVQA-Hard | 29.9 | 27.9 | 0.9 | 0.0  
InfographicVQA | 24.4 | 24.3 | 4.6 | 0.0  
MMMU | 18.0 | 9.9 | 17.6 | 0.1  
TextVQA | 62.0 | 61.3 | 5.8 | 0.0  
  
Table 10: Outer gradients compression impact on ML performance for a 2B model trained on 26B tokens, while injecting hardware failures at the MTBIchip=1\texttt{MTBI}_{\text{chip}}=1 year and Nchip=1.2​mN_{\text{chip}}=1.2\text{m} simulated chips.

### D.4 Number of learners impact on asynchronous resiliency

In Table 11, we compare the ML performance on text and vision benchmark without simulated hardware failures and with (Nchip=1.2​mN_{\text{chip}}=1.2\text{m} to Nchip=2.4​mN_{\text{chip}}=2.4\text{m} simulated chips and MTBIchip\texttt{MTBI}_{\text{chip}} = 1y, as in subsection 5.2. Note that lower values of number of learners MM result in lower goodput because the blast radius is larger: for M=2M=2, the radius is half of the total chips. Larger numbers of learners significantly improves goodput, here up to M=16M=16 with 93% goodput at Nchip=1.2​mN_{\text{chip}}=1.2\text{m} chips. We show neutral ML performance from M=2M=2 to M=8M=8, and slightly degraded performance at M=16M=16 but that can be mitigated by multiple aspects, including training larger models with larger token budget.

| No HW failures |  | With Nchip=1.2​mN_{\text{chip}}=1.2\text{m} and  
---  
MTBIchip\texttt{MTBI}_{\text{chip}} = 1y  
| With Nchip=2.4​mN_{\text{chip}}=2.4\text{m} and  
---  
MTBIchip\texttt{MTBI}_{\text{chip}} = 1y  
# learners MM | 2 | 4 | 8 | 16 | 2 | 4 | 8 | 16 | 2 | 4 | 8 | 16  
Goodput | 100% | 100% | 100% | 100% | 70% | 82% | 88% | 93% | 54% | 73% | 80% | 86%  
Text (avg) | 55.1 | 55.9 | 54.4 | 54.5 | 54.4 | 55.0 | 55.4 | 53.9 | 54.4 | 54.5 | 54.6 | 53.4  
Arc-Challenge | 37.1 | 36.0 | 36.3 | 35.0 | 35.7 | 36.9 | 36.5 | 34.4 | 35.4 | 35.5 | 35.1 | 35.8  
Arc-Easy | 64.1 | 65.2 | 64.0 | 63.1 | 64.2 | 63.5 | 64.2 | 63.5 | 61.1 | 63.0 | 64.4 | 62.6  
BoolQ | 58.2 | 61.2 | 55.8 | 59.5 | 59.4 | 60.2 | 62.0 | 59.8 | 59.2 | 60.2 | 60.2 | 54.1  
HellaSwag | 52.7 | 52.9 | 51.4 | 50.2 | 51.4 | 51.9 | 51.8 | 50.0 | 51.6 | 51.5 | 50.9 | 49.7  
Piqa | 72.4 | 72.2 | 72.5 | 72.0 | 71.9 | 71.5 | 71.6 | 71.0 | 71.8 | 71.7 | 70.9 | 71.2  
Siqa | 46.3 | 47.2 | 46.5 | 45.7 | 45.4 | 46.0 | 45.8 | 46.1 | 45.9 | 45.5 | 45.4 | 44.6  
Winogrande | 54.8 | 56.2 | 54.1 | 55.6 | 53.0 | 55.1 | 55.6 | 52.6 | 55.8 | 54.3 | 55.1 | 56.0  
Vision (avg) | 36.8 | 36.8 | 37.6 | 33.6 | 35.8 | 35.6 | 36.6 | 35.1 | 37.2 | 35.3 | 37.3 | 34.3  
COCO-Captions | 62.8 | 58.6 | 62.9 | 58.4 | 59.5 | 61.8 | 61.5 | 56.8 | 59.0 | 59.8 | 63.1 | 55.0  
ChartQA | 12.0 | 14.0 | 13.3 | 13.4 | 14.1 | 13.4 | 12.7 | 13.6 | 13.5 | 12.4 | 13.5 | 12.5  
DocVQA | 47.9 | 50.8 | 48.1 | 43.2 | 48.1 | 47.9 | 47.7 | 47.1 | 47.8 | 48.3 | 50.4 | 46.8  
DocVQA-Hard | 28.6 | 31.1 | 30.8 | 26.5 | 28.5 | 28.7 | 29.9 | 26.8 | 30.8 | 25.6 | 30.7 | 28.2  
InfographicVQA | 26.4 | 24.9 | 24.4 | 21.3 | 23.8 | 24.0 | 24.4 | 23.2 | 24.2 | 23.8 | 25.1 | 23.0  
MMMU | 17.6 | 14.8 | 19.3 | 11.8 | 14.0 | 13.8 | 18.0 | 19.0 | 22.0 | 16.2 | 18.8 | 17.4  
TextVQA | 62.6 | 63.6 | 64.5 | 60.9 | 62.6 | 60.0 | 62.0 | 59.2 | 63.0 | 61.0 | 59.9 | 57.0  
Table 11: Simulated hardware failures impact for a 2B model trained on 26B tokens across different numbers of learners MM, from 2 to 16 for 1.2 and 2.4 million simulated chips.

### D.5 Token budget impact

We showcase in Table 12 the results of a 2B dense model trained with data-parallel and Decoupled DiLoCo methods with increasing token budget, at 26B tokens, 260B tokens (10×10\times), and 1.3T tokens (50×50\times). Notice that while Decoupled DiLoCo can be slightly weaker at low token budget, its performance surpasses data-parallel as the budget increases.

| Data-Parallel | Decoupled DiLoCo M=8M=8  
---|---|---  
# tokens |  26B |  260B |  1.3T |  26B |  260B |  1.3T  
Text (avg) |  55.1 |  61.8 |  64.5 |  54.4 |  62.1 |  64.9  
Arc-Challenge |  36.2 |  46.2 |  51.1 |  36.3 |  48.1 |  51.5  
Arc-Easy |  65.1 |  73.2 |  75.5 |  64.0 |  73.8 |  77.0  
BoolQ |  58.8 |  66.1 |  72.2 |  55.8 |  66.1 |  71.6  
HellaSwag |  52.1 |  63.1 |  66.8 |  51.4 |  62.7 |  66.8  
Piqa |  71.8 |  76.6 |  77.3 |  72.5 |  76.0 |  77.8  
Siqa |  46.0 |  48.1 |  47.6 |  46.5 |  48.3 |  48.7  
Winogrande |  55.6 |  59.0 |  61.4 |  54.1 |  59.6 |  60.6  
Vision (avg) |  38.2 |  47.8 |  50.6 |  37.6 |  47.4 |  52.1  
COCO-Captions |  63.8 |  75.2 |  78.5 |  62.9 |  74.2 |  83.7  
ChartQA |  12.2 |  17.4 |  19.3 |  13.3 |  17.4 |  18.1  
DocVQA |  50.9 |  65.6 |  70.3 |  48.1 |  63.3 |  71.3  
DocVQA-Hard |  30.9 |  45.3 |  46.1 |  30.8 |  41.6 |  48.5  
InfographicVQA |  25.8 |  35.5 |  40.3 |  24.4 |  38.8 |  40.6  
MMMU |  21.1 |  24.6 |  26.6 |  19.3 |  24.9 |  29.0  
TextVQA |  62.9 |  71.1 |  73.4 |  64.5 |  71.5 |  73.2  
Table 12: Token budget for a 2B model trained on 26B to 1.3T tokens for data-parallel and Decoupled DiLoCo M=8M=8. Our method assimilates better than the baseline increased amount of tokens during its training.

## Appendix E Infrastructure Details

### E.1 Deterministic Replay via Event Tapes

#### Non-Determinism and Event Logging.

Decoupled DiLoCo’s availability in the event of failures is based on allowing bounded levels of nondeterminism in the training algorithm. Because the syncer aggregates updates as soon as a minimum quorum KK is reached (see Algorithm 2), the exact subset of learners used in any given outer optimization step depends entirely on unpredictable wall-clock timing, network jitter, and hardware failures. Additionally, the exact step on which a learner applies a global fragment update to its local model depends on both random failures and the relative speed of the learner and syncer steps at that moment. To isolate algorithmic behavior from this system noise, the syncer records an _event tape_ , 𝒯\mathcal{T}, during training. This tape captures the full causal state of the system by logging the following metadata on each worker at each local step: a vector clock (Mattern, 1989) which encodes the exact communication pattern between workers as described in Section 4, per-learner token counters, and failure/recovery events.

#### Deterministic Replay.

Given a recorded tape 𝒯\mathcal{T}, the syncer and learners can completely bypass the dynamic quorum logic. By reading the tape, the syncer deterministically executes the exact same sequence of synchronization decisions, participant subsets, and token weightings, and each learner applies fragment updates from the syncer and fails or recovers on the same local steps. This mechanism guarantees bitwise-identical training trajectories regardless of the actual hardware conditions during the replay run.

#### Synthetic Tape Generation.

Beyond replaying historical runs, we built a discrete-event simulator to generate _synthetic tapes_ for controlled “what-if” experiments such as chaos engineering for LLMs as described in Section 3.3. The generator models the learners and syncer as state machines advancing through virtual time, allowing us to inject configurable disruptions. These include constant speed heterogeneity, transient slowdowns, and realistic large-scale hardware failures modeled either on arbitrarily chosen values as in this paper, or calibrated on production cluster data. Using the deterministic replay execution mode in which each worker faithfully follows the synthetic tapes, we can rigorously evaluate the system’s goodput and algorithmic resilience under specific, perfectly reproducible chaotic conditions.

### E.2 Consistent distributed checkpointing

One of the challenges of moving from a single-controller programming model to a model in which multiple workers execute independently is creation of consistent checkpoints. Although the resilience of Decoupled DiLoCo makes a full system restart less likely, resuming from a checkpoint remains an important fallback strategy in extreme failure scenarios outside the bounds defined by the algorithm. Additionally, in the case of deterministic replay where a specific set of learners must participate in each outer optimization, resuming from a checkpoint remains the main resilience strategy in the case of a failure.

Checkpointing must not block training progress, so each of the workers in the system checkpoints its own state asynchronously to its own directory. One naive strategy would be for each worker to checkpoint independently when its local step meets the criteria twmodTc=0t_{w}\bmod T_{c}=0. Unfortunately, this strategy provides no guarantee of capturing training progress, because if the causal dependencies between worker states are not captured in the set of checkpoints, resuming from these checkpoints could put the system in an invalid state. Finding a set of uncoordinated checkpoints from all workers that form a consistent global state (Mattern, 1989) can require rolling back all the way to the initial state of the system (Elnozahy et al., 2002).

To prevent losing an unbounded amount of progress when a failure occurs, the workers follow a version of the Chandy-Lamport distributed snapshotting algorithm (Chandy and Lamport, 1985), as shown in Figure 11. The syncer begins a snapshot when its local step tsmodTc=0t_{s}\bmod T_{c}=0, recording its current model parameters and outer optimizer state. The syncer proceeds with training as usual, sending messages that include its current vector clock to the learner workers. When a learner receives a vector clock from the syncer for which the syncer step meets the condition tsmodTc=0t_{s}\bmod T_{c}=0, this message serves as the snapshot marker (Chandy and Lamport, 1985). Receiving this snapshot marker indicates to learner workers that they should checkpoint their own model parameters and inner optimizer state. Since learner workers update their own vector clocks with the latest syncer step, the next message sent from each learner to the syncer serves to return the snapshot marker, communicating to the syncer that the learner has taken its own snapshot. For each learner, there may have been some messages that were not yet received by the syncer at the time that the syncer started its snapshot, which have vector clocks that happen before the marker that the learner recorded its own snapshot. As the syncer receives these messages, it records them in the snapshot. Upon resuming from checkpoint, these messages are replayed by the syncer as if they were resent from the learners, in order to prevent deadlocks which could otherwise occur. The syncer declares its local snapshot complete and writes the snapshot state to the checkpoint directory when it has received a returned marker from every learner.

To avoid the scenario where a checkpoint cannot complete because some learners fail and never send back the snapshot marker, the syncer omits failed learners from the checkpoint. Upon resumption, instead of loading state from the checkpoint, these failed learners are directed by the syncer to go through the learner recovery process detailed in Section E.3.

Learner 0SyncerLearner 1⟨\langleL0: 10, L1: 8, S: 8⟩\rangle⟨\langleL0: 9, L1: 9, S: 9⟩\rangle⟨\langleL0: 8, L1: 10, S: 8⟩\rangle⟨\langleL0: 10, L1: 10, S: 10⟩\rangle⟨\langleL0: 9, L1: 11, S: 9⟩\rangleStart Snapshot⟨\langleL0: 11, L1: 10, S: 10⟩\rangle⟨\langleL0: 10, L1: 11, S: 11⟩\rangle⟨\langleL0: 9, L1: 12, S: 10⟩\rangleSave CheckpointSave Checkpoint⟨\langleL0: 12, L1: 11, S: 11⟩\rangle⟨\langleL0: 11, L1: 12, S: 12⟩\rangle⟨\langleL0: 10, L1: 13, S: 11⟩\rangleSave Checkpoint

Figure 11: Using vector clocks to communicate training progress and coordinate checkpointing. The arrows shown in blue represent messages that will be saved in the syncer’s checkpoint and replayed, which may be a different number of messages from each learner when K<MK<M.

### E.3 Distributed Learner Recovery

When a learner is slow or temporarily interrupted from training, Decoupled DiLoCo allows it to resume stepping with slightly stale local state and be brought back up to date with the rest of the system through the normal cycle of fragment synchronization. However, when a learner is interrupted for too long, completely fails and restarts, or is newly introduced during compute scavenging as in Section 5.3, resuming contribution to the system requires it to first acquire a recent copy of the model parameters and inner optimizer state. This process is called learner recovery.

The syncer has no special logic to coordinate the recovery of learners - it simply always broadcasts the most recently synchronized fragment to all connected learners. When a recovering learner restarts, it connects to the syncer and waits for the first fragment sync to arrive, stamped with the syncer’s latest vector clock tst_{s}. It then connects to a healthy peer learner and sends tst_{s} as part of a recovery request. The peer will respond with a copy of its local model parameters and inner optimizer state, but only once it has itself observed the same vector clock tst_{s} coming from the syncer. Until that has happened, it continues training and syncing as normal, leaving the recovery request in a pending state. The transfer of model state from peer learner to recovering learner happens asynchronously but may take significant time, especially if the two learners are geographically distant. During this time the recovering learner buffers any fragment update messages it receives from the syncer, such that when the transfer finishes they can be replayed locally to bring the new learner up to date and able to immediately participate in training. The set of messages received after tst_{s} is guaranteed to be sufficient to fully synchronize the recovering learner, thanks to the peer learner using tst_{s} as a threshold before initiating the transfer.

We apply an upper bound on the number of syncer steps a recovery is allowed to take, equal to the synchronization cycle length HH. This ensures that the staleness of any fragment and optimizer state in the recovered learner has the same bounds as in any other learner in the system, preventing recovery events from becoming an unpredictable source of instability.

As shown in Figure 12, the asynchronous design of the learner recovery protocol gives Decoupled DiLoCo an advantage when scavenging with limited bandwidth between compute nodes. Elastic data-parallel training must block on the transmission of the current model parameters and momentum buffers to the new replica before training, meaning that every upsize event (when a faulty slice has been repaired or replaced), incurs downtime lower-bounded by the time to transmit 3×3\times the model parameters to the new replica. In contrast, Decoupled DiLoCo training is tolerant of the new replica using state up to HH steps stale, and transmission of state to the new replica does not block the rest of the system until this limit is reached. As long as there is enough bandwidth available to transmit an additional 3×NH\frac{3\times N}{H} model parameters per step, Decoupled DiLoCo will not incur any downtime on an upsize event.

Figure 12: Downtime incurred for elastic data-parallel vs Decoupled DiLoCo due to data transmission when upsizing a new replica. Elastic data-parallel always incurs some overhead for transmitting the most recent model copy, while the overhead for Decoupled DiLoCo is null until the time to transmit a model copy over the available bandwidth increases above 6×6\times the step time (with H=24H=24). When transmission of a single model copy takes H=24H=24 steps, Decoupled DiLoCo upsize downtime will catch up to data-parallel.

### E.4 Bandwidth profile

#### Simulated bandwidth profile

Following Douillard et al. (2025), we define “compute utilization” as the percentage of time doing computation vs doing communication during the all-reduce of the (inner/outer) gradients: Tmath/Tmath+Tcomm.\nicefrac{{T_{\text{math}}}}{{T_{\text{math}}+T_{\text{comm.}}}}. We show in Table 13 and Figure 13 the required bandwidth in Gbits/s to reach a certain level of compute utilization, across two step times (1s and 5s) and two cluster sizes (2 datacenters and 8 datacenters). Longer step time TmathT_{\text{math}} mechanically improves the compute utilization of all methods, and further allows Decoupled DiLoCo to overlap its communication over the computation. More datacenters on which to perform the ring all-reduce will increase the required bandwidth too by a factor of 2​(|DCs|−1)/|DCs|\nicefrac{{2(|\text{DCs}|-1)}}{{|\text{DCs|}}} with DCs| being the number of datacenters on the ring.

In these simulations, Decoupled DiLoCo, with either bf16 (16 bits) or int4 (4 bits) communication size, requires orders of magnitude less bandwidth than its data-parallel counterpart. This massively less need of bandwidth is a critical property for scavenging new resources (see Section 5.3) which can now exploit crumbs of compute in location where bandwidth was not properly allocated beforehand.

(a) 1s compute step time, 2 datacenters

(b) 5s compute step time, 2 datacenters

(c) 1s compute step time, 8 datacenters

(d) 5s compute step time, 8 datacenters

Figure 13: Compute utilization for the 5B parameter model across varying compute step times and datacenters over a range of bandwidth values in Gbits/s.

Method | 50% CU | 75% CU | 90% CU | 95% CU | 99% CU  
---|---|---|---|---|---  
Data-Parallel | 49 | 81 | 104 | 113 | 195  
Decoupled DiLoCo M=8 | 0.58 | 1.2 | 1.7 | 1.9 | 2.2  
Decoupled DiLoCo M=8 with int4 comm. | 0.15 | 0.29 | 0.43 | 0.49 | 0.53  
  
(a) 1s compute step time, 2 datacenters

Method | 50% CU | 75% CU | 90% CU | 95% CU | 99% CU  
---|---|---|---|---|---  
Data-Parallel | 9.5 | 16 | 21 | 23 | 49  
Decoupled DiLoCo M=8 | 0.12 | 0.23 | 0.34 | 0.39 | 0.43  
Decoupled DiLoCo M=8 with int4 comm. | 0.03 | 0.06 | 0.08 | 0.10 | 0.11  
  
(b) 5s compute step time, 2 datacenters

Method | 50% CU | 75% CU | 90% CU | 95% CU | 99% CU  
---|---|---|---|---|---  
Data-Parallel | 85 | 140 | 183 | 198 | 391  
Decoupled DiLoCo M=8 | 1.0 | 2.0 | 3.0 | 3.3 | 3.8  
Decoupled DiLoCo M=8 with int4 comm. | 0.26 | 0.51 | 0.75 | 0.84 | 0.94  
  
(c) 1s compute step time, 8 datacenters

Method | 50% CU | 75% CU | 90% CU | 95% CU | 99% CU  
---|---|---|---|---|---  
Data-Parallel | 17 | 28 | 37 | 40 | 49  
Decoupled DiLoCo M=8 | 0.21 | 0.40 | 0.59 | 0.68 | 0.75  
Decoupled DiLoCo M=8 with int4 comm. | 0.05 | 0.10 | 0.15 | 0.17 | 0.19  
  
(d) 5s compute step time, 8 datacenters

Table 13: Bandwidth requirements in Gbits/s to reach a certain level of compute utilization for the 5B parameter model under varying pure compute step times and datacenter scales.

#### Non-collocated Learners

We now perform for real a distributed experiment on a custom Chinchilla-like (Hoffmann et al., 2022) dense architecture with 12B parameters. We distribute this experiment with M=8M=8 learners across the USA, one in the Midwest, two in the South, three out West, and two in the Great Plains. We display in Figure 14 the step time, normalized by the speed of a collocated Decoupled DiLoCo experiment (e.g. all learners in the same datacenter). While Decoupled DiLoCo with non-collocated compute (in orange) is around the same speed as its counterpart with collocated compute (in blue), data-parallel with non-collocated compute (in green) is significantly slower (>10-20x), to the point of being unusable. Note that in this experiment, we didn’t make any effort to allocate more bandwidth than the base minimum available and is a somewhat contrived comparison not representative of large-scale effort where proper allocation of bandwidth would be ensured in advance. It is, however, clear evidence that Decoupled DiLoCo can enable more flexible training, for example to scavenge compute across regions.

We also show in Figure 15 a XProf (OpenXLA Contributors, 2026), representing the real XLA operations done by Decoupled DiLoCo with non-collocated compute (top) vs with collocated-compute (bottom). When compute is collocated, available bandwidth is significantly higher, and thus the syncer time (in shaded blue, at the bottom row) is smaller than with non-collocated compute. However, even in the distributed case, that synchronization time can reliably fit under an entire compute step, and be therefore fully hidden, leading to optimal compute utilization.

Figure 14: Normalized step time and step count to completion of a non-collocated data-parallel vs with Decoupled DiLoCo M=8M=8 both non-collocated and collocated. Figure 15: Visualization of the XLA operations done for Decoupled DiLoCo M=8M=8 in both non-collocated (top) and collocated (bottom) situations.

## Appendix F Evaluation Benchmarks

To provide a comprehensive assessment of our models, we evaluate performance across a diverse suite of text and vision benchmarks. The datasets used are detailed below.

### F.1 Text Benchmarks

Our text evaluation suite targets reasoning, commonsense, and reading comprehension capabilities:

  * •

ARC (Challenge and Easy): Evaluates question-answering capabilities using grade-school science questions (Clark et al., 2018).

  * •

BoolQ: Tests reading comprehension with yes/no questions derived from search queries (Clark et al., 2019).

  * •

HellaSwag: Assesses commonsense natural language inference and the ability of models to choose the most logical continuation of a text (Zellers et al., 2019).

  * •

PIQA: Focuses on physical commonsense reasoning, testing a model’s understanding of physical interactions (Bisk et al., 2020).

  * •

SIQA (SocialIQA): Measures commonsense reasoning specifically regarding social interactions and human behavior (Sap et al., 2019).

  * •

WinoGrande: An adversarial benchmark designed to test pronoun resolution and general commonsense reasoning (Sakaguchi et al., 2020).


### F.2 Vision Benchmarks

To measure multimodal understanding, we utilize the following vision-based evaluation datasets:

  * •

ChartQA: Tests visual and logical reasoning capabilities through question answering based on charts and graphs (Masry et al., 2022).

  * •

COCO-Captions: Assesses the model’s ability to generate accurate and descriptive captions for everyday images (Chen et al., 2015).

  * •

DocVQA: Evaluates visual question answering specifically on images of document pages (Mathew et al., 2021).

  * •

InfographicVQA: A benchmark focused on visual question answering based on complex infographics (Mathew et al., 2022).

  * •

MMMU: A massive multi-discipline multimodal evaluation benchmark designed to test expert-level artificial general intelligence (Yue et al., 2024).

  * •

TextVQA: Tests the ability of models to read and reason about text explicitly present within an image (Singh et al., 2019).


### F.3 Full results

We display in this section all the results shown in the main paper as figures. Find in Table 14 the table results from 7(a) across three scales.

Gemma 2B Gemma 5B Gemma 9B Benchmark DP DiLoCo DP DiLoCo DP DiLoCo Text benchmarks (avg) 52.3 51.6 62.5 62.8 67.8 68.0 Arc-Challenge 36.2 36.3 47.3 47.2 56.1 56.1 Arc-Easy 65.1 64.0 73.6 73.8 81.6 80.4 BoolQ 58.8 55.8 65.5 68.8 71.3 74.7 HellaSwag 52.1 51.4 65.8 64.9 72.1 72.0 Piqa 71.8 72.5 76.5 76.4 79.3 78.9 Siqa 46.0 46.5 48.9 46.9 49.8 50.4 Winogrande 55.6 54.1 59.6 61.3 64.2 63.8 Vision benchmarks (avg) 44.1 43.3 49.3 47.2 56.2 55.3 COCO-Captions 63.8 62.9 78.9 74.7 87.5 86.8 ChartQA 12.2 13.3 19.0 17.3 21.6 23.0 DocVQA 50.9 48.1 65.6 64.0 76.1 75.1 DocVQA-Hard 30.9 30.8 45.5 44.9 53.2 52.3 InfographicVQA 25.8 24.4 37.2 36.2 45.9 44.8 MMMU 21.1 19.3 26.2 24.8 32.9 31.3 TextVQA 62.9 64.5 72.8 68.7 76.2 73.7

Table 14: Performance comparison of data-parallel (DP) vs. Decoupled DiLoCo (M=8M=8) across model scales (2B, 5B, 9B).

### F.4 Scavenging full results

The full downstream evaluation results from the experiments in Section 5.3 are shown in Table 15(b). Additionally, whilst those experiments detail an _iso-FLOPs_ regime where total compute spent per run was fixed, Table 15(a) shows results from an alternative evaluation setting where the total number of steps is fixed instead, and scavenged compute is used to opportunistically improve downstream task performance given a set time budget. In total, 50% of the steps in each run under the _iso-step_ regime had compute increased by the scavenging factor. We observed that validation loss improved predictably at this scale when increasing the available compute during scavenging windows, and Table 15(a) shows that these gains translated to real improvements on downstream vision tasks. Text tasks did not on average appear to benefit from the increased compute, but this trend is reflected in the DP baseline, so is likely a property not of DiLoCo but of this model scale, dataset and task set more generally.

Metric | Data Parallel (DP) | DiLoCo  
---|---|---  
Scavenging | Scavenging  
+25% | +50% | +100% | +300% | +25% | +50% | +100% | +300%  
Text Benchmarks (Avg) | 60.7 | 60.9 | 60.8 | 52.3 | 61.0 | 61.3 | 61.1 | 60.8  
Arc-Chall. | 43.9 | 45.6 | 44.5 | 31.9 | 46.3 | 45.4 | 43.8 | 44.0  
Arc-Easy | 70.6 | 70.7 | 71.8 | 57.9 | 73.2 | 72.2 | 72.5 | 70.9  
BoolQ | 65.6 | 65.8 | 65.7 | 61.9 | 63.5 | 66.6 | 66.9 | 66.1  
HellaSwag | 61.3 | 61.9 | 62.5 | 46.3 | 62.0 | 62.3 | 62.7 | 61.9  
Piqa | 75.6 | 75.4 | 75.4 | 69.2 | 76.1 | 75.6 | 75.6 | 75.1  
Siqa | 48.3 | 48.0 | 47.5 | 45.0 | 48.1 | 47.6 | 47.9 | 48.0  
Winogrande | 59.4 | 59.0 | 58.2 | 54.1 | 58.0 | 59.2 | 58.5 | 59.6  
Vision Benchmarks (Avg) | 46.3 | 47.4 | 48.3 | 30.6 | 45.9 | 46.7 | 48.8 | 47.2  
MMMU | 22.7 | 22.8 | 25.2 | 11.7 | 21.8 | 25.9 | 24.2 | 25.3  
COCO-Cap. | 74.8 | 75.7 | 75.0 | 52.0 | 71.9 | 73.3 | 77.6 | 76.7  
ChartQA | 16.5 | 18.2 | 16.6 | 9.8 | 16.2 | 18.6 | 17.3 | 15.6  
DocVQA | 63.3 | 65.5 | 67.3 | 41.1 | 64.4 | 63.0 | 67.4 | 63.6  
DocVQA-Hard | 41.1 | 43.8 | 45.9 | 22.8 | 42.5 | 40.4 | 44.7 | 44.8  
InfographicVQA | 34.2 | 34.6 | 36.9 | 20.3 | 34.6 | 35.5 | 38.1 | 34.3  
TextVQA | 71.7 | 71.4 | 71.4 | 56.7 | 70.1 | 70.5 | 72.2 | 70.3  
Validation loss delta | −0.22%-0.22\% | −0.47%-0.47\% | −0.79%-0.79\% | −1.41%\mathbf{-1.41\%} | −0.17%-0.17\% | −0.42%-0.42\% | −0.78%-0.78\% | −1.30%-1.30\%  
  
(a) Iso-step scavenging performance. All runs take the same number of steps, so higher scavenging factors translate to greater FLOPs and tokens seen.

Metric | Data Parallel (DP) | DiLoCo  
---|---|---  
Scavenging | Scavenging  
+25% | +50% | +100% | +300% | +25% | +50% | +100% | +300%  
Text (Avg) | 60.5 | 60.0 | 60.1 | 60.4 | 60.8 | 60.7 | 60.9 | 60.7  
Arc-Chall. | 42.7 | 44.0 | 43.9 | 43.0 | 44.0 | 44.8 | 44.1 | 43.1  
Arc-Easy | 70.2 | 70.0 | 72.0 | 71.5 | 70.9 | 72.8 | 70.5 | 72.0  
BoolQ | 66.8 | 63.4 | 64.6 | 66.1 | 66.1 | 67.2 | 66.4 | 67.7  
HellaSwag | 60.9 | 60.7 | 61.2 | 60.2 | 61.9 | 61.0 | 61.2 | 60.9  
Piqa | 75.1 | 75.8 | 75.7 | 75.2 | 75.1 | 75.7 | 75.1 | 74.4  
Siqa | 48.6 | 47.5 | 46.9 | 48.1 | 48.0 | 48.2 | 47.5 | 48.2  
Winogrande | 59.1 | 58.6 | 56.6 | 58.6 | 59.6 | 59.1 | 60.1 | 58.4  
Vision (Avg) | 46.6 | 45.9 | 47.0 | 46.4 | 47.2 | 45.2 | 46.4 | 46.1  
MMMU | 23.8 | 22.2 | 25.3 | 25.4 | 25.3 | 25.3 | 22.8 | 20.3  
COCO-Cap. | 76.6 | 73.4 | 71.8 | 71.7 | 76.7 | 74.4 | 71.6 | 74.6  
ChartQA | 14.8 | 16.0 | 16.8 | 17.3 | 15.6 | 16.2 | 16.3 | 14.1  
DocVQA | 62.9 | 64.1 | 66.1 | 64.2 | 63.6 | 60.0 | 65.2 | 66.8  
DocVQA-Hard | 42.4 | 42.6 | 43.8 | 43.7 | 44.8 | 37.9 | 44.5 | 42.3  
InfographicVQA | 35.9 | 33.7 | 35.2 | 33.4 | 34.3 | 32.6 | 33.6 | 34.6  
TextVQA | 69.8 | 69.5 | 70.1 | 69.1 | 70.3 | 70.1 | 70.4 | 70.1  
Validation loss delta | +0.02%+0.02\% | −0.04%\mathbf{-0.04\%} | −0.02%-0.02\% | +0.07%+0.07\% | +0.09%+0.09\% | −0.01%-0.01\% | +0.08%+0.08\% | +0.23%+0.23\%  
  
(b) Iso-FLOPs scavenging performance. All runs share identical total tokens and theoretical FLOPs.

Table 15: Scavenging experiment downstream performance in iso-steps and iso-FLOPs regimes.

Experimental support, please [view the build logs](./2604.21428v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

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
