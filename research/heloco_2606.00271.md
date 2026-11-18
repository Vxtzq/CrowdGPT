Skip to main content

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

[](https://arxiv.org/IgnoreMe) [ ](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2606.00271v1 "Back to abstract page") [ Download PDF ](/pdf/2606.00271v1 "Download PDF") Contents Reading mode Colour theme

  1. Abstract
  2. 1 Introduction
  3. 2 Background and Related Work
     1. Low-communication training for language models.
     2. Asynchrony and stale-update correction.
     3. Heterogeneity, non-IID data, and directional conflict.
  4. 3 Methodology
     1. Problem setup.
     2. Momentum-guided worker initialization.
     3. Measuring update alignment.
     4. Correcting stale directions.
     5. Outer-model update.
  5. 4 Experiments and Analyses
     1. Setup.
     2. 4.1 Convergence Under Heterogeneity
        1. Non-IID data domain analysis.
     3. 4.2 Effect of worker staleness in non-IID setting
  6. 5 Conclusion
     1. Limitations and future work.
  7. References
  8. A Appendix
     1. A.1 Notation
     2. A.2 Theoretical Justification of the HeLoCo Correction
        1. Lemma.
        2. Proof.
        3. Interpretation.
     3. A.3 HeLoCo Algorithm
     4. A.4 Implementation and Compute Details
     5. A.5 Training Hyperparameters
     6. A.6 Deeper Insights into Non-IID Training


[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2606.00271v1 [cs.DC] 29 May 2026

#  HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity

Abdullah Al Asif1 Patrick Diem2 Juan Pablo Muñoz3 Felix Wolf2 Ali Jannesari1 Arya Mazaheri2,4   
1Iowa State University   
2Technical University of Darmstadt   
3Maro Systems   
4PanocularAI 

###### Abstract

Distributed Low-Communication (DiLoCo) training reduces communication overhead by allowing workers to perform multiple local optimization steps before sending pseudo-gradients to a global outer update. Its asynchronous variant further improves hardware utilization by removing synchronization barriers, but at the cost of stale pseudo-gradients computed from outdated model states. As a result, these updates can become misaligned with the current global optimization direction, particularly in heterogeneous systems. This issue becomes even more pronounced when data are non-IID, a setting that has not been well studied in asynchronous low-communication training. To address this limitation, we propose HeLoCo, a direction-aware correction method for asynchronous low-communication training that uses outer momentum as a reference for the current optimization trajectory and selectively adjusts incoming pseudo-gradients before the outer update. Updates that remain aligned are preserved, while directionally conflicting components are corrected. On multilingual language-model training with heterogeneous workers and non-IID data, HeLoCo consistently improves validation loss. It outperforms existing asynchronous DiLoCo-based baselines by up to 7.5% at a fixed token budget, exceeds asynchronous momentum look-ahead by up to 3.3% at a fixed wall-clock budget, and surpasses the synchronous baseline by up to 22.1% under severe system heterogeneity. Our analysis further shows how staleness, worker speed, and data heterogeneity shape update quality and convergence in highly decentralized and heterogeneous training setups.

## 1 Introduction

Training large language models (LLMs) typically relies on tightly coupled, high-performance clusters with high-bandwidth, low-latency interconnects. While effective within a single data center, this setup becomes challenging when compute resources are geographically distributed or heterogeneous. In such cases, communication slows down and synchronization can take up a lot of time (Douillard et al., 2023; Douillard et al., 2025). When multiple workers train together, some may end up waiting for others due to slow connections or imbalances in workload. To solve this issue, the Distributed Low Communication (DiLoCo) algorithm lets worker nodes perform several updates locally before sending their updates, called pseudo-gradients (Douillard et al., 2023). By decoupling local and global updates, DiLoCo reduces communication requirements, while preserving training efficiency, making it particularly well suited to loosely connected clusters.(Douillard et al., 2023; Jaghouar et al., 2024; Douillard et al., 2025).

Synchronous DiLoCo requires all workers to send their updates before starting the next global step. When workers operate at different speeds or are distributed across regions, these differences can introduce stragglers, leading to delays and idle time. Switching to asynchronous variants allows each worker’s updates to be applied immediately. This improves both hardware efficiency and overall performance. However, a key challenge in this setting is gradient staleness, meaning that some gradient estimates are based on outdated model versions, which introduces inconsistency when they are consumed by the outer optimizer. Prior work (Liu et al., 2024) shows that this inconsistency can degrade convergence. As a result, although the asynchronous variant of DiLoCo permits more frequent updates than its synchronous counterpart, it can potentially yield worse optimization performance. Figure 1 illustrates the gap in synchronization between fast and slow workers. Fast workers contribute updates based on the most recent global model, ensuring high relevance. In contrast, slow workers (often called "stragglers") submit gradients derived from outdated model parameters. Because the global optimizer has already progressed, these "stale" updates can be counterproductive or even lead to training instability.

Figure 1: Staleness asymmetry in asynchronous DiLoCo with heterogeneous workers.

Existing correction methods only partially address the problem. Traditional delayed-update methods apply an exponential discount to updates based on their age or delay compensation (Zinkevich et al., 2009; Zheng et al., 2017). Recent asynchronous DiLoCo schemes introduce corrections using delayed Nesterov-like and momentum lookahead techniques (Liu et al., 2024; Ajanthan et al., 2025). However, these methods only apply corrections to the entire pseudo-gradient, which proves to be too coarse for heterogeneous systems and non-IID (Independent and Identically Distributed) dataset distributions. An update from a slower worker is not only more delayed but may also be less compatible with the current distribution utilized by the global trajectory (Li et al., 2020). These two effects do not necessarily impact all parameter blocks of the pseudo-gradient equally; some may align with the current direction while others may contradict it, potentially hindering the convergence of the outer method.

To address this issue, we propose HeLoCo (Heterogeneity-aware Low-Communication training), a direction-aware correction algorithm designed for asynchronous low-communication training. The core idea is to use outer momentum as a reference for determining the trustworthiness of each parameter block, where each block is an individual model tensor, such as a weight matrix. This approach allows processing of various stale gradient directions in alignment with the current outer update process. We further include a lightweight momentum-based (Zhang et al., 2019) initialization step to improve directional consistency.

Our contributions are as follows:

  * •

We identify the directional mismatch as a key failure mode in asynchronous DiLoCo: under system heterogeneity and non-IID data, individual tensor blocks within a stale pseudo-gradient can conflict with the outer update direction.

  * •

We propose HeLoCo, a momentum-guided correction method that initializes workers with a look-ahead model and selectively keeps or adjusts the pseudo-gradient before the outer update.

  * •

Empirically, our method reduces validation loss by up to 7.5% over asynchronous Nesterov and 3.3% over asynchronous MLA at a fixed token budget, and by 22.1% over synchronous Nesterov under both system and data heterogeneity.


## 2 Background and Related Work

Stale pseudo-gradients in asynchronous distributed training have been studied from the perspective of delay compensation and momentum correction, but the question of whether different parts of the same update deserve different treatment has received little attention. We review the three lines of work most relevant to this gap.

#### Low-communication training for language models.

DiLoCo showed that separating local SGD steps from a global outer optimizer dramatically cuts synchronization frequency without sacrificing model quality (Douillard et al., 2023). OpenDiLoCo confirmed this recipe holds even across continents (Jaghouar et al., 2024). Follow-up work pushed further: Streaming DiLoCo overlaps communication with computation to reduce peak bandwidth (Douillard et al., 2025), DiLoCoX scales to larger models via parallelism and compression (Qi et al., 2025), and NoLoCo (Kolehmainen et al., 2025) explores dropping global synchronization altogether. These works share a common focus on _when_ and _how often_ workers communicate. We ask a different question: given that some updates will inevitably arrive late, can we do better than applying them as-is?

#### Asynchrony and stale-update correction.

Stale gradients are as old as asynchronous optimization itself. Early work showed they can be tolerated in sparse settings (Recht et al., 2011), and parameter-server systems exposed their broader effects (Zinkevich et al., 2009; Dean et al., 2012). A subtler insight came later: asynchronous execution does not just add noise, it implicitly introduces a momentum-like bias (Mitliagkas et al., 2016). This makes the outer momentum a natural reference for diagnosing staleness. Taylor-style compensation was one early attempt to exploit this (Zheng et al., 2017). More recently, MLA (Ajanthan et al., 2025) corrects staleness by extrapolating the negative momentum direction uniformly across the entire pseudo-gradient. We argue, however, that tensor blocks within the same stale update can have fundamentally different alignment with the outer trajectory, and a single uniform correction is too coarse to handle this.

#### Heterogeneity, non-IID data, and directional conflict.

The staleness problem is further compounded when workers operate at different speeds and train on non-IID data — conditions that are unavoidable in globally distributed training. In such heterogeneity, stale updates do not just arrive late; they carry gradients shaped by a different data distribution and a different point in the loss landscape (McMahan et al., 2017). This directional distortion has been well documented in federated learning, where proximal regularization (Li et al., 2020), control variates (Karimireddy et al., 2020), and normalized averaging (Wang et al., 2020) were each designed to counter it. In multi-task optimization, PCGrad made a related observation: gradient conflicts are better resolved geometrically, block by block, than through a single global correction (Yu et al., 2020). Our method brings this geometric intuition into asynchronous DiLoCo, using outer momentum as a reference to selectively preserve, dampen, or reorient each tensor block of a stale update — depending on how well it agrees with the direction the outer optimizer is already moving.

## 3 Methodology

Section 1 highlights the shortcomings of asynchronous training with low communication. The local update might be based on an old version of the global model, leading to issues in scenarios where there are heterogeneous worker update speeds and non-IID data. Despite this, delayed pseudo-gradients preserve some of the local gradient information; however, some directions in the tensor may not correspond with the latest global model update directions. In order to solve this issue, HeLoCo employs two strategies: one for initialization, allowing workers to start with a look-ahead model state, and another for alignment, ensuring returned gradients properly align with the current update direction.

#### Problem setup.

We consider distributed training over KK workers. Worker ii samples data from a local distribution 𝒟i\mathcal{D}_{i}, and the target objective is:

| minθ⁡F⁡(θ)=∑i=1Kpi​Fi​(θ),Fi​(θ)=𝔼z∼𝒟i​[ℓ⁡(θ,z)],\min_{\theta}F(\theta)=\sum_{i=1}^{K}p_{i}F_{i}(\theta),\qquad F_{i}(\theta)=\mathbb{E}_{z\sim\mathcal{D}_{i}}[\ell(\theta;z)], |  | (1)  
---|---|---|---  
  
where θ\theta denotes the model parameters, pi≥0p_{i}\geq 0, ∑ipi=1\sum_{i}p_{i}=1, and ℓ⁡(θ,z)\ell(\theta;z) is the training loss. The weights pip_{i} define the intended contribution of each local objective to the global objective. The local distributions 𝒟i\mathcal{D}_{i} may be non-IID, and workers may have different computation or communication speeds. By contrast, in asynchronous training, update arrivals are driven by worker speed, so faster workers can influence the outer model more frequently than slower workers.

At outer step tt, the global model maintains parameters θt\theta_{t} and an outer momentum buffer mtm_{t}. Here, _outer_ refers to the global DiLoCo-level optimization step performed when a worker returns a pseudo-gradient, while _inner_ refers to the local optimization steps performed independently on each worker. When worker ii becomes available at outer step sis_{i}, the synchronizer initializes it with an outer model state θ¯si\bar{\theta}_{s_{i}}. The worker initializes its local model from this state and performs HH inner optimization steps:

| θi,0(si)=θ¯si,θi,h+1(si)=θi,h(si)−αhgi(θi,h(si)),h=0,…,H−1,\theta_{i,0}^{(s_{i})}=\bar{\theta}_{s_{i}},\qquad\theta_{i,h+1}^{(s_{i})}=\theta_{i,h}^{(s_{i})}-\alpha_{h}g_{i}(\theta_{i,h}^{(s_{i})}),\quad h=0,\ldots,H-1, |  | (2)  
---|---|---|---  
  
where αh\alpha_{h} is the inner learning rate and gi​(⋅)g_{i}(\cdot) is a stochastic gradient on worker ii’s local data. After local training, the worker returns the pseudo-gradient:

| Δi(si)=θ¯si−θi,H(si).\Delta_{i}^{(s_{i})}=\bar{\theta}_{s_{i}}-\theta_{i,H}^{(s_{i})}. |  | (3)  
---|---|---|---  
  
This sign convention makes Δi(si)\Delta_{i}^{(s_{i})} the descent displacement accumulated by local training, so the synchronizer applies it through a negative outer step.

If the update arrives at outer step tt, its staleness is defined as:

| τi​(t)=t−si.\tau_{i}(t)=t-s_{i}. |  | (4)  
---|---|---|---  
  
Thus, Δi(si)\Delta_{i}^{(s_{i})} was computed from the initial model θ¯si\bar{\theta}_{s_{i}}, while the outer model now holds θt\theta_{t}. This delay is the source of the stale-update error that HeLoCo is designed to reduce.

#### Momentum-guided worker initialization.

The first use of outer momentum occurs before local training begins. Instead of initializing a worker with the current outer model θr\theta_{r} at outer step rr, HeLoCo initializes it with a look-ahead model:

| θ¯r=θr−ηr​μ​mr,\bar{\theta}_{r}=\theta_{r}-\eta_{r}\mu m_{r}, |  | (5)  
---|---|---|---  
  
where ηr\eta_{r} is the outer learning rate at the time of initialization and μ\mu is the outer momentum coefficient. This model is a one-step prediction of where the outer model is moving under its current momentum. Since the worker will return after some delay, starting local training from θ¯r\bar{\theta}_{r} reduces the positional gap between the worker’s initial model and the outer model state at arrival.

For a worker initialized at step sis_{i}, the returned pseudo-gradient is therefore:

| Δi(si)=(θsi−ηsi​μ​msi)−θi,H(si).\Delta_{i}^{(s_{i})}=\left(\theta_{s_{i}}-\eta_{s_{i}}\mu m_{s_{i}}\right)-\theta_{i,H}^{(s_{i})}. |  | (6)  
---|---|---|---  
  
This look-ahead initialization reduces the mismatch at the beginning of local training. However, it cannot guarantee that the returned pseudo-gradient will remain directionally useful after several outer updates. This motivates a second decision when the pseudo-gradient returns: which tensor blocks should be kept unchanged, and which should be corrected before the outer update?

#### Measuring update alignment.

When Δi(si)\Delta_{i}^{(s_{i})} arrives at outer step tt, HeLoCo compares each tensor block of the update with the corresponding tensor block of the current outer models’ momentum mtm_{t}. For readability, we write Δb\Delta_{b} for a block of the arriving pseudo-gradient and (mt)b(m_{t})_{b} for the corresponding momentum block. All norms and inner products are computed after vectorizing each tensor block; ∥⋅∥\|\cdot\| denotes the Euclidean norm, equivalently the Frobenius norm for tensor-shaped parameters.

If both blocks have non-negligible norm, define:

| u^b=Δb‖Δb‖,v^b=(mt)b‖(mt)b‖.\hat{u}_{b}=\frac{\Delta_{b}}{\|\Delta_{b}\|},\qquad\hat{v}_{b}=\frac{(m_{t})_{b}}{\|(m_{t})_{b}\|}. |  | (7)  
---|---|---|---  
  
The directional compatibility score is defined as:

| cb=u^b⊤​v^b.c_{b}=\hat{u}_{b}^{\top}\hat{v}_{b}. |  | (8)  
---|---|---|---  
  
A positive value indicates that the stale tensor still agrees with the current global model trajectory. A negative value indicates that the tensor contains a component opposing the momentum direction. A value near zero indicates that the tensor is not directly conflicting but is only weakly supported by the current trajectory.

This tensor-wise view is the key design choice in HeLoCo. A stale pseudo-gradient is not treated as uniformly good or uniformly harmful. Within the same worker update, some tensors may remain aligned with the current global models’ trajectory, while others may conflict with it. A single scalar penalty on the entire pseudo-gradient cannot make this distinction.

#### Correcting stale directions.

HeLoCo applies a geometric correction to each tensor block. If either ‖Δb‖\|\Delta_{b}\| or ‖(mt)b‖\|(m_{t})_{b}\| is below the numerical threshold ε\varepsilon, the momentum direction is unreliable for that block and the block is passed through unchanged. Let cokc_{\mathrm{ok}} denote the threshold for sufficient alignment. If the block is sufficiently aligned, cb≥cokc_{b}\geq c_{\mathrm{ok}}, it is also preserved:

| Δ^b=Δb.\widehat{\Delta}_{b}=\Delta_{b}. |  | (9)  
---|---|---|---  
  
This leaves unchanged the tensor blocks that already agree with the current outer update direction.

If cb<0c_{b}<0, the block contains an anti-momentum component. HeLoCo reduces only that conflicting component rather than discarding the entire tensor:

| Δ^b=Δb−βb​cb​‖Δb‖​v^b,\widehat{\Delta}_{b}=\Delta_{b}-\beta_{b}c_{b}\|\Delta_{b}\|\hat{v}_{b}, |  | (10)  
---|---|---|---  
  
where

| βb=min⁡{ks​(−cb)​confb,βmax}.\beta_{b}=\min\left\\{k_{s}(-c_{b})\mathrm{conf}_{b},\,\beta_{\max}\right\\}. |  | (11)  
---|---|---|---  
  
Here ks>0k_{s}>0 controls the shrinkage strength and βmax∈(0,1]\beta_{\max}\in(0,1] prevents overcorrection. Since cb<0c_{b}<0, the correction reduces the component that points against the current outer update direction. Geometrically, this can be interpreted as a partial projection-based correction along the normalized momentum direction v^b\hat{v}_{b}. The incoming block Δb\Delta_{b} can be decomposed into a parallel component (Δb⊤​v^b)​v^b(\Delta_{b}^{\top}\hat{v}_{b})\hat{v}_{b} and an orthogonal residual Δb−(Δb⊤​v^b)​v^b\Delta_{b}-(\Delta_{b}^{\top}\hat{v}_{b})\hat{v}_{b}. HeLoCo attenuates only the anti-momentum part of the parallel component, while leaving the orthogonal residual unchanged so that local information not directly conflicting with the current trajectory is retained.

For weakly aligned blocks, 0≤cb<cok0\leq c_{b}<c_{\mathrm{ok}}, HeLoCo does not remove a component. Instead, it gently reorients the block toward the current momentum direction while preserving its magnitude:

| u~b\displaystyle\tilde{u}_{b} | =(1−λb)​u^b+λb​v^b,\displaystyle=(1-\lambda_{b})\hat{u}_{b}+\lambda_{b}\hat{v}_{b}, |  | (12)  
---|---|---|---|---  
| Δ^b\displaystyle\widehat{\Delta}_{b} | =‖Δb‖​u~bmax⁡{‖u~b‖,ε},\displaystyle=\|\Delta_{b}\|\frac{\tilde{u}_{b}}{\max\\{\|\tilde{u}_{b}\|,\varepsilon\\}}, |  | (13)  
  
with

| λb=min⁡{kd​(1−cb)​confb, 1}.\lambda_{b}=\min\left\\{k_{d}(1-c_{b})\mathrm{conf}_{b},\,1\right\\}. |  | (14)  
---|---|---|---  
  
The coefficient kd>0k_{d}>0 controls the reorientation strength. The less aligned the block is, the more strongly it is pulled toward the momentum direction. In the non-degenerate case where ‖u~b‖≥ε\|\tilde{u}_{b}\|\geq\varepsilon, this operation preserves ‖Δ^b‖=‖Δb‖\|\widehat{\Delta}_{b}\|=\|\Delta_{b}\| and changes only the direction of the tensor.

The confidence factor used in both correction cases is defined as:

| confb=‖Δb‖‖Δb‖+κ​‖(mt)b‖+ε,\mathrm{conf}_{b}=\frac{\|\Delta_{b}\|}{\|\Delta_{b}\|+\kappa\|(m_{t})_{b}\|+\varepsilon}, |  | (15)  
---|---|---|---  
  
where κ>0\kappa>0 controls the relative scale of the momentum norm. This factor makes the correction conservative when the arriving block is small relative to the accumulated momentum and stronger when the arriving block is large.

After all blocks are processed, the corrected pseudo-gradient is reassembled as:

| Δ^={Δ^b}b.\widehat{\Delta}=\\{\widehat{\Delta}_{b}\\}_{b}. |  | (16)  
---|---|---|---  
  
Appendix A.2 provides a theoretical justification for the tensor-wise correction used by HeLoCo.

#### Outer-model update.

The outer update applies the corrected pseudo-gradient immediately. If the asynchronous training rule uses a worker or delay weight, which we denote by ρt\rho_{t}; otherwise, ρt=1\rho_{t}=1. The weighted corrected update is:

| Gt=ρt​Δ^.G_{t}=\rho_{t}\widehat{\Delta}. |  | (17)  
---|---|---|---  
  
The global model then updates its momentum and parameters as

| mt+1\displaystyle m_{t+1} | =μ​mt+(1−μ)​Gt,\displaystyle=\mu m_{t}+(1-\mu)G_{t}, |  | (18)  
---|---|---|---|---  
| θt+1\displaystyle\theta_{t+1} | =θt−ηt​(Gt+μ​mt+1).\displaystyle=\theta_{t}-\eta_{t}\left(G_{t}+\mu m_{t+1}\right). |  | (19)  
  
The updated momentum mt+1m_{t+1} is used in the parameter step, allowing the arriving pseudo-gradient to first update the outer direction estimate and then contribute to the model update.

The HeLoCo communication method works the same way as DiLoCo. Each worker processes a single model, updates it locally, and creates a single pseudo-gradient vector. However, the way they prepare for data exchange is different. Before local training starts, the synchronizer sets up the look-ahead model. After this, to prevent incorrect updates, the synchronizer modifies the gradients using momentum before applying the outer update procedure. This adjustment only needs to access the existing momentum buffer and requires an extra pass through the gradient vector, adding a cost of O⁡(d)O(d) per arrival. Appendix A.1 summarizes the notation used in this section, and Appendix A.3 provides the HeLoCo pseudocode.

## 4 Experiments and Analyses

#### Setup.

We evaluate HeLoCo using TinyGPT (Radford et al., 2019) on the multilingual mC4 (Raffel et al., 2020) dataset. Across experiments, we compare Nesterov, MLA, and HeLoCo under different worker-speed and data heterogeneity settings. Appendix A.4 provides implementation, hardware, and runtime details.

### 4.1 Convergence Under Heterogeneity

For the convergence experiments, we compare methods based on two key factors: how fast the workers are (either all the same speed or different speeds) and how the data is organized (either similar or different types). In the setting with different speeds, five workers take between 0.74 and 7.50 seconds for each step. This leads to an uneven update process: the fastest worker provides 58.2% of all updates and has an average delay of 0.72. Meanwhile, the two slowest workers together account for only 12% of the updates, with delays of 14.38 and 16.50, respectively. We also look at Dynamic Local Updates (DyLU) (Wang and Joshi, 2019), which helps balance how often workers provide updates by changing the number of local steps they take. With DyLU, contributions from workers become almost equal (19-22%), and the average staleness is approximately τ¯≈4\bar{\tau}\approx 4. Crucially, DyLU is orthogonal to HeLoCo, as DyLU balances worker participation by changing the local-update schedule, while HeLoCo corrects returned pseudo-gradients before the outer update. Figure 2 compares training loss across IID and non-IID data settings under both heterogeneous and homogeneous worker configurations. When we use IID data on heterogeneous device 2(a), HeLoCo outperforms the other methods after about 6000 steps, achieving a final loss of 7.35. In comparison, MLA has a final loss of 7.41, and Nesterov has 7.89. Since the data are IID, this improvement mainly comes from the time delays introduced by workers with different speeds. In the homogeneous IID setting 2(b), the difference between HeLoCo and MLA decreases, as expected. When workers move at similar speeds, and delays are minimal, most parts of the data align well with the recent global model updates and remain unchanged during corrections.

The non-IID results highlight the importance of correcting finer-grained components when both statistical heterogeneity and system heterogeneity are present. In the heterogeneous non-IID setting, HeLoCo shows a clear advantage, with a final loss of 7.91, compared to 7.96 for MLA and 8.29 for Nesterov. In this scenario, outdated gradients often mix different levels of alignment, so correcting at the global update level is not precise enough. In the homogeneous, non-IID setting, our method again outperforms MLA and Nesterov. This shows that having non-IID data alone can cause enough differences to make per-block corrections beneficial, even with low delays. The DyLU variants also provide a helpful comparison. DyLU lowers staleness variance by equalizing how often updates occur, but it also reduces local training time for some workers. In both IID and non-IID settings, DyLU variants do not consistently perform better than the corresponding non-DyLU HeLoCo runs. This suggests that, in our case, keeping local optimization strong and fixing stale updates at the outer optimizer is more effective than cutting local training time to reduce staleness.

Figure 2: Validation loss over training steps on C4 under IID and non-IID data distributions with heterogeneous and homogeneous worker speeds. DyLU variants are shown only for heterogeneous settings.

#### Non-IID data domain analysis.

Figure 3: Evaluation loss comparison across five languages (de, en, es, fr, it) on multilingual C4 under non-IID training with five workers.

Figure 3 shows the results for different languages, where each language represents a distinct data domain with its own distribution, and how HeLoCo improves with staleness. Each language has a fixed-speed worker, keeping staleness the same during training. English dataset, trained by the most active worker (58.2% of training updates, average staleness 0.72), shows only a slight gain over MLA (7.91 vs. 7.93). This happens because most parameter blocks are aligned with the global model and need little correction. German and French, handled by slower workers (average staleness of 14.38 and 5.12, respectively), show larger improvements of 0.13 and 0.24, respectively. In these cases, stale gradients cause misalignment, and HeLoCo’s correction focuses only on the misaligned parts, preserving useful information rather than discarding everything. The Async-DyLU row confirms this: when DyLU forces all workers to similar staleness (τ¯≈4\bar{\tau}\approx 4), our approach still outperforms MLA, although by a smaller margin. This shows that per-block correction helps even when the staleness is the same. The synchronous row serves as a staleness-free baseline, where HeLoCo still outperforms MLA due to differences in the data. The slowest workers (contributing only 6.4% and 5.6% of total training updates) show fewer benefits from correction. This is expected, since infrequent updates mean the real issue shifts from misalignment to insufficient updates. These findings suggest that this method works best under moderate staleness: enough misalignment to benefit from correction, but enough updates for those corrections to matter.

### 4.2 Effect of worker staleness in non-IID setting

Building on Section 4.1, which evaluates a single fixed worker pace in the heterogeneous setting, this section examines how varying worker pace configurations affects optimization and loss dynamics under non-IID data distributions. Specifically, we analyze cases c) and d) of Figure 2 in greater depth to demonstrate HeLoCo’s robustness across a spectrum of heterogeneous worker configurations. We compare HeLoCo against three baselines: (i) synchronous DiLoCo with Nesterov, (ii) asynchronous DiLoCo with Nesterov, and (iii) asynchronous DiLoCo with MLA. Following the findings of Section 4.1, we exclude DyLu variants as they yield no performance gain in any configuration, and use synchronous Nesterov as the primary reference to assess HeLoCo and MLA in terms of loss progression over both steps and wall-clock time.

Each worker is assigned a speed drawn from {1,2,6,15}\\{1,2,6,15\\} (seconds per step). We construct scenarios with progressively increasing staleness, culminating in the extreme configuration (1,15,15,15,15)(1,15,15,15,15), in which one fast worker and four highly stale workers operate concurrently. Training runs for 300 outer steps of 80 inner steps each (24k total steps). Hyperparameters are listed in Appendix A.5 and match those from (Liu et al., 2024).

Figure 4: Validation loss over steps (left) and wall-clock time (right) for different worker pace configurations in the non-IID setting. Worker-pace colors indicate staleness level: green for low staleness, yellow for moderate staleness, and red for high staleness.

Figure 4 and Table 1 (left) show training loss over a fixed token budget across staleness configurations. HeLoCo consistently achieves lower loss than MLA in most settings. In the homogeneous case (1,1,1,1,1)(1,1,1,1,1) and several other low-staleness configurations, HeLoCo also outperforms synchronous Nesterov, indicating that tensor-wise correction is beneficial even in the absence of staleness. As staleness grows, the performance gap between methods widens. Asynchronous Nesterov exhibits unstable behavior and diverges across all configurations. In high-staleness regimes, synchronous Nesterov surpasses both MLA and asynchronous Nesterov in token-budget loss; however, HeLoCo remains consistently more robust, achieving lower loss than MLA across nearly all configurations. The maximum token-budget improvement of HeLoCo over MLA is 3.3%. In extreme cases such as (1,15,15,15,15)(1,15,15,15,15), synchronous Nesterov achieves substantially lower loss than both asynchronous methods at a fixed token budget.

This picture changes when runtime is taken into account. Due to synchronization barriers, the wall-clock time of synchronous training is bottlenecked by the slowest worker. As a result, asynchronous variants exhibit an advantage in loss-vs.-time performance across most configurations, as shown in Figure 4 and Table 1 (right). HeLoCo achieves the best trade-off between convergence speed and loss in all but two configurations; in the extreme staleness setting (1,15,15,15,15)(1,15,15,15,15), it falls short of the synchronous baseline. At its best, HeLoCo’s time-budget loss is 22.1% lower than that of synchronous Nesterov. While HeLoCo typically improves over MLA, the margin is configuration-dependent: in moderate-staleness regimes the gains are consistent, whereas in the most extreme staleness cases a slight degradation is observed, suggesting that adaptive mechanisms for managing highly stale updates remain an open direction.

Table 1: Validation loss for a fixed token budget (left) and a fixed time budget (right) across worker pace configurations. Left: cell color indicates the best (green) and worst (red) method per row. Right: color indicates HeLoCo’s improvement (green) or degradation (red) relative to each baseline at matched wall-clock time. _Legend:_ L-HeLoCo: HeLoCo loss; L-AMLA: async MLA loss; L-AN: async Nesterov loss; L-SN: sync Nesterov loss; Δ​X\Delta X: relative improvement of HeLoCo over XX at 24k steps; TΔ​X\Delta X: relative improvement of HeLoCo over XX at time TT. Worker pace | L-HeLoCo | L-AMLA | L-AN | L-SN | Δ\DeltaAMLA | Δ\DeltaAN | Δ\DeltaSN | T | TΔ\DeltaAMLA | TΔ\DeltaAN | TΔ\DeltaSN  
---|---|---|---|---|---|---|---|---|---|---|---  
|  |  |  |  | (%) | (%) | (%) | (s) | (%) | (%) | (%)  
(1,6,6,6,6) | 6.98 | 7.22 | 7.20 | 6.62 | 3.33 | 3.04 | -5.43 | 14400 | 3.33 | 3.04 | 5.80  
(1,2,2,2,2) | 6.54 | 6.72 | 7.04 | 6.62 | 2.62 | 7.08 | 1.22 | 8000 | 2.62 | 7.08 | 3.04  
(1,1,6,6,6) | 6.69 | 6.85 | 7.29 | 6.62 | 2.45 | 8.23 | -5.48 | 9600 | 0.92 | 3.05 | 4.68  
(1,1,1,6,6) | 6.66 | 6.80 | 7.15 | 6.62 | 2.03 | 6.75 | -0.62 | 7200 | 2.03 | 6.75 | 13.04  
(1,1,2,2,2) | 6.64 | 6.78 | 7.07 | 6.62 | 1.99 | 6.00 | -0.34 | 6880 | 1.99 | 6.00 | 3.13  
(1,1,1,1,1) | 6.50 | 6.63 | 7.03 | 6.62 | 1.84 | 7.50 | 1.78 | 4800 | 1.84 | 7.50 | 1.78  
(1,15,15,15,15) | 7.64 | 7.78 | 7.81 | 6.62 | 1.75 | 2.15 | -15.39 | 19200 | 1.75 | 2.15 | -0.26  
(1,1,1,2,2) | 6.64 | 6.74 | 7.06 | 6.62 | 1.48 | 5.98 | -0.22 | 6080 | 1.48 | 5.98 | 10.46  
(1,1,1,1,6) | 6.60 | 6.69 | 7.10 | 6.62 | 1.36 | 7.10 | 0.40 | 5760 | 1.36 | 7.10 | 15.78  
(1,1,1,1,15) | 6.66 | 6.75 | 7.13 | 6.62 | 1.30 | 6.67 | -0.54 | 5920 | 1.30 | 6.67 | 22.07  
(1,1,1,1,2) | 6.61 | 6.69 | 7.05 | 6.62 | 1.23 | 6.22 | 0.16 | 5360 | 1.23 | 6.22 | 10.80  
(1,1,1,15,15) | 6.84 | 6.89 | 7.21 | 6.62 | 0.75 | 5.22 | -3.22 | 7680 | 0.75 | 5.22 | 18.85  
(1,1,15,15,15) | 7.36 | 7.29 | 7.47 | 6.62 | -1.00 | 1.42 | -11.17 | 10960 | -1.00 | 1.42 | 7.81  
  
## 5 Conclusion

We presented HeLoCo, a correction method for asynchronous low-communication training that targets the core failure mode of gradient staleness: directional mismatch between stale pseudo-gradients and the current outer optimization trajectory. Rather than applying a single global correction, HeLoCo operates at the tensor-block level, using outer momentum as a trust reference to selectively preserve, attenuate, or interpolate each arriving update. The result is a method that remains aligned with the outer optimizer under both system and data heterogeneity, without discarding gradient information unnecessarily. Our experiments confirm that staleness is not a uniform problem: the benefit of correction scales with per-worker staleness and is most pronounced when system heterogeneity and non-IID data compound. In this regime, HeLoCo consistently outperforms asynchronous baselines at both fixed token and wall-clock time budgets, and often matches or exceeds synchronous training without its synchronization overhead. These results suggest a broader principle for asynchronous low-communicatoin distributed training, in which reducing communication frequency is not enough and the direction of each arriving update must also be actively managed.

#### Limitations and future work.

We conducted the evaluation using five workers and 15M-parameter models on subsets of C4 dataset. This controlled setting allows us to study HeLoCo under varying heterogeneous conditions, but it does not yet capture the scale of practical low-communication LLM training. Future work includes testing HeLoCo with more workers, larger models, longer training periods, and more diverse datasets. An interesting next step is to combine HeLoCo with decoupled DiLoCo-style systems (Douillard et al., 2026) to explore larger and more diverse deployments. Moreover, our results indicate that extreme staleness is still challenging asynchronous settings. As mentioned in Appendix A.6, highly stale pseudo-gradients may become unreliable, which suggests that discarding them may sometimes be preferable to correcting them. One promising idea is to merge HeLoCo with adaptive staleness-aware filtering, where we adjust, down-weight, or discard a returned pseudo-gradient based on how stale it is and its alignment statistics.

### Acknowledgements

We gratefully acknowledge the support from Germany’s Federal Ministry of Breakthrough Innovation (SPRIN-D) through the Composite Learning Challenge under the SymphonyLearn project, and the hessian.AI Service Center (funded by the Federal Ministry of Research, Technology and Space, BMFTR, grant no. 16IS22091), the hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and Innovation, grant no. S-DIW04/0013/003). We would also like to thank the National Science Foundation (NSF) for supporting this project under grant 2211982 and 2426580. This work utilized the Delta system at the National Center for Supercomputing Applications (NCSA) through allocation CIS240626. We also acknowledge support from the National Science Foundation under grants 2138259, 2138286, 2138307, 2137603, and 2138296.

## References

  * Ajanthan et al. (2025) T. Ajanthan, S. Ramasinghe, G. Avraham, Y. Zuo, and A. Long Momentum look-ahead for asynchronous distributed low-communication training.  In ICLR 2025 Workshop on Modularity for Collaborative, Decentralized, and Continual Deep Learning,  Cited by: §1, §2. 
  * Dean et al. (2012) J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, A. Senior, P. Tucker, K. Yang, Q. V. Le, et al. Large scale distributed deep networks.  In Advances in Neural Information Processing Systems,  Vol. 25.  Cited by: §2. 
  * Douillard et al. (2025) A. Douillard, Y. Donchev, K. Rush, S. Kale, Z. Charles, Z. Garrett, G. Teston, D. Lacey, R. McIlroy, J. Shen, et al. Streaming diloco with overlapping communication: towards a distributed free lunch.  arXiv preprint arXiv:2501.18512.  Cited by: §1, §2. 
  * Douillard et al. (2023) A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, and J. Shen DiLoCo: distributed low-communication training of language models.  arXiv.  External Links: [Document](https://dx.doi.org/10.48550/arxiv.2311.08105) Cited by: §1, §2. 
  * Douillard et al. (2026) A. Douillard, K. Rush, Y. Donchev, Z. Charles, N. Fallen, A. Dubey, I. Gog, J. Dean, B. Woodworth, Z. Garrett, et al. Decoupled diloco for resilient distributed pre-training.  arXiv preprint arXiv:2604.21428.  Cited by: §5. 
  * Jaghouar et al. (2024) S. Jaghouar, J. M. Ong, and J. Hagemann OpenDiLoCo: an open-source framework for globally distributed low-communication training.  arXiv preprint arXiv:2407.07852.  Cited by: §1, §2. 
  * Karimireddy et al. (2020) S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. U. Stich, and A. T. Suresh SCAFFOLD: stochastic controlled averaging for federated learning.  In International Conference on Machine Learning,  pp. 5132–5143.  Cited by: §2. 
  * Kolehmainen et al. (2025) J. Kolehmainen, N. Blagoev, J. Donaghy, O. Ersoy, and C. Nies NoLoCo: no-all-reduce low communication training method for large models.  arXiv preprint arXiv:2506.10911.  Cited by: §2. 
  * Li et al. (2020) T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith Federated optimization in heterogeneous networks.  Proceedings of Machine learning and systems 2, pp. 429–450.  Cited by: §1, §2. 
  * Liu et al. (2024) B. Liu, R. Chhaparia, A. Douillard, S. Kale, A. A. Rusu, J. Shen, A. Szlam, and M. Ranzato Asynchronous local-sgd training for language modeling.  arXiv preprint arXiv:2401.09135.  Cited by: §A.6, Table 3, §1, §1, §4.2. 
  * McMahan et al. (2017) B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas Communication-Efficient Learning of Deep Networks from Decentralized Data.  In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics,  Proceedings of Machine Learning Research, Vol. 54, pp. 1273–1282.  Cited by: §2. 
  * Mitliagkas et al. (2016) I. Mitliagkas, C. Zhang, S. Hadjis, and C. Ré Asynchrony begets momentum, with an application to deep learning.  In 2016 54th Annual Allerton Conference on Communication, Control, and Computing (Allerton),  pp. 997–1004.  Cited by: §2. 
  * Qi et al. (2025) J. Qi, W. Zhu, L. Li, M. Wu, Y. Wu, W. He, X. Gao, J. Zeng, and M. Heinrich DiLoCoX: a low-communication large-scale training framework for decentralized cluster.  arXiv preprint arXiv:2506.21263.  Cited by: §2. 
  * Radford et al. (2019) A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever Language models are unsupervised multitask learners.  OpenAI blog 1 (8), pp. 9.  Cited by: §4. 
  * Raffel et al. (2020) C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu Exploring the limits of transfer learning with a unified text-to-text transformer.  Journal of Machine Learning Research 21 (140), pp. 1–67.  External Links: [Link](http://jmlr.org/papers/v21/20-074.html) Cited by: §4. 
  * Recht et al. (2011) B. Recht, C. Re, S. Wright, and F. Niu Hogwild!: a lock-free approach to parallelizing stochastic gradient descent.  Advances in neural information processing systems 24.  Cited by: §2. 
  * Wang and Joshi (2019) J. Wang and G. Joshi Adaptive communication strategies to achieve the best error-runtime trade-off in local-update sgd.  Proceedings of Machine Learning and Systems 1, pp. 212–229.  Cited by: §4.1. 
  * Wang et al. (2020) J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor Tackling the objective inconsistency problem in heterogeneous federated optimization.  In Advances in Neural Information Processing Systems,  Vol. 33, pp. 7611–7623.  Cited by: §2. 
  * Yu et al. (2020) T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn Gradient surgery for multi-task learning.  In Advances in Neural Information Processing Systems,  Vol. 33, pp. 5824–5836.  Cited by: §A.2, §2. 
  * Zhang et al. (2019) M. Zhang, J. Lucas, J. Ba, and G. E. Hinton Lookahead optimizer: k steps forward, 1 step back.  Advances in neural information processing systems 32.  Cited by: §1. 
  * Zheng et al. (2017) S. Zheng, Q. Meng, T. Wang, W. Chen, N. Yu, Z. Ma, and T. Liu Asynchronous stochastic gradient descent with delay compensation.  In International conference on machine learning,  pp. 4120–4129.  Cited by: §1, §2. 
  * Zinkevich et al. (2009) M. Zinkevich, J. Langford, and A. J. Smola Slow learners are fast.  In Advances in Neural Information Processing Systems,  Vol. 22, pp. 2331–2339.  Cited by: §1, §2. 


## Appendix A Appendix

### A.1 Notation

Table 2 summarizes the notation used in this paper

Table 2: Symbols used in the methodology and appendix algorithm. Symbol |  Description  
---|---  
Distributed training objective  
KK |  Total number of workers participating in distributed training.  
i,ji,j |  Worker indices. Typically, ii denotes the worker whose update arrives, while jj denotes an available worker being initialized.  
𝒟i\mathcal{D}_{i} |  Local data distribution of worker ii. These distributions may be non-IID across workers.  
F⁡(θ)F(\theta) |  Global training objective.  
Fi​(θ)F_{i}(\theta) |  Local objective of worker ii, defined over its local data distribution.  
pip_{i} |  Nonnegative weight of worker ii in the global objective, where ∑i=1Kpi=1\sum_{i=1}^{K}p_{i}=1.  
zz |  Training sample drawn from a local data distribution.  
ℓ⁡(θ,z)\ell(\theta;z) |  Loss of model parameters θ\theta on sample zz.  
θ\theta |  Generic model parameter vector.  
dd |  Total number of scalar model parameters, used in the overhead discussion.  
Asynchronous training setup  
tt |  Current server step at which a returned pseudo-gradient is processed.  
rr |  Server step at which an available worker is initialized.  
sis_{i} |  Initialization step of worker ii, i.e., the server step when worker ii receives its initial model.  
τi​(t)\tau_{i}(t) |  Staleness of worker ii’s update at server step tt, defined as τi​(t)=t−si\tau_{i}(t)=t-s_{i}.  
θt\theta_{t} |  Server model at server step tt.  
θr\theta_{r} |  Server model at initialization step rr.  
mtm_{t} |  Server momentum buffer at step tt, representing the current server trajectory.  
mrm_{r} |  Server momentum buffer at initialization step rr.  
μ\mu |  Outer momentum coefficient used by the server.  
ηt\eta_{t} |  Outer learning rate at server step tt.  
ηr\eta_{r} |  Outer learning rate used when constructing the look-ahead model at initialization step rr.  
Local worker training  
θ¯si\bar{\theta}_{s_{i}} |  Initial model sent to worker ii at step sis_{i}. In HeLoCo, this is the look-ahead model.  
θ¯r\bar{\theta}_{r} |  Look-ahead model used to initialize an available worker, defined as θ¯r=θr−ηr​μ​mr\bar{\theta}_{r}=\theta_{r}-\eta_{r}\mu m_{r}.  
HH |  Number of local optimization steps performed by a worker before returning its pseudo-gradient.  
hh |  Index of a local optimization step, where h=0,…,H−1h=0,\ldots,H-1.  
θi,h(si)\theta_{i,h}^{(s_{i})} |  Local model of worker ii after hh inner steps, initialized from server step sis_{i}.  
θi,0(si)\theta_{i,0}^{(s_{i})} |  Initial local model of worker ii, equal to θ¯si\bar{\theta}_{s_{i}}.  
θi,H(si)\theta_{i,H}^{(s_{i})} |  Final local model of worker ii after HH local steps.  
αh\alpha_{h} |  Inner learning rate used at local step hh.  
gi​(⋅)g_{i}(\cdot) |  Stochastic gradient computed on worker ii’s local data.  
Δi(si)\Delta_{i}^{(s_{i})} |  Worker pseudo-gradient, defined as Δi(si)=θ¯si−θi,H(si)\Delta_{i}^{(s_{i})}=\bar{\theta}_{s_{i}}-\theta_{i,H}^{(s_{i})}.  
Tensor-wise compatibility and correction  
bb |  Index of a tensor block, such as a weight tensor or bias tensor.  
Δb\Delta_{b} |  Tensor block bb of the arriving pseudo-gradient.  
(mt)b(m_{t})_{b} |  Tensor block bb of the current server momentum.  
∥⋅∥\|\cdot\| |  Euclidean norm after vectorization; equivalent to the Frobenius norm for tensor-shaped parameters.  
u^b\hat{u}_{b} |  Unit direction of the pseudo-gradient block, defined as u^b=Δb/‖Δb‖\hat{u}_{b}=\Delta_{b}/\|\Delta_{b}\|.  
v^b\hat{v}_{b} |  Unit direction of the momentum block, defined as v^b=(mt)b/‖(mt)b‖\hat{v}_{b}=(m_{t})_{b}/\|(m_{t})_{b}\|.  
cbc_{b} |  Directional compatibility score between the pseudo-gradient block and the momentum block, defined as cb=u^b⊤​v^bc_{b}=\hat{u}_{b}^{\top}\hat{v}_{b}.  
cokc_{\mathrm{ok}} |  Threshold above which a tensor block is considered sufficiently aligned and is kept unchanged.  
ε\varepsilon |  Small numerical threshold used to avoid division by zero and to detect near-zero unreliable directions.  
Δ^b\widehat{\Delta}_{b} |  Corrected version of tensor block Δb\Delta_{b}.  
Δ^\widehat{\Delta} |  Full corrected pseudo-gradient after all tensor blocks are processed.  
confb\mathrm{conf}_{b} |  Confidence factor controlling how strongly tensor block bb should be corrected.  
κ\kappa |  Positive constant controlling how the momentum norm affects confb\mathrm{conf}_{b}.  
ksk_{s} |  Positive coefficient controlling shrinkage strength when a tensor block conflicts with momentum.  
βb\beta_{b} |  Tensor-wise shrinkage coefficient used when cb<0c_{b}<0.  
βmax\beta_{\max} |  Upper bound on βb\beta_{b} to prevent overcorrection.  
kdk_{d} |  Positive coefficient controlling reorientation strength for weakly aligned tensor blocks.  
λb\lambda_{b} |  Tensor-wise interpolation coefficient used when 0≤cb<cok0\leq c_{b}<c_{\mathrm{ok}}.  
u~b\tilde{u}_{b} |  Interpolated direction between the pseudo-gradient direction and the momentum direction.  
Outer model update  
ρt\rho_{t} |  Optional arrival weight applied to the corrected pseudo-gradient. If unused, ρt=1\rho_{t}=1.  
GtG_{t} |  Weighted corrected update, defined as Gt=ρt​Δ^G_{t}=\rho_{t}\widehat{\Delta}.  
mt+1m_{t+1} |  Updated global momentum after incorporating the arriving corrected update.  
θt+1\theta_{t+1} |  Updated global model after applying the outer update.  
Local variables in the pseudocode  
uu |  Tensor block of the returned pseudo-gradient, u=(Δi(si))bu=(\Delta_{i}^{(s_{i})})_{b}.  
vv |  Corresponding momentum block, v=(mt)bv=(m_{t})_{b}.  
u^\hat{u} |  Normalized update direction, u^=u/‖u‖\hat{u}=u/\|u\|.  
v^\hat{v} |  Normalized momentum direction, v^=v/‖v‖\hat{v}=v/\|v\|.  
u~\tilde{u} |  Direction obtained by blending u^\hat{u} with v^\hat{v} for weakly aligned blocks.  
  
### A.2 Theoretical Justification of the HeLoCo Correction

This appendix gives a local theoretical justification for the tensor-wise correction used by HeLoCo. The result is not a convergence proof for the full asynchronous training process. Instead, it shows that, for each non-degenerate tensor block, the correction does not reduce the signed component along the current outer momentum direction and does not increase the block norm.

The correction is motivated by the same geometry used in gradient surgery methods such as PCGrad Yu et al. [2020]. PCGrad treats two gradients as conflicting when their cosine similarity is negative and removes the conflicting projection component. HeLoCo uses this idea in a different setting: it compares each stale pseudo-gradient block with the corresponding outer momentum block. Unlike PCGrad, HeLoCo applies a damped correction in the anti-aligned case and also adjusts weakly aligned blocks by blending their direction with the momentum direction. For the proof only, we use shorthand notation; these symbols correspond to the notation in Table 2.

#### Lemma.

Fix a tensor block bb. Let

| u=Δb,v=(mt)bu=\Delta_{b},\qquad v=(m_{t})_{b} |   
---|---|---  
  
denote the returned pseudo-gradient block and the corresponding outer momentum block. Assume ‖u‖,‖v‖>ε\|u\|,\|v\|>\varepsilon, βmax≤1\beta_{\max}\leq 1, and ε<1/2\varepsilon<1/\sqrt{2}. Define

| u^=u‖u‖,v^=v‖v‖,c=u^⊤​v^.\hat{u}=\frac{u}{\|u\|},\qquad\hat{v}=\frac{v}{\|v\|},\qquad c=\hat{u}^{\top}\hat{v}. |   
---|---|---  
  
Here cc is the cosine similarity between the returned block and the momentum block. Let Δ^b\widehat{\Delta}_{b} be the corrected block produced by HeLoCo. Then

| Δ^b⊤​v^≥u⊤​v^,‖Δ^b‖≤‖u‖.\widehat{\Delta}_{b}^{\top}\hat{v}\geq u^{\top}\hat{v},\qquad\|\widehat{\Delta}_{b}\|\leq\|u\|. |   
---|---|---  
  
Thus, HeLoCo does not reduce the signed component of the block along the outer momentum direction and does not amplify the returned block.

#### Proof.

If c≥cokc\geq c_{\mathrm{ok}}, HeLoCo leaves the block unchanged, so both claims hold with equality.

If c<0c<0, HeLoCo applies

| Δ^b=u−βb​c​‖u‖​v^,0≤βb≤1.\widehat{\Delta}_{b}=u-\beta_{b}c\|u\|\hat{v},\qquad 0\leq\beta_{b}\leq 1. |   
---|---|---  
  
Since u⊤​v^=c​‖u‖u^{\top}\hat{v}=c\|u\|, we have

| Δ^b⊤​v^=(1−βb)​u⊤​v^≥u⊤​v^,\widehat{\Delta}_{b}^{\top}\hat{v}=(1-\beta_{b})u^{\top}\hat{v}\geq u^{\top}\hat{v}, |   
---|---|---  
  
because u⊤​v^<0u^{\top}\hat{v}<0. Also,

| ‖Δ^b‖2=‖u‖2​(1−βb​(2−βb)​c2)≤‖u‖2.\|\widehat{\Delta}_{b}\|^{2}=\|u\|^{2}\left(1-\beta_{b}(2-\beta_{b})c^{2}\right)\leq\|u\|^{2}. |   
---|---|---  
  
Therefore, counter-directed blocks become less opposed to the momentum direction without norm amplification.

If 0≤c<cok0\leq c<c_{\mathrm{ok}}, HeLoCo forms

| u~=(1−λb)​u^+λb​v^,0≤λb≤1.\tilde{u}=(1-\lambda_{b})\hat{u}+\lambda_{b}\hat{v},\qquad 0\leq\lambda_{b}\leq 1. |   
---|---|---  
  
Since 0≤c≤10\leq c\leq 1, we have

| ‖u~‖2=1−2​λb​(1−λb)​(1−c)≥12.\|\tilde{u}\|^{2}=1-2\lambda_{b}(1-\lambda_{b})(1-c)\geq\frac{1}{2}. |   
---|---|---  
  
Because ε<1/2\varepsilon<1/\sqrt{2}, the numerical guard is inactive in this case, and

| Δ^b=‖u‖​u~‖u~‖.\widehat{\Delta}_{b}=\|u\|\frac{\tilde{u}}{\|\tilde{u}\|}. |   
---|---|---  
  
Thus ‖Δ^b‖=‖u‖\|\widehat{\Delta}_{b}\|=\|u\|. Let

| c′=u~⊤​v^‖u~‖c^{\prime}=\frac{\tilde{u}^{\top}\hat{v}}{\|\tilde{u}\|} |   
---|---|---  
  
be the cosine similarity after correction. Then

| c′=(1−λb)​c+λb(1−λb)2+λb2+2​λb​(1−λb)​c.c^{\prime}=\frac{(1-\lambda_{b})c+\lambda_{b}}{\sqrt{(1-\lambda_{b})^{2}+\lambda_{b}^{2}+2\lambda_{b}(1-\lambda_{b})c}}. |   
---|---|---  
  
For 0≤c≤10\leq c\leq 1 and 0≤λb≤10\leq\lambda_{b}\leq 1, the numerator is at least cc and the denominator is at most 11, so c′≥cc^{\prime}\geq c. Therefore,

| Δ^b⊤​v^=‖u‖​c′≥‖u‖​c=u⊤​v^.\widehat{\Delta}_{b}^{\top}\hat{v}=\|u\|c^{\prime}\geq\|u\|c=u^{\top}\hat{v}. |   
---|---|---  
  
Hence weakly aligned blocks are moved toward the outer momentum direction without changing their norm.

The lemma applies to non-degenerate blocks for which the cosine comparison is well defined. When ‖u‖<ε\|u\|<\varepsilon or ‖v‖<ε\|v\|<\varepsilon, HeLoCo skips the correction because the update or momentum direction is numerically unreliable. This completes the proof.

#### Interpretation.

The lemma shows that HeLoCo does not blindly suppress stale pseudo-gradients. Well-aligned blocks are preserved, anti-aligned blocks have their opposing component damped, and weakly aligned blocks are adjusted toward the momentum direction without increasing their magnitude. Since outer momentum summarizes recent outer updates, this local guarantee supports the use of momentum as the reference direction for correcting stale pseudo-gradient blocks.

### A.3 HeLoCo Algorithm

Algorithm 1 Asynchronous DiLoCo with HeLoCo

1: Input:  Initial model θ(0)\theta^{(0)}, outer momentum m0=0m_{0}{=}0, outer learning rate η\eta, momentum coefficient μ\mu, kk workers, data shards {𝒟1,…,𝒟k}\\{\mathcal{D}_{1},\ldots,\mathcal{D}_{k}\\}

2: Output:  Updated global model θt+1\theta_{t+1} and outer momentum mt+1m_{t+1}

3: for outer step t=1,2,…t=1,2,\ldots do

4: if worker jj is available at step rr then

5: θ¯r←θr−ηr​μ​mr\bar{\theta}_{r}\leftarrow\theta_{r}-\eta_{r}\mu m_{r} ⊳\triangleright HeLoCo: look-ahead initialization 

6: Send θ¯r\bar{\theta}_{r} to worker jj; record sj←rs_{j}\leftarrow r

7: end if

8: for inner step h=1​…​Hh=1\ldots H do

9: x∼𝒟jx\sim\mathcal{D}_{j}

10: ℒ←f⁡(x,θj(t))\mathcal{L}\leftarrow f(x,\theta_{j}^{(t)})

11: θj(t)←InnerOpt​(θj(t),∇ℒ)\theta_{j}^{(t)}\leftarrow\texttt{InnerOpt}(\theta_{j}^{(t)},\nabla_{\mathcal{L}})

12: end for

13: if worker ii returns Δi(si)\Delta_{i}^{(s_{i})} at step tt then

14: τi←t−si\tau_{i}\leftarrow t-s_{i} ⊳\triangleright Measure staleness 

15: G^t←BlockCorrect​(Δi(si),mt,ρt)\widehat{G}_{t}\leftarrow\textsc{BlockCorrect}(\Delta_{i}^{(s_{i})},m_{t},\rho_{t}) ⊳\triangleright HeLoCo: per-block correction (Algorithm 2) 

16: mt+1←μ​mt+(1−μ)​G^tm_{t+1}\leftarrow\mu m_{t}+(1-\mu)\widehat{G}_{t}

17: θt+1←θt−η⁡(G^t+μ​mt+1)\theta_{t+1}\leftarrow\theta_{t}-\eta\bigl(\widehat{G}_{t}+\mu m_{t+1}\bigr)

18: end if

19: end for

Algorithm 2 Tensor Directional Correction

1: Input:  Pseudo-gradient Δi(si)\Delta_{i}^{(s_{i})}, outer momentum mtm_{t}, arrival weight ρt\rho_{t}; alignment threshold cokc_{\mathrm{ok}}, correction constants ks,kd,κ,βmax,εk_{s},k_{d},\kappa,\beta_{\max},\varepsilon

2: Output:  Corrected update G^t\widehat{G}_{t}

3: Δ^←Δi(si)\widehat{\Delta}\leftarrow\Delta_{i}^{(s_{i})}

4: for each tensor block bb do

5: u←(Δi(si))bu\leftarrow(\Delta_{i}^{(s_{i})})_{b}; v←(mt)bv\leftarrow(m_{t})_{b}

6: if ‖u‖<ε\|u\|<\varepsilon or ‖v‖<ε\|v\|<\varepsilon then

7: Δ^b←u\widehat{\Delta}_{b}\leftarrow u ⊳\triangleright Skip: direction unreliable 

8: else

9: u^←u/‖u‖\hat{u}\leftarrow u/\|u\|; v^←v/‖v‖\hat{v}\leftarrow v/\|v\|

10: cb←u^⊤​v^c_{b}\leftarrow\hat{u}^{\top}\hat{v} ⊳\triangleright Cosine alignment with momentum 

11: confb←‖u‖/(‖u‖+κ​‖v‖+ε)\mathrm{conf}_{b}\leftarrow\|u\|/(\|u\|+\kappa\|v\|+\varepsilon)

12: if cb≥cokc_{b}\geq c_{\mathrm{ok}} then

13: Δ^b←u\widehat{\Delta}_{b}\leftarrow u ⊳\triangleright Keep well-aligned blocks 

14: else if cb<0c_{b}<0 then

15: βb←min⁡{ks​(−cb)​confb,βmax}\beta_{b}\leftarrow\min\\{k_{s}(-c_{b})\,\mathrm{conf}_{b},\,\beta_{\max}\\}

16: Δ^b←u−βb​cb​‖u‖​v^\widehat{\Delta}_{b}\leftarrow u-\beta_{b}c_{b}\|u\|\hat{v} ⊳\triangleright Shrink the anti-momentum component 

17: else

18: λb←min⁡{kd​(1−cb)​confb, 1}\lambda_{b}\leftarrow\min\\{k_{d}(1-c_{b})\,\mathrm{conf}_{b},\,1\\}

19: u~←(1−λb)​u^+λb​v^\tilde{u}\leftarrow(1-\lambda_{b})\hat{u}+\lambda_{b}\hat{v}

20: Δ^b←‖u‖​u~/max⁡{‖u~‖,ε}\widehat{\Delta}_{b}\leftarrow\|u\|\,\tilde{u}/\max\\{\|\tilde{u}\|,\varepsilon\\} ⊳\triangleright Weakly aligned: rotate toward momentum 

21: end if

22: end if

23: end for

24: G^t←ρt​Δ^\widehat{G}_{t}\leftarrow\rho_{t}\,\widehat{\Delta}

25: return G^t\widehat{G}_{t}

This appendix gives the procedural form of HeLoCo. The pseudocode shows the two actions performed by the synchronizer during asynchronous training: initializing an available worker with the look-ahead outer model, and updating the outer model when a pseudo-gradient returns. Because workers run independently, their updates need not arrive in the same order in which they were initialized. The synchronizer therefore records the initialization step sis_{i} for each worker and computes the staleness τi​(t)=t−si\tau_{i}(t)=t-s_{i} when the update is processed.

### A.4 Implementation and Compute Details

All experiments were implemented in Python using PyTorch 2.10 and CUDA 12.6. We use a TinyGPT-style decoder-only Transformer with 15M parameters and the GPT-2 tokenizer. The main software dependencies are Transformers, NumPy, Pandas, and Matplotlib. Since different experiment groups were run on different GPU nodes, we report the hardware configuration separately for each group.

For the experiments in Section 4.1, each run uses five workers. Each worker performs 20 local steps before returning a pseudo-gradient to the server, and training runs for 100 outer updates. These experiments were run on a node with 4 NVIDIA A100 GPUs, 128 GB memory, and 8 CPU cores. The five workers were run in parallel, with each complete run taking approximately 60 minutes.

For the remaining results, we use a longer training setting with 80 local steps per worker update and 300 outer updates. To study different levels of system heterogeneity, we vary the worker paces, which directly changes the amount of staleness in the asynchronous update stream. These experiments were run on nodes with 8 NVIDIA H100 GPUs (80 GB memory each), 112 CPU cores and 2 TB RAM. A training run with 24k steps requires approximately 15 minutes of runtime for one experiment, while using 5 GPUs - one for each worker. Within each experimental group, all methods use the same model, data preprocessing, optimizer settings, and evaluation protocol. To save compute time, a full model evaluation on the evaluation dataset was omitted after each outer update during the middle of each training run. This was only performed at the beginning and at the end of the training runs, which is why gaps appear in the middle of the loss plots.

### A.5 Training Hyperparameters

The hyperparameters used for all experiments in Sections 4.2 and A.6 are summarized in table 3.

Table 3: Run configuration for non-iid experiments. There are different possibilities to manipulate weights. base means the weighting factor is calculated as k/k\nicefrac{{\sqrt{k}}}{{k}}, where k is the number of workers. This calculation was taken from the original async DiLoCo paper and showed the best results in our case. average means a factor of 1/k\nicefrac{{1}}{{k}} is used. For the inner learning rate the identical cosine scheduler from [Liu et al., 2024] is used. Additional HeLoCo-specific parameters are: kd=1.0k_{\text{d}}=1.0, cok=0.2c_{\text{ok}}=0.2, , κ=3.0\kappa=3.0, βmax=0.5\beta_{\max}=0.5, ks=0.5k_{\text{s}}=0.5 Method | Weight Factor | Outer learning rate | Momentum  
---|---|---|---  
async-HeLoCo | base | 0.7 | 0.9  
async-mla | base | 0.7 | 0.9  
async-nesterov | base | 0.07 | 0.9  
sync-nesterov | average | 0.7 | 0.9  
  
### A.6 Deeper Insights into Non-IID Training

Fixed shard-to-worker setup

In Section 4.2, we observed that HeLoCo achieves a lower overall loss than the other algorithms in the non-IID setting where shards are permanently assigned to workers. This conclusion was based on the average loss across all languages. In the following, we analyze in more detail how different levels of staleness affect the representation of individual shards (i.e., languages) within the global model.

Figure 5: Influence of the number and degree of stale workers on the validation loss for different languages in the server model (fixed setup). In each plot, worker 0 (lowest staleness) and worker 4 (highest staleness) are shown.

In Figure 5, we first analyze the behavior when introducing a single stale worker, starting from the homogeneous case and examining what happens when the staleness of one worker is increased. When a worker with pace 2 (seconds/iteration) is added, the loss of the fast worker (pace 1 second/iteration) changes only marginally, while the loss associated with worker 4 (the stale worker) increases immediately. As the staleness is increased further, this effect becomes more pronounced. The loss associated with the stale worker continues to deteriorate, and its corresponding language becomes increasingly underrepresented in the global model. This behavior remains stable for up to three stale workers with pace 2.

Once the number of stale workers reaches four, or when three workers exhibit a staleness of 6 or higher, a different behavior emerges for both MLA and HeLoCo. The loss associated with the fast worker (worker 0) begins to increase again during training, while the losses of the slow workers continue to decrease. This suggests that beyond a certain level of staleness, the global model starts to forget information associated with the fast workers language in favor of the stale workers. In other words, the model appears to lose previously acquired knowledge for well-represented shards while adapting to underrepresented ones.

Interestingly, this effect is not visible when considering only the mean validation loss across all languages, as shown in Figure 6. The degradation in performance for worker 0 is compensated by improvements for the stale workers, resulting in a relatively stable average loss. For future work, it may be interesting to investigate this behavior in more detail to better understand under which conditions this forgetting effect occurs and how it might be mitigated through appropriate training strategies.

Figure 6: Influence of the number and degree of stale workers on the mean validation loss of the server model.

Flexible shard-to-worker setup

In this section, we investigate a flexible shard-to-worker assignment strategy, in which each worker dynamically selects the shard that has been processed the least overall. This setup corresponds to the scenario used in the original async DiLoCo work of [Liu et al., 2024]. The key difference in our experiments is that the shards remain highly non-IID.

Compared to the fixed shard-to-worker setup, Figure 7 shows that the discrepancy between fast and slow workers has a much smaller impact on the language-specific losses. The flexible assignment strategy leads to a more balanced representation of the different shards, because fast workers are no longer restricted to a single shard and can contribute to multiple languages over time.

Figure 7: Influence of the number and degree of stale workers on the validation loss for different languages in the server model (flexible setup). In each plot, worker 0 (lowest staleness) and worker 4 (highest staleness) are shown.

Finally, we investigate whether discarding stale gradients can improve the overall training performance. This means for example, that in the case (1,1,6,6,6)(1,1,6,6,6) the pseudo-gradient of three stale workers is set to 0 in order to suppress their influence on the server model.

Figure 8: Influence of dropping pseudo-gradients from stale workers on the mean validation loss over all languages.

Figure 8 shows that discarding gradients in scenarios with more than three stale workers and a staleness of 6 or greater consistently improves the performance of MLA-based algorithms, while it leads to degraded loss for asynchronous Nesterov. This suggests that neither MLA-based HeLoCo nor the pure MLA approach is able to effectively exploit highly stale updates in this setting, in contrast to asynchronous Nesterov. Rather than contributing useful information to training, these stale updates appear to impair the quality of the global model. Overall, this behavior indicates that there is still room for improvement, and that developing an adaptive mechanism to determine which workers provide beneficial updates and which do not could be a promising direction for future work.

Experimental support, please [view the build logs](./2606.00271v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.


**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our [**major funders**](https://info.arxiv.org/about/major_funders.html), [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe.html) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
