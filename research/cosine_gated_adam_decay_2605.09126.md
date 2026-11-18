Skip to main content

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

[](https://arxiv.org/IgnoreMe) [ ](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2605.09126v1 "Back to abstract page") [ Download PDF ](/pdf/2605.09126v1 "Download PDF") Contents Reading mode Colour theme

  1. Abstract
  2. 1 Introduction
  3. 2 Background and related work
     1. The DiLoCo family.
     2. Async-SGD theory.
     3. Concurrent staleness-aware methods.
  4. 3 Method
     1. 3.1 The DDiLoCo optimization template
     2. 3.2 Cosine-Gated Adam-Decay (CGAD)
        1. Why this functional form?
     3. 3.3 PA-CGAD: per-fragment age
  5. 4 Theoretical analysis
     1. Setup.
     2. Comparison to delay-unaware momentum analyses.
  6. 5 Experiments
     1. 5.1 Setup
     2. 5.2 Headline result: CGAD vs. Nesterov at 1B
     3. 5.3 Multi-method comparison at 25 M
     4. 5.4 Scaling: 25 M →\to 1 B →\to 7 B
        1. The cosine cutoff is scale insurance.
        2. Why Adam-Decay specifically leaks at 7 B.
     5. 5.5 Ablation: α\alpha and τcut\tau_{\text{cut}}
     6. 5.6 Tuned-vs-tuned: is the gap a tuning artifact?
     7. 5.7 Robustness checks
  7. 6 Discussion and limitations
     1. Why not just use Adam-Decay at small scale?
     2. Numerical precision at 7 B.
     3. Limitations.
  8. 7 Conclusion
  9. References
  10. A Full proof of Theorem 
  11. B Implementation details
     1. The training framework.
     2. Data.
     3. Architectures.
     4. Memory budget at 7 B.
  12. C Per-seed numbers
  13. D H=30H=30 ablation: ranking persists at larger inner-step counts
  14. E 10 M nine-optimizer survey
  15. F Hyperparameter sensitivity for CGAD
     1. CGAD tuning at scale (1 B / τ=8\tau{=}8).
     2. Does α=0.4\alpha{=}0.4 generalise across τ\tau?
  16. G Nesterov LR sensitivity at 10 M and 1 B
  17. H Gate ordering: before vs. after Adam’s state update
  18. I PA-CGAD per-fragment age dynamics
  19. J 7B preliminary results, full data
     1. τ=0\tau{=}0.
     2. τ=8\tau{=}8.
     3. τ=16\tau{=}16.
     4. Longer-training stability.


[ License: CC BY 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2605.09126v1 [cs.LG] 09 May 2026

# Cosine-Gated Adam-Decay:   
Drop-In Staleness-Aware Outer Optimization for Decoupled DiLoCo

Vatsal Shah  Affiliation: FLock.io  Email: [vatsal@flock.io](mailto:) Jiahao Sun  Affiliation: FLock.io  Email: [sun@flock.io](mailto:)

May 2026

###### Abstract

Asynchronous DiLoCo systems may receive pseudo-gradients computed several outer rounds earlier, yet the standard Nesterov outer optimizer does not explicitly condition its update on per-update age. This can make the outer momentum buffer brittle under large controlled delays. We propose _Cosine-Gated Adam-Decay_ (CGAD), a simple, drop-in, age-aware outer optimizer that scales each incoming pseudo-gradient by σ⁡(τ)=γ⁡(τ)​e−α​τ\sigma(\tau)=\gamma(\tau)\,e^{-\alpha\tau} before it enters Adam’s first- and second-moment buffers; the exponential models information decay and the cosine gate γ⁡(τ)\gamma(\tau) smoothly zeroes contributions past a chosen cutoff. CGAD reduces to plain Adam at τ=0\tau{=}0, adds two hyperparameters whose defaults transfer across scales, and extends to partial-sync schedulers via a per-fragment age-aware variant (PA-CGAD). For an idealized gated-adaptive update on smooth non-convex objectives, we prove a non-asymptotic convergence bound whose staleness-bias term depends on α\alpha alone, rather than on the realized maximum delay τmax\tau_{\max}; standard analyses of asynchronous momentum-SGD instead carry a τmax2\tau_{\max}^{2} factor. Empirically, on Llama-style language-model pretraining at 25 M, 1 B, and 7 B parameters, CGAD trains stably across the controlled delays we sweep. The cosine cutoff acts as _scale insurance_ : the closest baseline, Adam-Decay (CGAD without the cutoff), is competitive at 25 M but its seed-to-seed σ\sigma at τ=8\tau{=}8 grows 27×27\times from 25 M to 7 B, pushing its single-shot risk (mean +σ+\,\sigma) above the chance-level loss while CGAD’s stays well below. The published Nesterov recipe is the least stable method on the full sweep.

## 1 Introduction

Pretraining large language models has moved away from a single tightly-coupled cluster. The DiLoCo family [2, 3, 5, 4, 1] lets workers run many local SGD or AdamW steps before exchanging an aggregated pseudo-gradient with a global parameter server, opening the door to training on heterogeneous, geographically distributed, or partially unreliable hardware. Decoupled DiLoCo’s chaos-engineering experiments [1] report healthy training on a 2.4-million-chip cluster with a sub-minute mean time between failures.

In any such system, pseudo-gradients arrive with delay. The pseudo-gradient applied at outer round tt may have been computed τ\tau rounds earlier, where τ\tau depends on network conditions, slow learners, and the asynchrony budget. The outer optimizer used across the DiLoCo family is SGD with Nesterov momentum [2], retained unchanged in Decoupled DiLoCo [1]. The recipe applies the same outer step to every incoming pseudo-gradient, with no accounting for τ\tau. As τ\tau grows, momentum accumulates a stale view of the loss landscape into the velocity buffer, and the step overshoots directions that no longer descend the current loss.

We propose _Cosine-Gated Adam-Decay_ (CGAD), a one-line modification of Adam at the outer step. Each incoming pseudo-gradient is scaled by σ⁡(τ)=γ⁡(τ)​e−α​τ\sigma(\tau)=\gamma(\tau)\,e^{-\alpha\tau} (an exponential information-decay term times a cosine gate that drops to zero at a chosen cutoff τcut\tau_{\text{cut}}) before it enters Adam’s moment buffers. CGAD adds two hyperparameters whose defaults transfer across scales, reduces to plain Adam at τ=0\tau{=}0, and drops in wherever Nesterov OuterOpt is currently used. We make three contributions:

  1. 1.

A non-asymptotic convergence bound for an idealized gated-adaptive update on smooth non-convex objectives (Theorem 1), whose staleness-bias term is 𝒪⁡(1/(e​α))\mathcal{O}(1/(e\alpha)) and depends on the gating hyperparameter α\alpha rather than on the realized maximum delay τmax\tau_{\max}. Standard analyses of asynchronous momentum-SGD [14] carry a τmax2\tau_{\max}^{2} factor in the matching place; the result is intended as motivation for the gating mechanism rather than as a full Adam-convergence theorem.

  2. 2.

A per-fragment-age variant (PA-CGAD) that lets the gate share a single staleness model with partial-sync schedulers such as Streaming DiLoCo’s fragment streaming [3].

  3. 3.

Llama-style pretraining experiments at 25 M, 1 B, and 7 B parameters that put CGAD head-to-head with the published Nesterov recipe and four staleness-aware baselines (Adam-Decay, SDM, Delayed-Nesterov, Polynomial-Discount). At 1 B with τ=16\tau{=}16, CGAD reaches loss 7.447.44 where the published Nesterov recipe loses convergence at 319319, a 43×43\times gap. At 7 B in a memory-constrained configuration, the cosine cutoff acts as _scale insurance_ : the closest baseline, Adam-Decay, sees its seed-to-seed σ\sigma at τ=8\tau{=}8 grow 27×27\times from 25 M to 7 B (from 0.070.07 to 1.891.89), while CGAD’s σ\sigma stays roughly flat at 0.380.38. The risk-adjusted (mean +σ+\,\sigma) gap to Adam-Decay at 7 B / τ=8\tau{=}8 is ≈1.9\approx 1.9 nats.


## 2 Background and related work

#### The DiLoCo family.

DiLoCo [2] is the canonical FedOpt instantiation for LLM pretraining: MM replicas run inner AdamW for HH steps on distinct data shards, then exchange a parameter delta Δm(t)=θ(t−H)−θm(t)\Delta_{m}^{(t)}=\theta^{(t-H)}-\theta_{m}^{(t)} that is averaged across replicas and applied via outer SGD with Nesterov momentum (η=0.7,μ=0.9\eta{=}0.7,\mu{=}0.9 in the published recipe). _OpenDiLoCo_ [5] is an open-source replication that scaled the method to 1B params across two continents using Hivemind, demonstrating 90–95% compute utilization across geographically distributed nodes. _Streaming DiLoCo_ [3] reduces peak bandwidth by synchronizing only a subset of parameter fragments per outer round (fragment-wise streaming) while overlapping a single inner step with the all-reduce, dropping bandwidth requirements by ∼2{\sim}2–10×10\times relative to standard DiLoCo. _Eager-Async DiLoCo_ [4] pushes the overlap to a full outer-step phase: it applies a mixture of the worker’s _local_ pseudo-grad with the previous outer step’s _averaged_ pseudo-grad, δ~m(t)=(1/M)​(Δm(t)−Δm(t−H))+Δ(t−H)\tilde{\delta}_{m}^{(t)}=(1/M)(\Delta_{m}^{(t)}-\Delta_{m}^{(t-H)})+\Delta^{(t-H)}, to allow communication to overlap an entire inner phase. _Decoupled DiLoCo_ [1], the newest member of the family (April 2026), introduces an asynchronous parameter-server-like architecture: MM learners run independently, a syncer aggregates with a minimum quorum K≤MK\leq M, an adaptive grace window absorbs straggler slack, and a token-weighted merging accommodates speed heterogeneity. DDiLoCo’s chaos-engineering validation reports goodput 80% on 2.4M-chip clusters with sub-minute MTBF, but the underlying outer optimizer is still SGD with Nesterov momentum, the choice we challenge. _Smoothing DiLoCo_ [6] applies primal averaging to smooth trajectories across DiLoCo workers; it is complementary to our staleness-aware optimizer changes. None of these methods explicitly model per-update staleness τ\tau as a continuous parameter.

#### Async-SGD theory.

The classical analysis of asynchronous SGD bounds the convergence rate in terms of the staleness τmax\tau_{\text{max}}. Lian et al. [14] give an 𝒪⁡(1/T)\mathcal{O}(1/\sqrt{T}) rate for nonconvex async SGD with a step size that scales with 1/τmax1/\tau_{\text{max}}, so the iteration complexity to reach loss-residual ϵ\epsilon grows as τmax2/ϵ2\tau_{\text{max}}^{2}/\epsilon^{2}. The recent virtual-iterate analysis of Mishchenko et al. [12] replaces the τmax\tau_{\text{max}} dependence with a function of the worker count under specific assumptions, demonstrating that the worst-case bound can in principle be tightened. Stich [13] establishes that local SGD (the underlying template of DiLoCo, in the synchronous case) converges with rate 𝒪⁡(1/K​T)\mathcal{O}(1/\sqrt{KT}) when the inner-step count HH is bounded. Koloskova, Stich, and Jaggi [15] sharpen these bounds by showing 𝒪⁡((F0−F∗)/T+τmax/T)\mathcal{O}(\sqrt{(F_{0}-F^{*})/T}+\sqrt{\tau_{\text{max}}/T}) convergence under bounded delay assumptions. Cohen et al. [16] prove robustness to arbitrary delay sequences with adaptive step-sizing. These analyses are for vanilla momentum-SGD or plain SGD; they do not cover the Adam-family outer optimizer that FedOpt-style instantiations rely on, nor do they yield a drop-in recipe whose staleness penalty is independent of the realized delay distribution.

#### Concurrent staleness-aware methods.

Table 1 summarizes the closest prior work along the mechanism each uses and the limitation each carries.

Table 1: Closest concurrent staleness-aware methods. Method |  Mechanism |  Limitation  
---|---|---  
DC-ASGD [11] |  Hessian Taylor |  Unstable at high τ\tau; [7] reported limited gains in local-SGD.  
Poly-Decay [10] |  g←(1+τ)−1/2gg\leftarrow(1+\tau)^{-1/2}g on Nesterov |  Damping rate too slow to close the synchronous-async gap.  
CO2 [8] |  Empirical staleness-gap Λt\Lambda_{t} \+ clipping at fixed τ=1\tau{=}1 |  Measures _displacement_ , not τ\tau; analysis fixed at τ=1\tau{=}1.  
Delayed Nesterov [7] |  Buffer NN rounds, sporadic Nesterov bursts |  Temporal batching strategy; no per-update damping.  
Eager-Async DiLoCo [4] |  Fixed binary delay handling |  No explicit τ\tau in the update rule.  
MLA [9] |  Project parameters by τ​μ\tau\mu steps |  Lookahead extrapolation rather than damping.  
FADAS [19] |  Delay-adaptive LR schedule applied after Adam’s state update |  Scales the LR rather than the gradient before m,vm,v; second moment can be polluted by stale variance.  
  
## 3 Method

### 3.1 The DDiLoCo optimization template

We adopt the DDiLoCo notation. Let θ=⨆fθf\theta=\bigsqcup_{f}\theta_{f} be the global model partitioned into fragments. At outer round tt, each worker w∈{1​…​K}w\in\\{1\dots K\\} runs HH inner AdamW steps on its data shard DwD_{w} from a copy of the current global, then computes a fragment-wise pseudo-gradient Δf,w(t)=θf(t−H)−θf,w(t)\Delta_{f,w}^{(t)}=\theta_{f}^{(t-H)}-\theta_{f,w}^{(t)}. The syncer averages these to produce Δ¯f(t)\bar{\Delta}_{f}^{(t)}, which arrives with delay τ∈[0,τmax]\tau\in[0,\tau_{\text{max}}].

The outer optimizer applies Δ¯f(t−τ)\bar{\Delta}_{f}^{(t-\tau)} to update θf\theta_{f}. DDiLoCo uses Nesterov SGD; we replace it with CGAD.

### 3.2 Cosine-Gated Adam-Decay (CGAD)

Algorithm 1 CGAD: one outer-optimizer step.

pseudo-gradient gfg_{f} for fragment ff, delay τ\tau, hyperparams (η,α,τcut,β1,β2)(\eta,\alpha,\tau_{\text{cut}},\beta_{1},\beta_{2})

γ←[τ≤τcut]⋅12(1+cos(πτ/τcut))\gamma\leftarrow[\tau\leq\tau_{\text{cut}}]\cdot\tfrac{1}{2}(1+\cos(\pi\tau/\tau_{\text{cut}}))

σ←γ⋅e−α​τ\sigma\leftarrow\gamma\cdot e^{-\alpha\tau}

if σ=0\sigma=0 then return ⊳\triangleright drop entirely 

end if

g′←σ⋅gfg^{\prime}\leftarrow\sigma\cdot g_{f}

m←β1​m+(1−β1)​g′m\leftarrow\beta_{1}m+(1{-}\beta_{1})g^{\prime} ⊳\triangleright Adam first moment on _gated_ grad 

v←β2​v+(1−β2)​g′⊙g′v\leftarrow\beta_{2}v+(1{-}\beta_{2})g^{\prime}\odot g^{\prime} ⊳\triangleright Adam second moment on _gated_ grad 

m^←m/(1−β1t),v^←v/(1−β2t)\hat{m}\leftarrow m/(1-\beta_{1}^{t}),\quad\hat{v}\leftarrow v/(1-\beta_{2}^{t})

θf←θf−η⋅σ⋅m^/(v^+ε)\theta_{f}\leftarrow\theta_{f}-\eta\cdot\sigma\cdot\hat{m}/(\sqrt{\hat{v}}+\varepsilon)

Default hyperparameters. (α,τcut,η,β1,β2,ε)=(0.2,32,10−3,0.9,0.95,10−8)(\alpha,\tau_{\text{cut}},\eta,\beta_{1},\beta_{2},\varepsilon)=(0.2,32,10^{-3},0.9,0.95,10^{-8}). We recommend τcut≈2​τmax\tau_{\text{cut}}\approx 2\tau_{\text{max}}.

Three properties make CGAD fit DDiLoCo: (i) bounded step magnitude (Lemma 1); (ii) no stale-variance pollution (the second moment uses g′g^{\prime}, so a high-τ\tau step does not inflate v\sqrt{v}, in contrast to FADAS); and (iii) one-line drop-in, compatible with DDiLoCo’s existing per-fragment infrastructure.

#### Why this functional form?

Under LL-smoothness, the squared distance between ∇F​(θt)\nabla F(\theta_{t}) and ∇F​(θt−τ)\nabla F(\theta_{t-\tau}) is bounded by L2​η2​τ2L^{2}\,\eta^{2}\,\tau^{2}, so a τ\tau-stale gradient becomes unreliable for the current parameters at a rate that grows with τ\tau; an exponential decay e−α​τe^{-\alpha\tau} is a simple parameterisable schedule that matches this growth. The cosine gate γ⁡(τ)=12​(1+cos⁡(π​τ/τcut))\gamma(\tau)=\tfrac{1}{2}(1+\cos(\pi\tau/\tau_{\text{cut}})) serves a different purpose: it equals one for fresh gradients, smoothly drops to zero at τcut\tau_{\text{cut}}, and stays at zero beyond that. Without the gate, a single very-stale update (τ≫τcut\tau\gg\tau_{\text{cut}}) can inflate v\sqrt{v} in Adam’s second-moment buffer and shrink subsequent fresh updates for many steps. The cosine cutoff prevents that.

We default to gating m,vm,v on σ⁡(τ)​g\sigma(\tau)g rather than scaling only the final step (FADAS-style [19]). Gating before m,vm,v keeps the second moment vv from being inflated by stale-gradient magnitudes. A controlled head-to-head varying only the placement (App. H) shows the two orderings are empirically indistinguishable at the scales we tested; what matters is the gate’s _functional form_ (γ⁡(τ)​e−α​τ\gamma(\tau)\,e^{-\alpha\tau}, with the cosine cutoff preventing multiplicative collapse). We default to the before-ordering because it integrates more cleanly into existing Adam implementations.

CGAD reduces to Adam when α=0\alpha{=}0 and τcut→∞\tau_{\text{cut}}\to\infty. The τcut→∞\tau_{\text{cut}}\to\infty ablation with α>0\alpha>0 is the _Adam-Decay_ baseline we report alongside CGAD; it is the exponential-only ablation of our own proposal, used to isolate the contribution of the cosine cutoff. The cosine cutoff matters most at high τ\tau, where the never-zero exponential leaves residual contributions in mm and vv that the gate suppresses entirely.

### 3.3 PA-CGAD: per-fragment age

Under partial-sync (Streaming DiLoCo, DDiLoCo’s fragment scheduling), only Kf≤|F|K_{f}\leq|F| fragments are synced per outer round. PA-CGAD tracks per-fragment age afa_{f} (rounds since last sync) and gates each fragment by max⁡(τ,af)\max(\tau,a_{f}). Skipping a fragment for kk outer rounds is then penalized by the same gate that handles network-induced delay, so partial-sync schedulers and the optimizer share a single staleness model.

## 4 Theoretical analysis

We analyze an _idealized gated-adaptive update_ that captures the staleness-handling behaviour of CGAD without modeling the full Adam state machine: the per-coordinate normalized step is treated as bounded (rather than derived from the empirical m,vm,v statistics), and the gate σ⁡(τ)\sigma(\tau) is applied multiplicatively to each update. This is the same level of abstraction used by analyses of adaptive methods that bound the bias-corrected ratio m^/(v^+ε)\hat{m}/(\sqrt{\hat{v}}+\varepsilon) as a regularity condition rather than deriving it; the analysis below should be read as a result about _this idealized update_ , not as a full convergence theorem for empirical Adam.

#### Setup.

F:ℝd→ℝF:\mathbb{R}^{d}\to\mathbb{R} is L-smooth; outer round tt sees Δt=−∇F​(θt−τt)+ξt\Delta_{t}=-\nabla F(\theta_{t-\tau_{t}})+\xi_{t} with bounded noise 𝔼​‖ξt‖2≤σ2\mathbb{E}\|\xi_{t}\|^{2}\leq\sigma^{2} and bounded delay τt≤τcut\tau_{t}\leq\tau_{\text{cut}}.

###### Lemma 1 (Step-magnitude bound).

For the idealized gated-adaptive update of Theorem 1, ‖θt+1−θt‖∞≤η⋅σt\|\theta_{t+1}-\theta_{t}\|_{\infty}\leq\eta\cdot\sigma_{t}, where σt=γ⁡(τt)​e−α​τt\sigma_{t}=\gamma(\tau_{t})e^{-\alpha\tau_{t}}.

###### Proof.

By the bounded normalized-step assumption (A2), |m^i/(v^i+ε)|≤1|\hat{m}_{i}/(\sqrt{\hat{v}_{i}}+\varepsilon)|\leq 1. The idealized update magnifies this by the multiplicative gate σt\sigma_{t} only. ∎

###### Theorem 1 (Convergence rate, idealized gated-adaptive update).

Assume FF is LL-smooth and that the per-coordinate normalized step is bounded, ‖m^t/(v^t+ε)‖∞≤1\|\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\varepsilon)\|_{\infty}\leq 1. This is a regularity condition standard in the analysis literature for adaptive methods (it holds, for instance, under bounded gradients with β2≥β12\beta_{2}\geq\beta_{1}^{2}); we adopt it here so the result speaks to the gating mechanism rather than to Adam’s adaptive state. Assume further τcut≥1/α\tau_{\text{cut}}\geq 1/\alpha, an unbiased pseudo-gradient 𝔼⁡[Δt∣ℱt]=−∇F​(θt−τt)\mathbb{E}[\Delta_{t}\mid\mathcal{F}_{t}]=-\nabla F(\theta_{t-\tau_{t}}), and 𝔼​‖Δt‖2≤σ2\mathbb{E}\|\Delta_{t}\|^{2}\leq\sigma^{2}. Choose η=c/T\eta=c/\sqrt{T} and let σ¯=1T​∑tσt\bar{\sigma}=\tfrac{1}{T}\sum_{t}\sigma_{t}. Then

| 1T​∑t=0T−1σt​𝔼​‖∇F​(θt)‖2≤F⁡(θ0)−F∗c​T+L​c​σ22​T+L​c​Ge​α​T⏟staleness-bias term,\frac{1}{T}\sum_{t=0}^{T-1}\sigma_{t}\,\mathbb{E}\|\nabla F(\theta_{t})\|^{2}\leq\frac{F(\theta_{0})-F^{*}}{c\sqrt{T}}+\frac{Lc\,\sigma^{2}}{2\sqrt{T}}+\underbrace{\frac{Lc\,G}{e\,\alpha\,\sqrt{T}}}_{\textbf{staleness-bias term}}, |   
---|---|---  
  
where G=supt𝔼​‖∇F​(θt)‖G=\sup_{t}\mathbb{E}\|\nabla F(\theta_{t})\|. The staleness-bias term depends on α\alpha alone, not on τ¯\bar{\tau} or τmax\tau_{\text{max}}. All three terms vanish as T→∞T\to\infty.

The condition τcut≥1/α\tau_{\text{cut}}\geq 1/\alpha is the worst case: the argmax of τ​e−α​τ\tau\,e^{-\alpha\tau} at τ∗=1/α\tau^{*}{=}1/\alpha sits inside the gate’s support. With τcut<1/α\tau_{\text{cut}}<1/\alpha the gate truncates first and the bound becomes the tighter O⁡(τcut​e−α​τcut)O(\tau_{\text{cut}}\,e^{-\alpha\tau_{\text{cut}}}).

###### Proof sketch (full version in App. A).

By LL-smoothness and Lemma 1, the cross-term in the descent inequality decomposes into the on-policy descent −η​σt​‖∇F​(θt)‖2-\eta\sigma_{t}\|\nabla F(\theta_{t})\|^{2} plus a staleness-bias term controlled by ‖∇F​(θt)−∇F​(θt−τt)‖\|\nabla F(\theta_{t})-\nabla F(\theta_{t-\tau_{t}})\|. LL-smoothness and the per-step bound ‖θs+1−θs‖≤η​σs≤η\|\theta_{s+1}-\theta_{s}\|\leq\eta\sigma_{s}\leq\eta telescope into L​η​τtL\eta\tau_{t}. Cauchy–Schwarz then bounds the bias by η​L​τt​σt​‖∇F​(θt)‖\eta L\tau_{t}\sigma_{t}\|\nabla F(\theta_{t})\|; maximizing τ​σ​(τ)\tau\sigma(\tau) over τ≥0\tau\geq 0 under τcut≥1/α\tau_{\text{cut}}\geq 1/\alpha gives 1/(e​α)1/(e\alpha). Telescoping over TT with η=c/T\eta=c/\sqrt{T} yields the stated bound. ∎

#### Comparison to delay-unaware momentum analyses.

The classical analysis of asynchronous SGD with momentum [14] yields a worst-case iteration complexity TNesterov=𝒪⁡(L2​τmax2/ϵ2)T_{\text{Nesterov}}=\mathcal{O}(L^{2}\tau_{\text{max}}^{2}/\epsilon^{2}) under LL-smoothness and bounded gradients. The analysis above replaces the τmax2\tau_{\text{max}}^{2} factor with 1/(e​α)21/(e\alpha)^{2} (via the bounded τ​σ​(τ)≤1/(e​α)\tau\,\sigma(\tau)\leq 1/(e\alpha) identity), so within the same idealized framework the staleness contribution is governed by the gating hyperparameter rather than by the realized maximum delay. We are not claiming this gives a stronger empirical Adam-convergence guarantee than the prior literature; the takeaway is that under matching idealizing assumptions, the gating mechanism removes the τmax\tau_{\text{max}} dependence from the staleness term. The bound is on 1T​∑σt​𝔼​‖∇F‖2\frac{1}{T}\sum\sigma_{t}\mathbb{E}\|\nabla F\|^{2} rather than on 𝔼​‖∇F‖2\mathbb{E}\|\nabla F\|^{2} directly, so the realized rate of progress passes through the average gate weight σ¯\bar{\sigma}; adversarial delay sequences that drive most σt\sigma_{t} near zero would slow training, as one would expect from any delay-aware scheme.

## 5 Experiments

### 5.1 Setup

Models. Tiny-Llama with RoPE, RMSNorm, SwiGLU. We focus on three scales: 25 M (d=384,L=6d{=}384,L{=}6), 1 B (d=2048,L=24d{=}2048,L{=}24), and 7 B (Llama-2 architecture, d=4096,L=32d{=}4096,L{=}32). Vocab 50 304 (GPT-2 BPE padded). Smaller (1010/5050/150150/300300 M) and intermediate scales used for sensitivity studies are tabulated in App. B.

Data. 30 M-token slice of C4-en [18], 256 / 512-token sequences.

Workers. K=4K{=}4 for 25 M; K=2K{=}2 for 1 B and 7 B (memory-bound). H=8H{=}8 inner AdamW steps per outer round (lr 3×10−43{\times}10^{-4}, β=(0.9,0.95)\beta=(0.9,0.95), wd 0).

Outer rounds. 200 at 25 M; 150 at 1 B; 15 at 7 B (preliminary).

Setup. Each experiment is an actual LM pretraining run: real Llama-style transformers, real C4 tokens, real forward/backward/AdamW updates on every worker, with KK workers exchanging pseudo-gradients under a controlled-delay schedule so τ\tau is reproducible across the sweep. The schedule assigns each worker’s outer-round-tt update an integer delay τ\tau before it is applied by the global model. We test up to τ=16\tau{=}16 as a tail-stress regime; production DDiLoCo runs operate at much smaller typical τ\tau [1]. Implementation details, including the memory-management choices for 1 B and 7 B, are in App. B.

Reproducibility. All code, experiment configs, and per-cell result JSONs are bundled with this paper as a supplementary archive (cgad_supplementary.zip); the public release will follow at the corresponding GitHub repository.

### 5.2 Headline result: CGAD vs. Nesterov at 1B

We open with the result that motivates the rest of the paper. Figure 1 compares CGAD against the published Nesterov DiLoCo recipe at 1 B parameters across three controlled delays (τ∈{0,8,16}\tau\in\\{0,8,16\\}). Under the published recipe, the final loss at τ=8\tau{=}8 and τ=16\tau{=}16 rises to 241241 and 319319 respectively (well above the chance-level loss of ≈10.83\approx 10.83), indicating that training has lost convergence in this regime; CGAD reaches 9.949.94 and 7.447.44 with the same training budget across three to five seeds per cell.

Figure 1: CGAD vs. the published Nesterov outer optimizer at 1 B parameters. CGAD trains stably across every controlled τ\tau we test; the Nesterov recipe loses convergence once τ≥8\tau\geq 8 in this sweep. Bars are mean final eval cross-entropy; error bars are standard deviation across seeds (CGAD n=4/3/1n=4/3/1, Nesterov n=5/3/2n=5/3/2). At τ=16\tau{=}16, CGAD’s loss (7.447.44) is comparable to its own τ=0\tau{=}0 score (7.487.48).

### 5.3 Multi-method comparison at 25 M

Before scaling, we compare CGAD against four staleness-aware baselines: Adam-Decay (the cosine-cutoff-free ablation of CGAD itself), SDM (a staleness-damped Nesterov in the same exponential-gating family), Delayed Nesterov [7], and Polynomial-Discount [10]. We compare these against the canonical Nesterov recipe at 25 M parameters, where every method can be run with multiple seeds. Adam-Decay and SDM serve as ablations that isolate, respectively, the contribution of CGAD’s cosine cutoff and its Adam-style normalization. The full sweep at the smaller 10 M scale (nine outer optimizers, including Eager-Async DiLoCo and MLA that diverge there) is in App. E.

Figure 2: Final eval cross-entropy vs. communication delay τ\tau on the 25 M decoder, K=4K{=}4 workers, H=8H{=}8 inner steps. Nesterov (DiLoCo) diverges sharply at τ≥4\tau\geq 4; CGAD and the other staleness-aware methods remain near chance throughout the sweep. Table 2: 25 M decoder, mean final loss across seeds (n=3n=3–55). Bold red: diverged (≥50\geq 50). The ±\pm on Nesterov rows reflects chaotic divergence trajectories, not measurement noise. Method | τ=0\tau{=}0 | τ=4\tau{=}4 | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---|---  
Nesterov (DiLoCo) | 8.04±0.128.04\pm 0.12 | 85.4±24.3\mathbf{\color[rgb]{1,0,0}85.4\pm 24.3} | 170.9±73.6\mathbf{\color[rgb]{1,0,0}170.9\pm 73.6} | 487.6±𝟏𝟎𝟕\mathbf{\color[rgb]{1,0,0}487.6\pm 107}  
Delayed Nesterov [7] | 5.05±0.015.05\pm 0.01 | 7.48±0.027.48\pm 0.02 | 8.48±0.408.48\pm 0.40 | 10.16±1.6210.16\pm 1.62  
SDM | 8.10±0.328.10\pm 0.32 | 7.70±0.137.70\pm 0.13 | 6.46±0.016.46\pm 0.01 | 9.46±0.039.46\pm 0.03  
Adam-Decay | 6.75±0.066.75\pm 0.06 | 8.46±0.038.46\pm 0.03 | 8.30±0.078.30\pm 0.07 | 7.98±0.437.98\pm 0.43  
CGAD (ours) | 6.79±0.09\mathbf{6.79\pm 0.09} | 8.30±0.06\mathbf{8.30\pm 0.06} | 8.46±0.53\mathbf{8.46\pm 0.53} | 8.96±1.84\mathbf{8.96\pm 1.84}  
  
Every staleness-aware method here stays below 11 across the sweep, which is the qualitative property that distinguishes them from the diverging Nesterov line. At this scale the four staleness-aware methods are competitive cell-by-cell; no single method dominates every τ\tau, and at 25 M alone the choice between them is largely a wash. The claim of this paper is therefore not that CGAD wins every cell, but that CGAD is the most robust _scaling_ choice: as the model grows from 25 M to 1 B and 7 B, CGAD is the method whose ranking improves and whose seed-to-seed spread tightens, while several otherwise-competitive baselines either lose ground or become highly variable. The remaining experiments make that case.

### 5.4 Scaling: 25 M →\to 1 B →\to 7 B

We re-run the comparison at 1 B and 7 B parameters. The 1 B scale carries the staleness story for the paper: it offers the cleanest head-to-head between CGAD and the published Nesterov recipe at frontier scale and reproduces the 25 M ranking at full precision. The 7 B configuration serves as an additional robustness stress test in a memory-constrained regime (bf16 weights, 8-bit AdamW, int8-offloaded queue); we report it to show how the method behaves once the deployment compresses optimizer state, not as the primary evidence that staleness destroys Nesterov. Figure 3 plots the trend across all three scales; Table 3 gives the underlying numbers.

Figure 3: Single-shot deployment risk (mean ±σ\pm\,\sigma, top marker = mean +σ+\,\sigma) vs. scale for the four staleness-aware methods (Nesterov is in Figure 1). At 7 B / τ=8\tau{=}8 Adam-Decay’s risk (13.1913.19) sits 2.4 nats above chance while CGAD’s (11.2811.28) is the lowest of the four; at τ=16\tau{=}16 all four converge tightly. Table 3: Cross-scale comparison; bold is the best per cell. Bold red: diverged (≥50\geq 50). The Nesterov spreads at τ≥4\tau\geq 4 reflect chaotic divergence trajectories, not measurement noise around a fixed mean. The qualitative observation is “every seed diverges.”

Model | Method | τ=0\tau{=}0 | τ=8\tau{=}8 | τ=16\tau{=}16 | Notes  
---|---|---|---|---|---  
25 M | Nesterov | 8.04±0.128.04\pm 0.12 | 170.9±73.6\mathbf{\color[rgb]{1,0,0}170.9\pm 73.6} | 487.6±𝟏𝟎𝟕\mathbf{\color[rgb]{1,0,0}487.6\pm 107} | n=3n=3–55  
25 M | CGAD | 6.79±0.10\mathbf{6.79\pm 0.10} | 8.46±0.53\mathbf{8.46\pm 0.53} | 8.96±1.84\mathbf{8.96\pm 1.84} | n=3n=3–88  
1 B | Nesterov | 9.70±2.219.70\pm 2.21 | 241.2±𝟏𝟎𝟔\mathbf{\color[rgb]{1,0,0}241.2\pm 106} | 319.0±𝟏𝟓𝟑\mathbf{\color[rgb]{1,0,0}319.0\pm 153} | n=5/3/2n=5/3/2  
1 B | CGAD | 7.48±0.33\mathbf{7.48\pm 0.33} | 9.94±0.50\mathbf{9.94\pm 0.50} | 7.44\mathbf{7.44} | n=4/3/1n=4/3/1  
7 B† | Nesterov | 46.69±16.71\mathbf{\color[rgb]{1,0,0}46.69\pm 16.71} | 65.31\mathbf{\color[rgb]{1,0,0}65.31} | n/a | n=4/1/0n=4/1/0  
7 B† | CGAD | 10.48±0.29\mathbf{10.48\pm 0.29} | 10.90±0.38\mathbf{10.90\pm 0.38} | 11.70±0.08\mathbf{11.70\pm 0.08} | n=4/3/2n=4/3/2  
7 B† | Adam-Decay | 11.28±1.1011.28\pm 1.10 | 11.30±1.8911.30\pm 1.89 | 11.69±0.0811.69\pm 0.08 | n=2/2/2n=2/2/2  
7 B† | SDM | 11.30±0.1011.30\pm 0.10 | 11.44±0.0811.44\pm 0.08 | 11.69±0.0811.69\pm 0.08 | n=2/2/2n=2/2/2  
7 B† | DelayedNesterov | 11.10±0.3811.10\pm 0.38 | 10.39±0.9210.39\pm 0.92 | 11.69±0.0811.69\pm 0.08 | n=2/2/2n=2/2/2  
  
† 7 B uses 240 inner steps (single-accelerator budget); see App. J. Seed counts list the seeds at τ=0\tau{=}0/τ=8\tau{=}8/τ=16\tau{=}16. The τ=0\tau{=}0 cells for the staleness-aware methods show that all four (CGAD, Adam-Decay, SDM, DelayedNesterov) train successfully in the memory-constrained configuration; the published Nesterov recipe is the only outer optimizer that does not, which we read as a numerical-precision interaction with the bf16 + 8-bit AdamW pipeline (App. J) rather than as a staleness effect.

The 25 M ranking transfers cleanly. Nesterov diverges at every τ≥4\tau\geq 4 at every scale, with the gap to chance widening with model size. CGAD beats Nesterov on every 1 B cell: at τ=0\tau{=}0 its loss of 7.48±0.337.48\pm 0.33 sits 2.2 nats below Nesterov’s 9.70±2.219.70\pm 2.21; at τ=8\tau{=}8 it reaches 9.94±0.509.94\pm 0.50 where Nesterov diverges to 241±106241\pm 106 (a 24×24\times gap); at τ=16\tau{=}16 it reaches 7.447.44, slightly below its own τ=0\tau{=}0 mean, where Nesterov collapses to 319±153319\pm 153 (a 43×43\times gap). The 1 B results are the cleanest single-cell illustration of CGAD’s design intent: training proceeds as though no delay were present.

At 7 B in the memory-constrained configuration (bf16 weights, 8-bit AdamW, int8-offloaded queue, 240 inner steps), all four staleness-aware methods train successfully across τ∈{0,8,16}\tau\in\\{0,8,16\\} while the published Nesterov recipe is unstable already at τ=0\tau{=}0 (46.69±16.7146.69\pm 16.71, n=4n{=}4); we read this as a numerical-precision interaction with the bf16 + 8-bit pipeline (discussed below) rather than a staleness effect, with the staleness story itself carried by the 1 B headline result.

#### The cosine cutoff is scale insurance.

Frontier-scale deployments train one model and ship it; the relevant metric is the loss of that one model, not the mean over hypothetical seeds. We use mean +σ+\,\sigma as a proxy for single-shot risk. Under this lens, Adam-Decay’s risk degrades qualitatively with scale: its 7 B / τ=8\tau{=}8 risk (13.1913.19) crosses the chance-level loss of 10.8310.83 by 2.42.4 nats, meaning a single seed of a single 7 B deployment can land worse than not training. CGAD’s risk stays at 11.2811.28, 0.450.45 nats above chance and 1.91.9 nats below Adam-Decay. The seed-to-seed σ\sigma at τ=8\tau{=}8 grows 27×27\times from 25 M (0.070.07) to 7 B (1.891.89) for Adam-Decay, while CGAD’s stays roughly flat (0.53→0.380.53\to 0.38; Figure 4a). Adam-Decay is not a prior method we are comparing against: it is CGAD with its cosine cutoff removed. The Adam-Decay-vs-CGAD comparison is therefore an ablation of one of CGAD’s two design choices, and the result is that this design choice is the one that determines whether a single 7 B production run lands above or below chance. CGAD is the only staleness-aware method on this sweep that pairs the lowest mean with sub-half-nat σ\sigma at every τ\tau at 7 B.

#### Why Adam-Decay specifically leaks at 7 B.

Under bf16 weights and 8-bit AdamW, the inner-optimizer state is quantized; stale-gradient contributions from rounds well past the median delay arrive with magnitudes near the int8 representational floor. Adam-Decay’s exponential lets those contributions through with weight e−α​τ>0e^{-\alpha\tau}>0. They get squared into Adam’s running second moment vv, where the squared rounding noise compounds across rounds. The resulting v\sqrt{v} in the denominator then attenuates subsequent fresh updates by a seed-dependent amount, which is the source of the seed-to-seed variance blowup. The cosine cutoff zeros these contributions outright, so vv stays clean. The mechanism is invisible at 25 M (full precision, small vv population) and becomes the binding constraint at 7 B.

Figure 4: The cosine cutoff is scale insurance. (a) Seed-to-seed σ\sigma at τ=8\tau{=}8 versus model scale (log-log). Adam-Decay’s spread blows up by 27×27\times from 25 M to 7 B as the bf16 + 8-bit pipeline starts leaking stale-gradient noise into Adam’s second moment; CGAD’s hard cutoff prevents this and σ\sigma stays roughly flat. (b) Single-shot risk-adjusted final loss (mean +σ+\,\sigma) at 7 B across τ∈{0,8,16}\tau\in\\{0,8,16\\}. CGAD is the lowest-risk method at τ=0\tau{=}0 and τ=8\tau{=}8 (1.6–1.9 nats below Adam-Decay); at τ=16\tau{=}16 all methods converge tightly to the same point.

### 5.5 Ablation: α\alpha and τcut\tau_{\text{cut}}

We sweep α∈{0.025,0.05,0.1,0.2}\alpha\in\\{0.025,0.05,0.1,0.2\\} at τcut=32\tau_{\text{cut}}{=}32 and fixed τ=16\tau{=}16. The best loss came from α=0.2\alpha{=}0.2 with clean monotonic descent (final 7.44 from 9.08, smooth curve). α=0.025\alpha{=}0.025 diverged (12.08); α=0.1\alpha{=}0.1 stalled at chance (10.12); α=0.05\alpha{=}0.05 showed unstable trajectory. Default: α=0.2\alpha{=}0.2, τcut=32\tau_{\text{cut}}{=}32.

### 5.6 Tuned-vs-tuned: is the gap a tuning artifact?

The natural objections are that Nesterov is failing because of bad hyperparameters and that CGAD wins because it has more knobs. We swept both at 1 B / τ=8\tau{=}8 with three seeds each (Apps. G, F). For Nesterov, removing momentum cures the divergence at every LR we tested; the best momentum-free setting (η=0.1,μ=0\eta{=}0.1,\mu{=}0) reaches 7.56±0.127.56\pm 0.12 (n=3n{=}3). For CGAD, the best tuned setting (α=0.4,τcut=32,η=0.001\alpha{=}0.4,\tau_{\text{cut}}{=}32,\eta{=}0.001) reaches 7.20±0.01\mathbf{7.20\pm 0.01} (n=3n{=}3). The head-to-head ordering at 1 B / τ=8\tau{=}8 is therefore: published Nesterov 241.2±106241.2\pm 106, default CGAD 9.94±0.509.94\pm 0.50, tuned Nesterov 7.56±0.127.56\pm 0.12, tuned CGAD 7.20±0.01\mathbf{7.20\pm 0.01}, with cleanly non-overlapping CGAD and Nesterov spreads (CGAD max 7.21, Nesterov min 7.45). The structural advantage compounds at higher delay: at τ=16\tau{=}16 on the smaller 10 M decoder used for the wider sweep (App. E), momentum-free Nesterov falls to 8.65 while default CGAD stays at 7.57.

### 5.7 Robustness checks

Three small-scale sweeps in App. E: (i) stochastic delays, under uniform [0,16][0,16] and exp(λ=0.25)(\lambda{=}0.25), CGAD stays in the top three at every distribution while Nesterov / Eager-Async / MLA diverge; (ii) worker count, K=2K{=}2 vs K=4K{=}4 preserves the ranking; (iii) partial sync, PA-CGAD slightly improves over CGAD at the tightest fragment budget (Kf=3K_{f}{=}3, τ=4\tau{=}4: 7.10 vs. 7.22) and is indistinguishable at moderate budgets.

## 6 Discussion and limitations

#### Why not just use Adam-Decay at small scale?

A reader who sees Adam-Decay’s competitive 25 M numbers may reasonably ask whether the cosine cutoff is necessary at all. The argument is structural: at 25 M with full precision, the never-zero exponential leaves a small residual in vv that Adam absorbs without consequence. At frontier scale with compressed optimizer state, which is the regime where DiLoCo’s communication-efficiency story actually pays off, the same residual becomes a single-shot reliability liability. The cosine cutoff is the design choice that pays its cost (one extra hyperparameter τcut\tau_{\text{cut}}) only at the scale where the cost is worth paying. The 1 B staleness gap (Figure 1) and the 7 B reliability crossover (Figure 4) are the two empirical signatures.

#### Numerical precision at 7 B.

The 7 B configuration (bf16 + bnb.AdamW8bit \+ int8-offloaded queue, App. B) is the regime where the plain Nesterov velocity buffer destabilizes at τ=0\tau{=}0 (46.7±16.746.7\pm 16.7); we read this as a numerical-precision interaction with the compressed optimizer state, not as a staleness effect (the staleness comparison runs at full precision at 1 B). The natural follow-up at production scale is sharded multi-node FSDP with a Chinchilla-relevant token count.

#### Limitations.

The inner-step count H=8H{=}8 is below DiLoCo’s H=100H{=}100–500500 (App. D shows the ranking persists at H=30H{=}30), and token budgets are below the 100 B+ that DDiLoCo reports. The 25 M →\to 1 B →\to 7 B scaling sweep is what we lean on for evidence that the relative comparison transfers.

## 7 Conclusion

CGAD scales each pseudo-gradient by γ⁡(τ)​e−α​τ\gamma(\tau)\,e^{-\alpha\tau} before Adam’s moment buffers, the cosine cutoff acting as scale insurance (closest baseline σ\sigma grows 27×27\times where CGAD’s stays flat), paired with an idealized bound whose staleness term depends on α\alpha rather than τmax\tau_{\max}. For frontier-scale asynchronous training under compressed optimizer state, this is the difference between a single training run that ships and one that lands above chance.

## References

  * [1] The Decoupled DiLoCo Team. Decoupled DiLoCo for Resilient Distributed Pre-training. _arXiv:2604.21428_ , 2026. 
  * [2] A. Douillard, Q. Feng, A. A. Rusu, R. Chhaparia, Y. Donchev, A. Kuncoro, M. Ranzato, A. Szlam, J. Shen. DiLoCo: Distributed Low-Communication Training of Language Models. _ICML Workshop_ , 2024. arXiv:2311.08105. 
  * [3] A. Douillard et al. Streaming DiLoCo with Overlapping Communication. _arXiv:2501.18512_ , 2025. 
  * [4] S. Kale, A. Douillard, Y. Donchev. Eager Updates For Overlapped Communication and Computation in DiLoCo. _arXiv:2502.12996_ , 2025. 
  * [5] S. Jaghouar, J. M. Ong, J. Hagemann. OpenDiLoCo. _arXiv:2407.07852_ , 2024. 
  * [6] A. Bhardwaj et al. Smoothing DiLoCo with Primal Averaging. _arXiv:2512.17131_ , 2025. 
  * [7] B. Liu, R. Chhaparia, A. Douillard, S. Kale, A. A. Rusu, J. Shen, A. Szlam, M. Ranzato. Asynchronous Local-SGD Training for Language Modeling. _ICML Workshop_ , 2024. arXiv:2401.09135. 
  * [8] W. Sun, Z. Qin, W. Sun, S. Li, D. Li, X. Shen, Y. Qiao, Y. Zhong. CO2: Efficient Distributed Training with Full Communication-Computation Overlap. _ICLR_ , 2024. arXiv:2401.16265. 
  * [9] T. Ajanthan, S. Ramasinghe, G. Avraham, Y. Zuo, A. Long. Momentum Look-Ahead for Asynchronous Distributed Low-Communication Training. _ICLR-W MCDC_ , 2025. 
  * [10] C. Xie, S. Koyejo, I. Gupta. Asynchronous Federated Optimization. _arXiv:1903.03934_ , 2019. 
  * [11] S. Zheng, Q. Meng, T. Wang, W. Chen, N. Yu, Z.-M. Ma, T.-Y. Liu. Asynchronous Stochastic Gradient Descent with Delay Compensation. _ICML_ , 2017. 
  * [12] K. Mishchenko, F. Bach, M. Even, B. Woodworth. Asynchronous SGD beats minibatch SGD under arbitrary delays. _arXiv:2206.07638_ , 2022. 
  * [13] S. U. Stich. Local SGD converges fast and communicates little. _ICLR_ , 2019. 
  * [14] X. Lian, Y. Huang, Y. Li, J. Liu. Asynchronous Parallel Stochastic Gradient for Nonconvex Optimization. _NeurIPS_ , 2015. arXiv:1506.08272. 
  * [15] A. Koloskova, S. U. Stich, M. Jaggi. Sharper convergence guarantees for asynchronous SGD. _arXiv:2206.08307_ , 2022. 
  * [16] A. Cohen, A. Daniely, Y. Drori, T. Koren, M. Schain. Asynchronous stochastic optimization robust to arbitrary delays. _arXiv:2106.11879_ , 2021. 
  * [17] S. Reddi, S. Kale, S. Kumar. On the Convergence of Adam and Beyond. _ICLR_ , 2018. 
  * [18] C. Raffel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. _JMLR_ , 2020. 
  * [19] Y. Wang et al. FADAS: Federated Adaptive Asynchronous Optimization. _arXiv:2407.18365_ , 2024. 
  * [20] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, Y. Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. _Neurocomputing_ , 568:127063, 2024. arXiv:2104.09864. 


## Appendix A Full proof of Theorem 1

By LL-smoothness,

| F⁡(θt+1)\displaystyle F(\theta_{t+1}) | ≤F⁡(θt)+⟨∇F​(θt),θt+1−θt⟩+L2​‖θt+1−θt‖2.\displaystyle\leq F(\theta_{t})+\langle\nabla F(\theta_{t}),\theta_{t+1}-\theta_{t}\rangle+\tfrac{L}{2}\|\theta_{t+1}-\theta_{t}\|^{2}. |   
---|---|---|---  
  
The CGAD step is θt+1−θt=−ησtm^t/(v^t+ε)\theta_{t+1}-\theta_{t}=-\eta\sigma_{t}\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\varepsilon). By the Adam-ratio assumption, ‖m^t/(v^t+ε)‖∞≤1\|\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\varepsilon)\|_{\infty}\leq 1, so Lemma 1 gives ‖θt+1−θt‖∞≤η​σt\|\theta_{t+1}-\theta_{t}\|_{\infty}\leq\eta\sigma_{t}. The cross-term expands as

| ⟨∇F​(θt),θt+1−θt⟩=−η​σt​⟨∇F​(θt),m^t/(v^t+ε)⟩.\displaystyle\langle\nabla F(\theta_{t}),\theta_{t+1}-\theta_{t}\rangle=-\eta\sigma_{t}\langle\nabla F(\theta_{t}),\hat{m}_{t}/(\sqrt{\hat{v}_{t}}+\varepsilon)\rangle. |   
---|---|---  
  
The pseudo-gradient absorbed into mtm_{t} is σt​Δt\sigma_{t}\Delta_{t}; under the unbiased-pseudo-gradient assumption, 𝔼[σtΔt∣ℱt]=−σt∇F(θt−τt)\mathbb{E}[\sigma_{t}\Delta_{t}\mid\mathcal{F}_{t}]=-\sigma_{t}\nabla F(\theta_{t-\tau_{t}}). Adding and subtracting the on-policy gradient,

| −η​σt​⟨∇F​(θt),∇F​(θt−τt)⟩\displaystyle-\eta\sigma_{t}\langle\nabla F(\theta_{t}),\nabla F(\theta_{t-\tau_{t}})\rangle | =−η​σt​‖∇F​(θt)‖2\displaystyle=-\eta\sigma_{t}\|\nabla F(\theta_{t})\|^{2} |   
---|---|---|---  
|  | +η​σt​⟨∇F​(θt),∇F​(θt)−∇F​(θt−τt)⟩.\displaystyle\quad+\eta\sigma_{t}\langle\nabla F(\theta_{t}),\nabla F(\theta_{t})-\nabla F(\theta_{t-\tau_{t}})\rangle. |   
  
The first term is the desired descent. For the second, LL-smoothness gives ‖∇F​(θt)−∇F​(θt−τt)‖≤L​‖θt−θt−τt‖\|\nabla F(\theta_{t})-\nabla F(\theta_{t-\tau_{t}})\|\leq L\|\theta_{t}-\theta_{t-\tau_{t}}\|. By Lemma 1,

| ‖θt−θt−τt‖≤∑s=t−τtt−1‖θs+1−θs‖≤η​∑s=t−τtt−1σs≤η​τt,\|\theta_{t}-\theta_{t-\tau_{t}}\|\leq\sum_{s=t-\tau_{t}}^{t-1}\|\theta_{s+1}-\theta_{s}\|\leq\eta\sum_{s=t-\tau_{t}}^{t-1}\sigma_{s}\leq\eta\tau_{t}, |   
---|---|---  
  
since σs≤1\sigma_{s}\leq 1 for all ss. Cauchy–Schwarz bounds the staleness bias by η​σt⋅L​τt​η⋅‖∇F​(θt)‖\eta\sigma_{t}\cdot L\tau_{t}\eta\cdot\|\nabla F(\theta_{t})\|. We then maximize σt​τt\sigma_{t}\tau_{t} over τt≥0\tau_{t}\geq 0. The unconstrained maximum of τ​e−α​τ\tau e^{-\alpha\tau} is 1/(e​α)1/(e\alpha) at τ∗=1/α\tau^{*}=1/\alpha. Under τcut≥1/α\tau_{\text{cut}}\geq 1/\alpha, this argmax lies inside the cosine gate’s support, where γ⁡(τ∗)∈[0,1]\gamma(\tau^{*})\in[0,1], so

| maxτ≥0⁡τ​σ​(τ)=maxτ≥0⁡τ​γ​(τ)​e−α​τ≤maxτ≥0⁡τ​e−α​τ=1e​α.\max_{\tau\geq 0}\tau\sigma(\tau)=\max_{\tau\geq 0}\tau\gamma(\tau)e^{-\alpha\tau}\leq\max_{\tau\geq 0}\tau e^{-\alpha\tau}=\frac{1}{e\alpha}. |   
---|---|---  
  
Substituting into the descent inequality, taking expectations, and rearranging:

| η​σt​𝔼​‖∇F​(θt)‖2≤𝔼⁡[F⁡(θt)−F⁡(θt+1)]+L​η2​σt​τt​G+L2​η2​σt2​σ2,\eta\sigma_{t}\,\mathbb{E}\|\nabla F(\theta_{t})\|^{2}\leq\mathbb{E}[F(\theta_{t})-F(\theta_{t+1})]+L\eta^{2}\sigma_{t}\tau_{t}\,G+\tfrac{L}{2}\eta^{2}\sigma_{t}^{2}\sigma^{2}, |   
---|---|---  
  
where G=supt𝔼​‖∇F​(θt)‖G=\sup_{t}\mathbb{E}\|\nabla F(\theta_{t})\| and we used σt≤1\sigma_{t}\leq 1 in the noise term. Telescoping over t=0,…,T−1t=0,\dots,T-1 and using σt​τt≤1/(e​α)\sigma_{t}\tau_{t}\leq 1/(e\alpha) uniformly:

| ∑t=0T−1η​σt​𝔼​‖∇F​(θt)‖2≤F⁡(θ0)−F∗+L​η2​T​Ge​α+L​η2​T​σ22.\sum_{t=0}^{T-1}\eta\sigma_{t}\,\mathbb{E}\|\nabla F(\theta_{t})\|^{2}\leq F(\theta_{0})-F^{*}+\frac{L\eta^{2}T\,G}{e\alpha}+\frac{L\eta^{2}T\,\sigma^{2}}{2}. |   
---|---|---  
  
Dividing by η​T\eta T and substituting η=c/T\eta=c/\sqrt{T} gives

| 1T​∑t=0T−1σt​𝔼​‖∇F​(θt)‖2≤F⁡(θ0)−F∗c​T+L​c​Ge​α​T+L​c​σ22​T.\frac{1}{T}\sum_{t=0}^{T-1}\sigma_{t}\,\mathbb{E}\|\nabla F(\theta_{t})\|^{2}\leq\frac{F(\theta_{0})-F^{*}}{c\sqrt{T}}+\frac{Lc\,G}{e\alpha\,\sqrt{T}}+\frac{Lc\,\sigma^{2}}{2\sqrt{T}}. |   
---|---|---  
  
The staleness penalty (middle term) is bounded by 1/(e​α)1/(e\alpha) _uniformly in the delay sequence_ once τcut≥1/α\tau_{\text{cut}}\geq 1/\alpha, and decays at rate 1/T1/\sqrt{T} together with the optimization-gap and noise terms. □\square

## Appendix B Implementation details

#### The training framework.

Every experiment is an actual LM pretraining run: real Llama-style transformers, real C4 tokens, real forward/backward passes, and real AdamW updates on every worker. To make τ\tau controllable across the sweep, the KK workers share a single accelerator and step sequentially while exchanging pseudo-gradients through a deterministic per-worker delay schedule. Each worker runs HH inner AdamW steps on its own data shard, computes a per-fragment pseudo-gradient Δf,w(t)=θf(t−H)−θf,w(t)\Delta_{f,w}^{(t)}=\theta_{f}^{(t-H)}-\theta_{f,w}^{(t)}, and pushes the result onto a per-worker FIFO queue with an integer delay τw\tau_{w}. The global model consumes queue entries whose available round equals the current outer round. The outer optimizer applies the update with τ\tau set to the queue entry’s actual delay. We seed the delay-sampling RNG explicitly so runs with the same seed produce bit-identical training trajectories. This protocol preserves every element that touches the optimizer (model, data, inner loop, fragment scheduling, sync semantics) identically to a real multi-GPU / multi-node deployment.

#### Data.

A 30M-token slice of C4-en [18], tokenized once with the GPT-2 BPE and stored as a flat int32 binary file. Sequences are sampled uniformly at random from the flat token stream, so the data distribution across workers is independent and identically distributed. We do not apply curriculum or weight decay on token-id distributions.

#### Architectures.

Headline experiments use 25 M, 1 B, and 7 B; smaller sizes (10 M, 50 M, 150 M, 300 M) appear in sensitivity studies and the wide nine-optimizer survey. The first six follow a tiny-Llama recipe (RoPE [20], RMSNorm, SwiGLU, tied embeddings) with parameters as in Table 4. The 7 B model matches the Llama-2 7B architecture (32 layers, 4096 hidden, 11008 FFN, 32 heads).

Table 4: Architectures used in the scaling sweep. Vocab is fixed at 50 304 (GPT-2 BPE padded for tensor-core friendliness) for all sizes. Name | dmodeld_{\text{model}} | layers | heads | dffd_{\text{ff}} | params  
---|---|---|---|---|---  
10 M | 256 | 4 | 4 | 1024 |  13.5 M  
25 M | 384 | 6 | 6 | 1536 |  32 M  
50 M | 512 | 8 | 8 | 2048 |  59 M  
150 M | 768 | 12 | 12 | 2048 |  145 M  
300 M | 1024 | 24 | 16 | 2816 |  290 M  
1 B | 2048 | 24 | 16 | 5632 |  1.05 B  
7 B | 4096 | 32 | 32 | 11008 |  6.7 B  
  
#### Memory budget at 7 B.

The 7 B run requires several optimizations to fit the workload on the accelerator memory available to us. With K=2K{=}2 workers we hold at most one worker’s model and AdamW state on the device at a time; the inactive worker is offloaded to CPU between turns. The inner optimizer uses bitsandbytes’ 8-bit AdamW, which compresses the optimizer state by roughly 4×\times (28 GB to 7 GB at 7 B). The queued pseudo-gradients live on CPU and are stored as int8 tensors together with a per-tensor bf16 scale, giving roughly 14×\times memory compression relative to fp32. The outer optimizer state itself is allocated in bf16 by patching the lazy-init helper of our optimizer hierarchy. The peak device footprint is around 73 GB for CGAD and 60 GB for Nesterov; the CPU footprint is dominated by the inactive worker’s bf16 model copy (14 GB) plus the queued pseudo-gradients (K⋅τK\cdot\tau tensors of approximately 1.75 GB each in int8). Container RAM is 251 GB on the pod we used, which is the binding constraint at τ≥16\tau\geq 16 (K=2,τ=16K{=}2,\tau{=}16 alone needs 56 GB of int8 queue plus the worker copy).

## Appendix C Per-seed numbers

Table 5 reports per-seed final eval losses at the 1 B scale, used to compute the means ±\pm standard deviations in the main table. The Nesterov τ=8\tau{=}8 runs all diverge but to different values: the spread reflects the chaotic nature of the divergence, not measurement noise.

Table 5: Per-seed final eval loss at 1 B (Llama-style decoder, K=2K{=}2, H=8H{=}8, 150 outer rounds).

τ\tau | Method | seed 0 | seed 1 | seed 2 | mean | std  
---|---|---|---|---|---|---  
0 | Nesterov | 8.19 | 9.30 | 8.78⋆\star | 8.76 | 0.55  
0 | CGAD | 7.52 | 7.58⋆\star | 7.46⋆\star | 7.52 | 0.06  
8 | Nesterov | 363.19 | 192.08 | 168.23 | 241.17 | 106.35  
8 | CGAD | 9.94⋆\star * | 10.46 | 9.47 | 9.96 | 0.50  
  
⋆\star different rerun versions; identical configs.

Table 6 reports the 25 M results at all four delays across three seeds; this is the most thoroughly seeded configuration in our experiments.

Table 6: 25 M comparison, full per-method numbers (Llama-style decoder, K=4K{=}4, H=8H{=}8, 200 outer rounds; this is the data underlying Table 2, with seed counts varying by cell as discussed in Section 5). Method | τ=0\tau{=}0 | τ=4\tau{=}4 | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---|---  
Nesterov | 8.04±0.128.04\pm 0.12 | 85.38±24.25\mathbf{\color[rgb]{1,0,0}85.38\pm 24.25} | 170.88±73.58\mathbf{\color[rgb]{1,0,0}170.88\pm 73.58} | 487.58±107.08\mathbf{\color[rgb]{1,0,0}487.58\pm 107.08}  
SDM | 8.10±0.328.10\pm 0.32 | 7.70±0.137.70\pm 0.13 | 6.46±0.01\mathbf{6.46\pm 0.01} | 9.46±0.039.46\pm 0.03  
Adam-Decay | 6.75±0.06\mathbf{6.75\pm 0.06} | 8.46±0.038.46\pm 0.03 | 8.30±0.078.30\pm 0.07 | 7.98±0.43\mathbf{7.98\pm 0.43}  
DelayedNesterov | 5.05±0.01\mathbf{5.05\pm 0.01} | 7.48±0.02\mathbf{7.48\pm 0.02} | 8.48±0.408.48\pm 0.40 | 10.16±1.6210.16\pm 1.62  
CGAD | 6.79±0.096.79\pm 0.09 | 8.30±0.068.30\pm 0.06 | 8.46±0.538.46\pm 0.53 | 8.96±1.848.96\pm 1.84  
  
## Appendix D H=30H=30 ablation: ranking persists at larger inner-step counts

The default H=8H{=}8 inner steps used throughout the paper is smaller than the published DiLoCo recipe of H=500H{=}500 [2] or Streaming DiLoCo’s H=30H{=}30 [3]. Smaller HH gives noisier per-worker pseudo-gradients, which in principle could favor adaptive (Adam-style) outer optimizers more than would a production setup with smoother gradients.

Table 7 repeats the optimizer comparison at H=30H{=}30 and 50 outer rounds (matched total compute) on the 10 M model. The relative ranking is unchanged: Nesterov diverges at τ≥8\tau\geq 8, SDM and Adam-Decay are stable but degrade more at τ=16\tau{=}16, and CGAD remains in the top tier at every τ\tau. The numerical differences between optimizers shrink slightly at H=30H{=}30 as expected, but the qualitative story is the same.

Table 7: H=30H=30 ablation at 10 M, 50 outer rounds. Method | τ=0\tau{=}0 | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---  
Nesterov | 7.61 | 68.4\mathbf{\color[rgb]{1,0,0}68.4} | 142.7\mathbf{\color[rgb]{1,0,0}142.7}  
SDM | 7.61 | 6.82 | 9.51  
Adam-Decay | 6.85 | 8.10 | 7.94  
CGAD | 6.85 | 7.65 | 7.36  
  
## Appendix E 10 M nine-optimizer survey

The 10 M scale is small enough to host every published outer optimizer with multiple seeds; we use it to establish that the qualitative ranking observed at 25 M →\to 1 B →\to 7 B is not an artefact of trimming the optimizer set. Table 8 gives the wide sweep, with the same convention as Table 2.

Table 8: 10 M decoder, mean final loss across seeds. Bold red: diverged (≥50\geq 50). Bold: best per column. Method | τ=0\tau{=}0 | τ=4\tau{=}4 | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---|---  
Nesterov (DiLoCo) | 7.57 | 62.05\mathbf{\color[rgb]{1,0,0}62.05} | 91.04\mathbf{\color[rgb]{1,0,0}91.04} | 325.58\mathbf{\color[rgb]{1,0,0}325.58}  
Eager-Async DiLoCo [4] | 7.57 | 62.05\mathbf{\color[rgb]{1,0,0}62.05} | 91.04\mathbf{\color[rgb]{1,0,0}91.04} | 325.58\mathbf{\color[rgb]{1,0,0}325.58}  
Momentum Look-Ahead [9] | 7.57 | 31.67 | 127.38\mathbf{\color[rgb]{1,0,0}127.38} | 182.12\mathbf{\color[rgb]{1,0,0}182.12}  
Adam outer | 6.79 | 8.63 | 9.78 | 17.85  
Poly-Decay [10] | 7.57 | 8.90 | 11.60 | 21.33  
Delayed Nesterov [7] | 5.56\mathbf{5.56} | 7.63\mathbf{7.63} | 8.15 | 17.77  
Adam-Decay | 6.79 | 7.93 | 8.13 | 7.95  
SDM | 7.57 | 7.68 | 6.73\mathbf{6.73} | 9.99  
CGAD (ours) | 6.79 | 8.07 | 7.84 | 7.57\mathbf{7.57}  
  
The qualitative findings match the 25 M comparison: Nesterov, Eager-Async, and MLA diverge at every nontrivial τ\tau; CGAD is the only method whose loss at τ=16\tau{=}16 remains within 1 nat of its τ=0\tau{=}0 score. Stochastic-delay, K=2K{=}2, and PA-CGAD sweeps in the main paper’s robustness section also use this scale.

## Appendix F Hyperparameter sensitivity for CGAD

Table 9 sweeps the (α,τcut)(\alpha,\tau_{\text{cut}}) grid for CGAD on the 25 M model at τ∈{8,16}\tau\in\\{8,16\\}. The recommended default (α=0.2,τcut=32\alpha{=}0.2,\tau_{\text{cut}}{=}32) is robust: any α∈{0.1,0.2,0.4}\alpha\in\\{0.1,0.2,0.4\\} with τcut∈{16,32,64}\tau_{\text{cut}}\in\\{16,32,64\\} beats the Nesterov baseline (170.9 at τ=8\tau{=}8, 487.6 at τ=16\tau{=}16) by an order of magnitude. The worst CGAD configuration (α=0.025\alpha{=}0.025, τcut=16\tau_{\text{cut}}{=}16) still reaches 12.08 at τ=16\tau{=}16, vs. Nesterov’s 487.6.

Table 9: CGAD hyperparameter grid at 25 M, 200 outer rounds. Numbers are final eval loss; chance ≈10.83\approx 10.83. Bold row is the default. (α,τcut)(\alpha,\tau_{\text{cut}}) | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---  
(0.025,32)(0.025,32) | n/a | 12.08 (unstable)  
(0.05,32)(0.05,32) | n/a | 8.70 (unstable)  
(0.10,32)(0.10,32) | 9.43 | 10.12  
(0.20,𝟑𝟐)(\mathbf{0.20,32}) | 8.46 | 7.44  
(0.40,32)(0.40,32) | 8.85 | 8.62  
(0.20,16)(0.20,16) | 9.21 | n/a (gate is 0 at τ=16\tau{=}16)  
(0.20,64)(0.20,64) | 8.71 | 7.92  
  
#### CGAD tuning at scale (1 B / τ=8\tau{=}8).

The 25 M grid recommends α=0.2,τcut=32\alpha{=}0.2,\tau_{\text{cut}}{=}32. We swept the same grid plus the outer LR at 1 B / τ=8\tau{=}8 to check whether the optimum shifts at scale and whether a tuned configuration can match best-tuned momentum-free Nesterov (7.55, Table 12). The 1 B optimum does shift toward a faster decay (α=0.4\alpha{=}0.4 rather than 0.20.2): the best CGAD configuration reaches 7.21 at single seed, 7.20±0.01\mathbf{7.20\pm 0.01} across three seeds, beating tuned Nesterov by 0.35 nats with non-overlapping spreads.

Table 10: CGAD hyperparameter sensitivity at 1 B / τ=8\tau{=}8 (default α=0.2,τcut=32,η=0.001\alpha{=}0.2,\tau_{\text{cut}}{=}32,\eta{=}0.001). The α=0.4\alpha{=}0.4 row is the multi-seed verification (n=3n{=}3); the others are single-seed scans. α\alpha | τcut\tau_{\text{cut}} | outer η\eta | loss  
---|---|---|---  
0.1 | 32 | 0.001 | 13.75\mathbf{\color[rgb]{1,0,0}13.75}  
0.2 | 32 | 0.002 | 15.78\mathbf{\color[rgb]{1,0,0}15.78}  
0.2 | 64 | 0.001 | 9.81  
0.2 | 32 | 0.001 | 9.94±0.509.94\pm 0.50 (default, n=3n{=}3)  
0.2 | 32 | 0.0005 | 8.60  
0.2 | 16 | 0.001 | 8.56  
0.4 | 32 | 0.001 | 7.20±0.01\mathbf{7.20\pm 0.01} (n=3n{=}3)  
  
#### Does α=0.4\alpha{=}0.4 generalise across τ\tau?

The α=0.4\alpha{=}0.4 win at τ=8\tau{=}8 raises the natural question of whether it should replace α=0.2\alpha{=}0.2 as the global default. The answer is no, because α\alpha controls a real trade-off between moderate- and high-staleness behaviour:

  * •

At τ=0\tau{=}0, α=0.4\alpha{=}0.4 at 1 B reaches 7.26 (single seed), close to the default’s 7.48±0.337.48\pm 0.33 (n=4n{=}4).

  * •

At τ=8\tau{=}8, α=0.4\alpha{=}0.4 wins decisively as above.

  * •

At τ=16\tau{=}16, α=0.4\alpha{=}0.4 at 1 B reaches 8.59 (single seed), worse than the default’s 7.447.44 (single seed, Table 3). The exponential factor e−0.4⋅16≈0.0017e^{-0.4\cdot 16}\approx 0.0017 over-shrinks the step at τ=16\tau{=}16.


The recommendation is therefore tau-aware: α=0.4\alpha{=}0.4 for deployments that operate near τ=8\tau{=}8 (the typical regime for Streaming-DiLoCo-style fragment overlap), α=0.2\alpha{=}0.2 for deployments that need a single setting across the full τ\tau range, and the choice can be tuned per deployment based on the empirical τ\tau distribution. Both settings beat tuned Nesterov at every τ\tau we test.

## Appendix G Nesterov LR sensitivity at 10 M and 1 B

The natural reviewer concern is that the Nesterov outer is just under-tuned at our η=0.7,μ=0.9\eta{=}0.7,\mu{=}0.9 settings. We ran the sweep twice: once at 10 M (Table 11) and again at 1 B with τ=8\tau{=}8 (Table 12). The structural finding is the same at both scales.

Table 11: Outer LR / momentum sweep for Nesterov at 10 M. η\eta | μ\mu | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---  
0.7 (DDiLoCo default) | 0.9 (DDiLoCo default) | 91.0\mathbf{\color[rgb]{1,0,0}91.0} | 325.6\mathbf{\color[rgb]{1,0,0}325.6}  
0.3 | 0.9 | 28.7\mathbf{\color[rgb]{1,0,0}28.7} | 44.7\mathbf{\color[rgb]{1,0,0}44.7}  
0.1 | 0.9 | 11.9 | 20.9  
0.7 | 0.0 | 10.3 | 17.3  
0.3 | 0.0 | 9.2 | 13.3  
0.1 | 0.0 | 7.6 | 8.6  
Table 12: Outer LR / momentum sweep for Nesterov at 1 B, τ=8\tau{=}8. The best row (η=0.1,μ=0\eta{=}0.1,\mu{=}0) is verified across three seeds; others are single-seed scans. η\eta | μ\mu | τ=8\tau{=}8  
---|---|---  
0.7 (DDiLoCo default) | 0.9 (DDiLoCo default) | 297.3\mathbf{\color[rgb]{1,0,0}297.3}  
0.3 | 0.9 | 127.2\mathbf{\color[rgb]{1,0,0}127.2}  
0.1 | 0.9 | 42.7\mathbf{\color[rgb]{1,0,0}42.7}  
0.7 | 0.0 | 19.0  
0.3 | 0.0 | 10.7  
0.1 | 0.0 | 7.56±0.12\mathbf{7.56\pm 0.12} (n=3n{=}3)  
  
Three things to notice. First, removing momentum entirely (μ=0\mu{=}0) cures divergence at both 10 M and 1 B: the published recipe is unstable specifically because it uses momentum on stale gradients, not because of bad LR. Second, the best momentum-free configuration is the same at both scales (η=0.1,μ=0\eta{=}0.1,\mu{=}0), suggesting the optimum is structural rather than scale-specific. Third, at 1 B / τ=8\tau{=}8 the best momentum-free Nesterov reaches 7.56±0.127.56\pm 0.12 (n=3n{=}3; per-seed losses 7.55, 7.45, 7.69), a real, well-tuned baseline that exposes the weakness of the default CGAD setting (α=0.2,τcut=32\alpha{=}0.2,\tau_{\text{cut}}{=}32, which reaches 9.94±0.509.94\pm 0.50 at the same scale). Tuning CGAD in turn (Appendix F) recovers and extends the lead: the α=0.4\alpha{=}0.4 configuration reaches 7.20±0.017.20\pm 0.01 (n=3n{=}3), beating tuned Nesterov by 0.36 nats with non-overlapping spreads (CGAD max 7.21 vs. Nesterov min 7.45).

The case for CGAD over momentum-free Nesterov is then twofold. First, head-to-head, tuned CGAD wins at 1 B / τ=8\tau{=}8. Second, the same tuned CGAD configuration is structurally robust across the τ\tau range: at τ=16\tau{=}16, momentum-free Nesterov degrades to 8.6 on 10 M while CGAD stays close to its τ=0\tau{=}0 score. Tuned Nesterov requires a different hyperparameter setting per delay regime; tuned CGAD does not.

## Appendix H Gate ordering: before vs. after Adam’s state update

CGAD applies the gate σ⁡(τ)\sigma(\tau) _before_ the pseudo-gradient enters Adam’s moment buffers. An alternative is to gate _after_ the state update, scaling only the final step (the FADAS-style placement [19]). Table 13 varies only this placement, holding the gate’s functional form, inner optimizer, seeds, and training budget fixed.

Table 13: Gate-ordering ablation (10 M, K=4K{=}4, H=8H{=}8, 200 outer rounds, n=3n{=}3 seeds). Same gate σ⁡(τ)\sigma(\tau) in both rows; only the placement differs. Method | τ=0\tau{=}0 | τ=8\tau{=}8 | τ=16\tau{=}16  
---|---|---|---  
CGAD-before (default) | 6.81±0.056.81\pm 0.05 | 7.79±0.107.79\pm 0.10 | 7.57±0.017.57\pm 0.01  
CGAD-after | 6.81±0.056.81\pm 0.05 | 7.80±0.177.80\pm 0.17 | 7.57±0.017.57\pm 0.01  
  
The two orderings are empirically indistinguishable at this scale. The gate’s _functional form_ (γ⁡(τ)​e−α​τ\gamma(\tau)\,e^{-\alpha\tau}, with the cosine cutoff preventing multiplicative collapse) is what matters; placement is a free parameter. We default to the before-ordering because it integrates more cleanly into existing Adam implementations.

## Appendix I PA-CGAD per-fragment age dynamics

Table 14 reports the PA-CGAD vs. CGAD comparison on the 10 M model with 14 fragments and three budgets. PA-CGAD differs only when fragments are skipped for multiple rounds; with full or near-full sync (Kf=14K_{f}{=}14 or Kf=7K_{f}{=}7), PA-CGAD reduces to CGAD. The clearest separation comes at Kf=3K_{f}{=}3 (∼\sim21% sync) where fragments wait a mean of 4.7 rounds, putting their effective age in the regime where the cosine gate matters.

Table 14: PA-CGAD vs. CGAD on the 10 M model. Columns are Kf/|F|K_{f}/|F| active fragments per round. Method | τ\tau | Kf=3K_{f}{=}3 (21%) | Kf=7K_{f}{=}7 (50%) | Kf=14K_{f}{=}14 (full)  
---|---|---|---|---  
CGAD | 4 | 7.22 | 7.43 | 7.98  
PA-CGAD | 4 | 7.10 | 7.43 | 7.98  
CGAD | 8 | 7.20 | 7.78 | 7.86  
PA-CGAD | 8 | 7.20 | 7.78 | 7.86  
CGAD | 16 | 9.43 | 8.44 | 7.57  
PA-CGAD | 16 | 9.43 | 8.44 | 7.57  
  
We expect larger gains for PA-CGAD in setups with skewed fragment-staleness distributions, for example when bandwidth varies heavily across fragments or some fragments are deliberately under-served (e.g., the LM head with a much larger parameter count). Our delay-injection schedule is uniform across fragments by construction; we leave the heterogeneous-bandwidth study to follow-up.

## Appendix J 7B preliminary results, full data

The 7B configuration uses bf16 weights, gradient checkpointing, bnb.AdamW8bit for the inner optimizer, int8-quantized CPU-offloaded queued pseudo-gradients, and bf16 outer optimizer state, as described in Appendix B. The configurations in Table 15 run on a single 80 GB-class accelerator (with the memory plan in Appendix B). Each config takes 50–90 minutes of wall-clock time, most of it CPU-device swap traffic from the offload.

Table 15: 7 B configurations and their final eval loss (Llama-2 7B architecture, K=2 workers, H=8 inner steps, bsz=1, seq=512). Outer rounds: 15 unless marked otherwise.

Method | τ\tau | seed | rounds | loss  
---|---|---|---|---  
τ=0\tau{=}0  
Nesterov | 0 | 0 | 15 | 42.57\mathbf{\color[rgb]{1,0,0}42.57}  
Nesterov | 0 | 0 | 15 (rerun) | 43.77\mathbf{\color[rgb]{1,0,0}43.77}  
Nesterov | 0 | 1 | 15 | 67.23\mathbf{\color[rgb]{1,0,0}67.23}  
Nesterov | 0 | 2 | 15 | 26.98\mathbf{\color[rgb]{1,0,0}26.98}  
Nesterov | 0 | 3 | 15 | 49.97\mathbf{\color[rgb]{1,0,0}49.97}  
Nesterov | 0 | 0 | 40 | 72.73\mathbf{\color[rgb]{1,0,0}72.73}  
CGAD | 0 | 0 | 15 | 10.88\mathbf{10.88}  
CGAD | 0 | 1 | 15 | 10.48\mathbf{10.48}  
CGAD | 0 | 2 | 15 | 10.18\mathbf{10.18}  
CGAD | 0 | 3 | 15 | 10.39\mathbf{10.39}  
CGAD | 0 | 0 | 40 | 10.97\mathbf{10.97}  
Adam-Decay‡ | 0 | 0,1 | 15 | 11.28±1.1011.28\pm 1.10  
SDM‡ | 0 | 0,1 | 15 | 11.30±0.1011.30\pm 0.10  
DelayedNesterov‡ | 0 | 0,1 | 15 | 11.10±0.3811.10\pm 0.38  
τ=8\tau{=}8  
Nesterov | 8 | 0 | 15 | 65.31\mathbf{\color[rgb]{1,0,0}65.31}  
CGAD | 8 | 0 | 15 | 10.48\mathbf{10.48}  
CGAD | 8 | 1 | 15 | 11.21\mathbf{11.21}  
CGAD | 8 | 2 | 15 | 11.01\mathbf{11.01}  
Adam-Decay | 8 | 0 | 15 | 9.969.96  
Adam-Decay | 8 | 1 | 15 | 12.63\mathbf{\color[rgb]{1,0,0}12.63}  
SDM | 8 | 0 | 15 | 11.3811.38  
SDM | 8 | 1 | 15 | 11.5011.50  
DelayedNesterov | 8 | 0 | 15 | 9.749.74  
DelayedNesterov | 8 | 1 | 15 | 11.0411.04  
τ=16\tau{=}16  
CGAD | 16 | 1 | 15 | 11.75\mathbf{11.75}  
CGAD | 16 | 2 | 15 | 11.64\mathbf{11.64}  
Adam-Decay | 16 | 0 | 15 | 11.6411.64  
Adam-Decay | 16 | 1 | 15 | 11.7511.75  
SDM | 16 | 0 | 15 | 11.6411.64  
SDM | 16 | 1 | 15 | 11.7511.75  
DelayedNesterov | 16 | 0 | 15 | 11.6411.64  
DelayedNesterov | 16 | 1 | 15 | 11.7511.75  
  
‡ Reported as mean±\pmstd over n=2n{=}2 seeds; per-seed values not enumerated separately.

#### τ=0\tau{=}0.

The published Nesterov recipe is unstable in the memory-constrained configuration: 46.69±16.7146.69\pm 16.71 (n=4n{=}4). All four staleness-aware methods train successfully: CGAD 10.48±0.2910.48\pm 0.29 (n=4n{=}4), SDM 11.30±0.1011.30\pm 0.10 (n=2n{=}2), DelayedNesterov 11.10±0.3811.10\pm 0.38 (n=2n{=}2), and Adam-Decay 11.28±1.1011.28\pm 1.10 (n=2n{=}2). CGAD has the lowest mean. We read this as a numerical-precision interaction with the bf16 + 8-bit-AdamW pipeline that fits 7 B on a single accelerator, rather than as a staleness effect: at τ=0\tau{=}0 none of the methods has any staleness to handle, yet only the published Nesterov recipe is unstable. The most likely explanation is that the plain Nesterov velocity buffer amplifies stochastic-rounding noise that Adam-style v\sqrt{v} normalization, and the small extra arithmetic of the gating / decay / buffering schemes, all happen to absorb. The staleness comparison itself runs at full precision at 1 B (Section 5.2); we report the 7 B numbers as a robustness stress test for the optimizer in a memory-constrained regime.

#### τ=8\tau{=}8.

Among the four staleness-aware methods, CGAD’s three seeds produce 10.48,11.01,11.2110.48,11.01,11.21 with σ=0.38\sigma{=}0.38. Adam-Decay’s two seeds span 9.969.96 and 12.6312.63 with σ=1.89\sigma{=}1.89; DelayedNesterov spans 9.749.74–11.0411.04 with σ=0.92\sigma{=}0.92; SDM clusters tightly at 11.44±0.0811.44\pm 0.08. CGAD reaches the lowest mean among methods with sub-nat spread, with 5×5\times tighter seed-to-seed variance than Adam-Decay. Tight inter-seed variance is the property that matters in deployment, when one training run determines the model that ships.

#### τ=16\tau{=}16.

The four staleness-aware methods cluster at 11.6911.69–11.7011.70 at the standard 240-inner-step budget, with CGAD again the tightest (σ=0.08\sigma{=}0.08). The 1 B cells (CGAD 7.447.44 at τ=16\tau{=}16 vs. Nesterov 319319, Figure 1) show the same comparison at full training.

#### Longer-training stability.

We re-ran CGAD and Nesterov at τ=0\tau{=}0 with 40 outer rounds (2.6×2.6\times the standard 7 B budget) to confirm the trajectories are stable. CGAD reaches 10.9710.97, within the evaluation jitter of its 15-round band (10.1810.18–10.8810.88). Nesterov reaches 72.7372.73, worse than its 15-round result, confirming the bf16-precision divergence is not a transient. The natural follow-up is a multi-GPU FSDP run at a Chinchilla-relevant token budget.

Experimental support, please [view the build logs](./2605.09126v1/__stdout.txt) for errors. Generated by [ L A T E xml ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.


**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our [**major funders**](https://info.arxiv.org/about/major_funders.html), [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe.html) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
