# Hypothesis-driven experimental program for learned goals

The experiments should isolate **goal selection** from navigation skill. A clean implementation would first pretrain an agent to navigate to either of two targets when given an explicit goal identifier. The navigation policy can then be frozen or mostly frozen, and SFT or RL is used only to learn which proxy signal determines that goal identifier.

The same hypothesis can be tested at three levels:

1. a one-step contextual choice task;
2. a fork gridworld with one meaningful left/right decision;
3. the full navigation environment.

The one-step task gives the cleanest causal evidence. The fork gridworld tests sequential behavior without introducing much navigation difficulty. The full environment tests whether the result survives in a richer setting.

Let $Y\in\{-1,+1\}$ denote the intended goal, such as the left or right target. Let $P_1,\ldots,P_m$ be proxy signals that may correlate with $Y$ during training.

For each trained policy $\pi$, the main evaluation should use OOD states in which the proxies disagree. Define the behavioral reliance on proxy $j$ as

$$
\rho_j(\pi)
=
\Pr_{x\sim\mathcal{D}_{\mathrm{conflict}}}
\left[
\arg\max_a\pi(a\mid x)
=
g_j(x)
\right],
$$

where $g_j(x)$ is the action recommended by proxy $j$.

Training reward is not sufficient to identify the learned goal. The primary outcomes should therefore be:

* conflict-state goal choice;
* the effect of intervening on each proxy;
* navigation capability conditional on the selected goal;
* and the evolution of these quantities throughout training.

For controlled signal complexity, the cleanest primary family is a degree-$k$ interaction code. Sample independent signs

$$
R_1,\ldots,R_{k-1}
\sim
\operatorname{Uniform}\bigl(\{-1,+1\}\bigr),
\qquad k\geq 2,
$$

and define

$$
R_k
=
Y\prod_{i=1}^{k-1}R_i,
\qquad
\prod_{i=1}^{0}R_i:=1.
$$

Then $Y=\prod_{i=1}^{k}R_i$.

Decoding $Y$ requires a degree-$k$ interaction, while every strict subset of the channels contains no information about $Y$. This provides a cleaner complexity ladder than continuous polynomial channels that might accidentally reveal the action through a single-channel threshold.

# Hypothesis 1: The model learns the simplest sufficiently accurate goal

## Hypothesis

When several signals predict reward, a finite-capacity model will tend to rely on the simplest signal that achieves sufficiently good training performance, rather than necessarily learning the most accurate available signal.

Suppose the model receives:

* a simple but imperfect proxy $P$ with $\Pr(P=Y)=q$;

* and a perfectly accurate but more complex degree-$k$ signal $Y=\prod_{i=1}^{k}R_i$.

The prediction is that the simple proxy will dominate when it is accurate enough and the exact signal is sufficiently difficult to decode.

Qualitatively,

$$
\text{high }q
+
\text{high }k
+
\text{low capacity}
\quad
\Longrightarrow
\quad
\text{simple proxy reliance}.
$$

Conversely,

$$
\text{low }q
+
\text{low }k
+
\text{high capacity}
\quad
\Longrightarrow
\quad
\text{intended-goal reliance}.
$$

## Intuition

A signal must provide both **predictive value** and an **accessible computation**.

The simple proxy can improve behavior as soon as the model learns a low-complexity mapping such as $A=P$.

The exact signal requires the model to implement

$$
A
=
R_1R_2\cdots R_k.
$$

Even though this signal is perfectly accurate, its accuracy is irrelevant if the architecture or update cannot efficiently extract it.

A useful heuristic is that the learner prefers signals with high training improvement per unit of effective complexity:

$$
\text{learning value of signal }j
\approx
\frac{
\text{reduction in training loss from }j
}{
\text{effective complexity of using }j
}.
$$

If $P$ is correct on $95\%$ of training examples, the complex signal can improve behavior only on the remaining $5\%$. That residual improvement may not be sufficient to drive the additional representation learning needed to decode the degree-$k$ signal.

“Acceptable accuracy” should not be assumed to be a universal threshold. It should emerge from the interaction of $q$, $k$, training duration, model capacity, update capacity, and the cost of proxy mistakes.

## Experiment design

Give every model the same two sources: $P$ and $(R_1,\ldots,R_k)$.

Sweep:

$$
q
\in
\{0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0\},
$$

$$
k
\in
\{1,2,3,4,5\},
$$

and model or update capacity

$$
B
\in
\{B_1,\ldots,B_m\}.
$$

Train the model in environments where $P$ has accuracy $q$. Evaluate on a balanced conflict set satisfying $P=-Y$.

Measure:

$$
\rho_P(\pi),
\qquad
\rho_Y(\pi),
$$

and the causal effect of independently flipping $P$ or the degree-$k$ channels.

The main result should be a phase diagram:

$$
(q,k,B)
\longmapsto
\rho_Y-\rho_P.
$$

### Predicted result

The boundary between proxy reliance and intended-goal reliance should shift systematically:

* increasing $q$ should favor the simple proxy;
* increasing $k$ should favor the simple proxy;
* increasing model or update capacity should favor the exact signal;
* increasing training time should favor the exact signal only if the proxy makes enough errors to generate continued learning pressure.

### Evidence against the hypothesis

The hypothesis would be weakened if:

* reliance on the proxy is unrelated to $q$;
* reliance on the exact signal is unrelated to independently measured decoding complexity;
* or models consistently learn the exact signal even when the simple proxy is nearly perfect and the exact decoder is near the edge of their capacity.

# Hypothesis 2: Goal complexity is relative to the model architecture

## Hypothesis

There is no architecture-independent ordering of goal complexity. What a model is capable of learning depends on the representation and update operations available to that particular architecture.

A goal $g$ may be easy for one architecture and difficult for another, even when both architectures have the same number of parameters.

Define the architecture-relative complexity of a strategy as

$$
C_{\mathcal{M},\epsilon}(g)
=
\min
\left\{
B:
\exists\theta
\text{ with update budget }B
\text{ such that }
\operatorname{Err}(\pi_\theta,g)\leq\epsilon
\right\}.
$$

The model should only reliably use goal $g$ when its available capacity exceeds this threshold:

$$
B
\gtrsim
C_{\mathcal{M},\epsilon}(g).
$$

## Intuition

Polynomial degree, number of channels, or program description length are properties of the signal construction. They do not by themselves tell us how difficult the signal is for an MLP.

A shallow-wide network, deep-narrow network, residual network, or network with a different activation may implement the same interaction with very different parameter and optimization costs.

The relevant question is therefore not:

$$
\text{How mathematically complex is this signal?}
$$

but:

$$
\begin{aligned}
&\text{How difficult is this signal for this policy architecture,}\\
&\text{from this initialization, under this update rule?}
\end{aligned}
$$

This also separates two notions of capacity:

* **model capacity**: everything the network could represent;
* **update capacity**: what post-training is allowed to change.

A large pretrained navigator may contain all required capabilities while a tiny goal-selection update is sufficient to redirect them. A small model trained from scratch must simultaneously represent navigation, signal decoding, and goal selection.

## Experiment design

First run a standalone calibration experiment.

For every signal $g_j$ and architecture $\mathcal{M}$, train a supervised decoder $\widehat{Y}=D_\theta(Z_j)$.

Sweep:

* depth;
* width;
* activation;
* residual versus non-residual architecture;
* full training versus final-layer-only training;
* adapter rank or number of trainable scalars.

Measure $C_{\mathrm{param}}(g_j)$, the minimum total parameter count reaching a target accuracy; $C_{\mathrm{update}}(g_j)$, the minimum trainable update budget; and $C_{\mathrm{steps}}(g_j)$, the number of optimization steps needed.

Next, place the same signals in competition inside the goal-selection environment. Test whether the proxy that controls behavior is predicted by the standalone calibration.

For RL actor-capacity experiments, keep the critic architecture fixed and sufficiently large. Otherwise a low-capacity critic can confound the result by producing poor advantage estimates.

### Predicted result

The intended signal should begin controlling behavior near the independently measured decoding threshold:

$$
B
\approx
C_{\mathcal{M},\epsilon}(g).
$$

Matched-parameter architectures may show different thresholds. For example, a deeper model may use a high-order interaction at a lower total parameter count, while a shallow model may rely on the simple proxy.

### Evidence against the hypothesis

The hypothesis would be weakened if:

* standalone decoder complexity does not predict goal reliance;
* architecture changes strongly affect decoder accuracy but not behavior;
* or agents reliably use signals that their corresponding standalone decoders cannot generalize from.

# Hypothesis 3: Models learn goals in an easy-to-hard sequence

## Hypothesis

During training, a model will often learn a simple imperfect proxy before learning a more complex accurate goal.

A typical trajectory may be

$$
\text{no useful strategy}
\rightarrow
\text{simple proxy}
\rightarrow
\text{complex intended goal}.
$$

The final goal therefore may not describe the model’s behavior throughout training.

## Intuition

A simple proxy can generate a useful gradient immediately. If $P$ is directly observed and positively correlated with $Y$, the model can reduce loss by assigning it a single large weight.

A degree-$k$ exact signal requires coordinated learning across several channels. Its usefulness may not become available until lower-level representations have formed.

Once the simple proxy has been learned, the remaining loss is concentrated on conflict examples $\{x:P(x)\neq Y(x)\}$.

If these examples occur with probability $\alpha$, then only approximately $N\alpha$ out of $N$ training examples distinguish the intended goal from the proxy.

The model can then follow one of two paths:

1. **replacement**: the exact goal eventually displaces the proxy;
2. **lock-in**: the proxy reduces loss enough that the remaining signal is too weak to induce the more complex computation.

This predicts that identical final training accuracies may hide very different learning histories and OOD behaviors.

## Experiment design

Train models with:

* one simple proxy $P$;
* one degree-$k$ exact signal;
* fixed proxy reliability $q$;
* no curriculum.

Save checkpoints at logarithmically spaced times:

$$
t
\in
\{1,2,4,8,16,\ldots,T\}.
$$

At every checkpoint, measure:

$$
\rho_P(t),
\qquad
\rho_Y(t),
\qquad
J_{\mathrm{train}}(t),
\qquad
J_{\mathrm{OOD}}(t).
$$

Define the acquisition time of goal $j$ as

$$
T_j^{\mathrm{acquire}}
=
\min
\left\{
t:\rho_j(t)\geq\tau
\right\}.
$$

Define the replacement time as

$$
T_{P\rightarrow Y}^{\mathrm{replace}}
=
\min
\left\{
t:\rho_Y(t)>\rho_P(t)
\right\}.
$$

Run the experiment across $q$, $k$, $B$, and training-set size.

### Predicted result

For intermediate regimes, the simple proxy should become behaviorally active before the exact signal:

$$
T_P^{\mathrm{acquire}}
<
T_Y^{\mathrm{acquire}}.
$$

Increasing complexity $k$ should delay or prevent replacement. Increasing the frequency of conflict examples should accelerate it.

Training reward may plateau before goal identity stabilizes:

$$
J_{\mathrm{train}}(t_1)
\approx
J_{\mathrm{train}}(t_2),
$$

while

$$
\rho_P(t_1)
\gg
\rho_P(t_2).
$$

### Evidence against the hypothesis

The simple-first account would be weakened if complex exact signals are consistently acquired before simpler proxies, or if acquisition order is unrelated to independently measured signal learnability.

# Hypothesis 4: Distinguishing examples, not ordinary training examples, identify the goal

## Hypothesis

The amount of data relevant to goal identification is not the total number of training examples. It is the number and diversity of examples on which candidate goals disagree.

Suppose the intended goal $g^\star$ and proxy $p$ disagree with probability $\alpha=\Pr[g^\star(x)\neq p(x)]$. Then only approximately $N_{\mathrm{conflict}}\approx N\alpha$ training examples provide direct evidence distinguishing them.

The intended goal should be learned only when the effective distinguishing evidence exceeds a threshold that grows with its complexity.

## Intuition

On agreement examples, both goals make the same prediction. Such examples teach the model how to behave, but they do not identify why that behavior is correct.

If the model sees one million examples on which $g^\star(x)=p(x)$, those examples may provide no more evidence about goal identity than one such example.

Conflict examples are different: $g^\star(x)\neq p(x)$.

They reveal which candidate rule is actually preferred by the label or reward.

But raw conflict count is also insufficient. Repeating one conflict state can encourage memorization. Generalizing the intended rule may require conflicts spread across the dimensions along which the proxy can fail.

The relevant quantity is therefore closer to the number of distinct, representative conflict contexts than merely total data volume.

## Experiment design

Construct datasets with the same:

* total number of examples $N$;
* proxy accuracy $q$;
* and number of conflict examples $N_{\mathrm{conflict}}$.

Vary only conflict diversity.

### Condition A: concentrated conflict

All disagreements occur in a small number of repeated states.

### Condition B: diverse conflict

Disagreements are distributed across many target locations, map geometries, and nuisance values.

### Condition C: structured holdout

Training conflicts cover some types of disagreement, while evaluation uses a new type.

Sweep $N_{\mathrm{conflict}}$, the number of unique conflict states $U_{\mathrm{conflict}}$, and intended-signal complexity $k$.

Test on unseen conflict states.

### Predicted result

OOD intended-goal reliance should depend more strongly on $U_{\mathrm{conflict}}$ than on repeated presentations of the same conflict.

A threshold-like result may appear:

$$
N_{\mathrm{conflict}}
\cdot
\text{conflict diversity}
\gtrsim
\text{goal-complexity-dependent threshold}.
$$

### Evidence against the hypothesis

The hypothesis would be weakened if agreement examples alone cause reliable intended-goal selection, or if repeated conflict examples generalize just as well as equally numerous diverse conflicts.

# Hypothesis 5: RL has an update-capacity advantage only when demonstrations contain substantial nuisance information

## Hypothesis

RL will require less update capacity than SFT when demonstrations contain high-entropy details that predict the demonstrated sequence but are irrelevant to reward.

This advantage should shrink, disappear, or reverse when SFT receives minimal, clean action labels.

The prediction is therefore conditional:

$$
U_{\min}^{\mathrm{RL}}
<
U_{\min}^{\mathrm{SFT}}
$$

primarily when:

* update capacity is strongly constrained;
* reward is reliable;
* demonstrations contain many reward-irrelevant choices;
* and training repeatedly resamples those choices.

## Intuition

The *Learning to Reason in 13 Parameters* paper proposes that SFT must model many bits in a demonstration, only some of which matter for task performance. In its account of RL, freshly sampled trajectories contain large amounts of entropy, but reward cleanly separates reward-correlated structure from irrelevant variation: correlated effects accumulate while uncorrelated effects cancel. The authors explicitly present this as a hypothesis for why RL performs better under extremely small trainable updates, and report a substantial empirical advantage for RL over SFT at tiny update sizes on their reasoning tasks. ([arXiv][1])

The direct analogue in a gridworld is not ordinary action-label SFT. If SFT receives the correct action at every state, it has a dense and highly informative signal. It may learn more efficiently than terminal-reward RL.

The closer analogue is **trajectory imitation**. There may be many equally rewarded ways to reach a goal:

* different shortest paths;
* neutral detours;
* arbitrary choices in symmetric regions;
* different timing or stylistic action sequences.

SFT is trained to reproduce the particular sampled trajectory. RL is trained to reproduce the successful outcome.

Under a tiny update budget, SFT may use capacity to encode incidental trajectory structure, while RL only needs to alter the policy in directions consistently associated with reward.

## Experiment design

Start from the same pretrained navigation agent and restrict goal-learning updates to $U\in\{1,4,16,64,256,1024,\text{full}\}$ trainable scalars or adapter parameters.

Compare four methods.

### 1. Clean action SFT

Train on minimal pairs $(o_t,a_t^\star)$.

There is one canonical intended action for each labeled state.

### 2. Nuisance-rich trajectory SFT

Train on full successful trajectories containing randomized reward-equivalent choices.

Control the nuisance entropy $H_{\mathrm{nuisance}}$.

### 3. On-policy imitation

Label states visited by the current agent. This controls for the off-policy state-distribution difference between standard SFT and RL.

### 4. RL

Train with terminal or verifiable reward.

Measure the minimum update budget required to reach $\rho_Y\geq 0.9$ on OOD conflict tests.

Report results against:

* environment interactions;
* labeled actions;
* optimizer steps;
* wall-clock compute;
* and update size.

### Predicted result

With clean action labels, SFT may match or outperform RL.

As $H_{\mathrm{nuisance}}$ increases, the RL update-capacity curve should degrade more slowly than trajectory SFT.

The strongest form of the result would be:

$$
\frac{
U_{\min}^{\mathrm{SFT}}
}{
U_{\min}^{\mathrm{RL}}
}
$$

increasing with demonstration nuisance entropy.

### Evidence against the hypothesis

The proposed signal-separation account would be weakened if:

* nuisance-rich and clean SFT require the same update size;
* RL has the same advantage when SFT labels are minimal and deterministic;
* or on-policy imitation fully reproduces RL’s advantage, suggesting that data distribution rather than reward filtering explains the result.

# Hypothesis 6: Independent noise cancels, but persistent noise becomes a learnable proxy

## Hypothesis

The effect of noise depends more on its temporal structure than on its marginal variance.

Independent, zero-mean noise should average away with repeated data. State-static or systematically biased noise should not. Instead, persistent noise can become a stable proxy that the model learns.

## Intuition

Let each sample contain useful signal $\mu$ and independent noise $\epsilon_i$ with $\mathbb{E}[\epsilon_i]=0$.

Across $N$ samples,

$$
\sum_{i=1}^{N}\mu
=
N\mu,
$$

while typically

$$
\sum_{i=1}^{N}\epsilon_i
=
O_p(\sqrt{N}).
$$

Thus the signal-to-noise ratio grows approximately as $\sqrt{N}$.

Now consider state-static noise $\epsilon_s$ that is reused every time state $s$ appears. After $N_s$ visits, its contribution is $N_s\epsilon_s$, which scales linearly rather than as $\sqrt{N_s}$.

The learner has no way to know that $\epsilon_s$ was intended to be “noise.” It is a stable feature of the environment.

Similarly, biased noise satisfying $\mathbb{E}[\epsilon\mid Y]\neq 0$ is not merely noise. It is a proxy.

This predicts that two perturbations with the same variance can have radically different effects depending on whether they are resampled, episode-static, state-static, or correlated with the target.

## Experiment design

Hold the clean proxy and marginal noise variance fixed. Compare:

1. **step-resampled noise**

$$
\widetilde Z_t
=
Z+\epsilon_t;
$$

2. **episode-static noise**

$$
\widetilde Z_t
=
Z+\epsilon_{\mathrm{episode}};
$$

3. **state-static noise**

$$
\widetilde Z(s)
=
Z(s)+\epsilon_s;
$$

4. **biased noise**

$$
\mathbb{E}[\epsilon\mid Y]\neq 0.
$$

Run each under:

* clean SFT;
* nuisance-rich SFT;
* on-policy imitation;
* RL.

Also separately perturb:

* proxy observations;
* SFT labels;
* RL rewards.

These are not equivalent interventions.

### Predicted result

For observation noise with matched variance:

$$
\text{step-resampled}
<
\text{episode-static}
<
\text{state-static}
$$

in its tendency to produce stable OOD proxy reliance.

Independent zero-mean noise should diminish with more samples for both SFT and RL. RL should not receive a unique advantage merely from averaging.

Biased or state-static noise may be learned by either algorithm whenever it predicts the respective training objective.

### Evidence against the hypothesis

The hypothesis would be weakened if matched-variance resampled and state-static noise have indistinguishable effects, or if increasing sample count fails to improve robustness to independent zero-mean noise.

# Hypothesis 7: Counterevidence unlearns a goal more effectively than absence or decorrelation

## Hypothesis

Removing an old proxy from the input is not the same as unlearning it. Making the proxy uninformative is also weaker than making it systematically wrong.

The expected ordering is:

$$
\text{reversal}
>
\text{decorrelation}
>
\text{removal}
$$

in the speed with which prior proxy reliance is erased.

## Intuition

Consider a simple proxy weight $w_P$ in a linearized goal selector. Its preferred value is related to the covariance between the proxy and the intended action: $w_P^\star\propto\operatorname{Cov}(P,Y)$.

During initial training, $\operatorname{Cov}(P,Y)>0$, so the model develops positive reliance on $P$.

If the proxy becomes neutral, $\operatorname{Cov}(P,Y)=0$, the new optimum removes this reliance, but the corrective signal may be weak.

If the proxy becomes anti-correlated, $\operatorname{Cov}(P,Y)<0$, the learner receives consistent evidence that proxy-following is harmful. The required update is no longer merely “ignore $P$,” but “treat $P$ as evidence for the opposite action.”

If the proxy channel is removed entirely, the parameters that previously processed it may receive little or no task gradient. Behavior can appear corrected because the proxy is unavailable, while the old mapping remains dormant.

## Experiment design

Use a two-phase or three-phase protocol.

### Phase A: install a proxy

Train with

$$
\Pr(P=Y)=q_A,
\qquad
q_A\in\{0.9,0.99,1.0\}.
$$

Continue until proxy reliance reaches a fixed level, $\rho_P\geq 0.9$.

### Phase B: unlearning intervention

Compare four conditions.

#### Removal

The proxy channel disappears.

#### Decorrelation

The proxy remains present but $\Pr(P=Y)=0.5$.

#### Reversal

The proxy becomes anti-correlated: $\Pr(P=Y)<0.5$.

#### Replacement

A new proxy becomes reliable while the old one becomes unreliable.

Measure reliance after every update.

Define reliance half-life:

$$
T_{1/2}
=
\min
\left\{
t:
\rho_P(t)
\leq
\frac{1}{2}\rho_P(0)
\right\}.
$$

After Phase B, reintroduce the original proxy in a balanced conflict environment to test whether it was erased or merely suppressed.

Disable weight decay in the core comparison, then add it as an ablation. Otherwise unused parameters may decay even without counterevidence.

### Predicted result

The expected ordering is

$$
T_{1/2}^{\mathrm{reversal}}
<
T_{1/2}^{\mathrm{decorrelation}}
<
T_{1/2}^{\mathrm{removal}}.
$$

Models trained under removal may immediately resume old behavior when the proxy is restored.

Reversal should produce the least rebound because it actively changes the mapping between the proxy and action.

### Evidence against the hypothesis

The hypothesis would be weakened if removal erases proxy reliance as thoroughly as reversal, or if anti-correlated evidence does not accelerate unlearning relative to neutral data.

# Hypothesis 8: Learned goals exhibit hysteresis and rebound

## Hypothesis

Once a model has strongly learned one goal, later training may suppress that goal without fully erasing it. Under a subsequent perturbation, the model may return to the old goal more quickly than it originally learned it.

This is a goal-level hysteresis prediction: behavior depends on training history, not only the current data distribution.

## Intuition

Suppose the model first learns goal $g_0$, then is trained to follow goal $g_1$.

Even if behavior after the second phase follows $g_1$, the parameters may remain close to a representation that implements $g_0$. The conditional complexity of restoring the old goal may therefore be small:

$$
C(g_0\mid\theta_{\mathrm{after}\ g_1})
<
C(g_0\mid\theta_{\mathrm{random}}).
$$

The old goal is not being learned from scratch. It is being reactivated.

The *Language Models Resist Alignment* paper develops a compression model in which changes in normalized compression rates under perturbation depend inversely on dataset sizes, conditional on its Pareto-distribution assumptions. The paper also reports behavioral rebound under reverse fine-tuning and stronger rebound with larger pretraining volumes and model sizes. These results motivate a toy analogue here, but they do not establish that the same scaling must hold for MLP goal selectors. ([arXiv][2])

## Experiment design

Use three training stages.

### Stage 0: old-goal pretraining

Train the model to follow proxy goal $g_0$.

Sweep old-goal training volume:

$$
N_0
\in
\{N_0^{(1)},\ldots,N_0^{(m)}\}.
$$

### Stage 1: intended-goal alignment

Train every model on the same quantity $N_1$ of evidence supporting $g_1$.

Continue until models reach comparable behavioral reliance on $g_1$, with $\rho_{g_1}\geq 0.9$.

### Stage 2: perturbation

Apply:

* neutral data;
* weakly conflicting data;
* removal of the intended signal;
* or partial reversal.

Measure whether behavior returns toward $g_0$.

Compare against a control model that had never learned $g_0$.

Define rebound:

$$
R_{g_0}
=
\rho_{g_0}^{\text{after perturbation}}
-
\rho_{g_0}^{\text{before perturbation}}.
$$

Define reactivation speed as the number of examples needed to restore fixed reliance on $g_0$, denoted $T_{\mathrm{reactivate}}(g_0)$.

Compare this with learning from scratch:

$$
T_{\mathrm{reactivate}}(g_0)
\quad\text{versus}\quad
T_{\mathrm{scratch}}(g_0).
$$

Sweep model capacity as well as $N_0$.

### Predicted result

Models previously trained on $g_0$ should satisfy

$$
T_{\mathrm{reactivate}}(g_0)
<
T_{\mathrm{scratch}}(g_0).
$$

Increasing $N_0$ should increase the asymmetry, at least until the old goal is fully represented.

Larger models may exhibit stronger rebound because they can retain both old and new goal computations rather than overwriting one with the other.

### Evidence against the hypothesis

The hysteresis account would be weakened if training history has no effect after controlling current behavior, or if old-goal reactivation takes as much data as learning the goal from scratch.

# Hypothesis 9: High-capacity models may learn several goals and a selector rather than one goal

## Hypothesis

As capacity increases, the model may stop choosing between proxy strategies and instead learn multiple strategies together with a context-dependent selector.

A high-capacity policy may implement

$$
\pi(a\mid x,C)
=
\begin{cases}
\pi_{g_0}(a\mid x), & C=0,\\
\pi_{g_1}(a\mid x), & C=1.
\end{cases}
$$

The question “which goal did the model learn?” may therefore have a categorical answer at low capacity and a conditional answer at high capacity.

## Intuition

A low-capacity model may be forced to compress behavior into one rule:

$$
g_0
\quad\text{or}\quad
g_1.
$$

A higher-capacity model can represent:

* both goal computations;
* a feature that identifies the context;
* and a gate that selects between them.

This can improve in-distribution performance while making OOD behavior harder to predict. When the context is ambiguous or removed, the model must fall back to one strategy. That fallback may reveal:

* the simpler goal;
* the earlier-learned goal;
* the more strongly pretrained goal;
* or a learned default branch.

Thus increased model capacity may reduce straightforward proxy misgeneralization while increasing latent behavioral multiplicity.

## Experiment design

Construct two contexts, $C\in\{0,1\}$.

In context $C=0$, proxy $P_0$ is correct.

In context $C=1$, proxy $P_1$ is correct.

Provide a context feature during training.

Sweep model capacity and update capacity.

Evaluate in four OOD conditions:

1. normal context;
2. context feature removed;
3. context feature randomized;
4. context says $0$ while reward structure matches context $1$.

Use causal interventions to determine whether the policy stores both proxy rules.

A representation-free behavioral test is to check whether changing only $C$ reverses which proxy controls the action:

$$
\rho_{P_0}(\pi\mid C=0)
\gg
\rho_{P_0}(\pi\mid C=1).
$$

### Predicted result

Low-capacity models should tend to select one globally useful proxy.

Higher-capacity models should show stronger conditional switching, with $\rho_{P_C}(\pi\mid C)\approx 1$.

When context is removed, the high-capacity model’s fallback may reveal historical or simplicity bias even though it performed perfectly during training.

### Evidence against the hypothesis

The hypothesis would be weakened if increasing capacity never increases context-sensitive goal selection, or if all capacities converge to a single proxy despite strong benefits from learning both.

# Recommended experimental order

The hypotheses should not all be tested simultaneously.

## Stage 1: validate the signal and model complexity manipulations

Run Hypothesis 2 first. Establish that the degree-$k$ family produces measurable architecture-relative decoding thresholds.

## Stage 2: map goal selection

Run Hypotheses 1 and 4. Produce the central phase diagrams over:

$$
\text{proxy accuracy}
\times
\text{signal complexity}
\times
\text{capacity}
\times
\text{conflict evidence}.
$$

## Stage 3: study training dynamics

Use the same conditions to test Hypothesis 3 with dense checkpointing.

## Stage 4: compare learning algorithms and noise regimes

Run Hypotheses 5 and 6, beginning with clean SFT, nuisance-rich SFT, on-policy imitation, and RL.

## Stage 5: study persistence

Use agents with clearly identified learned goals to test Hypotheses 7 and 8.

## Stage 6: test conditional multiplicity

Run Hypothesis 9 after single-goal reliance can already be measured reliably.

The resulting project would make several distinct, falsifiable claims rather than relying on one overarching theory:

$$
\boxed{
\begin{aligned}
&\text{Which goal is easiest to learn?}\\
&\text{Which goals are accessible to which architectures?}\\
&\text{In what order are goals acquired?}\\
&\text{What training evidence distinguishes them?}\\
&\text{When does RL require less update capacity than SFT?}\\
&\text{Which kinds of noise cancel?}\\
&\text{How are goals unlearned or reactivated?}
\end{aligned}
}
$$

Together, these experiments would provide a controlled empirical map from training conditions to learned behavioral objectives.

[1]: https://arxiv.org/html/2602.04118v1 "Learning to Reason in 13 Parameters"
[2]: https://arxiv.org/html/2406.06144 "Language Models Resist Alignment: Evidence From Data Compression"
