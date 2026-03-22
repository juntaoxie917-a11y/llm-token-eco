# 1. Model Setup: Vertical Supply + Downstream Logit Competition

## Players and Timing

There are two firms:

- **Teacher (T)**: supplies (i) a downstream product/API and (ii) upstream distillation tokens.
- **Student (S)**: trains via distillation and supplies a downstream product/API.

A single market is served in the downstream stage. The game unfolds in three stages:

1. **Upstream pricing**: Teacher sets per-token price $p \ge 0$ for distillation tokens.
2. **Student training**: Student chooses training volume $D \ge 0$.
3. **Downstream competition**: Teacher and student simultaneously set downstream prices $P_T, P_S \ge 0$. Market shares are determined by a logit demand system, and payoffs are realized.

> *Rationale.* This timing makes the key strategic channel explicit: the teacher’s upstream price affects student training scale $D$, which affects student quality and hence downstream competition.

------

# 2. Technology: Distillation Performance and Quality

## 2.1 Student performance as a function of training scale

Let the student’s realized cross-entropy loss after distillation be:
$$
L_S = L_S(N_S, D),
$$
with $N_S$ fixed and exogenous.

Assume standard scaling-law properties:

- $\frac{\partial L_S}{\partial D} < 0$ (more training tokens improves performance)
- $\frac{\partial^2 L_S}{\partial D^2} > 0$ (diminishing returns)

> *Note.* You can later specify $L_S$ via a distillation scaling law; at this stage we only need curvature/monotonicity.

## 2.2 Mapping loss to downstream “quality”

Define downstream quality as a decreasing function of loss:
$$
q_S(D) \equiv \psi(L_S(N_S,D)), \quad \psi'(\cdot) < 0.
$$
Similarly, the teacher has fixed downstream quality $q_T$ (teacher is fully pretrained):
$$
q_T \equiv \bar q_T.
$$
A convenient normalization is $\psi(L)=-L$, so that higher quality corresponds to lower loss.

> *Rationale.* Logit demand typically takes a linear-in-quality form. Using $q=-L$ makes the link between ML performance and demand transparent.

------

# 3. Student Training Cost (Upstream Tokens + Compute)

The student’s training cost has two components:

- Upstream token expenditure: $pD$
- Student-side compute cost: $kD$, with $k \ge 0$

Total training cost:
$$
C_S(D) = (p+k)D.
$$

> *Rationale.* With fixed model size $N_S$, training compute is approximately proportional to the number of processed tokens, hence linear in $D$.

------

# 4. Downstream Demand: Logit Shares with Outside Option

There is a unit mass of potential customers. Each customer chooses between:

- Teacher product $T$
- Student product $S$
- Outside option $0$

Utilities are:
$$
u_T = q_T - \tau P_T + \varepsilon_T,
\quad
u_S = q_S(D) - \tau P_S + \varepsilon_S,
\quad
u_0 = u_0 + \varepsilon_0,
$$
where:

- $\tau>0$ is price sensitivity,
- $u_0$ is the baseline utility of the outside option,
- $(\varepsilon_T,\varepsilon_S,\varepsilon_0)$ are i.i.d. Type-I extreme value shocks.

This yields logit market shares:
$$
s_T =
\frac{\exp(q_T - \tau P_T)}
{\exp(u_0) + \exp(q_T - \tau P_T) + \exp(q_S(D) - \tau P_S)},
$$

> *Key point.* This embeds an outside option mechanically, which addresses your simulation finding: without an outside option, demand may be too inelastic and monopoly pricing can become pathological.

------

# 5. Payoffs

## 5.1 Downstream profits

Assume constant marginal cost in downstream provision:

- Teacher marginal cost $m_T \ge 0$
- Student marginal cost $m_S \ge 0$

Downstream profits:
$$
\pi_T^{down} = (P_T - m_T)\, s_T,
\quad
\pi_S^{down} = (P_S - m_S)\, s_S.
$$

## 5.2 Teacher total profit (upstream + downstream)

Teacher earns upstream profit from selling tokens, plus downstream profit:
$$
\Pi_T = (p-c)D + (P_T - m_T)s_T,
$$
where $c \ge 0$ is the teacher’s marginal cost per distillation token (teacher inference cost per token, in your application).

## 5.3 Student total profit (downstream minus training cost)

$$
\Pi_S = (P_S - m_S)s_S - (p+k)D.
$$

> *Rationale.* The student incurs training cost upfront and earns downstream profit afterward.

------

# 6. Equilibrium Concept

A **Subgame Perfect Nash Equilibrium (SPNE)** consists of:

- Teacher upstream price $p^*$,
- Student training choice $D^*(p^*)$,
- Downstream prices $P_T^*(p^*), P_S^*(p^*)$,

such that:

1. Given $(p,D)$, $(P_T^*,P_S^*)$ form a Nash equilibrium of the downstream pricing subgame.
2. Given $p$, student chooses $D$ to maximize $\Pi_S$, anticipating the downstream equilibrium prices and shares.
3. Teacher chooses $p$ to maximize $\Pi_T$, anticipating the student’s $D(p)$ and downstream equilibrium.

---

# 3. Payoffs with Market Size

Let $M>0$ denote the size of the downstream market. Then downstream profits are proportional to market shares.

The teacher’s total profit is
$$
\Pi_T=(p-c)D+M(P_T-m_T)s_T,
$$
and the student’s total profit is
$$
\Pi_S=M(P_S-m_S)s_S-(p+k)D.
$$
Here:

- $p$ is the upstream per-token price set by the teacher,
- $c$ is the teacher’s marginal cost per distillation token,
- $k$ is the student’s marginal compute cost per training token,
- $P_T$ and $P_S$ are downstream prices,
- $m_T$ and $m_S$ are downstream marginal costs,
- $s_T$ and $s_S$ are logit market shares.

Introducing $M$ is useful because it separates two economically distinct scales:

1. the **training scale** $D$, measured in tokens; and
2. the **commercial scale** of downstream demand, measured by market size $M$.

This distinction will matter once we study the teacher’s incentive to distort upstream pricing in order to affect downstream competition.

------

# 4. Step A: Downstream Pricing Subgame

We now solve the downstream pricing game in Stage 3, taking the upstream price $p$ and the student’s training choice $D$ as given.

At this stage, student training expenditure and teacher token revenue are sunk. Hence the firms choose downstream prices $P_T$ and $P_S$ to maximize downstream operating profits only.

The logit shares are
$$
s_T=
\frac{\exp(q_T-\tau P_T)}
{\exp(u_0)+\exp(q_T-\tau P_T)+\exp(q_S(D)-\tau P_S)},
$$
The outside-option share is
$$
s_0=
\frac{\exp(u_0)}
{\exp(u_0)+\exp(q_T-\tau P_T)+\exp(q_S(D)-\tau P_S)}.
$$
The downstream profit functions are therefore
$$
\pi_T^{down}=M(P_T-m_T)s_T,
\qquad
\pi_S^{down}=M(P_S-m_S)s_S.
$$
Since $M$ is multiplicative, it does not affect the downstream first-order conditions. It only scales equilibrium profits.

------

## 4.1. Demand Derivatives under Logit Shares

For the multinomial logit system, the own-price derivative of firm $i$'s share is
$$
\frac{\partial s_i}{\partial P_i}
=
-\tau s_i(1-s_i),
\qquad i\in\{T,S\}.
$$
The cross-price derivative is
$$
\frac{\partial s_i}{\partial P_j}
=
\tau s_i s_j,
\qquad i\neq j.
$$
These are standard logit demand properties.

------

## 4.2. Teacher’s Downstream Pricing Problem

Given $D$ and $P_S$, the teacher solves
$$
\max_{P_T\ge 0}\; M(P_T-m_T)s_T.
$$
The first-order condition is
$$
\frac{\partial \pi_T^{down}}{\partial P_T}
=
M\left[
s_T+(P_T-m_T)\frac{\partial s_T}{\partial P_T}
\right]
=0.
$$
Substituting the logit derivative yields
$$
s_T-(P_T-m_T)\tau s_T(1-s_T)=0.
$$
Since $s_T>0$, this simplifies to
$$
1-\tau (P_T-m_T)(1-s_T)=0,
$$
or equivalently,
$$
\boxed{
P_T-m_T=\frac{1}{\tau(1-s_T)}.
}
\tag{T-FOC}
$$

------

## 4.3. Student’s Downstream Pricing Problem

Similarly, given $D$ and $P_T$, the student solves
$$
\max_{P_S\ge 0}\; M(P_S-m_S)s_S.
$$
The first-order condition is
$$
\frac{\partial \pi_S^{down}}{\partial P_S}
=
M\left[
s_S+(P_S-m_S)\frac{\partial s_S}{\partial P_S}
\right]
=0.
$$
Substituting the logit derivative gives
$$
s_S-(P_S-m_S)\tau s_S(1-s_S)=0,
$$
which simplifies to
$$
\boxed{
P_S-m_S=\frac{1}{\tau(1-s_S)}.
}
\tag{S-FOC}
$$

------

# 5. Equilibrium Markup System

The downstream pricing equilibrium is therefore characterized by the two markup equations
$$
\boxed{
P_T^*=m_T+\frac{1}{\tau(1-s_T^*)},
\qquad
P_S^*=m_S+\frac{1}{\tau(1-s_S^*)},
}
\tag{1}
$$
together with the logit share equations
$$
s_T^*=
\frac{\exp(q_T-\tau P_T^*)}
{\exp(u_0)+\exp(q_T-\tau P_T^*)+\exp(q_S(D)-\tau P_S^*)},
$$
This is a standard fixed-point system. For each given $D$, it determines the downstream equilibrium prices and shares:
$$
P_T^*(D),\quad P_S^*(D),\quad s_T^*(D),\quad s_S^*(D).
$$

------

# 6. Economic Interpretation of the Markup Formula

Equation (1) shows that each firm charges a markup over marginal cost that depends on its equilibrium market share.

In particular,
$$
P_i^*-m_i=\frac{1}{\tau(1-s_i^*)},
\qquad i\in\{T,S\}.
$$
This implies:

1. **Markup increases with market share.**
   A firm with a larger customer base faces less elastic residual demand and therefore charges a higher price.
2. **The outside option disciplines pricing.**
   Since $s_i<1$, the markup remains finite. This is precisely why the logit structure avoids the pathological pricing outcome that may arise in models without an outside option.
3. **Student training affects downstream competition through quality.**
   Since $q_S=q_S(D)$, a larger training scale $D$ improves student quality, which tends to raise $s_S^*$, lower $s_T^*$, and thereby reshape downstream markups and profits.

These observations are central for the next stage of the analysis.

------

# 7. Reduced-Form Downstream Profits

Given the downstream equilibrium, define reduced-form operating profits as
$$
\pi_T^{down*}(D)
=
M\big(P_T^*(D)-m_T\big)s_T^*(D),
$$
These reduced-form profits inherit their dependence on $D$ entirely through student quality $q_S(D)$.

This allows us to write the student’s Stage-2 problem as
$$
\max_{D\ge 0}\;
\Pi_S(D)
=
\pi_S^{down*}(D)-(p+k)D,
$$
and the teacher’s Stage-1 problem as
$$
\max_{p\ge 0}\;
\Pi_T(p)
=
(p-c)D(p)+\pi_T^{down*}(D(p)).
$$
Thus, once the downstream subgame is solved, the entire model collapses into a sequential problem in $D$ and $p$.

---

# Revised Step B: Reduced-Form Comparative Statics of the Downstream Subgame

We now study how the downstream equilibrium changes when the student’s training scale $D$ changes.

Recall that student quality is
$$
q_S(D)=\psi(L_S(N_S,D)),
\qquad \psi'(\cdot)<0,
$$
and, by the scaling-law assumption,
$$
\frac{dq_S}{dD}>0.
$$
Hence a higher training scale $D$ improves the student’s downstream quality.

The teacher’s downstream quality $q_T$ is fixed.

For each given $D$, consider the downstream pricing game between the teacher and the student. Let
$$
\pi_T^{down}(P_T,P_S;q_S)
= M(P_T-m_T)s_T(P_T,P_S;q_S),
$$
where $q_S=q_S(D)$.

We impose the following regularity assumption.

------

## Assumption B1 (Well-defined downstream equilibrium)

For every $q_S$ in the relevant range, the downstream pricing subgame admits a unique interior Nash equilibrium
$$
\bigl(P_T^*(q_S),\,P_S^*(q_S)\bigr),
$$
and the equilibrium is continuously differentiable in $q_S$.

Accordingly, define the equilibrium downstream profit functions
$$
\pi_T^{down*}(q_S)
:=
\pi_T^{down}\!\bigl(P_T^*(q_S),P_S^*(q_S);q_S\bigr),
$$
By composition with $q_S(D)$, we may also write
$$
\pi_i^{down*}(D)=\pi_i^{down*}(q_S(D)),
\qquad i\in\{T,S\}.
$$

------

## 9.1 Direct demand-shifting effect

Holding downstream prices fixed, an increase in student quality shifts demand toward the student.

From the logit system,
$$
s_S=
\frac{\exp(q_S-\tau P_S)}
{\exp(u_0)+\exp(q_T-\tau P_T)+\exp(q_S-\tau P_S)},
$$
so
$$
\frac{\partial s_S}{\partial q_S}
=
s_S(1-s_S)>0.
$$
Likewise,
$$
\frac{\partial s_T}{\partial q_S}
=
-s_T s_S<0.
$$
Therefore, at fixed prices, a quality improvement benefits the student and harms the teacher through the demand system.

This is the primitive competitive force in the model.

------

## 9.2 Why the usual envelope shortcut is not sufficient here

A crucial point is that the downstream outcome is a **Nash equilibrium**, not a single-agent optimum. Therefore, when $q_S$ changes, both equilibrium prices may adjust:
$$
P_T^*=P_T^*(q_S),\qquad P_S^*=P_S^*(q_S).
$$
As a result, one cannot in general write
$$
\frac{d\pi_S^{down*}}{dq_S}
=
M(P_S^*-m_S)\frac{\partial s_S}{\partial q_S},
$$
or
$$
\frac{d\pi_T^{down*}}{dq_S}
=
M(P_T^*-m_T)\frac{\partial s_T}{\partial q_S},
$$
because such formulas ignore the effect of the opponent’s equilibrium price adjustment.

The correct total derivative is obtained by the chain rule.

For the student,
$$
\frac{d\pi_S^{down*}}{dq_S}
=
\frac{\partial \pi_S^{down}}{\partial q_S}
+
\frac{\partial \pi_S^{down}}{\partial P_T}\frac{dP_T^*}{dq_S}
+
\frac{\partial \pi_S^{down}}{\partial P_S}\frac{dP_S^*}{dq_S}.
$$
At the student’s own optimum, the first-order condition implies
$$
\frac{\partial \pi_S^{down}}{\partial P_S}=0,
$$
so
$$
\boxed{
\frac{d\pi_S^{down*}}{dq_S}
=
\frac{\partial \pi_S^{down}}{\partial q_S}
+
\frac{\partial \pi_S^{down}}{\partial P_T}\frac{dP_T^*}{dq_S}.
}
\tag{B.1}
$$
Similarly, for the teacher,
$$
\frac{d\pi_T^{down*}}{dq_S}
=
\frac{\partial \pi_T^{down}}{\partial q_S}
+
\frac{\partial \pi_T^{down}}{\partial P_T}\frac{dP_T^*}{dq_S}
+
\frac{\partial \pi_T^{down}}{\partial P_S}\frac{dP_S^*}{dq_S},
$$
and the teacher’s own first-order condition implies
$$
\frac{\partial \pi_T^{down}}{\partial P_T}=0,
$$
hence
$$
\boxed{
\frac{d\pi_T^{down*}}{dq_S}
=
\frac{\partial \pi_T^{down}}{\partial q_S}
+
\frac{\partial \pi_T^{down}}{\partial P_S}\frac{dP_S^*}{dq_S}.
}
\tag{B.2}
$$
These expressions are the correct Nash-equilibrium analogues of the envelope formula.

------

## 9.3 Sign structure of the direct terms

The direct terms are immediate from the logit system.

For the student,
$$
\frac{\partial \pi_S^{down}}{\partial q_S}
=
M(P_S-m_S)\frac{\partial s_S}{\partial q_S}
=
M(P_S-m_S)s_S(1-s_S)>0,
$$
evaluated at the equilibrium.

For the teacher,
$$
\frac{\partial \pi_T^{down}}{\partial q_S}
=
M(P_T-m_T)\frac{\partial s_T}{\partial q_S}
=
-\,M(P_T-m_T)s_Ts_S<0,
$$
evaluated at the equilibrium.

Thus, **holding prices fixed**, a higher student quality strictly raises the student’s downstream profit and strictly lowers the teacher’s downstream profit.

------

## 9.4 Reduced-form monotonicity under strategic-substitutability conditions

To obtain sign results for the **full equilibrium** derivatives in (B.1)–(B.2), we need additional assumptions on downstream price reactions.

A convenient sufficient condition is the following.

------

## Assumption B2 (Regular reaction of downstream equilibrium)

In the downstream pricing game:

1. the student’s equilibrium price weakly increases with its own quality,
   $$
   \frac{dP_S^*}{dq_S}\ge 0;
   $$

2. the cross-price effect of the student’s price on the teacher’s profit is weakly positive,
   $$
   \frac{\partial \pi_T^{down}}{\partial P_S}\ge 0;
   $$

3. the cross-price effect of the teacher’s price on the student’s profit is weakly positive,
   $$
   \frac{\partial \pi_S^{down}}{\partial P_T}\ge 0.
   $$

These inequalities are natural in differentiated-product price competition: when the student becomes more attractive, it can typically sustain a higher downstream price, and a rival’s higher price weakly benefits the other firm.

Under Assumptions B1–B2, equation (B.2) implies
$$
\frac{d\pi_T^{down*}}{dq_S}
=
\underbrace{\frac{\partial \pi_T^{down}}{\partial q_S}}_{<0}
+
\underbrace{\frac{\partial \pi_T^{down}}{\partial P_S}}_{\ge 0}
\underbrace{\frac{dP_S^*}{dq_S}}_{\ge 0}.
$$
Hence the total sign is, in general, ambiguous unless the direct demand-loss effect dominates the strategic price-adjustment effect.

Accordingly, we state the teacher-side monotonicity as an assumption or sufficient-condition result, rather than as an unconditional theorem.

------

## Assumption B3 (Teacher profit decreases with student quality)

The teacher’s reduced-form downstream equilibrium profit is decreasing in student quality:
$$
\frac{d\pi_T^{down*}}{dq_S}<0.
$$
Economically, this requires that the teacher’s direct loss of market share from a better student dominate any offsetting gain created by equilibrium price adjustments.

This assumption is natural in the present application and can be verified numerically for the calibrated parameter region used in the simulation section.

Under Assumption B3 and $\frac{dq_S}{dD}>0$, we obtain
$$
\boxed{
\frac{d\pi_T^{down*}}{dD}
=
\frac{d\pi_T^{down*}}{dq_S}\frac{dq_S}{dD}
<0.
}
\tag{B.3}
$$
Likewise, if one additionally assumes that the student’s reduced-form downstream equilibrium profit is increasing in its own quality,
$$
\frac{d\pi_S^{down*}}{dq_S}>0,
$$
then
$$
\boxed{
\frac{d\pi_S^{down*}}{dD}
=
\frac{d\pi_S^{down*}}{dq_S}\frac{dq_S}{dD}
>0.
}
\tag{B.4}
$$

------

## 9.5 Interpretation

Step B establishes the economically relevant reduced-form property needed for the upper stages of the game:

- more student training raises student quality;
- higher student quality improves the student’s downstream position;
- under the maintained monotonicity condition, higher student quality reduces the teacher’s downstream equilibrium profit.

Therefore, training by the student intensifies downstream competition, which is the key force behind the teacher’s incentive to distort the upstream token price.

---

# Revised Step C: The Student’s Training Choice

We now turn to Stage 2. Given the upstream token price $p$, the student chooses training scale $D$, anticipating the downstream pricing equilibrium.

From Step B, for each $D$ the downstream subgame induces reduced-form equilibrium profit
$$
\pi_S^{down*}(D)
=
\pi_S^{down*}(q_S(D)).
$$
Therefore the student solves
$$
\max_{D\ge 0}\;
\Pi_S(D;p)
=
\pi_S^{down*}(D)-(p+k)D.
\tag{C.1}
$$
This formulation makes clear that the student trades off the downstream benefit of more training against the marginal training cost $p+k$.

------

## 10.1 Regularity assumptions for the student problem

To characterize the student’s optimal training choice, we impose the following standard conditions.

### Assumption C1 (Regularity of the student’s reduced-form problem)

The reduced-form downstream profit function $\pi_S^{down*}(D)$ is continuously differentiable, and:

1. it is strictly increasing in $D$,
   $$
   \frac{d\pi_S^{down*}(D)}{dD}>0;
   $$

2. it is strictly concave in $D$,
   $$
   \frac{d^2\pi_S^{down*}(D)}{dD^2}<0;
   $$

3. its marginal benefit crosses the constant marginal training cost:
   $$
   \lim_{D\to 0^+}\frac{d\pi_S^{down*}(D)}{dD}>p+k,
   \qquad
   \lim_{D\to\infty}\frac{d\pi_S^{down*}(D)}{dD}<p+k.
   $$

Condition (1) says that more training improves the student’s downstream position.
 Condition (2) imposes diminishing returns to training.
 Condition (3) ensures that the optimum is interior rather than at a boundary.

These are reduced-form assumptions on the composed object $\pi_S^{down*}(D)$. In the present model they are economically natural and can also be checked numerically in the calibrated region used in the simulations.

------

## 10.2 Existence and uniqueness of the student’s optimal training level

Under Assumption C1, the student’s objective $\Pi_S(D;p)$ is strictly concave in $D$. Therefore, for every given upstream price $p$, the student has a unique optimal training choice $D^*(p)\in(0,\infty)$.

The first-order condition is
$$
\frac{d\Pi_S(D;p)}{dD}
=
\frac{d\pi_S^{down*}(D)}{dD}-(p+k)=0,
$$
or equivalently,
$$
\boxed{
\frac{d\pi_S^{down*}(D^*(p))}{dD}=p+k.
}
\tag{C.2}
$$
Equation (C.2) states that the student chooses training so that marginal downstream benefit equals marginal training cost.

Because the objective is strictly concave, this first-order condition is both necessary and sufficient.

------

## 10.3 Comparative statics with respect to the upstream token price

We next study how the student’s optimal training level responds to the upstream price $p$.

Differentiating (C.2) with respect to $p$ yields
$$
\frac{d^2\pi_S^{down*}(D^*(p))}{dD^2}\frac{dD^*(p)}{dp}=1.
$$
Hence
$$
\boxed{
\frac{dD^*(p)}{dp}
=
\frac{1}{\frac{d^2\pi_S^{down*}(D^*(p))}{dD^2}}
<0,
}
\tag{C.3}
$$
where the inequality follows from strict concavity:
$$
\frac{d^2\pi_S^{down*}}{dD^2}<0.
$$
Thus a higher upstream token price reduces the student’s equilibrium training scale.

This result is central to the vertical-control mechanism: by charging a higher token price, the teacher can discourage student training.

------

## 10.4 Interpretation

Step C delivers the student-side transmission channel of the model.

A higher token price raises the student’s marginal cost of training. Since the reduced-form gain from training is concave, the student optimally responds by lowering $D$. Therefore the upstream price acts as a control variable that shapes the student’s downstream competitiveness through training.

------

# Revised Step D: The Teacher’s Upstream Pricing Problem

We now return to Stage 1. The teacher chooses the upstream token price $p$, anticipating both the student’s training response $D^*(p)$ and the downstream pricing equilibrium induced by that training level.

The teacher’s total profit can be written as
$$
\Pi_T(p)
=
(p-c)D^*(p)+\pi_T^{down*}(D^*(p)).
\tag{D.1}
$$
The first term is upstream token profit. The second term is the teacher’s reduced-form downstream equilibrium profit.

------

## 11.1 First-order condition

Assume for the moment that the teacher’s optimum is interior. Differentiating (D.1) with respect to $p$, we obtain
$$
\frac{d\Pi_T}{dp}
=
D^*(p)
+
(p-c)\frac{dD^*(p)}{dp}
+
\frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}.
$$
Hence the teacher’s first-order condition is
$$
\boxed{
D^*(p)
+
\left[(p-c)+\frac{d\pi_T^{down*}}{dD}\right]\frac{dD^*(p)}{dp}
=0.
}
\tag{D.2}
$$
This is the fundamental pricing condition of the model.

------

## 11.2 Interpretation of the teacher’s first-order condition

Equation (D.2) shows that the teacher faces two effects when increasing $p$.

First, there is the standard **direct upstream gain**:
$$
D^*(p),
$$
which reflects higher revenue on infra-marginal token sales.

Second, there is an **indirect quantity effect** through the student’s response:
$$
\frac{dD^*(p)}{dp}<0.
$$
This affects teacher profit through two channels:

1. a standard upstream quantity-loss term,
   $$
   (p-c)\frac{dD^*(p)}{dp};
   $$

2. a downstream strategic term,
   $$
   \frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}.
   $$

Under Step C,
$$
\frac{dD^*(p)}{dp}<0.
$$
If, in addition, Assumption B3 from Step B holds, namely
$$
\frac{d\pi_T^{down*}}{dD}<0,
$$
then
$$
\frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}>0.
\tag{D.3}
$$
Therefore, when the teacher dislikes student training because it strengthens a downstream rival, a higher upstream price yields an additional strategic benefit: it reduces $D$, which softens downstream competition.

This is the formal manifestation of the **raising-rivals’-cost effect** in the present model.

------

## 11.3 Comparison with the pure upstream monopoly benchmark

It is useful to compare the competition model with a benchmark in which the teacher sells tokens upstream but does **not** compete with the student downstream.

In that benchmark, the teacher solves
$$
\max_{p\ge 0}\;(p-c)D^*(p),
$$
so the interior first-order condition is
$$
D^*(p)+(p-c)\frac{dD^*(p)}{dp}=0.
\tag{D.4}
$$
By contrast, in the competition model the first-order condition is (D.2), which includes the additional term
$$
\frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}.
$$
Under Assumption B3 and Step C, this term is positive. Hence downstream competition adds an extra incentive for the teacher to raise the upstream token price.

At the level of marginal incentives, the competition model therefore pushes the teacher toward more aggressive upstream pricing than the pure-upstream benchmark.

This observation is economically robust. A formal comparison of the optimal prices, however, requires an additional monotonicity condition on the teacher’s objective, as stated below.

------

## Proposition 3 (Competition can strengthen the teacher’s incentive to raise the token price)

Let $p^M$ denote the teacher’s optimal upstream token price in the benchmark model without downstream competition, and let $p^C$ denote the corresponding optimal price in the competition model.

Suppose that:

1. the student’s best-response training level satisfies
   $$
   \frac{dD^*(p)}{dp}<0;
   $$

2. the teacher’s reduced-form downstream profit is decreasing in student training,
   $$
   \frac{d\pi_T^{down*}}{dD}<0;
   $$

3. the teacher’s objective in each model is single-peaked in $p$, or more generally the first-order conditions satisfy the standard monotonicity property ensuring that an upward shift in marginal payoff translates into a weakly higher maximizer.

Then
$$
\boxed{
p^C\ge p^M.
}
\tag{D.5}
$$

### Proof

In the benchmark model, the teacher’s first-order condition is
$$
D^*(p)+(p-c)\frac{dD^*(p)}{dp}=0.
$$
In the competition model, the first-order condition is
$$
D^*(p)+(p-c)\frac{dD^*(p)}{dp}
+
\frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}=0.
$$
By assumptions (1) and (2),
$$
\frac{d\pi_T^{down*}}{dD}\frac{dD^*(p)}{dp}>0.
$$
Hence, relative to the benchmark, the competition model contains an additional positive term in the teacher’s marginal incentive to raise $p$. Assumption (3) ensures that this upward shift in marginal payoff implies a weakly higher optimal upstream price. Therefore
$$
p^C\ge p^M.
$$
$\square$

### Intuition

Without downstream competition, the teacher raises the upstream price only to extract token revenue. With downstream competition, the teacher also values a higher upstream price because it discourages student training and thereby weakens a downstream rival. This strategic consideration tends to increase the teacher’s optimal token price.

------

## 11.4 Existence of an optimal upstream price

We next discuss existence.

Under Step C, the best-response function $D^*(p)$ is well defined and continuous. Under Assumption B1, the reduced-form downstream profit $\pi_T^{down*}(D)$ is continuous in $D$. Therefore the teacher’s objective
$$
\Pi_T(p)=(p-c)D^*(p)+\pi_T^{down*}(D^*(p))
$$
is continuous in $p$.

To ensure that an optimum exists on $[0,\infty)$, it is sufficient to rule out divergence of profits as $p\to\infty$. A convenient sufficient condition is:

### Assumption D1 (Boundedness at high prices)

As $p\to\infty$,

1. student training demand vanishes sufficiently fast,
   $$
   D^*(p)\to 0
   \quad\text{and}\quad
   (p-c)D^*(p)
   \text{ remains bounded or tends to }0;
   $$

2. downstream profit remains bounded,
   $$
   \sup_D |\pi_T^{down*}(D)|<\infty.
   $$

Under Assumption D1, $\Pi_T(p)$ attains a maximum on $[0,\infty)$. Therefore:

### Proposition 4 (Existence of an optimal upstream price)

Under Assumptions B1, C1, and D1, the teacher’s upstream pricing problem admits at least one optimal solution
$$
p^*\in[0,\infty).
$$

------

## 11.5 Uniqueness and curvature

Uniqueness of the teacher’s optimum generally requires stronger curvature assumptions.

Differentiating (D.2) again introduces the curvature of both:

1. student demand,
   $$
   D^{*\prime\prime}(p),
   $$

2. the teacher’s reduced-form downstream equilibrium profit,
   $$
   \frac{d^2\pi_T^{down*}}{dD^2}.
   $$

A sufficient condition for uniqueness is that the teacher’s objective $\Pi_T(p)$ be strictly concave in $p$. This can be ensured, for example, if:

- the best-response demand $D^*(p)$ is sufficiently regular and not too convex;
- the strategic downstream benefit from reducing $D$ is itself diminishing.

At the present stage, however, existence and the sign structure of the teacher’s incentives are more important than a full global uniqueness proof.

------

# Revised Section: The Teacher’s Distortion Relative to the Student’s Private Incentives

It is useful to compare the student’s and teacher’s incentives with respect to training scale $D$.

From Step C, the student chooses $D$ to satisfy
$$
\frac{d\pi_S^{down*}(D)}{dD}=p+k.
\tag{D.6}
$$
Thus the student equates the private marginal downstream gain from training with the marginal training cost.

The teacher, by contrast, does not choose $D$ directly. Instead, the teacher chooses $p$, which indirectly affects $D$, taking into account not only upstream token revenue but also the fact that lower student training may raise the teacher’s downstream equilibrium profit.

Under Assumption B3, the teacher dislikes increases in student training because they intensify downstream competition. Therefore the teacher internalizes a force that the student does not internalize when choosing $D$: the negative effect of student training on the teacher’s downstream profit.

This creates a wedge between:

1. the student’s privately optimal training level given $p$, and
2. the training level induced by the teacher’s strategically chosen upstream price.

That wedge is the central distortion generated by vertical control in the model.

------

# Revised Section: Welfare Interpretation

Although a full welfare analysis is beyond the present scope, the model already suggests a natural inefficiency.

A higher upstream token price has three effects:

1. it transfers surplus from the student to the teacher through upstream payments;
2. it reduces student training and hence lowers student quality;
3. it softens downstream competition.

Accordingly, when the teacher uses upstream pricing strategically against a downstream rival, the equilibrium is likely to feature **underinvestment in student training** relative to a benchmark in which the upstream supplier does not internalize the competitive harm caused by student quality improvement.

This observation can be formalized in a later welfare section or extension.

------

# Revised Summary of Step D

We have now characterized the teacher’s upstream pricing problem in the competition model.

The main results are as follows.

1. The teacher solves
   $$
   \max_{p\ge 0}\;
   \Pi_T(p)
   =
   (p-c)D^*(p)+\pi_T^{down*}(D^*(p)).
   $$

2. If the optimum is interior, the first-order condition is
   $$
   D^*(p)
   +
   \left[(p-c)+\frac{d\pi_T^{down*}}{dD}\right]\frac{dD^*(p)}{dp}
   =0.
   $$

3. Since
   $$
   \frac{dD^*(p)}{dp}<0,
   $$
   and, under Assumption B3,
   $$
   \frac{d\pi_T^{down*}}{dD}<0,
   $$
   the downstream strategic effect of a higher upstream token price is positive.

4. Hence downstream competition adds an extra incentive for the teacher to raise the upstream token price relative to a model without downstream competition; under additional monotonicity conditions, this implies
   $$
   p^C\ge p^M.
   $$

This is the core economic mechanism of the model: the teacher uses upstream pricing not only as a revenue-extraction device, but also as a strategic instrument to weaken a downstream rival.