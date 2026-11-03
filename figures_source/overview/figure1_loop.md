---
title: Figure 1 (Loop) — Online Test-Time Feedback Loop
section: sec-setup.tex (Problem Setup)
latex_label: fig:test_time_feedback
builder: figures_source/overview/figure1_loop_build.py
outputs:
  - figures_source/overview/figure1_loop.pdf
  - figures/figure1_loop.pdf
---

# Figure description: minimal online feedback loop

This figure should make the Section 2 loop obvious before the reader reaches
the design-space details. The visual job is not to enumerate all Section 3
knobs. It should show only the high-level loop:

1. A current generator supports steered generation.
2. Steering shifts the sampling distribution and outputs selected molecules for
   oracle evaluation.
3. The oracle returns raw scores under a finite budget.
4. Feedback is split into reward shaping and regularization tracks.
5. The tracks compile into a training objective.
6. Fine-tuning on that loss updates the generator used in the next round.

## Notation shown in the figure

Keep notation sparse:

- $p_{\theta_t}$: current generator.
- $S_t$: selected oracle batch of size $K$.
- $r(x)$: raw oracle score.
- $\tilde r(x)$: shaped training reward.
- $\mathcal{D}_t$: evaluated history.
- $B$: total oracle budget.

Avoid local Section 3 notation in the figure, such as Thompson-sampling
variables, CVaR thresholds, debiasing coefficients, replay fractions, or
gradient-step counts.

## Visual constraints

- Minimal two-region loop with one return arrow.
- No arrows through text or labels.
- Use enough spacing between blocks that the figure reads when scaled to
  NeurIPS single-column width.
- Keep colors restrained: one color for sampling/acquisition, one for
  feedback/update, one muted accent for the oracle.
- The two high-level regions should be legible:
  "Steered generation" and "Feedback-driven generator update".

## Caption intent

The caption should explain that oracle feedback has two roles: it selects the
current evaluated batch and supplies training data that changes the generator
for later rounds.
