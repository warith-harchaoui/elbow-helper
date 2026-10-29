# Landscape

[🇫🇷](PAYSAGE.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](LANDSCAPE.md)

How `elbow-helper` compares to *every other way of finding a knee*. Each approach is rated on **this project's job: reporting a knee only when the evidence supports it, and saying so explicitly otherwise**, not penalised for excelling at a different job (raw changepoint detection, exploratory visualization, statistical inference in R).

## At a glance

<!-- TABLE:START -->
| Elbow Detection Tool | Noise Robustness | Automatic Shape Inference | Explicit Abstention | Multiple Breakpoints | Statistical Significance Testing | Uncertainty Quantification | Model-Selection Rigor | Dependency Footprint | One-Call Ease of Use | Reproducibility | Published Math/Derivation |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **elbow-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| kneed | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| ruptures | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| kneebow | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| KElbowVisualizer | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| R segmented package | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Manual eyeballing | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| Ask an LLM | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/elbow-helper/main/assets/landscape.png)

The map is a 2-D summary of the eleven criteria, so read it as a shape, not a scoreboard. `elbow-helper` is at the top-right corner. The axes read **Horizontal: Nimble ↔ Precise** and **Vertical: Thorough ↔ Adaptable**.
<!-- FIGURE:END -->

## Positioning

`elbow-helper` does not compete with `ruptures` on multi-breakpoint search or with a statistician's judgement on a well-understood dataset. It answers a narrower, harder question: given one curve and no other context, is any candidate knee strong enough to trust? Most of the field either always returns a point estimate (`kneed`, `kneebow`, Yellowbrick's `KElbowVisualizer`) or asks a human to supply the judgement `elbow-helper` automates (manual eyeballing, an LLM prompt). Its closest analogue is not another knee locator but R's `segmented` package, which shares the same instinct: a knee claim should come with a standard error, not just a coordinate.

## Per-tool write-up

### kneed
The reference implementation `elbow-helper`'s own locator is ported from, credited in the Acknowledgements section of the README. Excellent at the one thing it does: given a curve and an explicit `curve`/`direction`, it returns a single point, deterministically, in one line of code. It carries no notion of confidence and no path to "there is no knee here" beyond an unhelpful `None`. `elbow-helper` wraps this exact geometric idea in the confirmation machinery kneed leaves to the caller.

### ruptures
The right tool when the question is genuinely "how many breakpoints and where" over a signal, with PELT, binary segmentation, and window-based search all built in. It is agnostic to what a "knee" even means, so a diminishing-returns curve and a mean-shift in noise are the same kind of object to it. `elbow-helper`'s own multi-knee research (`research/multiknee/`) tests the same PELT-adjacent dynamic program `ruptures` uses, adding the model-selection layer (mBIC, FWER) `ruptures` leaves to the user to configure.

### kneebow
A small package built around the same rotation idea `elbow-helper`'s locator uses: turn the curve until it points downward, then take the extremum. Geometric, fast, and short enough to read in one sitting. Like kneed, it commits to an answer on every call, with no smoothing-scale search, no bootstrap and no null-hypothesis check behind the number it returns. Its install is heavier than its size suggests, too: the published package declares numpy, matplotlib and scikit-learn as runtime dependencies, so it brings in the plotting and machine-learning stacks that `elbow-helper` deliberately does without.

### Yellowbrick KElbowVisualizer
The most popular way practitioners actually find k-means's elbow: it fits the estimator itself across a range of k, scores and plots the curve, then — with `locate_elbow=True`, its default — marks the bend using the same Kneedle construction `kneed` implements. So calling it a plot undersells it: it is a locator with a chart around it, and it inherits kneed's commit-on-every-call behaviour along with the geometry. What it is *not* is a general detector for a curve you already hold: it owns the k-sweep, and scores it with one of `distortion`, `silhouette` or `calinski_harabasz` rather than accepting arbitrary `(x, y)`. It also pulls scikit-learn and matplotlib as hard dependencies. `elbow-helper`'s own k-means worked example (`doc/ELBOW-en.tex`, Part I) targets exactly this curve, with the confirmation chain a Kneedle read never runs.

### R segmented package
The most statistically serious alternative here: proper broken-line regression, with standard errors, Davies' test, and multi-breakpoint support (`psi`), in a mature, peer-reviewed R package that Vito M. R. Muggeo has maintained and published against since 2003. What it asks of the caller is real statistical fluency, a model formula, starting breakpoint guesses, and R itself rather than a one-line Python call. `elbow-helper` automates the parts of this workflow, smoothing-scale search, bootstrap, null test, that `segmented` leaves to a human analyst's judgement.

### Manual eyeballing
The universal baseline: look at the plot, decide where it bends. A careful analyst can genuinely say "I don't see a clear knee here", which is more than most automated tools manage, but the judgement is not reproducible across people or even across the same person on different days. It does not scale past a handful of curves either.

### Ask an LLM
A modern variant of eyeballing: paste the data or a screenshot into a chat model and ask where the knee is. Large language models are fluent at describing shape in words and can hedge sensibly when asked to, but the answer is not deterministic across runs, carries no calibrated uncertainty, and rests on no reproducible derivation a reader could check.

## The thesis

Every alternative here is good at something `elbow-helper` does not try to be: `ruptures` at counting breakpoints, `segmented` at full statistical inference, Yellowbrick at owning the k-sweep end to end, an LLM at describing a shape in plain words. What none of them do by default is refuse. `elbow-helper` exists for the narrower case where a wrong answer is worse than no answer. It treats "there is no clear knee here" as a first-class, equally well-supported result.

## When to pick what

A comparison document that only ever recommends itself is advertising, so here is the routing, including the cases that belong elsewhere.

- **Pick `elbow-helper`** when a wrong knee costs more than no knee: an autoscaling threshold, a cluster count that ships, a "we have enough data now" call. What comes back is either a point estimate with a 90% interval, a bootstrap detection rate and a null-model p-value, or one of fifteen reason codes naming the gate that closed.
- **Pick `kneed`** when you already know the curve bends and you only want the coordinate. One call, two dependencies, nothing to configure. `elbow-helper`'s locator is a port of it, so this trades away the confirmation layer rather than the geometry.
- **Pick `ruptures`** when the question is "how many breakpoints, and where" over a long signal. `robust_knees` answers that question for diminishing-returns curves, but `ruptures` is the mature, widely cited segmentation library, with the cost functions and search strategies to match.
- **Pick R's `segmented`** when the breakpoint has to survive peer review: a standard error on the breakpoint itself, Davies' test, random effects. Nothing in Python replaces it, and `elbow-helper` does not try to.
- **Pick Yellowbrick's `KElbowVisualizer`** when you are already inside a scikit-learn pipeline and want the k-sweep fitted, scored and plotted by one object. That is work `elbow-helper` does not do at all.
- **Pick your own eyes** when there is one curve, you are going to plot it anyway, and nobody has to reproduce the decision later.

**When not to pick `elbow-helper`:**

- **Short applied curves, at the shipped defaults.** `min_samples` is 20 and `min_side_points` is 5, so a k-means scree run out to k = 10 is rejected outright with `INSUFFICIENT_DATA`. The defaults are calibrated for longer, noisier measurement curves. A retuned profile for short curves lives in `tests/test_real_world_examples.py` and works, but choosing those thresholds is exactly the judgement call this package otherwise makes for you.
- **When a number is required on every call.** The whole value here is that it sometimes refuses. If a downstream system cannot handle an abstention, `kneed` or `kneebow` will always answer and leave the trust question to you.
- **Signal segmentation.** Mean shifts in noise, several regimes over a time series, online changepoint detection: that is `ruptures`' problem, not this one's.

## Sources and as-of date

The claims above were last checked on **2026-10-27**. Versions and declared dependencies come from each project's own published package metadata on that date. The star ratings are this project's editorial judgement against the job stated at the top of the page, not a benchmark, and `elbow-helper`'s own row is a self-assessment worth reading as such.

Two rows name no software and so carry no version: "Manual eyeballing" and "Ask an LLM" are practices, and their ratings are argument rather than measurement.

- [kneed](https://pypi.org/project/kneed/) — 0.9.0; declares `numpy` and `scipy`; [documentation](https://kneed.readthedocs.io/en/latest/). Implements the Kneedle algorithm of Satopää, Albrecht, Irwin and Raghavan, *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior*, 31st International Conference on Distributed Computing Systems Workshops (2011), pp. 166-171.
- [ruptures](https://pypi.org/project/ruptures/) — 1.1.10; declares `numpy` and `scipy`; [documentation](https://centre-borelli.github.io/ruptures-docs/). Cites Truong, Oudre and Vayatis, *Selective review of offline change point detection methods*, Signal Processing 167 (2020) 107299, [doi:10.1016/j.sigpro.2019.107299](https://doi.org/10.1016/j.sigpro.2019.107299).
- [kneebow](https://pypi.org/project/kneebow/) — 1.0.4, released 2025-08-29; declares `numpy`, `matplotlib` and `scikit-learn`.
- [Yellowbrick](https://pypi.org/project/yellowbrick/) — 1.5; declares `matplotlib`, `scipy`, `scikit-learn`, `numpy` and `cycler`. The `locate_elbow` default and the three supported metrics are documented on the [KElbowVisualizer page](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html).
- [segmented](https://cran.r-project.org/package=segmented) on CRAN — 2.2-2, published 2026-09-11; author and maintainer Vito M. R. Muggeo.
- `elbow-helper` itself — this repository at 0.4.0. The thresholds quoted above are the defaults in `src/elbow_helper/config.py`, and the retuned short-curve profile is in `tests/test_real_world_examples.py`.

## Ideas worth stealing

Researched rather than guessed: what each neighbour does better, and what this library would have to change to match it. Nothing here is a commitment — it is the reading list the next version starts from.

- **Kneeliverse — an ensemble of detectors, with their agreement as the confidence** — It implements Discrete Curvature, DFDT, Kneedle, the L-method and Menger curvature side by side. This library estimates uncertainty from the curve; running several independent detectors and reporting where they agree is a second, cheaper estimate of the same thing — and disagreement is itself the honest signal.

- **`kneed` — the sensitivity parameter, and online multi-knee mode** — `S` lets a caller say how aggressive detection should be instead of accepting one built-in notion of "obvious". Online mode finds several knees in one curve, which a single-answer API cannot express.

- **`ruptures` — the general framing is change-point detection with a cost model** — An elbow is one change point under one cost. Saying so connects this library to a much larger, well-studied toolbox and tells a reader when their problem is actually the bigger one.

- **R's `segmented` — confidence intervals that fall out of the fit** — Fitting two joined lines gives a breakpoint *and* its standard error from the regression itself. That is the same uncertainty this library works to estimate, obtained as a by-product rather than as a separate step.
