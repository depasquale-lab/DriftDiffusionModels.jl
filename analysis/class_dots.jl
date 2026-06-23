### A Pluto.jl notebook ###
# v0.20.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 11111111-0000-0000-0000-000000000004
env_ready = begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(path = normpath(joinpath(@__DIR__, "..")))
    Pkg.add(Pkg.PackageSpec(name = "HiddenMarkovModels", version = "0.7"))
    Pkg.instantiate()
    "✓ environment active · DriftDiffusionModels dev'd, HiddenMarkovModels 0.7 (upstream)"
end

# ╔═╡ 11111111-0000-0000-0000-000000000005
begin
    env_ready                      # ensure the environment cell runs first
    using DriftDiffusionModels
    using CSV, DataFrames, Statistics, Random
    using Plots
    using PlutoUI
end

# ╔═╡ 11111111-0000-0000-0000-000000000006
helpers_ready = begin
    env_ready
    # Reuse the loader + plotting + analysis helpers that live with the repo.
    include(joinpath(@__DIR__, "..", "outreach", "load_class_data.jl"))
    include(joinpath(@__DIR__, "..", "outreach", "plots.jl"))
    include(joinpath(@__DIR__, "analysis_helpers.jl"))
    "✓ loaded load_class_data.jl + plots.jl + analysis_helpers.jl"
end

# ╔═╡ 11111111-0000-0000-0000-000000000002
md"""
# How Does Your Brain Decide?

### A live look at our own class data

Okay. Everybody in this room ran the dots task at
**[ryguy.io/dots](https://ryguy.io/dots/)**, and every button press got saved
(anonymously, just a random ID). Let's pull it up and see what it says about the
way you make decisions.

Everything on this page is live. As I move sliders or choose a dataset, it
recomputes right in front of the room.

The walkthrough has a simple arc:

1. Build an intuition for the model.
2. Use the math to make rough predictions before fitting.
3. Fit the model and ask whether it actually explains the data.
4. Look for where the model succeeds, where it fails, and how people differ.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000020
md"""
## First, the idea

When a choice is easy, you answer right away. When it's hard, you slow down. Why?
Your brain is piling up evidence over time, and it holds off until that pile tips
far enough one way to commit.

The Drift Diffusion Model is just that idea, drawn as a picture. A marker starts in
the middle. Evidence pushes it toward "left" or "right." There's noise, so it
wobbles on the way. The moment it touches a boundary, that's your answer.

So the teaching question for the next few cells is: **what does each part of that
picture control?**
"""

# ╔═╡ 11111111-0000-0000-0000-000000000021
md"""
## The knobs we get to turn

The whole model comes down to a few numbers. Each one should connect to a plain
psychological idea; otherwise the model is just curve-fitting with technical names.

| parameter | what it represents |
|:--|:--|
| **drift `k`** | how hard the evidence pulls the marker. Think of it as perceptual sensitivity. |
| **`α`** | do stronger signals keep helping (`α=1`), or hit diminishing returns (`α<1`)? |
| **boundary `B`** | distance to the boundaries: how much certainty you demand before committing. |
| **start `a₀`** | a lean toward one option before any evidence arrives. |
| **delay `τ`** | sensing and moving time that has nothing to do with deciding. |

As we go, keep asking: can we see a trace of each knob in the raw behavior before
we fit anything?
"""

# ╔═╡ 11111111-0000-0000-0000-000000000040
md"""
## The math under the cartoon

Here is the whole model in one line:

```math
dX_t = v\,dt + dW_t,\qquad v = s\,k\,c^\alpha
```

That says the evidence marker `X` changes by two ingredients. The first part,
`v dt`, is the real signal pushing it left or right. The second part, `dW_t`, is
random noise. The decision happens when `X` hits a boundary:

```math
X_t = B \Rightarrow \text{choose right}, \qquad X_t = 0 \Rightarrow \text{choose left}.
```

That looks formal, but it is just the picture above written precisely enough
that a computer can make predictions from it. This is the main reason math is
useful here: it forces a verbal explanation to say exactly what it predicts.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000041
md"""
Two useful consequences fall straight out:

```math
P(\text{right}) =
\frac{1-\exp(-2v\,a_0B)}{1-\exp(-2vB)}
```

and when there is no signal (`v = 0`), the average decision time is roughly

```math
\text{mean RT at } c=0 \approx \tau + \frac{B^2}{4}
```

So even before fitting, the data gives us clues:

- Bias at zero coherence hints at `a₀`.
- Slow hard trials hint at `B`.
- The steepness of the S-curve hints at `k`.

The next step is to see whether those clues are strong enough to guess reasonable
parameters by hand.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000051
md"""
### Reading the equation without getting lost

The symbols are doing specific jobs:

| symbol | plain meaning | where it shows up in behavior |
|:--|:--|:--|
| `X_t` | current evidence | the hidden state we never observe directly |
| `v` | drift rate | how strongly the stimulus pushes choices |
| `dW_t` | noise | why the same stimulus can produce different responses |
| `B` | boundary | the speed-accuracy trade-off |
| `τ` | non-decision delay | the part of reaction time not spent deciding |

This is a useful habit in mathematical psychology: every symbol should have both
a formal role in the equation and a behavioral interpretation.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000030
md"""
## Build intuition with a simulation

Before we look at your data, let's get a feel for the model. I'll move these
sliders and we'll watch simulated decisions pile up toward a boundary. Green paths
are heading to "right," red to "left," and every one starts on the grey dotted
line.

The point of this cell is not to find the right parameters yet. It is to connect
each knob with a visible change in simulated behavior.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000031
md"""
boundary **B** (caution): $(@bind simB Slider(0.5:0.1:4.0, default = 1.5, show_value = true))

drift **k** (sensitivity): $(@bind simK Slider(0.0:0.2:8.0, default = 3.0, show_value = true))

non-decision **τ** in seconds: $(@bind simτ Slider(0.0:0.02:0.6, default = 0.2, show_value = true))

**coherence c** (signal strength): $(@bind simC Slider(0.0:0.04:0.64, default = 0.16, show_value = true))
"""

# ╔═╡ 11111111-0000-0000-0000-000000000032
begin
    helpers_ready
    plot_race(B = simB, k = simK, c = simC, τ = simτ, npaths = 8)
end

# ╔═╡ 11111111-0000-0000-0000-000000000033
sim_stats = begin
    helpers_ready
    sim_model = CoherentDDM(B = simB, k = simK, α = 1.0, a₀ = 0.5, τ = simτ)
    sim_rng = MersenneTwister(7)
    sims = [simulateDDM(sim_model, rand(sim_rng, (-1, 1)), Float64(simC), 1e-4, sim_rng)
            for _ in 1:400]
    (acc = round(100 * mean(s.choice == s.s for s in sims); digits = 1),
     mrt = round(mean(s.rt for s in sims); digits = 2))
end

# ╔═╡ 11111111-0000-0000-0000-000000000034
md"""
With these settings the model lands at **$(sim_stats.acc)% correct**, with a mean
reaction time of **$(sim_stats.mrt) s**.

> **Watch the trade-off.** When I push the boundary **B** up, accuracy climbs,
> but the answers get slower. Pulling it down gives the opposite. That entire
> trade-off is one number. When coherence **c** goes all the way to 0, there is
> no signal at all, and even a perfect model is stuck right around 50/50.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000007
md"""
## 1 · So, how did we do?

Here's everyone's results. This is a first-pass sanity check before modeling: how accurate was
each participant, and how long did they usually take?
"""

# ╔═╡ 11111111-0000-0000-0000-000000000022
md"""
> A question for the room: whose accuracy is highest? And do the most accurate
> people also tend to be the slowest? Hang onto that. We'll come back to it.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000b
md"""
## 2 · Before fitting: can we guess the knobs?

I'll choose whose data to look at, then tune the knobs live to match the two
curves by hand. No optimizer yet: just a simple model, a plot, and some visible
structure in the data.

This is an important teaching step. If the fitted parameters later feel surprising,
we can compare them to the clues we saw before fitting.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000044
md"""
Here is a deliberately rough no-fitting guess, using the shortcuts above:

- `τ` starts near the fastest plausible responses.
- `B` comes from how slow the hardest trials are.
- `a₀` comes from side bias when coherence is zero.
- `k` comes from the slope of the choice curve.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000052
md"""
### Rules of thumb while tuning

These are the qualitative moves to watch for:

| if the hand-tuned model... | the likely knob to adjust |
|:--|:--|
| chooses right too often at `c = 0` | move `a₀` back toward 0.5 |
| is too inaccurate on easy trials | increase `k` or `B` |
| is accurate enough but too slow | lower `B` or lower `τ` |
| misses mostly at high coherence | change `α`, because the signal-strength scaling may be wrong |

The important part is not getting a perfect manual fit. It is learning which
features of the data constrain which psychological interpretation.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000004a
md"""
The score table is intentionally blunt: smaller error means the hand-tuned model
is closer to the binned data. If one curve gets better while the other gets
worse, that is already informative: the same parameter settings have to explain
both choices and reaction times.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000004b
md"""
## 3 · Now let's fit the model

Now we let the model do the careful version of what we just did by eye. We hand
the computer all of these decisions and ask which parameter settings make the
observed choices and reaction times most likely.

I'll start with **α fixed at 1**, so the first fitted model is the simpler one.
Then we can ask whether freeing `α` is worth the extra flexibility.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000d
md"""Let the model estimate the **α** knob too (instead of fixing α = 1):
$(@bind fit_alpha CheckBox(default = false))"""

# ╔═╡ 11111111-0000-0000-0000-00000000004d
md"""
Now we can compare the rough guesses with the parameter values from the actual
fit.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000011
md"""
## 4 · The two curves everyone draws

This is the standard first readout for a perceptual decision model. The dots are
the actual data. The dashed red line is what the model predicts.

Read the plot in two passes: first choices, then time. A good model should explain
both with the same parameter values.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000023
md"""
On the left is the psychometric curve: how often you chose "right" as the motion
runs from strongly-left over to strongly-right. Notice it crosses 50% right at
zero. No signal, so it's a coin flip. The steeper that S-shape, the sharper the
perception.

On the right is the chronometric curve. We're slowest on the hardest trials,
because weak evidence takes longer to crawl up to a boundary.

This is where the psychology enters: the model is not just saying "people are
slower on hard trials." It is proposing a mechanism for why.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000024
md"""
### Same idea, simpler: were we right?

Percent correct at each difficulty level. Look at the `c = 0` bar sitting right
down near the 50% line. With no signal, a coin flip really is the best anyone can
do. This plot is useful because it checks the task itself: if zero coherence is
not near chance, something about the task or data deserves a closer look.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000026
md"""
### The clue the whole model is built on: reaction times

Look at the shape. Hard decisions are slower and more spread out, and both piles
lean the same way, with a long tail of slow responses trailing off to the right.
That lopsided shape is exactly what a drifting, noisy marker produces. It's a big
reason scientists believed the picture in the first place.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000035
md"""
## 5 · Is the extra flexibility worth it?

Here's a trap worth knowing about. A model with more knobs can *always* fit the
data a little better. So how do we know the extra complexity is worth it?
Scientists use a score called **AIC**. It rewards good fit, then charges a penalty
for every extra parameter. Lower is better. Let's pit the simple model (`α` locked
at 1) against the flexible one (`α` free):

This is model comparison in miniature: not "which line looks nicest?", but "which
explanation is accurate enough to justify its complexity?"
"""

# ╔═╡ 11111111-0000-0000-0000-000000000037
md"""
> If the α-free row has the lower AIC, then the data support a diminishing-returns
> relationship between coherence and evidence. If it doesn't, the simpler model is
> the better explanation. The point is not to add knobs until the line looks nice;
> it is to ask which explanation earns its extra complexity.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000038
md"""
## 6 · The real test: can it regenerate us?

A model that only describes the data it was fit to has not proved much. A stronger
test is whether it can generate new data with the same structure. Here the fitted
model runs the same experiment again, and we compare its reaction times (red)
against the real ones (grey).

This is a posterior-predictive check in spirit: if the model really captured the
process, fake data from the model should resemble the real data in more than one
summary statistic.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000004c
md"""
## 7 · Where does the model miss?

The fit is only half the story. We also want to know where the model misses.
These bars are **data minus model**. A little noise is normal. A pattern is more
interesting, because it points to something the model does not capture yet.

This is the most scientific part of the notebook. A model that fails in a clear
way is useful, because it tells us what assumption to question next.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000028
md"""
## 8 · Is everyone the same?

Here's every single person's psychometric curve on one grid. Look around for a
second. Some of these S-shapes are steep (sharp perception), some are shallow
(noisier), and a few are shifted left or right, which is a side bias.

The teaching point is that group averages are useful, but they can hide different
strategies or abilities across individuals.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000060
md"""
### And the chronometric curves, person by person

Same idea for reaction time. Most people are slowest when the signal is weak (left
side) and speed up as it gets stronger. For some that drop is crisp; for others
it's buried in noise, which tells you something about how reliable their timing is.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000002a
md"""
### The speed/accuracy trade-off, one dot per person

Remember the question from the scoreboard? Here's the answer. Each dot is one
person: their typical reaction time along the bottom, their accuracy up the side.
The slowest people are often the most accurate. The model gives that pattern a
candidate explanation: different people may be setting different boundaries.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000003a
md"""
### And here's what the model says about each person

Now we fit the model to each person on their own. Suddenly the speed/accuracy story
has an actual number behind it: the boundary `B`. Cautious people (high `B`) land
higher up and slower. The full table of everyone's knobs is right below.

This is the payoff for interpretability: instead of only saying "this person was
slower," we can ask whether they were slower because they waited for more evidence,
had weaker drift, or had a longer non-decision delay.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000053
md"""
### A caution about individual fits

Each person contributed a limited number of trials, so the individual parameters
are useful teaching examples, not labels for the person. A high `B` here means
"this model explains these trials with a higher boundary." It does not mean the
person is cautious in every situation.

That distinction matters. Models can make psychological ideas measurable, but the
measurement is always tied to the task, the data quality, and the assumptions of
the model.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000002c
md"""
## 9 · Discussion prompts

- When I switch from "Everyone" to a single person back in section 2, whose data fits
  cleanest? Whose is the noisiest, and why might that be?
- When I tick the **α** box on the pooled data, how does the AIC table change?
- In the hand-tuning section, how close can we get to the fitted parameters without looking?
  Which knob is easiest to estimate from the plots? Which one needs the full model?
- Bigger questions: would practice change drift `k`? Would sleep loss stretch delay
  `τ`? Would instructions to prioritize accuracy move the boundary `B`?
"""

# ╔═╡ 11111111-0000-0000-0000-000000000099
md"""
---
# Appendix · setup & data

Everything below loads the tools and data. It's here for reproducibility, so a
scientist can re-run this and get identical numbers. Safe to ignore during the
talk.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000003
md"""
**Reproducible environment.** Activates this folder's package environment,
`dev`-installs `DriftDiffusionModels` from the repo working copy one level up,
and pins upstream `HiddenMarkovModels 0.7` (the release that exports
`initialization`). `Project.toml` / `Manifest.toml` lock everything else.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000008
datapath = joinpath(@__DIR__, "data", "class_responses.csv")

# ╔═╡ 11111111-0000-0000-0000-000000000009
class = begin
    helpers_ready
    load_class_data(datapath)
end

# ╔═╡ 11111111-0000-0000-0000-00000000000a
summary_df = begin
    helpers_ready
    rows = [(participant = pid,
             n_trials   = length(tr),
             accuracy_pct = round(100 * count(t -> t.choice == t.s, tr) / length(tr); digits = 1),
             median_rt_s  = round(median(t.rt for t in tr); digits = 3))
            for (pid, tr) in sort(collect(class); by = first)]
    DataFrame(rows)
end

# ╔═╡ 11111111-0000-0000-0000-000000000042
md"""
Dataset: $(@bind who Select(vcat(["__ALL__" => "Everyone (whole class pooled)"],
                      [pid => pid for pid in sort(collect(keys(class)))])))
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000e
trials = who == "__ALL__" ? pool_trials(class) : class[who]

# ╔═╡ 11111111-0000-0000-0000-000000000043
guess_model = begin
    helpers_ready
    ddm_guess_from_data(trials)
end

# ╔═╡ 11111111-0000-0000-0000-000000000045
parameter_table("back-of-envelope guess" => guess_model)

# ╔═╡ 11111111-0000-0000-0000-000000000046
md"""
Now I'll see whether I can beat that guess by hand.

boundary **B**: $(@bind handB Slider(0.5:0.1:5.5, default = round(guess_model.B; digits = 1), show_value = true))

drift **k**: $(@bind handK Slider(0.1:0.1:12.0, default = round(guess_model.k; digits = 1), show_value = true))

exponent **α**: $(@bind handα Slider(0.4:0.1:1.8, default = 1.0, show_value = true))

start **a₀**: $(@bind handA0 Slider(0.1:0.02:0.9, default = round(guess_model.a₀; digits = 2), show_value = true))

delay **τ**: $(@bind handτ Slider(0.0:0.02:0.8, default = round(guess_model.τ; digits = 2), show_value = true))
"""

# ╔═╡ 11111111-0000-0000-0000-000000000047
hand_model = CoherentDDM(B = handB, k = handK, α = handα, a₀ = handA0, τ = handτ, fit_α = false)

# ╔═╡ 11111111-0000-0000-0000-000000000048
plot_manual_match(trials, hand_model)

# ╔═╡ 11111111-0000-0000-0000-000000000049
manual_score(trials, hand_model)

# ╔═╡ 11111111-0000-0000-0000-00000000000f
model = begin
    m = CoherentDDM(fit_α = fit_alpha)
    fit!(m, trials)
    m
end

# ╔═╡ 11111111-0000-0000-0000-000000000010
md"""
Here's what the computer came back with for
$(who == "__ALL__" ? "the whole class" : who), fit to **$(length(trials)) decisions**:

| knob | value | what it's telling us |
|:--|--:|:--|
| boundary `B`  | $(round(model.B;  digits = 2))    | how much certainty they demand (caution) |
| drift `k`     | $(round(model.k;  digits = 2))    | how strongly the evidence drives the choice |
| `α`           | $(round(model.α;  digits = 2))    | diminishing returns of a stronger signal (1 means none) |
| start `a₀`    | $(round(model.a₀; digits = 2))    | side bias, where 0.5 is perfectly even |
| delay `τ`     | $(round(model.τ;  digits = 2)) s  | sensing and moving time, not "thinking" time |
"""

# ╔═╡ 11111111-0000-0000-0000-00000000004e
parameter_table("back-of-envelope guess" => guess_model,
                "hand-tuned sliders" => hand_model,
                "maximum-likelihood fit" => model)

# ╔═╡ 11111111-0000-0000-0000-000000000012
plot_summary(trials; model = model)

# ╔═╡ 11111111-0000-0000-0000-000000000025
plot_accuracy(trials)

# ╔═╡ 11111111-0000-0000-0000-000000000027
plot_rt_distributions(trials)

# ╔═╡ 11111111-0000-0000-0000-000000000036
model_cmp = begin
    helpers_ready
    cmp_df, _, _ = compare_alpha(trials)
    cmp_df
end

# ╔═╡ 11111111-0000-0000-0000-000000000039
plot_ppc(trials, model; reps = 20)

# ╔═╡ 11111111-0000-0000-0000-00000000004f
plot_residuals(trials, model)

# ╔═╡ 11111111-0000-0000-0000-000000000050
worst_residuals(trials, model)

# ╔═╡ 11111111-0000-0000-0000-000000000029
student_grid = begin
    helpers_ready
    plot_student_grid(class, plot_psychometric;
                      title = "Each participant's psychometric curve")
end

# ╔═╡ 11111111-0000-0000-0000-000000000061
chrono_grid = begin
    helpers_ready
    plot_student_grid(class, plot_chronometric;
                      title = "Each participant's chronometric curve")
end

# ╔═╡ 11111111-0000-0000-0000-00000000002b
speed_acc = let
    ids  = sort(collect(keys(class)))
    accs = [100 * count(t -> t.choice == t.s, class[id]) / length(class[id]) for id in ids]
    rts  = [median(t.rt for t in class[id]) for id in ids]
    scatter(rts, accs; legend = false, ms = 7, c = :purple,
        xlabel = "median reaction time (s)  →  slower",
        ylabel = "percent correct  →  more accurate",
        title = "Speed vs accuracy (one dot per participant)")
end

# ╔═╡ 11111111-0000-0000-0000-00000000003b
student_params = begin
    helpers_ready
    fit_all_students(class)
end

# ╔═╡ 11111111-0000-0000-0000-00000000003c
plot_param_scatter(student_params)

# ╔═╡ 11111111-0000-0000-0000-000000000013
md"""**Appendix table** · accuracy by coherence, by the numbers:"""

# ╔═╡ 11111111-0000-0000-0000-000000000014
accuracy_table = begin
    cohs, acc, n = accuracy_by_coherence(trials)
    DataFrame(coherence = cohs,
              accuracy_pct = round.(100 .* acc; digits = 1),
              n_trials = n)
end

# ╔═╡ 11111111-0000-0000-0000-000000000001
PlutoUI.TableOfContents(title = "Contents", depth = 2)

# ╔═╡ 11111111-0000-0000-0000-000000000015
md"""
---
*Made for a high-school outreach visit. Data is anonymous; the model code is the
live working copy of `DriftDiffusionModels.jl`.*
"""

# ╔═╡ Cell order:
# ╟─11111111-0000-0000-0000-000000000002
# ╟─11111111-0000-0000-0000-000000000020
# ╟─11111111-0000-0000-0000-000000000021
# ╟─11111111-0000-0000-0000-000000000040
# ╟─11111111-0000-0000-0000-000000000041
# ╟─11111111-0000-0000-0000-000000000051
# ╟─11111111-0000-0000-0000-000000000030
# ╟─11111111-0000-0000-0000-000000000031
# ╠═11111111-0000-0000-0000-000000000032
# ╠═11111111-0000-0000-0000-000000000033
# ╟─11111111-0000-0000-0000-000000000034
# ╟─11111111-0000-0000-0000-000000000007
# ╠═11111111-0000-0000-0000-000000000009
# ╠═11111111-0000-0000-0000-00000000000a
# ╟─11111111-0000-0000-0000-000000000022
# ╟─11111111-0000-0000-0000-00000000000b
# ╟─11111111-0000-0000-0000-000000000042
# ╠═11111111-0000-0000-0000-00000000000e
# ╠═11111111-0000-0000-0000-000000000043
# ╟─11111111-0000-0000-0000-000000000044
# ╠═11111111-0000-0000-0000-000000000045
# ╟─11111111-0000-0000-0000-000000000046
# ╟─11111111-0000-0000-0000-000000000052
# ╠═11111111-0000-0000-0000-000000000047
# ╠═11111111-0000-0000-0000-000000000048
# ╠═11111111-0000-0000-0000-000000000049
# ╟─11111111-0000-0000-0000-00000000004a
# ╟─11111111-0000-0000-0000-00000000004b
# ╟─11111111-0000-0000-0000-00000000000d
# ╠═11111111-0000-0000-0000-00000000000f
# ╟─11111111-0000-0000-0000-000000000010
# ╟─11111111-0000-0000-0000-00000000004d
# ╠═11111111-0000-0000-0000-00000000004e
# ╟─11111111-0000-0000-0000-000000000011
# ╠═11111111-0000-0000-0000-000000000012
# ╟─11111111-0000-0000-0000-000000000023
# ╟─11111111-0000-0000-0000-000000000024
# ╠═11111111-0000-0000-0000-000000000025
# ╟─11111111-0000-0000-0000-000000000026
# ╠═11111111-0000-0000-0000-000000000027
# ╟─11111111-0000-0000-0000-000000000035
# ╠═11111111-0000-0000-0000-000000000036
# ╟─11111111-0000-0000-0000-000000000037
# ╟─11111111-0000-0000-0000-000000000038
# ╠═11111111-0000-0000-0000-000000000039
# ╟─11111111-0000-0000-0000-00000000004c
# ╠═11111111-0000-0000-0000-00000000004f
# ╠═11111111-0000-0000-0000-000000000050
# ╟─11111111-0000-0000-0000-000000000028
# ╠═11111111-0000-0000-0000-000000000029
# ╟─11111111-0000-0000-0000-000000000060
# ╠═11111111-0000-0000-0000-000000000061
# ╟─11111111-0000-0000-0000-00000000002a
# ╠═11111111-0000-0000-0000-00000000002b
# ╟─11111111-0000-0000-0000-00000000003a
# ╟─11111111-0000-0000-0000-000000000053
# ╠═11111111-0000-0000-0000-00000000003b
# ╠═11111111-0000-0000-0000-00000000003c
# ╟─11111111-0000-0000-0000-00000000002c
# ╟─11111111-0000-0000-0000-000000000099
# ╟─11111111-0000-0000-0000-000000000003
# ╠═11111111-0000-0000-0000-000000000004
# ╠═11111111-0000-0000-0000-000000000005
# ╠═11111111-0000-0000-0000-000000000006
# ╠═11111111-0000-0000-0000-000000000008
# ╟─11111111-0000-0000-0000-000000000013
# ╠═11111111-0000-0000-0000-000000000014
# ╠═11111111-0000-0000-0000-000000000001
# ╟─11111111-0000-0000-0000-000000000015
