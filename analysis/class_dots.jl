### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 11111111-0000-0000-0000-000000000002
md"""
# 🎯 How Does Your Brain Decide?

### Analyzing *our own* class data with a model of decision-making

Welcome! Before today, everyone played a quick game at
**[ryguy.io/dots](https://ryguy.io/dots/)**: a cloud of dots flickered on screen,
some drifting **left** or **right** while the rest jittered randomly, and you had
to call the direction as fast and accurately as you could.

That game is a famous neuroscience experiment. In this notebook we'll use **your
real, anonymous data** to answer questions like:

- How does the difficulty of a choice change how *fast* and *accurate* we are?
- Can a single mathematical model capture how a whole class makes decisions?
- Is everyone the same — or do some people trade speed for accuracy differently?

No coding needed — just read along, drag the sliders, and look at the pictures. 👇
"""

# ╔═╡ 11111111-0000-0000-0000-000000000020
md"""
## 🧠 The big idea: your brain *collects evidence*

When a decision is hard, you don't decide instantly — you **gather evidence over
time** until you're sure enough to commit.

Think of guessing which lunch line is moving faster. You glance back and forth,
and each glance is a little clue. Clues pile up on one side until you're
convinced — *then* you switch lines. Easy call? You decide almost instantly. Close
call? You watch longer before committing.

Scientists model this with the **Drift Diffusion Model (DDM)**. Picture a marker
that starts in the middle and drifts toward one of two **finish lines** — "LEFT"
or "RIGHT":

- Every instant, the evidence nudges the marker toward the correct side…
- …but there's **noise**, so it also wobbles randomly.
- The moment it touches a finish line, you respond.

**Strong** motion → a steep, fast drift → quick, correct answers.
**Weak** motion → a shallow drift swamped by noise → slow answers and more mistakes.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000021
md"""
## 🎛️ The model's "knobs"

The DDM explains a person's choices and timing with just a few numbers. When we
*fit* the model, we're asking the computer to find the knob settings that best
reproduce what the class actually did:

| knob | plain-English meaning |
|:--|:--|
| **drift / `k`** | How strongly the evidence pulls the marker. Bigger = sharper senses. |
| **`α`** | Whether extra motion strength keeps helping (`α=1`) or hits diminishing returns (`α<1`). |
| **boundary / `B`** | How far the finish lines are — i.e. **how sure you insist on being** before answering. Far = careful but slow; close = fast but error-prone. |
| **start point / `a₀`** | A built-in lean toward one side before any dots even appear. |
| **non-decision time / `τ`** | Time spent *not* deciding — light hitting your eyes, the signal reaching your brain, your finger pressing the key. |

The big one to remember is **`B`, the boundary**: it captures the
**speed–accuracy trade-off** every one of us makes.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000003
md"""
---
## 1 · Setting up (the reproducible part)

*(You can skip the next three cells — they just load the tools. They're here so a
scientist could re-run this analysis and get identical results.)*

This activates this folder's package environment and `dev`-installs
`DriftDiffusionModels` from the repo working copy one level up. Its
`HiddenMarkovModels` dependency is pinned to the upstream `0.7` line (the release
that exports `initialization`). The `Project.toml` / `Manifest.toml` next to this
notebook lock exact versions of everything else.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000004
env_ready = begin
    import Pkg
    Pkg.activate(@__DIR__)
    # Track the working copy of the package — this is the "dev version".
    Pkg.develop(path = normpath(joinpath(@__DIR__, "..")))
    # Pin the upstream HMM 0.7 line (has `initialization`); the unbounded resolve
    # otherwise falls back to an older release that breaks precompilation.
    Pkg.add(Pkg.PackageSpec(name = "HiddenMarkovModels", version = "0.7"))
    Pkg.instantiate()
    "✓ environment active · DriftDiffusionModels dev'd, HiddenMarkovModels 0.7 (upstream)"
end

# ╔═╡ 11111111-0000-0000-0000-000000000005
begin
    env_ready                      # ensure the environment cell runs first
    using DriftDiffusionModels
    using CSV, DataFrames, Statistics
    using Plots
    using PlutoUI
end

# ╔═╡ 11111111-0000-0000-0000-000000000006
helpers_ready = begin
    env_ready
    # Reuse the loader + plotting helpers that live with the experiment, so this
    # notebook and the outreach scripts never drift apart.
    include(joinpath(@__DIR__, "..", "outreach", "load_class_data.jl"))
    include(joinpath(@__DIR__, "..", "outreach", "plots.jl"))
    "✓ loaded load_class_data.jl + plots.jl"
end

# ╔═╡ 11111111-0000-0000-0000-000000000007
md"""
## 2 · Who played, and how did we do?

The anonymous results (only random `P-XXXX` ids — no names) load below. Each
person did about 200 trials. Here's everyone's trial count, accuracy, and typical
reaction time:
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

# ╔═╡ 11111111-0000-0000-0000-000000000022
md"""
> 💬 **Quick discussion:** Whose accuracy is highest? Do the most accurate people
> also have the *slowest* reaction times? Hold that thought — we'll come back to
> it at the very end.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000b
md"""
---
## 3 · Fit the model to the data

Now the fun part. Pick whose data to look at — **the whole class together**, or
one person — and the computer finds the model knob-settings that best explain
those choices. Leave **α** fixed at first; tick the box later to let the model
estimate it (works best on the pooled class data).
"""

# ╔═╡ 11111111-0000-0000-0000-00000000000c
@bind who Select(vcat(["__ALL__" => "⭐ Everyone (whole class pooled)"],
                      [pid => pid for pid in sort(collect(keys(class)))]))

# ╔═╡ 11111111-0000-0000-0000-00000000000d
md"""Let the model estimate the **α** knob too (instead of fixing α = 1):
$(@bind fit_alpha CheckBox(default = false))"""

# ╔═╡ 11111111-0000-0000-0000-00000000000e
trials = who == "__ALL__" ? pool_trials(class) : class[who]

# ╔═╡ 11111111-0000-0000-0000-00000000000f
model = begin
    m = CoherentDDM(fit_α = fit_alpha)
    fit!(m, trials)
    m
end

# ╔═╡ 11111111-0000-0000-0000-000000000010
md"""
**The model's best-fit knobs** for $(who == "__ALL__" ? "the whole class" : who)
— based on **$(length(trials)) decisions**:

| knob | value | reminder |
|:--|--:|:--|
| boundary `B`  | $(round(model.B;  digits = 2))    | how sure they insist on being (caution) |
| drift `k`     | $(round(model.k;  digits = 2))    | how strongly evidence drives the choice |
| `α`           | $(round(model.α;  digits = 2))    | diminishing returns of stronger motion (1 = none) |
| start `a₀`    | $(round(model.a₀; digits = 2))    | side bias (0.5 = perfectly even) |
| delay `τ`     | $(round(model.τ;  digits = 2)) s  | eyes-to-finger time that isn't "thinking" |
"""

# ╔═╡ 11111111-0000-0000-0000-000000000011
md"""
---
## 4 · The two famous curves

These two pictures show up in basically every decision-making paper. The **dots
are our class data**; the **dashed red line is the model's prediction**. When the
line hugs the dots, the model is capturing how we actually decided.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000012
plot_summary(trials; model = model)

# ╔═╡ 11111111-0000-0000-0000-000000000023
md"""
**Left — the psychometric curve:** the chance of choosing "right" as the motion
goes from strongly-left (far left) to strongly-right (far right). Notice it
crosses **50% at zero coherence** — when there's no real signal, we're literally
guessing. The steeper the S, the sharper our senses.

**Right — the chronometric curve:** how long decisions took. We're **slowest on
the hardest trials** (weak motion) because the evidence trickles in and the marker
takes longer to crawl to a finish line.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000024
md"""
### A simpler view: were we right?

If the curves above feel busy, here's the plain version — percent correct at each
difficulty level. The bar at `c = 0` sits near the dashed 50% line: **with no
signal, a coin flip is the best anyone can do.**
"""

# ╔═╡ 11111111-0000-0000-0000-000000000025
plot_accuracy(trials)

# ╔═╡ 11111111-0000-0000-0000-000000000026
md"""
### The clue the model is built from: reaction times

Here are the actual reaction times on the **hardest** vs **easiest** trials. Two
things to notice: hard decisions are **slower and more spread out**, and both
piles are **lopsided** — a long tail of slow responses. That exact skewed shape is
what convinced scientists a "drifting marker" was a good description of the brain.
"""

# ╔═╡ 11111111-0000-0000-0000-000000000027
plot_rt_distributions(trials)

# ╔═╡ 11111111-0000-0000-0000-000000000028
md"""
---
## 5 · Is everyone the same?

Below is **every student's psychometric curve** on one grid. Some S-curves are
steep (sharp perception), some are shallow (noisier), and a few are shifted
left/right (a side bias — the `a₀` knob).
"""

# ╔═╡ 11111111-0000-0000-0000-000000000029
student_grid = let
    helpers_ready
    ids = sort(collect(keys(class)))
    panels = map(ids) do id
        p = plot_psychometric(class[id])
        plot!(p; title = id, legend = false, titlefontsize = 8,
              xlabel = "", ylabel = "")
        p
    end
    plot(panels...; layout = (3, 4), size = (1150, 760),
         plot_title = "Each student's psychometric curve")
end

# ╔═╡ 11111111-0000-0000-0000-00000000002a
md"""
### The speed–accuracy trade-off, one dot per student

Remember the question from the table? Here's the answer. Each dot is one student:
their **typical reaction time** (across) vs their **accuracy** (up). Often the
slowest people are the most accurate and the fastest make more mistakes — that's
the **boundary `B`** at work. There's usually no single "best" strategy, just
different trade-offs.
"""

# ╔═╡ 11111111-0000-0000-0000-00000000002b
speed_acc = let
    ids  = sort(collect(keys(class)))
    accs = [100 * count(t -> t.choice == t.s, class[id]) / length(class[id]) for id in ids]
    rts  = [median(t.rt for t in class[id]) for id in ids]
    scatter(rts, accs; legend = false, ms = 7, c = :purple,
        xlabel = "median reaction time (s)  →  slower",
        ylabel = "percent correct  →  more accurate",
        title = "Speed vs accuracy — one dot per student")
end

# ╔═╡ 11111111-0000-0000-0000-00000000002c
md"""
---
## 6 · Things to try 🔬

- In **Section 3**, switch from "Everyone" to individual students. Whose data does
  the model fit best? Whose is noisiest, and why might that be?
- Tick the **α** box on the pooled data. Does the red line fit better? What value
  does α take, and what does that say about strong vs weak motion?
- Find the student with the highest boundary `B` (hint: slow + accurate in the
  scatter). Find someone with a low `B` (fast + more errors).
- **Bigger questions:** Would tired people have a different `τ`? Would a video-gamer
  have a higher drift `k`? How would you design a follow-up experiment to test it?
"""

# ╔═╡ 11111111-0000-0000-0000-000000000013
md"""
---
### Appendix · accuracy by coherence, by the numbers
"""

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
*Made for a high-school outreach visit. The data is anonymous; the model code is
the live working copy of `DriftDiffusionModels.jl`. Curious how the dots game
works? It's one self-contained web page — ask to see the source.*
"""

# ╔═╡ Cell order:
# ╟─11111111-0000-0000-0000-000000000002
# ╟─11111111-0000-0000-0000-000000000020
# ╟─11111111-0000-0000-0000-000000000021
# ╟─11111111-0000-0000-0000-000000000003
# ╠═11111111-0000-0000-0000-000000000004
# ╠═11111111-0000-0000-0000-000000000005
# ╠═11111111-0000-0000-0000-000000000006
# ╟─11111111-0000-0000-0000-000000000007
# ╠═11111111-0000-0000-0000-000000000008
# ╠═11111111-0000-0000-0000-000000000009
# ╠═11111111-0000-0000-0000-00000000000a
# ╟─11111111-0000-0000-0000-000000000022
# ╟─11111111-0000-0000-0000-00000000000b
# ╠═11111111-0000-0000-0000-00000000000c
# ╟─11111111-0000-0000-0000-00000000000d
# ╠═11111111-0000-0000-0000-00000000000e
# ╠═11111111-0000-0000-0000-00000000000f
# ╟─11111111-0000-0000-0000-000000000010
# ╟─11111111-0000-0000-0000-000000000011
# ╠═11111111-0000-0000-0000-000000000012
# ╟─11111111-0000-0000-0000-000000000023
# ╟─11111111-0000-0000-0000-000000000024
# ╠═11111111-0000-0000-0000-000000000025
# ╟─11111111-0000-0000-0000-000000000026
# ╠═11111111-0000-0000-0000-000000000027
# ╟─11111111-0000-0000-0000-000000000028
# ╠═11111111-0000-0000-0000-000000000029
# ╟─11111111-0000-0000-0000-00000000002a
# ╠═11111111-0000-0000-0000-00000000002b
# ╟─11111111-0000-0000-0000-00000000002c
# ╟─11111111-0000-0000-0000-000000000013
# ╠═11111111-0000-0000-0000-000000000014
# ╠═11111111-0000-0000-0000-000000000001
# ╟─11111111-0000-0000-0000-000000000015
