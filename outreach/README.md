# Classroom Moving-Dots Experiment → CoherentDDM

A self-contained random-dot-motion (evidence accumulation) task for a class
demo. Students run ~200 trials in their browser before class; their
**anonymous** data auto-collects into one Google Sheet; you download it and fit
the `CoherentDDM` live.

```
student's browser  ─►  Google Form  ─►  Google Sheet  ─►  CSV  ─►  CoherentDDM
   rdm_task.html                                          load_class_data.jl
```

Each trial records exactly the four fields the model needs:
`rt` (reaction time, s), `choice` (−1 left / +1 right), `s` (true direction
−1/+1), `c` (coherence 0–1). No names, emails, or accounts — only a random
`P-XXXXXX` id per student.

---

## Two ways to collect — the form is optional

The task **always downloads** a per-student backup file (`rdm_P-XXXX.csv`) when
a student finishes. So pick whichever fits:

| | Setup | Student effort | You do |
|---|---|---|---|
| **A. Google Form** | ~5 min, once | none — auto-uploads | download one Sheet CSV → `load_class_data` |
| **B. Collect files** | none | send/upload their file | gather files in a folder → `load_student_csvs` |

For ~30 kids you won't see beforehand, **A** is smoother. **B** needs no Google
account at all. Steps 1–2 below are only for path **A**; skip to step 3 for **B**.

## 1. Make the Google Form (path A only, ~5 min)

1. Go to <https://forms.google.com> → blank form. Name it e.g. "Dots Data".
2. **Settings → Responses:** turn **OFF** "Collect email addresses" and
   "Limit to 1 response" (keeps it anonymous and lets a student redo it).
3. Add two **questions**, both **Required**:
   - Q1, type **Short answer**, title `participant_id`
   - Q2, type **Paragraph**, title `data`
4. Click **Responses → Link to Sheets** to create the collecting spreadsheet.
5. **Get all three values from one "pre-filled" link.** Google hides a field ID
   like `entry.123456789` behind each question; this is the easy way to see them.
   - Click the form's **⋮** menu → **Get pre-filled response**.
   - It opens like a student's view. Type a *memorable* dummy answer in each box
     so you can tell them apart: `PID_HERE` in the `participant_id` box,
     `DATA_HERE` in the `data` box.
   - Scroll down → **Get link** → **COPY LINK**. Paste it somewhere. It looks like:
     ```
     https://docs.google.com/forms/d/e/1FAIpQLSxxxx/viewform?usp=pp_url&entry.111111111=PID_HERE&entry.222222222=DATA_HERE
     ```
   - Read your values off it:
     - `ENTRY_PID`  = the `entry.…` followed by `=PID_HERE`  (here `entry.111111111`)
     - `ENTRY_DATA` = the `entry.…` followed by `=DATA_HERE` (here `entry.222222222`)
     - `FORM_ACTION` = the part **before** `?`, with `/viewform` changed to
       `/formResponse`: `https://docs.google.com/forms/d/e/1FAIpQLSxxxx/formResponse`

   The dummy `PID_HERE`/`DATA_HERE` text is only a label so you know which ID is
   which — nothing gets saved.

## 2. Plug the values into the task

Open `rdm_task.html` and edit only the `CONFIG` block at the top:

```js
FORM_ACTION: "https://docs.google.com/forms/d/e/1FAIpQL.../formResponse",
ENTRY_PID:   "entry.123456789",
ENTRY_DATA:  "entry.987654321",
```

Leave `FORM_ACTION: ""` to run in **test mode** (no upload; just downloads the
CSV) while you check it works.

## 3. Deploy

Drop `rdm_task.html` into your personal website folder and share the link, e.g.
`https://yoursite.com/rdm_task.html`. It's a single static file — no build, no
server. Works on phones and laptops (arrow keys on a laptop, tap left/right
sides on a phone).

**Test it yourself first:** open the live link, do a few trials, then check a
row appeared in the linked Google Sheet.

## 4. Before class: send the link

> "Open this link on a laptop or phone, do the ~8-minute dots task. It's
> anonymous — no name needed. Just finish to the 'thank you' screen."

You can watch rows arrive in the Sheet in real time.

## 5. In class: fit the model

Load the data (one line differs by collection path), then fit:

```julia
include("outreach/load_class_data.jl")     # needs:  ] add CSV DataFrames

# Path A — the Google Sheet (File → Download → Comma-separated values):
class = load_class_data("class_responses.csv")
# Path B — a folder of the per-student backup files:
# class = load_student_csvs("collected_files/")

summarize(class)                       # students, trial counts, accuracy

# Group-level fit (whole class pooled)
model = CoherentDDM()
fit!(model, pool_trials(class))
@show model.B model.k model.α model.a₀ model.τ

# One student
some_id = first(keys(class))
fit!(CoherentDDM(), class[some_id])
```

Both loaders return the same `Dict{id => Vector{CoherentDDMResult}}`, so
everything downstream is identical. `load_class_data` matches column names
loosely, so the default Google headers (`participant_id`, `data`, plus the auto
`Timestamp`) work without renaming.

## 6. In class: plot it live

```julia
include("outreach/plots.jl")           # needs:  ] add Plots

data  = pool_trials(class)
model = CoherentDDM(); fit!(model, data)

plot_summary(data; model=model)        # psychometric + chronometric, fit overlaid
# or individually:
plot_psychometric(data; model=model)   # P(choose right) vs signed coherence — the S-curve
plot_chronometric(data; model=model)   # mean RT vs coherence — slower when the signal is weak
```

Drop `; model=model` to show the raw data first, then re-run with the fit
overlaid — a nice "the model captures it" reveal.

### Nice things to show the students
- **Psychometric curve:** accuracy vs coherence `c` — flat at 50% for `c=0`,
  rising toward ceiling. (The `c=0` condition is literally guessing — a great
  talking point.)
- **Chronometric curve:** mean RT vs `c` — slower (more accumulation) when the
  signal is weak.
- The fitted **drift gain `k`**, **boundary `B`** (speed/accuracy trade-off),
  and **non-decision time `τ`** — connect each parameter to something a teenager
  feels: how strong the evidence is, how cautious they are, and how long their
  eyes+fingers take.
- Fit two groups (e.g. "go fast" vs "be accurate" instructions) and compare `B`.

## Tuning the task
All knobs live in the `CONFIG` block of `rdm_task.html`:
- `COHERENCES`, `REPS_PER_DIRECTION` — set the trial count
  (`len(COHERENCES) × 2 × REPS`; default `5 × 2 × 20 = 200`).
- `MAX_RT_SEC`, `FIXATION_SEC`, `ITI_SEC` — timing.
- `N_DOTS`, `DOT_SPEED_DPS`, `APERTURE_FRAC` — stimulus look.
- `PRACTICE_TRIALS` — easy warm-up trials (with feedback, not saved).

## Privacy notes
- No personal data is requested or stored — only a random id and the trial
  numbers.
- Each student's CSV also downloads to their device as a backup; if the upload
  ever fails, the thank-you screen tells them to send that file to you.
- Google Form responses live in your Google account; delete the form/Sheet
  after the demo if you like.
