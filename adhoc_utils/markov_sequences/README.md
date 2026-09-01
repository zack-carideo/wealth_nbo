# `seq_profile.py` — Markov sequence profiling

A standalone utility for asking "how do these states follow one another?"
Give it a table with one row per event, tell it which column identifies a
sequence (the customer) and which column holds the state (the product), and it
fits a first- or second-order Markov chain over the states, prints a readable
profile, and can emit a handful of per-customer features describing how
typical each customer's path is against that chain.

It is deliberately separate from the main pipeline: no config file, no coupling
to `preprocess.py`, no dependencies beyond pandas and numpy. Every knob is a
CLI flag or a function argument with a default.

```
adhoc_utils/markov_sequences/
├── seq_profile.py          the utility (CLI + importable module)
├── sample_sequences.csv    24-row example input, six customers
└── README.md               this file
```

## Contents

1. [Quick start](#quick-start)
2. [Input requirements](#input-requirements)
3. [Options reference](#options-reference)
4. [Running end to end](#running-end-to-end)
5. [What comes out](#what-comes-out)
6. [Mapping transitions and paths back to ids](#mapping-transitions-and-paths-back-to-ids)
7. [Python API](#python-api)
8. [Things that are easy to get wrong](#things-that-are-easy-to-get-wrong)
9. [Sample input walkthrough](#sample-input-walkthrough) (the original `SAMPLE_FORMAT.md`)

---

## Quick start

```bash
cd adhoc_utils/markov_sequences

python seq_profile.py --input sample_sequences.csv \
    --id customer_id --state product_type --time origination_date \
    --min-count 2 --path-length 2
```

That prints a text profile to stdout and writes nothing. Add `--save` to
persist the fitted chain and `--scores` to write per-customer features (see
[Running end to end](#running-end-to-end)).

**Environment.** Needs Python with pandas and numpy. On the machine this repo
was verified on, the `python` on PATH is 3.12 without numpy; use the 3.10
install that the rest of the pipeline uses:

```powershell
py -3.10 seq_profile.py --input sample_sequences.csv --id customer_id --state product_type
```

---

## Input requirements

A CSV (or, via the Python API, any DataFrame) with at least two columns: an id
and a state. A time column is optional and only adds gap statistics.

The frame is taken at its word. **Nothing is validated and nothing is
re-sorted.** Four things must already be true:

1. **Grouped.** All rows for an id are contiguous.
2. **Ordered within the group**, oldest first.
3. **One row per event.** Other columns (balances, flags) are ignored — the
   utility reads only the id, the state, and optionally the time.
4. **The state column is what you want to profile.** Anything else is along
   for the ride and can stay in the frame.

Nulls in the state column become the literal state `MISSING` so they show up
in the profile rather than vanishing. States are compared as strings.

If there is any doubt about provenance, sort defensively before calling the
utility. `mergesort` is stable, so same-day rows keep whatever order the source
gave them — matching the tie behaviour in `preprocess.py`:

```python
df = df.sort_values([id_col, time_col], kind="mergesort")
```

See [What unsorted input costs you](#what-unsorted-input-costs-you) for why
this matters.

---

## Options reference

### CLI flags

| Flag | Required | Default | What it does |
|---|---|---|---|
| `--input PATH` | yes | — | CSV to read. Must already be grouped by id and sorted by time within id. |
| `--id COL` | yes | — | Column that defines a sequence (e.g. `customer_id`). |
| `--state COL` | yes | — | Column holding the state to profile (e.g. `product_type`, `product_family`). |
| `--time COL` | no | `None` | Parsed with `pd.to_datetime`. When given, each transition also gets `gap_days_median` / `gap_days_mean` and the summary shows a median gap per transition. Has no effect on probabilities. |
| `--order N` | no | `1` | Markov order. `1` = the previous state predicts the next. `2` = the previous two states (joined as `a>b`) predict the next. Cell count grows as `states^(order+1)`; check `thin share` before trusting order 2. |
| `--alpha A` | no | `0.5` | Additive (Laplace) smoothing. `prob_smoothed = (n + alpha) / (from_n + alpha * n_to_states)`. Keeps log-probabilities finite for pairs never observed. Raw `prob` is unaffected. |
| `--min-count N` | no | `30` | An origin state seen fewer than `N` times is flagged `thin`. **Nothing is dropped** — the flag is there so you can decide. Drives the `thin share` line in the summary. |
| `--boundaries` | no | off | Insert a `<START>` sentinel before and `<END>` after every id's sequence. Makes "what do people start with" and "what do they stop after" readable as ordinary transitions, but the sentinels then count in state frequencies and sequence lengths. |
| `--path-length N` | no | `3` | Length of the contiguous runs reported under `COMMON RUNS`. `< 2` disables the section. |
| `--save PATH` | no | `None` | Write the fitted model (all tables + params) as JSON. Reload with `SequenceModel.load(path)`. |
| `--scores PATH` | no | `None` | Score the same input against the fitted model and write one row of features per id. |

One `fit()` argument has no CLI flag and always uses its default from the
command line: `top_paths=25` (how many runs are kept in the `paths` table).

### `fit()` arguments

Same knobs, same defaults, as keyword arguments:

```python
sp.fit(df, id_col, state_col, time_col=None, order=1, alpha=0.5,
       min_count=30, add_boundaries=False, path_length=3, top_paths=25)
```

### `score()` arguments

```python
sp.score(df, model, id_col=None, state_col=None)
```

`id_col` / `state_col` default to whatever the model was fit with. `order`,
`alpha` and `add_boundaries` are read from the model — you cannot override them
at scoring time, which is the point.

### `summarize()` arguments

```python
sp.summarize(model, top=12)
```

`top` caps each section of the text report (states, origins, transitions,
runs) at that many rows.

---

## Running end to end

The full cycle is fit → inspect → persist → score. From the command line, one
invocation does all four:

```bash
cd adhoc_utils/markov_sequences

python seq_profile.py \
    --input sample_sequences.csv \
    --id customer_id \
    --state product_type \
    --time origination_date \
    --order 1 \
    --alpha 0.5 \
    --min-count 2 \
    --path-length 2 \
    --save  markov_product_type.json \
    --scores sequence_scores.csv
```

What happens, in order:

1. **Read** `--input` with `pd.read_csv`. No sorting.
2. **Prepare.** Keep only id / state / time. Null states → `MISSING`. If
   `--boundaries`, pad each id with `<START>` / `<END>`.
3. **Pair up.** Build one row per observed transition. A window is valid only
   if the id is unchanged across it, which is why contiguity is required.
   Raises `ValueError("no transitions found ...")` if nothing is valid.
4. **Aggregate** into the five output tables (below).
5. **Print** `summarize(model)` to stdout.
6. **Save** the model to `--save`, if given.
7. **Score** the *same* input against the model and write `--scores`, if given.

On the sample file this prints:

```
sequence profile: product_type within customer_id
----------------------------------------------------------------
ids                6
rows               24
transitions        18  (order 1)
distinct states    12
length  min/med/p95/max   1 / 4 / 6 / 7
thin share         33.3%  (transitions from an origin seen < 2 times)

STATES
  checking_basic                  6   25.0%       5 ids
  savings_standard                5   20.8%       3 ids
  auto_loan                       3   12.5%       3 ids
  MISSING                         2    8.3%       2 ids
  ...

ORIGINS  (entropy_norm: 0 = one certain next state, 1 = anyone's guess)
  checking_basic           n=6       H=0.65  ->savings_standard     0.33
  savings_standard         n=4       H=0.43  ->savings_standard     0.50
  auto_loan                n=2       H=0.29  ->checking_basic       0.50
  MISSING                  n=1       H=0.00  ->auto_loan            1.00 *thin
  ...

TOP TRANSITIONS  (lift vs the destination's overall share)
  checking_basic         -> savings_standard       n=2       p=0.33  lift=1.20  med 536d
  savings_standard       -> savings_standard       n=2       p=0.50  lift=1.80  med 332d
  ...

COMMON RUNS  (length 2)
  checking_basic>savings_standard                      2  11.1%
  savings_standard>savings_standard                    2  11.1%
  ...

model -> markov_product_type.json
scores -> sequence_scores.csv
```

and `sequence_scores.csv` looks like:

```
customer_id,n_transitions,mean_logprob,min_logprob,unseen_rate,mean_entropy,self_rate
C001,3,-1.660,-1.846,0.0,0.458,0.0
C002,4,-1.543,-2.037,0.0,0.380,0.5
C004,2,-1.752,-2.037,0.0,0.325,0.0
C005,3,-1.657,-2.037,0.0,0.217,0.0
C006,6,-1.658,-2.037,0.0,0.337,0.0
```

C003 is absent — one row, no transitions. See
[Sample input walkthrough](#sample-input-walkthrough).

### The two-step version (fit on train, score later)

The CLI scores the same file it fit on. For anything that feeds a model, fit on
training ids, persist, and score other batches from the saved file so the
features do not depend on who else is in the scoring batch — the same
discipline as `percentile_grids.json`:

```python
import pandas as pd
import seq_profile as sp

train = pd.read_csv("train_acquisitions.csv")        # already sorted
train = train[train["target"].isna()]                # drop outcome rows first

model = sp.fit(train, id_col="customer_id", state_col="product_family",
               time_col="origination_date", min_count=30)
print(sp.summarize(model))
model.save("markov_product_family.json")

# later, on a different batch
model = sp.SequenceModel.load("markov_product_family.json")
holdout = pd.read_csv("holdout_acquisitions.csv")
holdout = holdout[holdout["target"].isna()]
scores = sp.score(holdout, model)                   # one row per customer_id
```

### Using it against pipeline data

The raw acquisitions file that `preprocess.py` reads is the natural input.
Three things to do first:

- **Sort** by `[id, time]` with `kind="mergesort"` (raw extracts rarely arrive
  grouped).
- **Filter out outcome rows** (`target` non-null). They have a null state and
  will otherwise enter the chain as `MISSING`. The CLI has no filter flag;
  either pre-filter the CSV or use the Python API.
- **Pick the state grain.** `product_type` gives a wide, sparse alphabet;
  `product_family` (the rollup used in preprocessing) gives far fewer cells
  and a lower `thin share`. Start with the family.

---

## What comes out

`fit()` returns a `SequenceModel` holding five DataFrames and a `params` dict.

| Table | One row per | Columns |
|---|---|---|
| `states` | state | `state`, `n` (rows), `n_ids` (ids that ever see it), `share` |
| `lengths` | id | `id`, `length` (rows for that id, incl. sentinels if `--boundaries`) |
| `transitions` | (from, to) pair | `from_state`, `to_state`, `n`, `from_n`, `prob`, `prob_smoothed`, `thin`, `lift`, and `gap_days_median` / `gap_days_mean` when a time column was given |
| `from_states` | origin state | `from_state`, `n`, `n_distinct_next`, `entropy_bits`, `entropy_norm` (0 = one certain next state, 1 = uniform over the alphabet), `top_next`, `top_next_prob`, `thin` |
| `paths` | contiguous run | `path` (states joined with `>`), `n`, `share` |
| `pairs` | observed transition | `<id_col>`, `from_state`, `to_state`, and `gap_days` when a time column was given. **In-memory only** — see below. |
| `path_rows` | observed run | `<id_col>`, `path`. **In-memory only.** |

`params` records every fit argument plus `n_to_states`, `to_states`, `n_ids`,
`n_rows`, `n_transitions`, so a loaded model is self-describing.

`pairs` and `path_rows` are the row-level frames the aggregates were counted
from, with the id still attached. They exist only on the object returned by
`fit()`; `save()` does not write them and a model from `load()` has them set
to `None`. That keeps the JSON proportional to the number of states, not the
number of rows.

`lift` is `prob / (destination's unconditional share of all transitions)`.
Above 1 means this origin makes that destination more likely than chance.

`score()` returns one row per id:

| Column | Meaning |
|---|---|
| `n_transitions` | steps in the chain actually scored |
| `mean_logprob` | average smoothed log P(step). Low = an unusual path. |
| `min_logprob` | the single most surprising step |
| `unseen_rate` | share of steps whose (from, to) pair was never observed at fit time |
| `mean_entropy` | average `entropy_norm` of the origins this id passed through — how predictable its positions were |
| `self_rate` | share of steps that repeated the same state |

Prefer these over one-hot encoding the transition pairs. A handful of
continuous columns costs far fewer degrees of freedom than a block of dummies,
which matters when events are scarce.

---

## Mapping transitions and paths back to ids

The summary tells you *that* `cd_standard -> heloc` happened; these tell you
*who*. Both return a small frame of `<id_col>, n` sorted by `n` descending —
`n` is how many times that id made the move, since one customer can repeat a
transition (C002 does `savings_standard -> savings_standard` twice).

```python
model = sp.fit(df, "customer_id", "product_type", path_length=2)

model.ids_for_transition("cd_standard", "heloc")
#   customer_id  n
# 0        C005  1

model.ids_for_transition("savings_standard", "savings_standard")
#   customer_id  n
# 0        C002  2

model.ids_for_path("checking_basic>savings_standard")
#   customer_id  n
# 0        C001  1
# 1        C006  1
```

Rules:

- **`ids_for_path` needs a run of the same length as `path_length` at fit
  time.** A model fit with `path_length=3` cannot answer a two-state path;
  refit, or use `ids_for_transition` for the pairwise case.
- **Order 2 origins are joined with `>`.** For a model fit with `order=2`,
  `ids_for_transition("checking_basic>savings_standard", "auto_loan")`.
- **Sentinels work like any other state** when fit with `add_boundaries=True`:
  `ids_for_transition("<START>", "checking_basic")` is "who opened with a
  basic checking account".
- **A pair or path never seen returns an empty frame**, not an error.
- **Loaded models raise.** The lookups need `model.pairs` / `model.path_rows`,
  which are not persisted (see [What comes out](#what-comes-out)). Calling
  either on a `SequenceModel.load()` result raises a `ValueError` saying so.
  If you need the mapping, refit on the frame; it is cheap.

For anything the two helpers don't cover, filter the frames directly:

```python
# every transition C006 made, with the gap in days
model.pairs[model.pairs["customer_id"] == "C006"]

# ids whose chain ever passes through MISSING, from either side
p = model.pairs
p.loc[(p.from_state == "MISSING") | (p.to_state == "MISSING"), "customer_id"].unique()
```

---

## Python API

```python
import seq_profile as sp

model = sp.fit(df, id_col="customer_id", state_col="product_family")

print(sp.summarize(model, top=12))     # text block, paste into a ticket
model.matrix()                          # wide from x to table of prob
model.matrix(value="n")                 # ... or counts, lift, prob_smoothed
model.prob("checking_basic", "heloc")   # smoothed P(to | from); never 0
model.transitions                       # the long table
model.from_states, model.states, model.paths, model.lengths, model.params

model.ids_for_transition("checking_basic", "heloc")   # who made this move
model.ids_for_path("a>b>c")                           # who walked this run
model.pairs, model.path_rows                          # the id-level frames

model.save("markov.json")
model = sp.SequenceModel.load("markov.json")   # pairs / path_rows are None here
scores = sp.score(df, model)
```

`model.prob()` for a pair never seen from a known origin returns the smoothing
floor `alpha / (from_n + alpha * n_to_states)`; for an origin never seen at all
it returns `1 / n_to_states`.

---

## Things that are easy to get wrong

- **A transition table fit on all your data and then used to build features is
  a mild leak into cross-validation.** It is unsupervised, so it is a weak one,
  but if you care: fit on training ids only and persist the model, exactly as
  you would a percentile grid.
- **`order=2` needs far more data than `order=1`.** The number of cells grows
  as `states^(order+1)`. Check the `thin share` line in the summary before
  believing anything.
- **Boundaries are off by default.** With them on, "what do people start
  with" and "what do they stop after" become readable, but the state
  frequencies and lengths now include the sentinels.
- **Nothing is sorted or validated.** Wrong order produces a well-formed table
  of nonsense with no error. See the next section.
- **Outcome rows have a null state.** Filter them before fitting or they show
  up as `MISSING` transitions that describe your labelling convention, not
  customer behaviour.

---

## Sample input walkthrough

*This section is the original `SAMPLE_FORMAT.md`, kept intact.*

### Sample input for `seq_profile.py`

Two files, identical content:

| File | Purpose |
|---|---|
| `sample_sequences.csv` | correctly formatted |
| `sample_sequences_unsorted.csv` | the same 24 rows shuffled, to show what breaks |

> `sample_sequences_unsorted.csv` is not currently checked in. Make it with
> `df.sample(frac=1, random_state=0).to_csv(...)` if you want to reproduce the
> comparison below.

```
python seq_profile.py --input sample_sequences.csv \
    --id customer_id --state product_type --time origination_date \
    --min-count 2 --path-length 2
```

### The four requirements

1. **Grouped.** All rows for an id are contiguous.
2. **Ordered within the group**, oldest first.
3. **One row per event.** Balances are the values at origination; the utility
   ignores them entirely and reads only the id and the state.
4. **The state column is what you want to profile.** Anything else is along for
   the ride and can stay in the frame.

Nothing here is validated. The frame is taken at its word.

### What each customer demonstrates

| id | Rows | Point |
|---|---|---|
| C001 | 4 | The ordinary case. Four states, three transitions. |
| C002 | 5 | Two rows on `2018-11-05` — a same-day bundle. Ties resolve to file order, so **decide that order upstream**. Also three consecutive `savings_standard`: self-transitions are real transitions and appear on the diagonal. |
| C003 | 1 | Contributes to state counts but produces no transition, so it is absent from `score()`. Left-join, never inner-join. |
| C004 | 3 | A genuinely null `product_type`. Becomes the state `MISSING` rather than disappearing — visible in the profile, which is the point. |
| C005 | 4 | Carries an **outcome row**: id, date, `target=True`, everything else null. |
| C006 | 7 | The longest chain, and the only source of several transition pairs. |

### Filter outcome rows before fitting

C005's target row has no `product_type`, so it enters the chain as `MISSING`
and invents a transition that describes your labelling convention rather than
customer behaviour:

```
heloc    -> MISSING     n=1
```

(An earlier version of this note also blamed `MISSING -> auto_loan` on the
outcome row. `model.ids_for_transition("MISSING", "auto_loan")` shows it
belongs to C004 — a real null state followed by a real product — and the
id-contiguity check already stops C005's trailing `MISSING` from pairing with
C006's first row. That is what the lookup is for.)

So:

```python
acq = df[df["target"].isna()]
model = sp.fit(acq, "customer_id", "product_type")
```

`MISSING` still appears afterwards, from C004. That is correct and is the
distinction worth holding onto: a null state is data, an outcome row is
bookkeeping.

### What unsorted input costs you

Shuffling the rows changes nothing that raises an error. It changes the answer:

| | sorted | shuffled |
|---|---|---|
| transitions found | 18 | 6 |
| distinct origins | 9 | 4 |

The id-contiguity check quietly rejects most adjacent pairs, and the few it
accepts are pairs that happened to land next to each other. The output is a
well-formed table of nonsense. If there is any doubt about provenance, sort
defensively before calling `fit`:

```python
df = df.sort_values([id_col, time_col], kind="mergesort")
```

`mergesort` is stable, so same-day rows keep whatever order the source gave
them — matching the tie behaviour in `preprocess.py`.
