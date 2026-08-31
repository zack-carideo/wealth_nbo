# Sample input for `seq_profile.py`

Two files, identical content:

| File | Purpose |
|---|---|
| `sample_sequences.csv` | correctly formatted |
| `sample_sequences_unsorted.csv` | the same 24 rows shuffled, to show what breaks |

```
python seq_profile.py --input sample_sequences.csv \
    --id customer_id --state product_type --time origination_date \
    --min-count 2 --path-length 2
```

## The four requirements

1. **Grouped.** All rows for an id are contiguous.
2. **Ordered within the group**, oldest first.
3. **One row per event.** Balances are the values at origination; the utility
   ignores them entirely and reads only the id and the state.
4. **The state column is what you want to profile.** Anything else is along for
   the ride and can stay in the frame.

Nothing here is validated. The frame is taken at its word.

## What each customer demonstrates

| id | Rows | Point |
|---|---|---|
| C001 | 4 | The ordinary case. Four states, three transitions. |
| C002 | 5 | Two rows on `2018-11-05` — a same-day bundle. Ties resolve to file order, so **decide that order upstream**. Also three consecutive `savings_standard`: self-transitions are real transitions and appear on the diagonal. |
| C003 | 1 | Contributes to state counts but produces no transition, so it is absent from `score()`. Left-join, never inner-join. |
| C004 | 3 | A genuinely null `product_type`. Becomes the state `MISSING` rather than disappearing — visible in the profile, which is the point. |
| C005 | 4 | Carries an **outcome row**: id, date, `target=True`, everything else null. |
| C006 | 7 | The longest chain, and the only source of several transition pairs. |

## Filter outcome rows before fitting

C005's target row has no `product_type`, so it enters the chain as `MISSING`
and invents two transitions that describe your labelling convention rather than
customer behaviour:

```
heloc    -> MISSING     n=1
MISSING  -> auto_loan   n=1      (this one crosses from C004 into C005)
```

So:

```python
acq = df[df["target"].isna()]
model = sp.fit(acq, "customer_id", "product_type")
```

`MISSING` still appears afterwards, from C004. That is correct and is the
distinction worth holding onto: a null state is data, an outcome row is
bookkeeping.

## What unsorted input costs you

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
