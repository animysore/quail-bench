# QUAIL-B reference

This page specifies what an adapter receives and returns, and exactly how
QUAIL-B computes each metric. The [README](../README.md) covers installation,
running the benchmark, the queries, and the results.

- [Adapter function](#adapter-function)
- [`QuerySpec`](#queryspec)
- [Input tables](#input-tables)
- [Prompt rendering](#prompt-rendering)
- [`RunOutput`](#runoutput)
- [Predicate answers](#predicate-answers)
- [Measurements](#measurements)
- [Prompt pieces](#prompt-pieces)
- [Metric requirements](#metric-requirements)
- [Metric definitions](#metric-definitions)
- [Run records and rescoring](#run-records-and-rescoring)
- [Extraction samples](#extraction-samples)

## Adapter function

```python
def run_query(
    query: quail_b.QuerySpec,
    tables: dict[str, pyarrow.Table],
) -> quail_b.RunOutput:
    ...
```

The adapter must finish one query or raise an exception. QUAIL-B stops the run
at the first exception and records the failure in `run.json`.

The smallest valid result is:

```python
quail_b.RunOutput(
    filter_answers=None,
    join_answers=None,
    rows=rows,
    runtime_s=runtime_s,
)
```

## `QuerySpec`

| Attribute | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Stable query ID, such as `IMDB-4` |
| `description` | `str` | Short description of the query shape |
| `plan` | `substrait.plan_pb2.Plan` | Parsed Substrait 0.103 plan |
| `plan_bytes` | `bytes` | Serialized form of the same plan |

The plan defines table scans, relation aliases, prompts, filters, joins,
equality conditions, and the final projection. It uses the standard Substrait
relational operators and these extension functions:

```text
ai_filter:str_str
ai_join:str_str_str
ai_classify:str_str_list_list
```

A classification appears as a `ProjectRel` that adds one label column to its
relation; its hint alias is the operator ID, such as `classify-1`, and its
hint output names end with `alias.column`, such as `r.sentiment`. A `FilterRel`
whose condition is a standard `SingularOrList` over that column keeps the
documents with an accepted label; its ID is `label-filter-N` and it asks the
model nothing. See [Classification](#classification).

Their declarations and URN are in
[`quail_b/substrait_extensions.yaml`](../quail_b/substrait_extensions.yaml).

## Input tables

`tables` maps physical table names from the plan to `pyarrow.Table` values. It
contains only the tables the current query scans. For IMDB-4:

```python
{
    "reviews": reviews_table,
    "aspects": aspects_table,
}
```

Each table contains its published `id` and document columns. Physical table
names are the dictionary keys. Relation aliases are the column names of `rows`
and the answer tables. For `reviews AS r`, the dictionary key is `reviews` and
the column name is `r`.

## Prompt rendering

The string argument of each AI function is a prompt template. Reference labels
correspond to the complete prompt produced from that template and its document
or documents.

Use the rendering functions in `quail_b.rendering`, or produce identical text:

```python
from quail_b.rendering import render_filter_prompt, render_join_prompt

filter_text = render_filter_prompt(template, document)
join_text = render_join_prompt(template, documents, anchor=0)
```

Every prompt starts with a document. The question follows it and begins
"Evaluate TRUE or FALSE for the following question:". Because the document
comes first, an engine can compute a document's KV once and reuse it across
every question asked of that document.

Classification prompts use `render_classify_prompt(template, document,
labels, descriptions)`, described under [Classification](#classification).

For joins, `documents` follows template placeholder order. `anchor` selects the
document placed first. Both renderers end with `ANSWER:`, and the model answer
must be read as `TRUE` or `FALSE`.

The published labels use these templates and this rendering. A different
prompt defines a different predicate, and its results are incomparable with
the labels.

## `RunOutput`

| Field | Type | Required for |
| --- | --- | --- |
| `filter_answers` | `dict[str, pa.Table] \| None` | Predicate and token metrics |
| `join_answers` | `dict[str, pa.Table] \| None` | Predicate and token metrics |
| `rows` | `pa.Table` | Every run |
| `runtime_s` | `float` | Every run |
| `measurements` | `dict` | Optional engine measurements |
| `prompt_pieces` | `dict \| None` | Token and KV metrics |
| `classify_answers` | `dict[str, pa.Table] \| None` | Label accuracy |
| `extract_answers` | `dict[str, pa.Table] \| None` | Optional extraction traces |

Extraction queries add nullable response/error struct columns to `rows`.
See [Extraction](extraction.md#adapter-output-and-scoring) for their schema,
optional answer tables, NULL handling and coverage checks.

### Result rows

`rows` contains one ID column per relation alias selected by the plan. Its
column names must match the selected aliases exactly. IMDB-4 selects `r.id`
and `a.id`:

```python
pa.table({
    "r": ["rv17", "rv42"],
    "a": ["as0", "as1"],
})
```

A query that returns a label column, such as IMDB-13's `r.sentiment`, adds
one string column named by the query, here `sentiment`, holding one of that
call's labels.

The harness rejects:

- missing or additional columns;
- null IDs;
- IDs absent from the corresponding input table;
- duplicate ID tuples.

Scoring ignores column order and row order.

### Runtime

`runtime_s` is a finite, nonnegative number of seconds. It includes query
execution through completion of asynchronous model or GPU work. It excludes
engine startup, model loading, result collection, scoring, and result saving.
Use the same timing boundary for every system you compare.

## Predicate answers

`rows` shows only the final result. `filter_answers` and `join_answers` show
how the engine got there: the value of each AI predicate for every tuple the
engine evaluated it on. QUAIL-B compares each value with its reference label to
measure the accuracy of each operator.

Both fields are optional. Each maps an operator ID from the plan to a table
with one row per evaluated tuple and a boolean `answer` column.

### Example

IMDB-4 has three AI operators:

```text
Project [r.id, a.id]
└── AI Join J1                   join-1
    ├── AI Selection F4          filter-2
    │   └── AI Selection F1      filter-1
    │       └── Scan reviews AS r
    └── Scan aspects AS a
```

Suppose `reviews` has three rows and `aspects` has two. The engine evaluates
`filter-1` on every review, `filter-2` on the reviews that pass `filter-1`, and
`join-1` on each remaining review paired with each aspect:

```python
filter_answers = {
    "filter-1": pa.table({
        "r": ["rv17", "rv42", "rv50"],
        "answer": [True, True, False],
    }),
    "filter-2": pa.table({
        "r": ["rv17", "rv42"],
        "answer": [True, False],
    }),
}
join_answers = {
    "join-1": pa.table({
        "r": ["rv17", "rv17"],
        "a": ["as0", "as6"],
        "answer": [True, False],
    }),
}
rows = pa.table({"r": ["rv17"], "a": ["as0"]})
```

A selection table has one ID column, named by the alias of the relation it
filters. A join table has one ID column for each input alias. Each table lists
only the tuples the engine evaluated, so a different plan or operator order
produces different tables for the same query.

### Rules

- IDs and answers must be nonnull, and every ID must exist in its input table.
- Each ID, or each ID pair for a join, appears at most once per table.
- Return a table for some operators to score only those operators.
- Return a table for every operator to enable the checks and metrics below.
  Use `{}` for a query with zero operators of one kind, and `None` to omit a
  kind entirely.

When every operator has a table, QUAIL-B also:

- checks that `rows` matches the result those answers produce: it compares the
  row count and verifies a sample of rows;
- counts evaluated join pairs from the join tables; and
- computes input and minimum tokens from these tables and `prompt_pieces`.

## Classification

`ai_classify(prompt, document, labels, descriptions)` returns one label from
the list for each document. `descriptions` has one entry per label; an empty
string means none. Labels are distinct, and no label continues another label
word for word.

### Prompt and reference label

`render_classify_prompt` produces the text the labels are scored after:

```text
DOCUMENT:
<document>

Answer with exactly one of the categories below for the following question: <question>

Categories:
- <label 1>: <description 1>
- <label 2>
ANSWER:
```

Each label follows as `" " + label`. The reference label is the label with the
largest sum of its tokens' log probabilities, each normalized over the full
vocabulary at temperature 1, from `Qwen/Qwen3-32B-FP8`. No end marker is
scored, and the earlier label wins a tie. `quail_b.predicates.CLASSIFY_JUDGE_SPEC`
records this definition. An engine may compute it any way, for example by
scoring only the first token when the first tokens differ, but its answers are
compared with this definition.

### Answers

`classify_answers` maps each classify operator ID to a table with the
relation's alias column and a string `label` column, one row per document the
engine classified. A document with no row cannot pass a label filter or appear
with its label. Every label must be one of the call's labels.

```python
classify_answers = {
    "classify-1": pa.table({
        "r": ["rv17", "rv42"],
        "label": ["negative", "mixed"],
    }),
}
```

With answer tables for every operator, `rows` must match the rows the answers
imply, label columns included.

### Label files

A classification label set stores `left_id` and a string `label` column where a
filter stores its boolean `answer`. Its manifest's predicate lists `labels`.

### Queries waiting for labels

A query marked `labels_pending` in the catalog has no published labels yet.
`quail_b.queries()` leaves it out; `queries(include_pending=True)` and
`get_query` return it. Running one needs a label collection that includes its
predicates, passed with `root` or `collection_id`. No query is pending now.

## Measurements

`measurements` holds numbers the engine reports. QUAIL-B recognizes three keys:

| Key | Type | Meaning |
| --- | --- | --- |
| `evaluated_document_pairs` | nonnegative `int` | Pairs across all joins |
| `fresh_tokens` | nonnegative `int` | Positions processed by model forward passes |
| `input_tokens` | nonnegative `int` | Full length of every evaluated prompt |

When every join has an answer table, QUAIL-B uses the sum of their row counts
in place of `evaluated_document_pairs`.

`input_tokens` counts every position of every evaluated prompt, including
positions read from KV. Report it when prompt pieces are unavailable. It
enables input token throughput and cost per million input tokens. Minimum
tokens and KV regret require `prompt_pieces`.

**Approximate KV regret.** Some engines, such as vLLM, tokenize each whole
prompt as one string. Their token count can differ slightly from the count the
pieces give, since a tokenizer can merge text across a piece boundary. Such an
engine reports `input_tokens` along with `prompt_pieces`. QUAIL-B then uses
the reported count, scales the minimum to match it, and marks KV regret as
approximate with `regret_approximate`.

QUAIL-B saves other JSON serializable measurements as engine metadata.

## Prompt pieces

`prompt_pieces` describes the token IDs around each document. With it, QUAIL-B
can count the tokens every evaluated prompt contains and the minimum an engine
must compute.

| Entry | Type | Meaning |
| --- | --- | --- |
| `tokenizer` | `str` | Hugging Face tokenizer name used for documents |
| `preamble` | `list[int]` | Tokens before each first document |
| `filters` | `list[dict]` | One `id` and `tail` token list per filter |
| `joins` | `list[dict]` | One join description per join |

Each filter description has:

```text
id: operator ID
tail: tokens after the document
```

Each join description has:

```text
id: operator ID
anchor: alias of the document placed first
frame: tokens after the anchor document
label: tokens before the partner document
tail: tokens after the partner document
```

Prompt pieces require:

- an answer table for every filter and join;
- every filter and join listed exactly once;
- `measurements["fresh_tokens"]`;
- the same tokenizer and token layout used during execution, or a reported
  `input_tokens` total when the engine tokenized full prompts itself.

QUAIL-B tokenizes the documents after execution and derives input tokens,
minimum tokens, recomputed tokens, and KV regret.

## Metric requirements

| Metric | Adapter data |
| --- | --- |
| Query time | `runtime_s` |
| Output precision, recall, F1, exact match | `rows` |
| Document throughput | `runtime_s`, for a query with zero joins |
| Join throughput | Answer tables for every join, or a reported pair count |
| Predicate-level accuracy | Predicate answers |
| Label accuracy | Classification answers |
| Extraction field agreement, reference field recall and coverage | Extraction result rows; optional matching extraction answer tables |
| Generated extraction tokens | Nonnegative integer `measurements["output_tokens"]` |
| Fresh tokens | `measurements["fresh_tokens"]` |
| Input tokens and their throughput | Prompt pieces, or reported input tokens |
| Minimum tokens and KV regret | All answer tables, prompt pieces, and fresh tokens; not yet defined for classification or extraction queries |
| GPU cost | `gpu_count` and `gpu_hourly_rate_usd` |
| Cost per million input tokens | GPU cost and a positive input token count |

A metric that lacks its data is `unavailable` in `report.md` and null in
`run.json` and `measurements.parquet`. QUAIL-B never reports a missing count as
zero.

The extraction metrics distinguish missing documents, generation errors and
successful NULL fields; see [their definitions](extraction.md#adapter-output-and-scoring).

## Metric definitions

### Output quality

QUAIL-B compares the distinct ID tuples in `rows` with the reference result.
Precision is the share of returned rows that are in the reference result.
Recall is the share of reference rows that were returned. F1 is their harmonic
mean. Exact match means the two row sets are equal.

### Predicate-level accuracy

Predicate-level accuracy is the share of the engine's filter and join answers
that match the reference labels. It counts only the tuples the engine
evaluated. `run.json` also records true and false positives and negatives for
each predicate.

### Label accuracy

Label accuracy is the share of the engine's classification answers that match
the reference labels, over the documents the engine classified. It is reported
beside predicate-level accuracy, not merged into it.

Accuracy is not a focus of this benchmark. See
[Reference answers](../README.md#reference-answers) for how the labels were
made.

### Throughput

Document throughput is input documents divided by `runtime_s`. Join throughput
is evaluated document pairs, summed across all joins, divided by `runtime_s`.

### Tokens and KV

| Metric | Definition |
| --- | --- |
| Input tokens | Full evaluated prompts, including tokens served from KV |
| Fresh tokens | Input positions processed by model forward passes |
| Minimum tokens | Input positions required with an unlimited prefix KV cache |
| Recomputed tokens | Fresh tokens minus minimum tokens |
| KV regret | Recomputed tokens divided by fresh tokens, as a percentage |
| Input token throughput | Input tokens divided by `runtime_s` |

The minimum counts each distinct prompt prefix once, so a document's questions
share the document and any leading tokens they have in common. For joins, each
pair's label, partner document, and answer cue count once per pair. Recomputed
tokens are therefore the work a perfect prefix KV cache would have avoided.

Input tokens depend on which prompts the plan evaluates. Fresh tokens measure
model computation. Two engines can therefore have the same input tokens and
different fresh tokens.

### Cost

```text
GPU cost = runtime_s / 3600 * gpu_count * gpu_hourly_rate_usd
cost per million input tokens = GPU cost / input tokens * 1,000,000
```

## Run records and rescoring

`run.json` records the QUAIL-B version, the corpus ID, the reference collection
ID, a hash of each query definition, each query's status, and its metrics. If
scoring fails, the query status is `scoring_failed` and its saved files remain
in place.

`quail-b report` rescores saved files against the recorded corpus and
reference collection. It requires corpus and query hashes that match the
installed QUAIL-B, and it rewrites `run.json`, `report.md`, and
`measurements.parquet`.

## Extraction samples

The fork's field-name extraction contract, synthetic queries, exact prompt,
response/error types and scoring rules are described in
[Field-name extraction samples](extraction.md). The suite is opt-in and does
not change published query or corpus identities.
