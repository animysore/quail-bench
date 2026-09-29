# Field-name extraction samples

The fork includes three synthetic correctness queries, separate from the
published academic workloads. Their Substrait plans, catalog, inputs and
hand-authored references are bundled in `quail_b/samples/extraction/`.
They require no data downloads. These small examples do not establish a
performance improvement or represent the published datasets.

| Query | Input | Operation | Reference output rows |
| --- | --- | --- | ---: |
| EXTRACT-1 | 8 synthetic documents | Extract vendor and total | 8 |
| EXTRACT-2 | Same 8 documents | Filter invoices, then extract vendor and total | 6 |
| EXTRACT-3 | 3 synthetic reviews | Extract movie title and rating | 3 |

For example, EXTRACT-1 has this SQL meaning:

```sql
SELECT d.id AS d, AI.EXTRACT(d.body, 'vendor', 'total') AS extracted
FROM extract_invoices AS d;
```

Field names are nonempty, distinct strings. Every successful field is a string
copied from the document, or NULL when the information is absent. Dates,
amounts and ratings are not converted to other types. Comparison is exact,
including whitespace, punctuation and case.

The returned `extracted` column has this Arrow type:

```text
struct<
  response: struct<vendor: string?, total: string?>?,
  error: string?
>?
```

A non-NULL document returns either a response with every requested field and
a NULL error, or a NULL response and a nonempty error message. A NULL document
returns a NULL outer struct without inference. Successful all-NULL fields,
failed generation and a missing result row are separate outcomes.

The prompt is defined by `quail_b.rendering.render_extract_prompt`. Adapters
must use this exact instruction, field order and document text, recording any
model-specific chat framing and generation limits in run metadata. References
are manually authored, not model-generated. In particular, the invoice text
`Total: GBP 70.` has reference value `GBP 70`; a model that retains the sentence
punctuation receives one field mismatch.

## Load or run the samples

```python
import quail_b
from quail_b.scoring import expected_rows

suite = quail_b.load_benchmark(
    ["EXTRACT-1", "EXTRACT-2", "EXTRACT-3"], scale_factor=1.0)
query = suite.queries[0]
print(query.plan)
print(suite.tables["extract_invoices"])
print(expected_rows(query, suite.ground_truth, suite.tables))
```

Run them through the same adapter contract as the published benchmark:

```python
quail_b.run(
    my_adapter,
    queries=["EXTRACT-1", "EXTRACT-2", "EXTRACT-3"],
    scale_factor=1.0,
    output_dir="results/extraction",
    gpu_count=1,
    gpu_hourly_rate_usd=3.9492,
    metadata={"engine": "my_engine", "model": "my_model"},
)
```

Use a new output directory. The fixed corpus accepts only scale factor 1.0.
Run sample queries separately from academic queries. The default query list
does not include samples; `quail_b.queries.queries(include_samples=True)` lists
them explicitly. The ordinary published corpus IDs and query hashes are unchanged.

The sample corpus ID hashes exact input IDs, texts and the fixture version.
The reference collection ID additionally hashes the reference values, fields,
extraction instruction and filter prompt. Local `data_dir` Parquet files must
match the bundled inputs exactly. Rescoring refuses changed corpus, query or
reference identities.

## Adapter output and scoring

Return `RunOutput.rows` with one ID column named by the relation alias and
the selected response/error struct columns. IDs are stable corpus IDs, not row
positions. Each extraction projection must appear in the result. Multiple
projections are supported on one input table, optionally after AI filters.
Joins, classification, filtering on extracted fields and LIMIT are not part
of the initial extraction benchmark plan format.

`RunOutput.extract_answers` optionally maps operator IDs such as `extract-1`
to tables with the alias ID column and that operator's named output column.
Include NULL input rows. If a table is provided, its IDs and values must exactly
match the corresponding result columns. Recorded filters must allow every
extracted document.
Without that table, field scoring uses the corresponding saved result rows.
Final row scoring always uses the saved result rows.

Scoring reports:

- **Field agreement:** correct fields divided by observed non-NULL documents
  times their field count. An error contributes all its fields to the
  denominator and zero to the numerator.
- **Reference field recall:** correct fields on reference-surviving documents
  divided by all requested fields of their non-NULL inputs. Missing documents
  and errors reduce this metric. A missed row cannot increase it.
- **Coverage:** expected, covered and missing reference documents; observed
  generation errors and NULL inputs. Counts repeat per extraction projection.
- **Output precision and recall:** exact IDs and all selected values, including
  NULL outer structs, compared with the reference result.

An absent response field in the Arrow schema is invalid. String fields use
`pa.string()`, not `pa.large_string()`. Nullable fields
present in the schema represent explicit NULL values; adapters must classify
missing model JSON keys as errors before constructing that typed Arrow value.
Errors cannot be reported as successful all-NULL fields.

Set `runtime_s` after GPU execution completes, excluding model startup,
result collection and scoring. Report `measurements["input_tokens"]` as the
complete length of every evaluated prompt, counting a shared prefix every time,
and `measurements["output_tokens"]` separately. `fresh_tokens` remains optional
uncached input work. Extraction does not yet define the minimum-prefix metric;
minimum and KV regret are unavailable, never zero by assumption.
When present, output token counts must be nonnegative integers.

Adapters that iterate over `QuerySpec._info.ai_operators` must handle extraction
operators or reject the query before loading an engine.

The supplied GPU rate is a reference rate. Cost is runtime hours multiplied
by GPU count and rate; it is not an account billing measurement.

The regular run/report path saves plans, rows, optional extraction answers,
metrics and configuration. `quail-b report results/extraction` reloads and
rescores them offline.

## Regenerate plans

Edit sample definitions in `tools/make_extract_plans.py` and run:

```sh
uv run python -m tools.make_extract_plans
```

`tools/make_substrait_plans.py` continues to generate only the published
catalog. Extraction uses a typed Substrait scalar call:
`ai_extract(document, fields, result_type)`. The third argument is type
metadata inferred from the fields, not another SQL argument. Both it and
`output_type` must declare the response/error struct. Nested names in
`RelRoot.names` and project hints are in depth-first order.
