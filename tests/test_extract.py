"""Synthetic extraction plans, identities, error handling, and saved scores."""

import importlib
import json
from importlib.resources import files

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail_b
from quail_b import extract_samples
from quail_b.extraction import reference_key, result_type
from quail_b.queries import QuerySpec, queries
from quail_b.rendering import extract_template, render_extract_prompt
from quail_b.run import _query_hash, _read_output, _score, _validate_output
from quail_b.scoring import expected_rows
from tools.make_extract_plans import write_plans
from tools.make_substrait_plans import (
    Classify,
    Extract,
    Filter,
    Scan,
    build_plan,
)

SAMPLE_IDS = ("EXTRACT-1", "EXTRACT-2", "EXTRACT-3")


@pytest.fixture
def suite():
    return quail_b.load_benchmark(SAMPLE_IDS, scale_factor=1.0)


def _output(suite, index=0, *, traced=False):
    spec = suite.queries[index]
    rows = expected_rows(spec, suite.ground_truth, suite.tables)
    filters = {}
    if spec._info.filters:
        labels = next(iter(suite.ground_truth.predicates.values()))
        filters = {"filter-1": labels.table.select(
            ["left_id", "answer"]).rename_columns(["d", "answer"])}
    return quail_b.RunOutput(
        filters if traced else None, {} if traced else None, rows,
        runtime_s=2.0,
        measurements={"input_tokens": 538, "output_tokens": 115},
        extract_answers={
            op.id: rows.select([op.relation, op.output])
            for op in spec._info.extracts} if traced else None)


def _score_output(suite, output, index=0):
    return _score(suite.queries[index], output, suite, 1, 3.9492)


def _replace_values(output, values, fields=("vendor", "total")):
    output.rows = pa.table({
        "d": output.rows["d"],
        "extracted": pa.array(values, result_type(fields)),
    })


def test_samples_are_explicit_and_keep_the_published_catalog(suite):
    assert len(queries()) == 41
    assert not set(SAMPLE_IDS) & queries().keys()
    assert tuple(name for name in queries(include_samples=True)
                 if name.startswith("EXTRACT-")) == SAMPLE_IDS
    assert suite.kind == "synthetic-extraction-v1"
    assert suite.ground_truth.reference_model is None
    assert suite.corpus_id.startswith("c_synthetic_extract_")
    assert [expected_rows(spec, suite.ground_truth, suite.tables).num_rows
            for spec in suite.queries] == [8, 6, 3]


def test_checked_in_sample_plans_equal_the_generator(tmp_path):
    write_plans(tmp_path)
    package = files("quail_b").joinpath("samples", "extraction")
    assert len(list(tmp_path.iterdir())) == 4
    for path in tmp_path.iterdir():
        assert path.read_text() == package.joinpath(path.name).read_text()
    assert package.joinpath("fixtures.json").is_file()


def test_extract_plan_has_nullable_nested_types_and_names(suite):
    spec = suite.queries[0]
    root = spec.plan.relations[0].root
    assert list(root.names) == [
        "d", "extracted", "response", "vendor", "total", "error"]
    project = root.input.project.input.project
    assert list(project.common.hint.output_names) == [
        "d.id", "d.body", "d.extracted", "response", "vendor", "total", "error"]
    call = project.expressions[0].scalar_function
    assert call.arguments[2].type == call.output_type
    assert spec._info.extracts[0].fields == ("vendor", "total")
    copy = QuerySpec.from_plan(spec.id, spec.description, spec.plan)
    assert copy.plan_bytes == spec.plan_bytes
    extension = files("quail_b").joinpath("substrait_extensions.yaml").read_text()
    assert "name: ai_extract" in extension


@pytest.mark.parametrize("fields, match", [
    ((), "nonempty"), ((" ",), "nonempty"),
    (("vendor", "vendor"), "distinct"),
])
def test_invalid_fields_are_rejected(fields, match):
    tree = Extract(Scan("documents", "d", "body"), fields, "extracted")
    with pytest.raises(ValueError, match=match):
        QuerySpec.from_plan("TEST", "", build_plan(tree, ("d", "d.extracted")))


@pytest.mark.parametrize("change, match", [
    ("output_type", "output type"), ("type_argument", "result type"),
    ("root_names", "root names"), ("hint_names", "field names"),
    ("document_index", "out of range"), ("urn", "wrong URN"),
])
def test_invalid_typed_plans_are_rejected(suite, change, match):
    plan = suite.queries[0].plan
    root = plan.relations[0].root
    project = root.input.project.input.project
    call = project.expressions[0].scalar_function
    if change == "output_type":
        call.output_type.Clear()
    elif change == "type_argument":
        del call.arguments[2]
    elif change == "root_names":
        root.names[3] = "wrong"
    elif change == "hint_names":
        project.common.hint.output_names[4] = "wrong"
    elif change == "document_index":
        call.arguments[0].value.selection.direct_reference.struct_field.field = -1
    else:
        plan.extension_urns[0].urn = "wrong"
    with pytest.raises(ValueError, match=match):
        QuerySpec.from_plan("TEST", "", plan)


def test_projection_and_filter_order_are_checked():
    tree = Extract(Scan("documents", "d", "body"), ("vendor",), "extracted")
    for candidate, selected, match in [
        (tree, ("d",), "never used"),
        (tree, ("d.extracted",), "relation's id"),
        (tree, ("d", "d.extracted", "d.extracted"), "repeat a column"),
        (Filter(tree, "Keep? {0}"), ("d", "d.extracted"), "precede extraction"),
        (Extract(Classify(Scan("documents", "d", "body"),
                          "{0}", ("yes", "no"), "label"),
                 ("vendor",), "extracted"),
         ("d", "d.label", "d.extracted"), "without classify"),
    ]:
        with pytest.raises(ValueError, match=match):
            QuerySpec.from_plan("TEST", "", build_plan(candidate, selected))


def test_multiple_extractions_keep_fields_and_change_query_identity():
    scan = Scan("extract_invoices", "d", "body", nullable_text=True)
    tree = Extract(Extract(scan, ("vendor",), "vendor_value"),
                   ("total",), "total_value")
    spec = QuerySpec.from_plan(
        "TEST", "", build_plan(tree, ("d", "d.vendor_value", "d.total_value")))
    assert [op.id for op in spec._info.extracts] == ["extract-1", "extract-2"]
    assert list(spec.plan.relations[0].root.names) == [
        "d", "vendor_value", "response", "vendor", "error",
        "total_value", "response", "total", "error"]
    first = QuerySpec.from_plan("TEST", "", build_plan(
        Extract(scan, ("vendor", "total"), "value"), ("d", "d.value")))
    second = QuerySpec.from_plan("TEST", "", build_plan(
        Extract(scan, ("total", "vendor"), "value"), ("d", "d.value")))
    assert _query_hash(first) != _query_hash(second)
    assert reference_key("extract_invoices", "body", ("vendor", "total")) != (
        reference_key("extract_invoices", "body", ("total", "vendor")))


def test_rendering_preserves_document_and_field_braces():
    fields = ("a{0}", 'quote"field', "日本語")
    document = 'Keep {0} and "\\n" literally.'
    assert render_extract_prompt(document, fields) == (
        "DOCUMENT:\n" + document + extract_template(fields)[3:])
    assert json.dumps(fields, ensure_ascii=False) in extract_template(fields)
    assert "Return only the JSON object.\nANSWER:" in extract_template(fields)


@pytest.mark.parametrize("traced", [False, True])
def test_reference_outputs_score_and_rescore_identically(suite, tmp_path, traced):
    outputs = {spec.id: _output(suite, index, traced=traced)
               for index, spec in enumerate(suite.queries)}
    record = quail_b.run(
        lambda spec, tables: outputs[spec.id], queries=SAMPLE_IDS,
        scale_factor=1.0, output_dir=tmp_path / "run",
        gpu_hourly_rate_usd=3.9492)
    assert record["status"] == "complete"
    assert record["benchmark_suite"] == "synthetic-extraction-v1"
    assert record["reference_source"] == "hand-authored synthetic fixtures"
    for item in record["queries"]:
        metrics = item["metrics"]
        counts = metrics["accuracy"]["extraction_accuracy"]
        assert counts["accuracy"] == counts["reference_field_recall"] == 1
        assert counts["error_documents"] == counts["missing_documents"] == 0
        assert metrics["accuracy"]["output_accuracy"]["exact_match"]
        assert metrics["input_tokens_per_second"] == 269
        assert metrics["output_tokens"] == 115
        assert metrics["cost_usd"] == pytest.approx(2 * 3.9492 / 3600)
        assert metrics["minimum_tokens"] is metrics["regret_tokens"] is None
        loaded = _read_output(tmp_path / "run" / item["id"], item)
        assert loaded.rows.equals(outputs[item["id"]].rows)
        if traced:
            assert loaded.extract_answers["extract-1"].equals(
                outputs[item["id"]].extract_answers["extract-1"])
    quail_b.report(tmp_path / "run")
    rescored = json.loads((tmp_path / "run/run.json").read_text())
    assert [item["metrics"] for item in rescored["queries"]] == [
        item["metrics"] for item in record["queries"]]
    assert "## Extraction" in (tmp_path / "run/report.md").read_text()
    flat = pq.read_table(tmp_path / "run/measurements.parquet")
    assert flat["extraction_fields_correct"].to_pylist() == [14, 12, 4]


def test_errors_do_not_receive_credit_for_missing_information(suite):
    output = _output(suite)
    values = output.rows["extracted"].to_pylist()
    values[6] = {"response": None, "error": "Extraction exceeded its output limit"}
    _replace_values(output, values)
    metrics = _score_output(suite, output)["accuracy"]
    counts = metrics["extraction_accuracy"]
    assert counts["correct"] == 12
    assert counts["evaluated"] == 14
    assert counts["error_documents"] == 1
    assert counts["null_inputs"] == 1
    assert counts["missing_documents"] == 0
    assert counts["reference_field_recall"] == pytest.approx(12 / 14)
    assert metrics["output_accuracy"]["matching_rows"] == 7


def test_missing_rows_are_coverage_loss_not_null_fields(suite):
    output = _output(suite)
    output.rows = output.rows.slice(1)
    counts = _score_output(suite, output)["accuracy"]["extraction_accuracy"]
    assert counts["missing_documents"] == 1
    assert counts["covered_documents"] == 7
    assert counts["accuracy"] == 1
    assert counts["reference_field_recall"] == pytest.approx(12 / 14)
    assert counts["error_documents"] == 0


def test_complete_trace_cannot_hide_a_missing_result_row(suite):
    output = _output(suite, traced=True)
    output.rows = output.rows.slice(1)
    with pytest.raises(ValueError, match="not implied"):
        _score_output(suite, output)


def test_exact_field_agreement_keeps_punctuation_and_case(suite):
    output = _output(suite)
    values = output.rows["extracted"].to_pylist()
    values[3]["response"]["total"] = "GBP 70."
    values[0]["response"]["vendor"] = "acme"
    _replace_values(output, values)
    counts = _score_output(suite, output)["accuracy"]["extraction_accuracy"]
    assert counts["correct"] == 12 and counts["evaluated"] == 14


@pytest.mark.parametrize("value, index, match", [
    (None, 0, "non-NULL document"),
    ({"response": None, "error": None}, 0, "response struct"),
    ({"response": None, "error": ""}, 0, "needs a message"),
    ({"response": {"vendor": None, "total": None}, "error": "bad"}, 0,
     "NULL response"),
    ({"response": {"vendor": None, "total": None}, "error": None}, 7,
     "NULL document"),
])
def test_invalid_null_and_error_states_are_rejected(suite, value, index, match):
    output = _output(suite)
    values = output.rows["extracted"].to_pylist()
    values[index] = value
    _replace_values(output, values)
    with pytest.raises(ValueError, match=match):
        _score_output(suite, output)


def test_missing_schema_fields_and_wrong_field_types_are_rejected(suite):
    output = _output(suite)
    values = output.rows["extracted"].to_pylist()
    for fields in [("vendor",), ("vendor", "total", "invented")]:
        _replace_values(output, values, fields)
        with pytest.raises(ValueError, match="every string field"):
            _score_output(suite, output)
    output.rows = pa.table({"d": ["I-101"], "extracted": [123]})
    with pytest.raises(ValueError, match="every string field"):
        _score_output(suite, output)


def test_unknown_and_duplicate_ids_and_operators_are_rejected(suite):
    for ids, match in [(["unknown"], "unknown document"),
                       (["I-101", "I-101"], "duplicate document"),
                       ([None], "cannot be null")]:
        output = _output(suite)
        output.rows = output.rows.slice(0, len(ids)).set_column(
            0, "d", pa.array(ids, pa.string()))
        with pytest.raises(ValueError, match=match):
            _score_output(suite, output)
    output = _output(suite, traced=True)
    output.extract_answers["unknown"] = output.rows
    with pytest.raises(ValueError, match="unknown extract operator"):
        _score_output(suite, output)


def test_saved_answers_must_agree_with_rows_and_filters(suite):
    output = _output(suite, traced=True)
    values = output.rows["extracted"].to_pylist()
    values[0]["response"]["vendor"] = "changed"
    _replace_values(output, values)
    with pytest.raises(ValueError, match="not implied"):
        _score_output(suite, output)
    output = _output(suite, 1, traced=True)
    answers = output.filter_answers["filter-1"]
    output.filter_answers["filter-1"] = answers.set_column(
        1, "answer", pa.array([False] * len(answers)))
    with pytest.raises(ValueError, match="recorded filters"):
        _score_output(suite, output, 1)


def test_filter_mistake_changes_output_accuracy_even_when_fields_match(suite):
    output = _output(suite, traced=True)
    labels = next(iter(suite.ground_truth.predicates.values())).table
    output.filter_answers = {"filter-1": pa.table({
        "d": labels["left_id"], "answer": [True] * 7 + [False]})}
    output.rows = output.rows.slice(0, 7)
    output.extract_answers = {"extract-1": output.rows}
    accuracy = _score_output(suite, output, 1)["accuracy"]
    assert accuracy["extraction_accuracy"]["accuracy"] == 1
    assert accuracy["answer_accuracy"]["false_positive"] == 1
    assert accuracy["output_accuracy"]["precision"] == pytest.approx(6 / 7, abs=1e-6)


def test_empty_output_and_no_generated_tokens_are_measured(suite):
    output = _output(suite)
    output.rows = output.rows.slice(0, 0)
    output.measurements = {"input_tokens": 0, "output_tokens": 0}
    counts = _score_output(suite, output)["accuracy"]["extraction_accuracy"]
    assert counts["accuracy"] is None
    assert counts["missing_documents"] == 8
    assert counts["reference_field_recall"] == 0
    output.measurements["output_tokens"] = True
    with pytest.raises(ValueError, match="output_tokens"):
        _score_output(suite, output)


def test_fixed_sample_scale_identity_and_local_inputs(suite, tmp_path):
    with pytest.raises(ValueError, match="scale_factor=1.0"):
        quail_b.load_benchmark(["EXTRACT-1"])
    with pytest.raises(ValueError, match="separately"):
        quail_b.load_benchmark(["EXTRACT-1", "IMDB-1"], scale_factor=1.0)
    with pytest.raises(ValueError, match="collection changed"):
        quail_b.load_benchmark(["EXTRACT-1"], scale_factor=1.0,
                               collection_id="wrong")
    for name, table in suite.tables.items():
        pq.write_table(table, tmp_path / (name + ".parquet"))
    loaded = quail_b.load_benchmark(
        SAMPLE_IDS, scale_factor=1.0, data_dir=tmp_path)
    assert loaded.corpus_id == suite.corpus_id
    changed = suite.tables["extract_invoices"].set_column(
        1, "body", pa.array(["changed"] * 8))
    pq.write_table(changed, tmp_path / "extract_invoices.parquet")
    with pytest.raises(ValueError, match="does not match synthetic corpus"):
        quail_b.load_benchmark(["EXTRACT-1"], scale_factor=1.0, data_dir=tmp_path)


def test_rescore_detects_changed_reference_and_corpus(suite, tmp_path, monkeypatch):
    root = tmp_path / "fixtures"
    root.mkdir()
    original = files("quail_b").joinpath("samples", "extraction", "fixtures.json")
    fixture = json.loads(original.read_text())
    (root / "fixtures.json").write_text(json.dumps(fixture))
    output = _output(suite)
    quail_b.run(lambda spec, tables: output, queries=["EXTRACT-1"],
                scale_factor=1.0, output_dir=tmp_path / "run")
    monkeypatch.setattr(extract_samples, "_root", lambda: root)
    fixture["tables"]["extract_invoices"]["rows"][0]["response"]["vendor"] = "new"
    (root / "fixtures.json").write_text(json.dumps(fixture))
    changed = quail_b.load_benchmark(["EXTRACT-1"], scale_factor=1.0)
    assert changed.corpus_id == suite.corpus_id
    assert changed.ground_truth.collection_id != suite.ground_truth.collection_id
    with pytest.raises(ValueError, match="collection changed"):
        quail_b.report(tmp_path / "run")
    fixture["tables"]["extract_invoices"]["rows"][0]["body"] = "new"
    (root / "fixtures.json").write_text(json.dumps(fixture))
    assert quail_b.load_benchmark(
        ["EXTRACT-1"], scale_factor=1.0).corpus_id != suite.corpus_id


def test_reference_keys_cannot_silently_become_null_fields(
        suite, tmp_path, monkeypatch):
    fixture = json.loads(files("quail_b").joinpath(
        "samples", "extraction", "fixtures.json").read_text())
    del fixture["tables"]["extract_invoices"]["rows"][0]["response"]["vendor"]
    (tmp_path / "fixtures.json").write_text(json.dumps(fixture))
    monkeypatch.setattr(extract_samples, "_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="every string or NULL field"):
        quail_b.load_benchmark(["EXTRACT-1"], scale_factor=1.0)


def test_prompt_changes_invalidate_extraction_query_hash(suite, monkeypatch):
    substrait = importlib.import_module("quail_b.substrait")
    spec = suite.queries[0]
    previous = _query_hash(spec)
    monkeypatch.setattr(substrait, "extract_template", lambda fields: "changed")
    assert _query_hash(spec) != previous


def test_missing_reference_document_fails_instead_of_scoring_as_null(suite):
    key = next(iter(suite.ground_truth.extracts))
    suite.ground_truth.extracts[key] = suite.ground_truth.extracts[key].slice(1)
    output = _output(suite, index=2)
    with pytest.raises(KeyError, match="entire input table"):
        expected_rows(suite.queries[0], suite.ground_truth, suite.tables)
    _validate_output(suite.queries[2], output, suite.tables)


def test_multiple_projection_scoring_counts_each_operator(suite):
    scan = Scan("extract_invoices", "d", "body", nullable_text=True)
    fields = ("vendor", "total")
    tree = Extract(Extract(scan, fields, "first"), fields, "second")
    spec = QuerySpec.from_plan(
        "TEST", "", build_plan(tree, ("d", "d.first", "d.second")))
    rows = expected_rows(spec, suite.ground_truth, suite.tables)
    output = quail_b.RunOutput(
        {}, {}, rows, runtime_s=1.0,
        extract_answers={"extract-1": rows.select(["d", "first"])})
    metrics = _score(spec, output, suite, 1, None)
    counts = metrics["accuracy"]["extraction_accuracy"]
    assert counts["correct"] == counts["evaluated"] == 28
    assert counts["expected_documents"] == counts["covered_documents"] == 16
    assert counts["null_inputs"] == 2
    assert [item["source"] for item in metrics["accuracy"]["per_predicate"]] == [
        "answers", "rows"]


def test_saved_bad_extraction_can_be_diagnosed_and_rescored(suite, tmp_path):
    output = _output(suite)
    values = output.rows["extracted"].to_pylist()
    values[0] = None
    _replace_values(output, values)
    with pytest.raises(ValueError, match="non-NULL document"):
        quail_b.run(lambda spec, tables: output, queries=["EXTRACT-1"],
                    scale_factor=1.0, output_dir=tmp_path / "run")
    record = json.loads((tmp_path / "run/run.json").read_text())
    assert record["status"] == "failed"
    assert record["queries"][0]["status"] == "scoring_failed"
    assert (tmp_path / "run/EXTRACT-1/rows.parquet").exists()
    with pytest.raises(ValueError, match="non-NULL document"):
        quail_b.report(tmp_path / "run")
