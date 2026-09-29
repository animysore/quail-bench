"""Validate and score field-name extraction without an inference engine."""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa

from quail_b.rendering import extract_template


def result_type(fields) -> pa.StructType:
    """Return the nullable response/error struct for the requested fields."""
    return pa.struct([
        ("response", pa.struct([(name, pa.string()) for name in fields])),
        ("error", pa.string()),
    ])


def reference_key(table: str, column: str, fields: tuple[str, ...]) -> str:
    """Identify extraction references by input, fields, and exact instruction."""
    value = json.dumps({
        "table": table, "column": column, "fields": fields,
        "prompt": extract_template(fields),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "extract_" + hashlib.sha256(value.encode()).hexdigest()


def _documents(spec, tables):
    relation, = spec._info.relations
    table = tables[relation.table]
    return dict(zip(
        map(str, table["id"].to_pylist()),
        table[relation.text_column].to_pylist()))


def _ids(table, alias, documents):
    if table[alias].null_count:
        raise ValueError("document IDs cannot be null")
    ids = [str(value) for value in table[alias].to_pylist()]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate document IDs in an answer table")
    if not set(ids) <= documents.keys():
        raise ValueError(f"unknown document ID for alias {alias}")
    return ids


def _values(table, alias, column, fields, documents):
    if set(table.column_names) != {alias, column} or len(table.columns) != 2:
        raise ValueError("extraction answers need an ID and their output column")
    if table[column].type != result_type(fields):
        raise ValueError("extraction must have every string field and error status")
    ids = _ids(table, alias, documents)
    values = table[column].to_pylist()
    for row_id, value in zip(ids, values):
        if documents[row_id] is None:
            if value is not None:
                raise ValueError("a NULL document must have a NULL extraction")
        elif value is None:
            raise ValueError("a non-NULL document needs a response or an error")
        elif value["error"] is None:
            if value["response"] is None:
                raise ValueError("a successful extraction needs a response struct")
        elif not value["error"] or value["response"] is not None:
            raise ValueError("an extraction error needs a message and NULL response")
    return dict(zip(ids, values))


def validate_output(spec, output, tables):
    """Validate extraction structures, IDs, and consistency with saved answers."""
    documents = _documents(spec, tables)
    alias = spec._info.base_alias
    extracts = {op.id: op for op in spec._info.extracts}
    columns = {alias, *(op.output for op in extracts.values())}
    if output.rows is None:
        raise ValueError("extraction scoring needs the saved result rows")
    if (set(output.rows.column_names) != columns
            or len(output.rows.columns) != len(columns)):
        raise ValueError("output columns must match the extraction query")
    if output.join_answers or output.classify_answers:
        raise ValueError("extraction queries have no joins or classifications")
    filters = {op.id: op for op in spec._info.filters}
    allowed = set(documents)
    for operator_id, table in (output.filter_answers or {}).items():
        if operator_id not in filters:
            raise ValueError(f"unknown filter operator {operator_id!r}")
        if set(table.column_names) != {alias, "answer"}:
            raise ValueError("filter answers need an ID and answer column")
        ids = _ids(table, alias, documents)
        if table["answer"].type != pa.bool_() or table["answer"].null_count:
            raise ValueError("predicate answers must be non-null booleans")
        allowed.intersection_update(
            row_id for row_id, answer in zip(ids, table["answer"].to_pylist())
            if answer)
    returned_ids = set(_ids(output.rows, alias, documents))
    if not returned_ids <= allowed:
        raise ValueError("a returned row did not pass its recorded filters")
    traces = {}
    for operator_id, table in (output.extract_answers or {}).items():
        if operator_id not in extracts:
            raise ValueError(f"unknown extract operator {operator_id!r}")
        operator = extracts[operator_id]
        traces[operator_id] = _values(
            table, alias, operator.output, operator.fields, documents)
        if not traces[operator_id].keys() <= allowed:
            raise ValueError("extraction answers include filtered-out documents")
    for operator in extracts.values():
        returned = _values(
            output.rows.select([alias, operator.output]), alias,
            operator.output, operator.fields, documents)
        if operator.id in traces and traces[operator.id] != returned:
            raise ValueError("a returned extraction is not implied by its answers")


def _reference(spec, operator, truth, documents):
    relation = spec._info.relation(operator.relation)
    key = reference_key(relation.table, relation.text_column, operator.fields)
    try:
        table = truth.extracts[key]
    except KeyError as error:
        raise KeyError(f"no extraction reference for {operator.id}") from error
    values = _values(
        table.rename_columns([operator.relation, operator.output]),
        operator.relation, operator.output, operator.fields, documents)
    if values.keys() != documents.keys():
        raise KeyError("extraction references must cover the entire input table")
    if any(value is not None and value["error"] is not None
           for value in values.values()):
        raise ValueError("reference extractions cannot contain generation errors")
    return key, values


def expected_rows(spec, truth, tables):
    """Return the reference projection with exact strings and SQL NULLs."""
    from quail_b.scoring import expected_survivors, output_columns

    documents = _documents(spec, tables)
    alias = spec._info.base_alias
    ids = expected_survivors(spec, truth, tables)[alias]
    columns = {alias: pa.array(ids, pa.string())}
    for operator in spec._info.extracts:
        _, values = _reference(spec, operator, truth, documents)
        columns[operator.output] = pa.array(
            [values[row_id] for row_id in ids], result_type(operator.fields))
    return pa.table(columns).select(output_columns(spec))


def evaluate(spec, output, truth, tables):
    """Score exact fields, failed generations, coverage, and final rows."""
    from quail_b.scoring import (
        BinaryCounts,
        _labels,
        _row_metrics,
        agreement,
    )

    documents = _documents(spec, tables)
    alias = spec._info.base_alias
    reference_rows = expected_rows(spec, truth, tables)
    expected = {row[alias]: row for row in reference_rows.to_pylist()}
    returned = {str(row[alias]): {**row, alias: str(row[alias])}
                for row in output.rows.to_pylist()}
    binary = BinaryCounts()
    per_predicate = []
    for operator in spec._info.filters:
        table = (output.filter_answers or {}).get(operator.id)
        if table is not None:
            labels = _labels(truth, operator.prompt)
            counts = agreement(table, [alias], labels)
            binary.merge(counts)
            per_predicate.append({
                "predicate_key": labels.key, "op": "filter", "alias": alias,
                **counts.as_dict()})
    total = dict.fromkeys((
        "correct", "evaluated", "expected_documents", "covered_documents",
        "missing_documents", "error_documents", "null_inputs",
        "reference_fields", "correct_reference_fields"), 0)
    for operator in spec._info.extracts:
        key, reference = _reference(spec, operator, truth, documents)
        table = (output.extract_answers or {}).get(operator.id)
        traced = table is not None
        if table is None:
            table = output.rows.select([alias, operator.output])
        values = _values(
            table, alias, operator.output, operator.fields, documents)
        counts = dict.fromkeys(total, 0)
        counts["expected_documents"] = len(expected)
        counts["covered_documents"] = len(expected.keys() & values.keys())
        counts["missing_documents"] = len(expected.keys() - values.keys())
        counts["reference_fields"] = sum(
            documents[row_id] is not None for row_id in expected
        ) * len(operator.fields)
        for row_id, value in values.items():
            if value is None:
                counts["null_inputs"] += 1
                continue
            counts["evaluated"] += len(operator.fields)
            if value["error"] is not None:
                counts["error_documents"] += 1
                continue
            correct = sum(value["response"][name] == reference[row_id]["response"][name]
                          for name in operator.fields)
            counts["correct"] += correct
            if row_id in expected:
                counts["correct_reference_fields"] += correct
        per_predicate.append({
            "predicate_key": key, "op": "extract", "alias": alias,
            "operator_id": operator.id, "source": "answers" if traced else "rows",
            **counts,
            "accuracy": (counts["correct"] / counts["evaluated"]
                         if counts["evaluated"] else None)})
        for name in total:
            total[name] += counts[name]
    total["accuracy"] = (
        total["correct"] / total["evaluated"] if total["evaluated"] else None)
    total["reference_field_recall"] = (
        total["correct_reference_fields"] / total["reference_fields"]
        if total["reference_fields"] else None)
    matched = sum(row_id in expected and row == expected[row_id]
                  for row_id, row in returned.items())
    return {
        "ground_truth_collection_id": truth.collection_id,
        "ground_truth_reference_model": truth.reference_model,
        "answer_accuracy": binary.as_dict() if binary.evaluated else None,
        "label_accuracy": None, "extraction_accuracy": total,
        "output_accuracy": _row_metrics(len(returned), len(expected), matched),
        "per_predicate": per_predicate,
        "input_document_rows": len(documents),
        "unique_input_documents": len(documents),
    }
