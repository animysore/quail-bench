"""Load the bundled synthetic extraction suite and hand-authored references."""

import hashlib
import json
from functools import cache
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.protobuf import json_format
from substrait import plan_pb2

from quail_b.extraction import reference_key, result_type
from quail_b.labels import GroundTruthCollection, PredicateLabels

INVOICE_PROMPT = "Does this document explicitly describe an invoice? {0}"
SUITE_NAME = "synthetic-extraction-v1"


def _root():
    return files("quail_b").joinpath("samples", "extraction")


@cache
def query_specs():
    """Return the opt-in sample catalog as validated Substrait query specs."""
    from quail_b.queries import QuerySpec

    catalog = json.loads(_root().joinpath("catalog.json").read_text())
    specs = {}
    for item in catalog:
        name = item["id"]
        if name in specs:
            raise ValueError(f"duplicate extraction sample ID {name}")
        plan = json_format.Parse(
            _root().joinpath(name + ".json").read_text(), plan_pb2.Plan())
        specs[name] = QuerySpec.from_plan(name, item["description"], plan)
    return specs


def _hash(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _validate_fixture(fixture):
    for name, table in fixture["tables"].items():
        fields = table["fields"]
        if (not fields or any(not isinstance(field, str) or not field.strip()
                              for field in fields)
                or len(fields) != len(set(fields))):
            raise ValueError("sample extraction fields must be nonempty and distinct")
        seen = set()
        for row in table["rows"]:
            if not isinstance(row["id"], str) or row["id"] in seen:
                raise ValueError("sample document IDs must be distinct strings")
            seen.add(row["id"])
            if row["body"] is None:
                if row["response"] is not None:
                    raise ValueError("a NULL sample document needs a NULL reference")
            elif (not isinstance(row["body"], str)
                  or not isinstance(row["response"], dict)
                  or set(row["response"]) != set(fields)
                  or any(value is not None and not isinstance(value, str)
                         for value in row["response"].values())):
                raise ValueError("sample references need every string or NULL field")
            if name == "extract_invoices" and not isinstance(row["is_invoice"], bool):
                raise ValueError("sample filter references must be booleans")
            if (name == "extract_invoices" and row["body"] is None
                    and row["is_invoice"]):
                raise ValueError("a NULL sample document cannot pass the filter")


def load_samples(specs, *, scale_factor, data_dir, collection_id, accuracy):
    """Load a fixed, content-identified corpus without network access."""
    from quail_b.benchmark import Benchmark

    if scale_factor != 1.0:
        raise ValueError("synthetic extraction samples require scale_factor=1.0")
    if any(spec.id not in query_specs() for spec in specs):
        raise ValueError("run synthetic samples separately from published queries")
    fixture = json.loads(_root().joinpath("fixtures.json").read_text())
    _validate_fixture(fixture)
    data = {
        name: [{"id": row["id"], "body": row["body"]} for row in value["rows"]]
        for name, value in fixture["tables"].items()
    }
    corpus_id = "c_synthetic_extract_" + _hash({
        "version": fixture["version"], "tables": data})
    names = {rel.table for spec in specs for rel in spec._info.relations}
    tables = {}
    for name in names:
        table = pa.Table.from_pylist(
            data[name], schema=pa.schema([("id", pa.string()), ("body", pa.string())]))
        if data_dir is not None:
            supplied = pq.read_table(Path(data_dir) / (name + ".parquet"))
            if not supplied.equals(table):
                raise ValueError(f"{name} does not match synthetic corpus {corpus_id}")
            table = supplied
        tables[name] = table
    extracts = {}
    reference_ids = {}
    for name, value in fixture["tables"].items():
        fields = tuple(value["fields"])
        rows = value["rows"]
        key = reference_key(name, "body", fields)
        reference_ids[name] = key
        extracts[key] = pa.table({
            "id": pa.array([row["id"] for row in rows], pa.string()),
            "value": pa.array([
                None if row["body"] is None else {
                    "response": row["response"], "error": None}
                for row in rows], result_type(fields)),
        })
    reference_id = "gt_synthetic_extract_" + _hash({
        "corpus_id": corpus_id, "fixtures": fixture, "extracts": reference_ids,
        "filter_prompt": INVOICE_PROMPT})
    if collection_id is not None and collection_id != reference_id:
        raise ValueError("extraction reference collection changed")
    truth = None
    if accuracy:
        filter_key = "synthetic.extraction.is_invoice"
        labels = PredicateLabels(
            filter_key, reference_id,
            {"kind": "filter", "template": INVOICE_PROMPT,
             "left_table": "extract_invoices", "left_column": "body"},
            {(row["id"], None): row["is_invoice"]
             for row in fixture["tables"]["extract_invoices"]["rows"]},
            {"hand_authored": len(data["extract_invoices"])})
        truth = GroundTruthCollection(
            reference_id, corpus_id, 1.0, None, {filter_key: labels}, extracts)
    return Benchmark(specs, tables, 1.0, corpus_id, truth, SUITE_NAME)
