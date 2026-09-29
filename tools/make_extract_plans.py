"""Generate the opt-in synthetic extraction plans and catalog."""

import json
import sys
from pathlib import Path

from quail_b.extract_samples import INVOICE_PROMPT
from tools.make_substrait_plans import (
    Extract,
    Filter,
    Scan,
    build_plan,
    plan_json,
)

PLANS_DIR = (
    Path(__file__).resolve().parents[1] / "quail_b" / "samples" / "extraction")


def write_plans(directory: Path):
    """Write three sample plans without modifying the published catalog."""
    invoices = Scan("extract_invoices", "d", "body", nullable_text=True)
    reviews = Scan("extract_reviews", "r", "body", nullable_text=True)
    definitions = [
        ("EXTRACT-1", "Synthetic invoices: extract vendor and total",
         Extract(invoices, ("vendor", "total"), "extracted"),
         ("d", "d.extracted")),
        ("EXTRACT-2", "Synthetic invoices: filter invoices, then extract fields",
         Extract(Filter(invoices, INVOICE_PROMPT), ("vendor", "total"), "extracted"),
         ("d", "d.extracted")),
        ("EXTRACT-3", "Synthetic reviews: extract movie title and rating",
         Extract(reviews, ("movie_title", "rating"), "extracted"),
         ("r", "r.extracted")),
    ]
    directory.mkdir(parents=True, exist_ok=True)
    catalog = []
    for name, description, tree, select in definitions:
        (directory / (name + ".json")).write_text(plan_json(build_plan(tree, select)))
        catalog.append({"id": name, "description": description})
    (directory / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    write_plans(Path(sys.argv[1]) if len(sys.argv) > 1 else PLANS_DIR)
