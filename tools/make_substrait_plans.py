"""Write the QUAIL-B query plans as Substrait ProtoJSON files.

    uv run python tools/make_substrait_plans.py [output_dir]

The output directory defaults to `quail_b/plans`. The checked-in files
there are the published query definitions; a test checks that they
equal this script's output, so a prompt or query shape is changed here
and in `quail_b/prompts.py`, then the files are regenerated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import json_format
from substrait import algebra_pb2 as algebra
from substrait import plan_pb2, type_pb2
from substrait.extensions import extensions_pb2

from quail_b import prompts
from quail_b.prompts import (
    AGENT_IMPLEMENTED_FIX,
    AGENT_RECOVERED,
    ASPECT_SENTIMENT,
    CARDIOVASCULAR_REACTION,
    DISCUSS_ASPECT,
    F1,
    F4,
    F5,
    F11,
    F12,
    F13,
    LEP1,
    LEP2,
    LEPJOIN,
    LEPS1,
    NEUROLOGICAL_REACTION,
    P_LOC,
    P_MSG,
    REACTION,
    REFUTE,
    SCENARIO_MATCH,
    SERIOUS_ADVERSE_EVENT,
    SUPPORT,
)
from quail_b.substrait import (
    AI_CLASSIFY_NAME,
    AI_EXTENSION_URN,
    AI_EXTRACT_NAME,
    AI_FILTER_NAME,
    AI_JOIN_NAME,
    AND_NAME,
    BOOLEAN_EXTENSION_URN,
    COMPARISON_EXTENSION_URN,
    EQUAL_NAME,
    SUBSTRAIT_VERSION,
    extract_type,
)

PLANS_DIR = Path(__file__).resolve().parents[1] / "quail_b" / "plans"

# Function anchors are fixed so expressions can reference them while
# the tree is built; only the functions a plan uses are declared.
_URN_ANCHORS = {
    AI_EXTENSION_URN: 1,
    COMPARISON_EXTENSION_URN: 2,
    BOOLEAN_EXTENSION_URN: 3,
}
_FUNCTIONS = {
    AI_FILTER_NAME: (1, AI_EXTENSION_URN),
    AI_JOIN_NAME: (2, AI_EXTENSION_URN),
    EQUAL_NAME: (3, COMPARISON_EXTENSION_URN),
    AND_NAME: (4, BOOLEAN_EXTENSION_URN),
    AI_CLASSIFY_NAME: (5, AI_EXTENSION_URN),
    AI_EXTRACT_NAME: (6, AI_EXTENSION_URN),
}


@dataclass(frozen=True)
class Scan:
    """Read one document table.

    Attributes:
        table: The table name.
        alias: The relation alias, unique within a query.
        text: The column the AI functions read.
        columns: Further columns, for ordinary join conditions.
    """

    table: str
    alias: str
    text: str
    columns: tuple[str, ...] = ()
    nullable_text: bool = False


@dataclass(frozen=True)
class Filter:
    """Keep the documents of one relation that answer a prompt TRUE."""

    input: Scan | Filter | Classify | LabelFilter
    prompt: str


@dataclass(frozen=True)
class Classify:
    """Add one label column to one relation from a fixed list of labels.

    Attributes:
        input: The relation's scan, filters, or classifications.
        prompt: The classification prompt.
        labels: The labels, in tie-breaking order.
        output: The name of the label column.
        descriptions: One description per label, or empty for none.
    """

    input: Scan | Filter | Classify | LabelFilter
    prompt: str
    labels: tuple[str, ...]
    output: str
    descriptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Extract:
    """Append named string fields and an error status to one input relation."""

    input: Scan | Filter | Extract
    fields: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class LabelFilter:
    """Keep the documents whose label column holds an accepted label."""

    input: Classify | LabelFilter
    output: str
    accepted: tuple[str, ...]


@dataclass(frozen=True)
class Join:
    """Keep the pairs of documents that answer a prompt TRUE.

    Attributes:
        left: The left input.
        right: The right input.
        aliases: The relations the prompt's `{0}` and `{1}` refer to.
        prompt: The join prompt.
        on: Column pairs, in `aliases` order, that must also be equal.
    """

    left: Scan | Filter | Join
    right: Scan | Filter | Join
    aliases: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Query:
    """One benchmark query.

    Attributes:
        id: The query ID.
        description: A short description for the catalog.
        tree: The operator tree.
        privacy: Whether the query reads the privacy policy corpus.
        select: Returned columns: an alias for its id, `alias.column`
            for a label column; None returns every relation's id.
        labels_pending: Whether its reference labels are unpublished.
    """

    id: str
    description: str
    tree: Scan | Filter | Join | Classify | LabelFilter
    privacy: bool = False
    select: tuple[str, ...] | None = None
    labels_pending: bool = False


def _string_type():
    return type_pb2.Type(
        string=type_pb2.Type.String(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _bool_type():
    return type_pb2.Type(
        bool=type_pb2.Type.Boolean(
            nullability=type_pb2.Type.NULLABILITY_REQUIRED
        )
    )


def _common(alias):
    return algebra.RelCommon(
        direct=algebra.RelCommon.Direct(),
        hint=algebra.RelCommon.Hint(alias=alias),
    )


def _field(index):
    return algebra.Expression(
        selection=algebra.Expression.FieldReference(
            direct_reference=algebra.Expression.ReferenceSegment(
                struct_field=algebra.Expression.ReferenceSegment.StructField(
                    field=index
                )
            ),
            root_reference=algebra.Expression.FieldReference.RootReference(),
        )
    )


def _literal(text):
    return algebra.Expression(
        literal=algebra.Expression.Literal(string=text)
    )


def _string_list(values):
    return algebra.Expression(
        literal=algebra.Expression.Literal(
            list=algebra.Expression.Literal.List(values=[
                algebra.Expression.Literal(string=value) for value in values
            ])
        )
    )


def _call(name, arguments, output_type=None):
    return algebra.Expression(
        scalar_function=algebra.Expression.ScalarFunction(
            function_reference=_FUNCTIONS[name][0],
            arguments=[
                algebra.FunctionArgument(value=argument)
                for argument in arguments
            ],
            output_type=output_type or _bool_type(),
        )
    )


class _Emitter:
    """Turn a query tree into Substrait relations.

    Operators are numbered in post-order (inputs before the operator,
    left before right), which is the order the benchmark's plan reader
    reports them in.
    """

    def __init__(self):
        self.counts = {"filter": 0, "join": 0, "classify": 0, "extract": 0,
                       "label-filter": 0}
        self.functions = set()
        self.extracted_fields = {}

    def output_names(self, fields, *, qualified=False):
        names = []
        for alias, column in fields:
            names.append(f"{alias}.{column}" if qualified else (
                alias if column == "id" else column))
            if (alias, column) in self.extracted_fields:
                names.extend((
                    "response", *self.extracted_fields[alias, column], "error"))
        return names

    def _operator_id(self, kind):
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def emit(self, node):
        """Return (rel, fields, text columns by alias) for one tree."""
        if isinstance(node, Scan):
            names = ("id", node.text, *node.columns)
            types = [_string_type() for _ in names]
            if node.nullable_text:
                types[1].string.nullability = type_pb2.Type.NULLABILITY_NULLABLE
            read = algebra.ReadRel(
                common=_common(node.alias),
                base_schema=type_pb2.NamedStruct(
                    names=names,
                    struct=type_pb2.Type.Struct(
                        types=types,
                        nullability=type_pb2.Type.NULLABILITY_REQUIRED,
                    ),
                ),
                named_table=algebra.ReadRel.NamedTable(names=[node.table]),
            )
            fields = tuple((node.alias, name) for name in names)
            return algebra.Rel(read=read), fields, {node.alias: node.text}

        if isinstance(node, Extract):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_EXTRACT_NAME)
            self.extracted_fields[alias, node.output] = node.fields
            common = _common(self._operator_id("extract"))
            output_fields = (*fields, (alias, node.output))
            common.hint.output_names.extend(
                self.output_names(output_fields, qualified=True))
            relation = algebra.ProjectRel(
                common=common, input=rel, expressions=[_call(
                    AI_EXTRACT_NAME, [
                        _field(fields.index((alias, text[alias]))),
                        _string_list(node.fields)],
                    extract_type(node.fields))])
            relation.expressions[0].scalar_function.arguments.add(
                type=extract_type(node.fields))
            return algebra.Rel(project=relation), output_fields, text

        if isinstance(node, Classify):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_CLASSIFY_NAME)
            descriptions = node.descriptions or ("",) * len(node.labels)
            output_fields = (*fields, (alias, node.output))
            common = _common(self._operator_id("classify"))
            common.hint.output_names.extend(
                ".".join(field) for field in output_fields)
            relation = algebra.ProjectRel(
                common=common,
                input=rel,
                expressions=[_call(AI_CLASSIFY_NAME, [
                    _literal(node.prompt),
                    _field(fields.index((alias, text[alias]))),
                    _string_list(node.labels),
                    _string_list(descriptions),
                ], _string_type())],
            )
            return algebra.Rel(project=relation), output_fields, text

        if isinstance(node, LabelFilter):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            relation = algebra.FilterRel(
                common=_common(self._operator_id("label-filter")),
                input=rel,
                condition=algebra.Expression(
                    singular_or_list=algebra.Expression.SingularOrList(
                        value=_field(fields.index((alias, node.output))),
                        options=[_literal(label) for label in node.accepted],
                    )
                ),
            )
            return algebra.Rel(filter=relation), fields, text

        if isinstance(node, Filter):
            rel, fields, text = self.emit(node.input)
            (alias,) = text
            self.functions.add(AI_FILTER_NAME)
            relation = algebra.FilterRel(
                common=_common(self._operator_id("filter")),
                input=rel,
                condition=_call(AI_FILTER_NAME, [
                    _literal(node.prompt),
                    _field(fields.index((alias, text[alias]))),
                ]),
            )
            return algebra.Rel(filter=relation), fields, text

        left, left_fields, left_text = self.emit(node.left)
        right, right_fields, right_text = self.emit(node.right)
        fields = left_fields + right_fields
        text = {**left_text, **right_text}
        first, second = node.aliases
        self.functions.add(AI_JOIN_NAME)
        conditions = [_call(AI_JOIN_NAME, [
            _literal(node.prompt),
            _field(fields.index((first, text[first]))),
            _field(fields.index((second, text[second]))),
        ])]
        for left_column, right_column in node.on:
            self.functions.update({EQUAL_NAME, AND_NAME})
            conditions.append(_call(EQUAL_NAME, [
                _field(fields.index((first, left_column))),
                _field(fields.index((second, right_column))),
            ]))
        expression = (
            conditions[0] if len(conditions) == 1
            else _call(AND_NAME, conditions)
        )
        relation = algebra.JoinRel(
            common=_common(self._operator_id("join")),
            left=left,
            right=right,
            expression=expression,
            type=algebra.JoinRel.JOIN_TYPE_INNER,
        )
        return algebra.Rel(join=relation), fields, text


def build_plan(tree, select=None) -> plan_pb2.Plan:
    """Return the Substrait plan selecting the id of each relation.

    Args:
        tree: The query tree.
        select: Returned columns, an alias for its id or `alias.column`
            for a label column, or None for every relation's id in scan
            order.
    """
    emitter = _Emitter()
    rel, fields, text = emitter.emit(tree)
    columns = [
        tuple(name.split(".", 1)) if "." in name else (name, "id")
        for name in (list(text) if select is None else select)
    ]
    names = emitter.output_names(columns)
    expressions = [_field(fields.index(column)) for column in columns]
    project = algebra.Rel(
        project=algebra.ProjectRel(
            common=algebra.RelCommon(
                emit=algebra.RelCommon.Emit(
                    output_mapping=[
                        len(fields) + index
                        for index in range(len(expressions))
                    ]
                )
            ),
            input=rel,
            expressions=expressions,
        )
    )
    used = sorted(emitter.functions, key=lambda name: _FUNCTIONS[name][0])
    urns = sorted(
        {_FUNCTIONS[name][1] for name in used},
        key=_URN_ANCHORS.__getitem__,
    )
    return plan_pb2.Plan(
        version=plan_pb2.Version(
            major_number=SUBSTRAIT_VERSION[0],
            minor_number=SUBSTRAIT_VERSION[1],
            patch_number=SUBSTRAIT_VERSION[2],
            producer="quail-b",
        ),
        extension_urns=[
            extensions_pb2.SimpleExtensionURN(
                extension_urn_anchor=_URN_ANCHORS[urn],
                urn=urn,
            )
            for urn in urns
        ],
        extensions=[
            extensions_pb2.SimpleExtensionDeclaration(
                extension_function=(
                    extensions_pb2.SimpleExtensionDeclaration.ExtensionFunction(
                        extension_urn_reference=_URN_ANCHORS[_FUNCTIONS[name][1]],
                        function_anchor=_FUNCTIONS[name][0],
                        name=name,
                    )
                )
            )
            for name in used
        ],
        relations=[
            plan_pb2.PlanRel(
                root=algebra.RelRoot(input=project, names=names)
            )
        ],
    )


def _filters(node, *prompts):
    for prompt in prompts:
        node = Filter(node, prompt)
    return node


def _reviews(alias="r"):
    return Scan("reviews", alias, "body")


def _aspects(alias="a"):
    return Scan("aspects", alias, "aspect")


def _reports():
    return Scan("reports", "r", "report")


def _terms(alias="m"):
    return Scan("terms", alias, "term")


def _claims(alias="c", *columns):
    return Scan("claims", alias, "claim", columns)


def _evidence(alias="e"):
    return Scan("evidence", alias, "text")


def _contexts():
    return Scan("citation_contexts", "d", "destination_context")


def _passages():
    return Scan("citation_passages", "s", "passage_text")


def _traces():
    return Scan("agent_traces", "t", "trace")


def _policies():
    return Scan("policies", "p", "policy_text")


def _classify(node, prompt, labels, output, descriptions=()):
    return Classify(node, prompt, labels, output, descriptions)


def _sentiment(node):
    return _classify(node, prompts.IMDB_SENTIMENT,
                     prompts.IMDB_SENTIMENT_LABELS, "sentiment")


def _critical(node):
    """Reviews whose sentiment is negative or mixed."""
    return LabelFilter(_sentiment(node), "sentiment", ("negative", "mixed"))


def _organ_class(node):
    return _classify(node, prompts.BIO_ORGAN_CLASS,
                     prompts.BIO_ORGAN_CLASS_LABELS, "organ_class")


def _outcome(node):
    return _classify(node, prompts.AGENT_OUTCOME,
                     prompts.AGENT_OUTCOME_LABELS, "outcome",
                     prompts.AGENT_OUTCOME_DESCRIPTIONS)


def _imdb_chain(first):
    """Chain r1-a1-r2-a2: both reviews discuss a1, and r2 is positive about a2."""
    return Join(
        Join(
            Join(first, _aspects("a1"), ("r1", "a1"), DISCUSS_ASPECT),
            _reviews("r2"), ("r2", "a1"), DISCUSS_ASPECT,
        ),
        _aspects("a2"), ("r2", "a2"), ASPECT_SENTIMENT,
    )


def _fever_chain(c1, e1, c2, e2):
    """Chain c1-e1-c2-e2: e1 supports c1 but refutes c2, and e2 supports c2."""
    return Join(
        Join(Join(c1, e1, ("c1", "e1"), SUPPORT), c2, ("c2", "e1"), REFUTE),
        e2, ("c2", "e2"), SUPPORT,
    )


QUERIES = (
    Query("IMDB-1", "filter: F1 (at least one positive aspect)",
          _filters(_reviews(), F1)),
    Query("IMDB-2", "join: J1 (reviews x aspects)",
          Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT)),
    Query("IMDB-3", "F1 -> J1, dependent",
          Join(_filters(_reviews(), F1), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-4", "F1 -> F4 -> J1, 2 filters then 1 join",
          Join(_filters(_reviews(), F1, F4), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-5", "F1 -> F4 -> F5 -> J1, 3 filters then 1 join",
          Join(_filters(_reviews(), F1, F4, F5), _aspects(), ("r", "a"),
               DISCUSS_ASPECT)),
    Query("IMDB-6", "F1 -> F4, 2 filters, no join",
          _filters(_reviews(), F1, F4)),
    Query("IMDB-7", "F1 -> F4 -> F5, 3 filters, no join",
          _filters(_reviews(), F1, F4, F5)),
    Query("IMDB-8", "2J, same anchor: J1 (DISCUSS_ASPECT) -> J2 "
          "(ASPECT_SENTIMENT), reviews x aspects x aspects",
          Join(Join(_reviews(), _aspects(), ("r", "a"), DISCUSS_ASPECT),
               _aspects("a2"), ("r", "a2"), ASPECT_SENTIMENT)),
    Query("IMDB-9", "3J chain r1-a1-r2-a2: two reviews discuss the same "
          "aspect, second review positive about another",
          _imdb_chain(_reviews("r1"))),
    Query("IMDB-10", "F1 -> 3J chain r1-a1-r2-a2",
          _imdb_chain(_filters(_reviews("r1"), F1))),
    Query("IMDB-11", "classify: the sentiment of every review",
          _sentiment(_reviews()), select=("r", "r.sentiment")),
    Query("IMDB-12", "F1 -> classify: the IMDb genre of each review's movie",
          _classify(_filters(_reviews(), F1), prompts.IMDB_GENRE,
                    prompts.IMDB_GENRE_LABELS, "genre"),
          select=("r", "r.genre")),
    Query("IMDB-13", "negative or mixed reviews -> J1: the aspects critical "
          "reviews discuss, with each review's sentiment",
          Join(_critical(_reviews()), _aspects(), ("r", "a"),
               DISCUSS_ASPECT),
          select=("r", "r.sentiment", "a")),
    Query("IMDB-14", "negative or mixed reviews -> classify: their main "
          "complaint, returned with their sentiment",
          _classify(_critical(_reviews()), prompts.IMDB_COMPLAINT,
                    prompts.IMDB_COMPLAINT_LABELS, "complaint"),
          select=("r", "r.sentiment", "r.complaint")),

    Query("BIO-1", "filter: serious adverse event",
          _filters(_reports(), SERIOUS_ADVERSE_EVENT)),
    Query("BIO-2", "join: J1 (reports x terms)",
          Join(_reports(), _terms(), ("r", "m"), REACTION)),
    Query("BIO-3", "serious adverse event -> reaction join",
          Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT), _terms(),
               ("r", "m"), REACTION)),
    Query("BIO-4", "3F + 2J: serious reports with neurological and "
          "cardiovascular reactions",
          Join(Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT),
                    _filters(_terms("n"), NEUROLOGICAL_REACTION),
                    ("r", "n"), REACTION),
               _filters(_terms("c"), CARDIOVASCULAR_REACTION),
               ("r", "c"), REACTION)),
    Query("BIO-5", "classify: the MedDRA system organ class of every "
          "reaction term",
          _organ_class(_terms()), select=("m", "m.organ_class")),
    Query("BIO-6", "serious adverse event reports x their cardiac or "
          "vascular reactions",
          Join(_filters(_reports(), SERIOUS_ADVERSE_EVENT),
               LabelFilter(_organ_class(_terms()), "organ_class",
                           ("cardiac disorders", "vascular disorders")),
               ("r", "m"), REACTION)),

    Query("FEV-1", "filter: F11 (about a person)", _filters(_claims(), F11)),
    Query("FEV-2", "join: J3 (claims x evidence)",
          Join(_claims(), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-3", "F11 -> J3, dependent",
          Join(_filters(_claims(), F11), _evidence(), ("c", "e"), SUPPORT)),
    Query("FEV-4", "F11 -> F12 -> J3, 2 filters then 1 join",
          Join(_filters(_claims(), F11, F12), _evidence(), ("c", "e"),
               SUPPORT)),
    Query("FEV-5", "2F + 1J: two-sided pushdown - F11 on claims, F13 on "
          "evidence, each filtered before J3",
          Join(_filters(_claims(), F11), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-6", "3F + 1J: two-sided pushdown, deeper - F11 -> F12 on "
          "claims, F13 on evidence, each filtered before J3",
          Join(_filters(_claims(), F11, F12), _filters(_evidence(), F13),
               ("c", "e"), SUPPORT)),
    Query("FEV-7", "2J, same anchor: J1 (SUPPORT) -> J2 (REFUTE), "
          "claims x evidence x evidence",
          Join(Join(_claims(), _evidence(), ("c", "e"), SUPPORT),
               _evidence("e2"), ("c", "e2"), REFUTE)),
    Query("FEV-8", "3J chain c1-e1-c2-e2: evidence supports c1 but refutes "
          "c2, c2 supported by different evidence",
          _fever_chain(_claims("c1"), _evidence("e1"), _claims("c2"),
                       _evidence("e2"))),
    Query("FEV-9", "4F + 3J: F11 on c1 and c2, F13 on e1 and e2, then the "
          "c1-e1-c2-e2 join chain",
          _fever_chain(_filters(_claims("c1"), F11),
                       _filters(_evidence("e1"), F13),
                       _filters(_claims("c2"), F11),
                       _filters(_evidence("e2"), F13))),
    # A claim names its Wikipedia page, and an evidence row's id is its
    # page name, so SUPPORT is asked only of a claim and its own page.
    Query("FEV-10", "2F + 1J over pairs: F11 on claims, F13 on evidence, "
          "SUPPORT asked only of a claim and its own Wikipedia page",
          Join(_filters(_claims("c", "evidence_wiki_url"), F11),
               _filters(_evidence(), F13), ("c", "e"), SUPPORT,
               on=(("evidence_wiki_url", "id"),))),
    Query("FEV-11", "classify: claim topic; the political and historical "
          "claims, with their topic",
          LabelFilter(_classify(_claims(), prompts.FEV_TOPIC,
                                prompts.FEV_TOPIC_LABELS, "topic"),
                      "topic", ("politics", "history")),
          select=("c", "c.topic")),

    Query("LEP-1", "filter: LEP1 (reasoning does not apply)",
          _filters(_contexts(), LEP1)),
    Query("LEP-2", "join: citation contexts x cited passages",
          Join(_contexts(), _passages(), ("d", "s"), LEPJOIN)),
    Query("LEP-3", "LEP1 -> join, dependent",
          Join(_filters(_contexts(), LEP1), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-4", "LEP1 -> LEP2 -> join, 2 filters then 1 join",
          Join(_filters(_contexts(), LEP1, LEP2), _passages(), ("d", "s"),
               LEPJOIN)),
    Query("LEP-5", "2F + 1J: two-sided pushdown - LEP1+LEP2 on excerpts, "
          "LEPS1 on passages, each filtered before the join",
          Join(_filters(_contexts(), LEP1, LEP2),
               _filters(_passages(), LEPS1), ("d", "s"), LEPJOIN)),
    Query("LEP-6", "constitutional or criminal law excerpts -> the "
          "passages they cite, with each excerpt's area of law",
          Join(LabelFilter(
                   _classify(_contexts(), prompts.LEP_AREA,
                             prompts.LEP_AREA_LABELS, "area"),
                   "area", ("constitutional law", "criminal law")),
               _passages(), ("d", "s"), LEPJOIN),
          select=("d", "d.area", "s")),
    Query("AGENT-1", "filter: recovered after an unsuccessful approach",
          _filters(_traces(), AGENT_RECOVERED)),
    Query("AGENT-2", "filter: implemented a plausible fix",
          _filters(_traces(), AGENT_IMPLEMENTED_FIX)),
    Query("AGENT-3", "recovered -> classify: whether the agent resolved "
          "the issue",
          _outcome(_filters(_traces(), AGENT_RECOVERED)),
          select=("t", "t.outcome")),
    Query("AGENT-4", "unresolved traces -> classify: why the agent failed",
          _classify(LabelFilter(_outcome(_traces()), "outcome",
                                ("not resolved",)),
                    prompts.AGENT_FAILURE, prompts.AGENT_FAILURE_LABELS,
                    "failure", prompts.AGENT_FAILURE_DESCRIPTIONS),
          select=("t", "t.failure")),

    # PrivacyPolicies: only when that corpus is available.
    Query("PRIV-1", "2 filters: P_MSG + P_LOC",
          _filters(_policies(), P_MSG, P_LOC), privacy=True),
    Query("PRIV-2", "2 filters + 1 join: P_MSG + P_LOC -> scenarios",
          Join(_filters(_policies(), P_MSG, P_LOC),
               Scan("scenarios", "s", "scenario"), ("p", "s"),
               SCENARIO_MATCH), privacy=True),
)


def plan_json(plan: plan_pb2.Plan) -> str:
    """Return a plan as ProtoJSON with sorted keys and a trailing newline."""
    return json_format.MessageToJson(
        plan,
        preserving_proto_field_name=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_plans(directory: Path) -> None:
    """Write every query plan and the catalog into a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    catalog = []
    for query in QUERIES:
        (directory / f"{query.id}.json").write_text(
            plan_json(build_plan(query.tree, query.select))
        )
        entry = {
            "id": query.id,
            "description": query.description,
            "privacy": query.privacy,
        }
        if query.labels_pending:
            entry["labels_pending"] = True
        catalog.append(entry)
    (directory / "catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n"
    )


def main(argv=None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    directory = Path(arguments[0]) if arguments else PLANS_DIR
    write_plans(directory)
    print(f"wrote {len(QUERIES)} plans to {directory}")


if __name__ == "__main__":
    main()
