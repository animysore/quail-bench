"""Read the benchmark information in a QUAIL-B Substrait plan."""

from __future__ import annotations

from dataclasses import dataclass

from substrait import algebra_pb2, plan_pb2, type_pb2

from quail_b.rendering import extract_template

SUBSTRAIT_VERSION = (0, 103, 0)
AI_EXTENSION_URN = "extension:org.fsdatalab.quail_b:functions_ai"
COMPARISON_EXTENSION_URN = "extension:io.substrait:functions_comparison"
BOOLEAN_EXTENSION_URN = "extension:io.substrait:functions_boolean"

AI_FILTER_NAME = "ai_filter:str_str"
AI_JOIN_NAME = "ai_join:str_str_str"
AI_CLASSIFY_NAME = "ai_classify:str_str_list_list"
AI_EXTRACT_NAME = "ai_extract:str_list_any"
EQUAL_NAME = "equal:any_any"
AND_NAME = "and:bool"

_FUNCTION_URNS = {
    AI_FILTER_NAME: AI_EXTENSION_URN,
    AI_JOIN_NAME: AI_EXTENSION_URN,
    AI_CLASSIFY_NAME: AI_EXTENSION_URN,
    AI_EXTRACT_NAME: AI_EXTENSION_URN,
    EQUAL_NAME: COMPARISON_EXTENSION_URN,
    AND_NAME: BOOLEAN_EXTENSION_URN,
}


@dataclass(frozen=True)
class _Relation:
    alias: str
    table: str
    text_column: str


@dataclass(frozen=True)
class _Filter:
    id: str
    relation: str
    prompt: str


@dataclass(frozen=True)
class _Join:
    id: str
    relations: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _Classify:
    """One ai_classify call that adds a label column to one relation.

    Attributes:
        id: The operator ID.
        relation: The alias of the classified documents.
        prompt: The prompt template, with the document as `{0}`.
        labels: The categories, in the order that breaks ties.
        descriptions: One description per label; empty means none.
        output: The name of the added label column.
    """

    id: str
    relation: str
    prompt: str
    labels: tuple[str, ...]
    descriptions: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class _LabelFilter:
    """Keep the documents whose label is one of the accepted labels."""

    id: str
    relation: str
    output: str
    accepted: tuple[str, ...]


@dataclass(frozen=True)
class _Extract:
    """One field-name extraction that adds a response/error struct."""

    id: str
    relation: str
    fields: tuple[str, ...]
    output: str

    @property
    def prompt(self) -> str:
        return extract_template(self.fields)


type _Operator = _Filter | _Join | _Classify | _LabelFilter | _Extract


@dataclass(frozen=True)
class _PlanInfo:
    relations: tuple[_Relation, ...]
    operators: tuple[_Operator, ...]
    select: tuple[str, ...]

    @property
    def filters(self) -> tuple[_Filter, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _Filter)
        )

    @property
    def joins(self) -> tuple[_Join, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _Join)
        )

    @property
    def ai_operators(self) -> tuple[_Filter | _Join | _Classify | _Extract, ...]:
        """The operators that ask the model, each with its prompt."""
        return tuple(
            operator
            for operator in self.operators
            if not isinstance(operator, _LabelFilter)
        )

    @property
    def classifies(self) -> tuple[_Classify, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _Classify)
        )

    @property
    def label_filters(self) -> tuple[_LabelFilter, ...]:
        return tuple(
            operator
            for operator in self.operators
            if isinstance(operator, _LabelFilter)
        )

    @property
    def extracts(self) -> tuple[_Extract, ...]:
        return tuple(op for op in self.operators if isinstance(op, _Extract))

    def classify_output(self, output: str) -> _Classify:
        return next(
            operator for operator in self.classifies
            if operator.output == output
        )

    @property
    def base_alias(self) -> str:
        return self.relations[0].alias

    def relation(self, alias: str) -> _Relation:
        return next(
            relation for relation in self.relations if relation.alias == alias
        )


@dataclass(frozen=True)
class _Decoded:
    fields: tuple[tuple[str, str], ...]
    tables: tuple[tuple[str, str], ...]
    text_columns: tuple[tuple[str, str], ...]
    operators: tuple[_Operator, ...]


def _with_text_column(
    text_columns: tuple[tuple[str, str], ...],
    field: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    """Record the column an AI function reads from one relation."""
    alias, column = field
    known = dict(text_columns)
    if known.get(alias, column) != column:
        raise ValueError(f"relation {alias!r} is used with two text columns")
    if alias in known:
        return text_columns
    return (*text_columns, (alias, column))


def _function_names(plan: plan_pb2.Plan) -> dict[int, str]:
    urns = {}
    for declaration in plan.extension_urns:
        anchor = declaration.extension_urn_anchor
        if anchor in urns:
            raise ValueError(f"duplicate Substrait extension URN anchor {anchor}")
        urns[anchor] = declaration.urn
    names = {}
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            raise ValueError("QUAIL-B plans support scalar functions only")
        function = declaration.extension_function
        if function.function_anchor in names:
            raise ValueError(
                f"duplicate Substrait function anchor {function.function_anchor}"
            )
        expected_urn = _FUNCTION_URNS.get(function.name)
        if expected_urn is None:
            raise ValueError(
                f"unsupported Substrait function {function.name!r}"
            )
        if urns.get(function.extension_urn_reference) != expected_urn:
            raise ValueError(
                f"Substrait function {function.name!r} has the wrong URN"
            )
        names[function.function_anchor] = function.name
    return names


def _selection_index(expression: algebra_pb2.Expression) -> int:
    if not expression.HasField("selection"):
        raise ValueError("Substrait argument must be a field selection")
    reference = expression.selection
    if not reference.HasField("direct_reference"):
        raise ValueError("Substrait field must use a direct reference")
    segment = reference.direct_reference
    if not segment.HasField("struct_field") or segment.struct_field.HasField(
        "child"
    ):
        raise ValueError("Substrait field must select one top-level column")
    return segment.struct_field.field


def _selected(
    fields: tuple[tuple[str, str], ...],
    expression: algebra_pb2.Expression,
) -> tuple[str, str]:
    index = _selection_index(expression)
    if not 0 <= index < len(fields):
        raise ValueError(f"Substrait field index {index} is out of range")
    return fields[index]


def _arguments(
    expression: algebra_pb2.Expression,
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        raise ValueError("Substrait predicate must be a scalar function")
    return [
        argument.value for argument in expression.scalar_function.arguments
    ]


def _literal_string(expression: algebra_pb2.Expression) -> str:
    if not expression.HasField("literal"):
        raise ValueError("Substrait AI prompt must be a string literal")
    literal = expression.literal
    if literal.WhichOneof("literal_type") != "string":
        raise ValueError("Substrait AI prompt must be a string literal")
    return literal.string


def _string_list(expression: algebra_pb2.Expression,
                 name: str = "ai_classify labels") -> tuple[str, ...]:
    if (not expression.HasField("literal")
            or expression.literal.WhichOneof("literal_type") != "list"):
        raise ValueError(f"{name} must be a list literal")
    values = []
    for value in expression.literal.list.values:
        if value.WhichOneof("literal_type") != "string":
            raise ValueError(f"{name} must be strings")
        values.append(value.string)
    return tuple(values)


def _validate_labels(labels, descriptions) -> None:
    if len(labels) < 2:
        raise ValueError("ai_classify needs at least two labels")
    if len(descriptions) != len(labels):
        raise ValueError("ai_classify needs one description per label")
    folded = [" ".join(label.split()).casefold() for label in labels]
    if any(not label for label in folded) or len(set(folded)) != len(folded):
        raise ValueError("ai_classify labels must be nonempty and distinct")
    for label in folded:
        for other in folded:
            # the score of a label that continues another is never higher
            if other != label and other.startswith(label + " "):
                raise ValueError(
                    f"ai_classify label {other!r} extends label {label!r}")


def _flatten(
    expression: algebra_pb2.Expression,
    functions: dict[int, str],
) -> list[algebra_pb2.Expression]:
    if not expression.HasField("scalar_function"):
        return [expression]
    function = expression.scalar_function
    if functions.get(function.function_reference) != AND_NAME:
        return [expression]
    conditions = []
    for argument in function.arguments:
        conditions.extend(_flatten(argument.value, functions))
    return conditions


def _decode(
    rel: algebra_pb2.Rel,
    functions: dict[int, str],
) -> _Decoded:
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        alias = read.common.hint.alias
        if not alias:
            raise ValueError("QUAIL-B reads need a relation alias in hint.alias")
        if not read.HasField("named_table") or not read.named_table.names:
            raise ValueError("QUAIL-B reads need a named table")
        if len(read.base_schema.names) != len(read.base_schema.struct.types):
            raise ValueError("Substrait read schema names and types do not match")
        fields = tuple((alias, name) for name in read.base_schema.names)
        if (alias, "id") not in fields:
            raise ValueError("QUAIL-B reads need an id column")
        return _Decoded(fields, ((alias, read.named_table.names[-1]),), (), ())

    if kind == "project":
        expressions = rel.project.expressions
        if len(expressions) == 1 and functions.get(
                expressions[0].scalar_function.function_reference
        ) == AI_EXTRACT_NAME:
            return _decode_extract(rel.project, functions)
        return _decode_classify(rel.project, functions)

    if kind == "filter" and rel.filter.condition.HasField("singular_or_list"):
        child = _decode(rel.filter.input, functions)
        membership = rel.filter.condition.singular_or_list
        alias, output = _selected(child.fields, membership.value)
        classify = next(
            (operator for operator in child.operators
             if isinstance(operator, _Classify)
             and (operator.relation, operator.output) == (alias, output)),
            None)
        if classify is None:
            raise ValueError("a label filter must test an ai_classify column")
        accepted = tuple(_literal_string(option) for option in membership.options)
        if not accepted or len(set(accepted)) != len(accepted):
            raise ValueError("a label filter needs distinct accepted labels")
        if not set(accepted) <= set(classify.labels):
            raise ValueError("a label filter accepts a label not in its call")
        operator = _LabelFilter(
            rel.filter.common.hint.alias, alias, output, accepted)
        return _Decoded(child.fields, child.tables, child.text_columns,
                        (*child.operators, operator))

    if kind == "filter":
        child = _decode(rel.filter.input, functions)
        condition = rel.filter.condition
        function = condition.scalar_function
        if functions.get(function.function_reference) != AI_FILTER_NAME:
            raise ValueError("QUAIL-B FilterRel must call ai_filter")
        arguments = _arguments(condition)
        if len(arguments) != 2:
            raise ValueError("ai_filter needs a prompt and document")
        prompt = _literal_string(arguments[0])
        field = _selected(child.fields, arguments[1])
        operator = _Filter(rel.filter.common.hint.alias, field[0], prompt)
        return _Decoded(
            child.fields,
            child.tables,
            _with_text_column(child.text_columns, field),
            (*child.operators, operator),
        )

    if kind == "join":
        if rel.join.type != algebra_pb2.JoinRel.JOIN_TYPE_INNER:
            raise ValueError("QUAIL-B JoinRel must be an inner join")
        left = _decode(rel.join.left, functions)
        right = _decode(rel.join.right, functions)
        fields = left.fields + right.fields
        conditions = _flatten(rel.join.expression, functions)
        ai_conditions = [
            condition
            for condition in conditions
            if condition.HasField("scalar_function")
            and functions.get(
                condition.scalar_function.function_reference
            ) == AI_JOIN_NAME
        ]
        if len(ai_conditions) != 1:
            raise ValueError("QUAIL-B JoinRel needs one ai_join condition")
        arguments = _arguments(ai_conditions[0])
        if len(arguments) != 3:
            raise ValueError("ai_join needs a prompt and two documents")
        prompt = _literal_string(arguments[0])
        document_fields = (
            _selected(fields, arguments[1]),
            _selected(fields, arguments[2]),
        )
        relation_aliases = tuple(field[0] for field in document_fields)
        text_columns = (*left.text_columns, *right.text_columns)
        for field in document_fields:
            text_columns = _with_text_column(text_columns, field)
        on = []
        for condition in conditions:
            function = condition.scalar_function
            name = functions.get(function.function_reference)
            if name == AI_JOIN_NAME:
                continue
            if name != EQUAL_NAME:
                raise ValueError("unsupported QUAIL-B join condition")
            equality = _arguments(condition)
            if len(equality) != 2:
                raise ValueError("equal needs two fields")
            first, second = (
                _selected(fields, expression) for expression in equality
            )
            if (first[0], second[0]) == relation_aliases:
                on.append((first[1], second[1]))
            elif (second[0], first[0]) == relation_aliases:
                on.append((second[1], first[1]))
            else:
                raise ValueError("join equality uses unrelated relations")
        operator = _Join(
            rel.join.common.hint.alias,
            relation_aliases,
            prompt,
            tuple(on),
        )
        return _Decoded(
            fields,
            (*left.tables, *right.tables),
            text_columns,
            (*left.operators, *right.operators, operator),
        )

    raise ValueError(f"unsupported QUAIL-B Substrait relation {kind!r}")


def _decode_classify(project: algebra_pb2.ProjectRel,
                     functions: dict[int, str]) -> _Decoded:
    """Decode a ProjectRel that adds one ai_classify label column."""
    child = _decode(project.input, functions)
    if len(project.expressions) != 1 or project.common.HasField("emit"):
        raise ValueError("a classify ProjectRel adds exactly one column")
    expression = project.expressions[0]
    function = expression.scalar_function
    if (not expression.HasField("scalar_function")
            or functions.get(function.function_reference) != AI_CLASSIFY_NAME):
        raise ValueError("an inner QUAIL-B ProjectRel must call ai_classify")
    arguments = _arguments(expression)
    if len(arguments) != 4:
        raise ValueError(
            "ai_classify needs a prompt, document, labels, and descriptions")
    prompt = _literal_string(arguments[0])
    field = _selected(child.fields, arguments[1])
    labels = _string_list(arguments[2])
    descriptions = _string_list(arguments[3])
    _validate_labels(labels, descriptions)
    names = tuple(project.common.hint.output_names)
    expected = tuple(".".join(name) for name in child.fields)
    if len(names) != len(expected) + 1 or names[:-1] != expected:
        raise ValueError("a classify ProjectRel must name every output field")
    alias, _, output = names[-1].partition(".")
    if alias != field[0] or not output or "." in output:
        raise ValueError("a label column must belong to the classified relation")
    if output == "id" or (alias, output) in child.fields:
        raise ValueError(f"label column {output!r} is already a field")
    operator = _Classify(project.common.hint.alias, alias, prompt, labels,
                         descriptions, output)
    return _Decoded(
        (*child.fields, (alias, output)),
        child.tables,
        _with_text_column(child.text_columns, field),
        (*child.operators, operator),
    )


def extract_type(fields) -> type_pb2.Type:
    """Return the nullable response/error type of a field-name extraction."""
    nullable = type_pb2.Type.NULLABILITY_NULLABLE
    string = type_pb2.Type(
        string=type_pb2.Type.String(nullability=nullable))
    response = type_pb2.Type(struct=type_pb2.Type.Struct(
        types=[string for _ in fields], nullability=nullable))
    return type_pb2.Type(struct=type_pb2.Type.Struct(
        types=[response, string], nullability=nullable))


def _output_names(fields, operators, *, qualified=False) -> tuple[str, ...]:
    """Name selected fields depth first, including nested extraction structs."""
    extracts = {(op.relation, op.output): op.fields for op in operators
                if isinstance(op, _Extract)}
    names = []
    for alias, column in fields:
        names.append(f"{alias}.{column}" if qualified else (
            alias if column == "id" else column))
        if (alias, column) in extracts:
            names.extend(("response", *extracts[alias, column], "error"))
    return tuple(names)


def _decode_extract(project, functions) -> _Decoded:
    child = _decode(project.input, functions)
    if project.common.HasField("emit"):
        raise ValueError("an extract ProjectRel adds exactly one column")
    expression = project.expressions[0]
    call = expression.scalar_function
    if (len(call.arguments) != 3
            or not all(arg.HasField("value") for arg in call.arguments[:2])
            or not call.arguments[2].HasField("type")):
        raise ValueError("ai_extract needs a document, fields, and result type")
    arguments = [arg.value for arg in call.arguments[:2]]
    field = _selected(child.fields, arguments[0])
    names = _string_list(arguments[1], "ai_extract fields")
    if not names or any(not name.strip() for name in names):
        raise ValueError("ai_extract needs nonempty field names")
    if len(set(names)) != len(names):
        raise ValueError("ai_extract field names must be distinct")
    if (call.output_type != extract_type(names)
            or call.arguments[2].type != extract_type(names)):
        raise ValueError("ai_extract has an invalid response/error output type")
    hints = tuple(project.common.hint.output_names)
    previous = _output_names(child.fields, child.operators, qualified=True)
    if len(hints) != len(previous) + len(names) + 3 or (
            hints[:len(previous)] != previous):
        raise ValueError("an extract ProjectRel must name every nested field")
    alias, _, output = hints[len(previous)].partition(".")
    if alias != field[0] or not output or "." in output:
        raise ValueError("an extract column must belong to its document relation")
    if output == "id" or (alias, output) in child.fields:
        raise ValueError(f"extract column {output!r} is already a field")
    if hints[len(previous) + 1:] != ("response", *names, "error"):
        raise ValueError("ai_extract field names do not match its output")
    operator = _Extract(project.common.hint.alias, alias, names, output)
    return _Decoded(
        (*child.fields, (alias, output)), child.tables,
        _with_text_column(child.text_columns, field),
        (*child.operators, operator))


def _field_alias(name: str) -> str:
    alias, separator, column = name.partition(".")
    if not separator or not alias or not column or "." in column:
        raise ValueError(f"invalid field name {name!r}")
    return alias


def _validate_info(info: _PlanInfo) -> None:
    aliases = [relation.alias for relation in info.relations]
    if not aliases or len(set(aliases)) != len(aliases):
        raise ValueError("QUAIL-B relation aliases must be nonempty and unique")
    operator_ids = [operator.id for operator in info.operators]
    if any(not operator_id for operator_id in operator_ids):
        raise ValueError("QUAIL-B operator IDs must be nonempty")
    if len(set(operator_ids)) != len(operator_ids):
        raise ValueError("QUAIL-B operator IDs must be unique")
    if len(info.relations) != len(info.joins) + 1:
        raise ValueError("QUAIL-B plan must join every relation")
    joined = {info.base_alias}
    for join in info.joins:
        referenced = set(join.relations)
        if len(referenced & joined) != 1 or len(referenced - joined) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        joined.update(referenced)
    first_join = {alias: len(info.operators) for alias in aliases}
    for index, operator in enumerate(info.operators):
        if isinstance(operator, _Join):
            for alias in operator.relations:
                first_join[alias] = min(first_join[alias], index)
    for index, operator in enumerate(info.operators):
        if (
            isinstance(operator, (_Filter, _Classify, _LabelFilter))
            and index > first_join[operator.relation]
        ):
            raise ValueError(
                f"filter {operator.id!r} must precede joins on its relation"
            )
    generated = (*info.classifies, *info.extracts)
    outputs = [operator.output for operator in generated]
    if len(set(outputs)) != len(outputs) or set(outputs) & set(aliases):
        raise ValueError("generated column names must be distinct from each "
                         "other and from relation aliases")
    selected = {_field_alias(name) for name in info.select}
    if not selected or not selected <= set(aliases):
        raise ValueError("QUAIL-B projection uses an unknown relation")
    selected_ids = {
        _field_alias(name) for name in info.select if name.endswith(".id")}
    if selected_ids != selected:
        raise ValueError("a query selecting a label column must select its "
                         "relation's id")
    for name in info.select:
        alias, column = name.split(".", 1)
        if column != "id" and not any(
                (operator.relation, operator.output) == (alias, column)
                for operator in generated):
            raise ValueError(f"QUAIL-B projection selects {name!r}; only ids "
                             "and generated columns can be selected")
    used = {operator.output for operator in info.label_filters}
    used.update(name.split(".", 1)[1] for name in info.select)
    for operator in generated:
        if operator.output not in used:
            raise ValueError(
                f"label column {operator.output!r} is never used")
    if info.extracts:
        if len(set(info.select)) != len(info.select):
            raise ValueError("an extraction projection cannot repeat a column")
        if len(info.relations) != 1 or info.classifies:
            raise ValueError("extraction queries use one table without classify")
        extracting = False
        for operator in info.operators:
            if isinstance(operator, _Extract):
                extracting = True
            elif extracting:
                raise ValueError("all filters must precede extraction")


def _inspect_plan(plan: plan_pb2.Plan) -> _PlanInfo:
    """Return the benchmark information in a supported Substrait plan."""
    version = plan.version
    if (
        version.major_number,
        version.minor_number,
        version.patch_number,
    ) != SUBSTRAIT_VERSION:
        raise ValueError(
            f"QUAIL-B requires Substrait {'.'.join(map(str, SUBSTRAIT_VERSION))}"
        )
    if len(plan.relations) != 1 or not plan.relations[0].HasField("root"):
        raise ValueError("QUAIL-B needs one Substrait root relation")
    root = plan.relations[0].root
    if not root.input.HasField("project"):
        raise ValueError("QUAIL-B root must contain a ProjectRel")
    project = root.input.project
    decoded = _decode(project.input, _function_names(plan))
    select = tuple(
        ".".join(_selected(decoded.fields, expression))
        for expression in project.expressions
    )
    expected_mapping = tuple(
        len(decoded.fields) + index
        for index in range(len(project.expressions))
    )
    if tuple(project.common.emit.output_mapping) != expected_mapping:
        raise ValueError("QUAIL-B ProjectRel has an invalid output mapping")
    expected_names = _output_names(
        (name.split(".", 1) for name in select), decoded.operators)
    if tuple(root.names) != expected_names:
        raise ValueError("QUAIL-B root names do not match selected relations")
    text_columns = dict(decoded.text_columns)
    relations = []
    for alias, table in decoded.tables:
        if alias not in text_columns:
            raise ValueError(f"relation {alias!r} is not read by an AI function")
        relations.append(_Relation(alias, table, text_columns[alias]))
    info = _PlanInfo(tuple(relations), decoded.operators, select)
    _validate_info(info)
    return info
