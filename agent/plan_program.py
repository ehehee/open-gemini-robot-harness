"""Parse the Code-as-Policies text Gemini POSTs to /plan into a list of
primitive calls -- WITHOUT exec/eval.

The post's interface has the model write Python-looking calls
(`x, y = robot.deproject(u=0.52, v=0.48, z_plane=0.015)` then
`robot.grasp(x=x, ...)`). Running model-written Python would hand it the whole
machine, so this walks the AST and accepts only:
  * `robot.<primitive>(kw=expr, ...)` as a statement
  * `name = robot.<pure primitive>(...)` / `a, b = robot.<pure primitive>(...)`
  * `name = expr`
where expr is a number, a variable assigned earlier, unary -/+, or + - * / of
those. Anything else is rejected with a message the model can act on.

Pure primitives (deproject*) are evaluated during parsing since they depend
only on the last observation, never on motion -- that's what lets the whole
program be previewed before anything moves.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class PlanError(ValueError):
    pass


@dataclass
class Call:
    name: str
    kwargs: dict[str, float]
    line: int
    source: str
    arm: str = "robot"  # which receiver: "robot" (single-arm) or "left"/"right"


def parse_plan(code: str, signatures: dict[str, dict[str, float | None]],
               pure: dict[str, Callable[..., tuple]],
               receivers: tuple[str, ...] = ("robot",)) -> list[Call | list[Call]]:
    """signatures: primitive -> {param: default} (None = required).
    pure: primitive -> function(**kwargs) returning a tuple, run at parse time.
    receivers: the names a call may be made on. With more than one (bimanual:
    "left", "right"), `parallel(left.X(...), right.Y(...))` is also allowed and
    comes back as a list of Calls to run at the same time."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise PlanError(f"syntax error on line {e.lineno}: {e.msg}") from None
    env: dict[str, float] = {}
    calls: list[Call] = []
    lines = code.splitlines()

    def value(node) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise PlanError(f"line {node.lineno}: '{node.id}' is not defined")
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](value(node.left), value(node.right))
        raise PlanError(f"line {getattr(node, 'lineno', '?')}: only numbers, variables and + - * / are allowed, "
                        f"got `{ast.unparse(node)}`")

    recv = " / ".join(f"`{r}.<primitive>(...)`" for r in receivers)

    def robot_call(node) -> Call:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in receivers):
            raise PlanError(f"line {node.lineno}: expected a {recv} call, got `{ast.unparse(node)}`")
        arm = node.func.value.id
        name = node.func.attr
        if name not in signatures:
            raise PlanError(f"line {node.lineno}: unknown primitive robot.{name}; "
                            f"available: {', '.join(sorted(signatures))}")
        sig = signatures[name]
        if len(node.args) > len(sig):
            raise PlanError(f"line {node.lineno}: robot.{name} got too many positional args")
        kwargs = {p: value(a) for p, a in zip(sig, node.args)}
        for kw in node.keywords:
            if kw.arg is None or kw.arg not in sig:
                raise PlanError(f"line {node.lineno}: robot.{name} has no parameter '{kw.arg}'; "
                                f"parameters: {', '.join(sig)}")
            kwargs[kw.arg] = value(kw.value)
        for p, default in sig.items():
            if p not in kwargs:
                if default is None:
                    raise PlanError(f"line {node.lineno}: robot.{name} is missing required '{p}'")
                kwargs[p] = default
        return Call(name, kwargs, node.lineno, lines[node.lineno - 1].strip(), arm)

    for stmt in tree.body:
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "parallel"):
            if len(receivers) < 2:
                raise PlanError(f"line {stmt.lineno}: parallel(...) is only available with two arms")
            if stmt.value.keywords or len(stmt.value.args) != 2:
                raise PlanError(f"line {stmt.lineno}: parallel(...) takes exactly two arm calls")
            group = [robot_call(a) for a in stmt.value.args]
            if group[0].arm == group[1].arm:
                raise PlanError(f"line {stmt.lineno}: parallel(...) needs one call per arm, got {group[0].arm} twice")
            if any(c.name in pure for c in group):
                raise PlanError(f"line {stmt.lineno}: deproject can't go inside parallel(...)")
            calls.append(group)
            continue
        if isinstance(stmt, ast.Expr):
            call = robot_call(stmt.value)
            if call.name in pure:
                continue  # result discarded; nothing to do
            calls.append(call)
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            names = ([target.id] if isinstance(target, ast.Name)
                     else [t.id for t in target.elts] if isinstance(target, ast.Tuple)
                     and all(isinstance(t, ast.Name) for t in target.elts) else None)
            if names is None:
                raise PlanError(f"line {stmt.lineno}: only `name = ...` or `a, b = ...` assignments are allowed")
            if isinstance(stmt.value, ast.Call):
                call = robot_call(stmt.value)
                if call.name not in pure:
                    raise PlanError(f"line {stmt.lineno}: robot.{call.name} returns nothing to assign")
                result = pure[call.name](**call.kwargs)
                if len(names) != len(result):
                    raise PlanError(f"line {stmt.lineno}: robot.{call.name} returns {len(result)} values, "
                                    f"assigned to {len(names)}")
                env.update(zip(names, (float(r) for r in result)))
            else:
                if len(names) != 1:
                    raise PlanError(f"line {stmt.lineno}: tuple assignment only from robot.deproject*")
                env[names[0]] = value(stmt.value)
        else:
            raise PlanError(f"line {stmt.lineno}: only {recv} calls, parallel(...) and simple assignments "
                            f"are allowed, got `{ast.unparse(stmt)}`")
    if not calls:
        raise PlanError("the plan contains no robot actions")
    return calls
