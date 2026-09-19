#!/usr/bin/env python3
"""Identify and rename leading-underscore class/function names.

Scope: production Python under sglang_omni/ and sglang_omni_router/python/.
Nested functions (and classes defined inside functions) keep their names.
Dunder names, name-mangled __foo names, and a lone "_" are never touched.
Vendor copies of upstream code are excluded from definition rewrites.

Colliding symbols (a public name already exists in the same scope) are left
alone. Attribute access for generic names that also appear as data fields is
rewritten only inside the defining class.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tokenize
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPO_ROOT / "sglang_omni",
    REPO_ROOT / "sglang_omni_router" / "python",
)
VENDOR_PREFIX = REPO_ROOT / "sglang_omni" / "vendor"
REFERENCE_ROOTS = (
    REPO_ROOT / "sglang_omni",
    REPO_ROOT / "sglang_omni_router",
    REPO_ROOT / "tests",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "examples",
    REPO_ROOT / "playground",
    REPO_ROOT / ".github",
)
SKIP_SCRIPT = Path(__file__).resolve()

# Attribute names that also exist as data fields. Do not globally rewrite
# `.name` for these; only rewrite inside the defining class / import sites.
ATTRIBUTE_COLLISION_NAMES = {
    "_batch_buckets",
    "_codec_lock",
    "_device",
    "_forward",
    "_inputs",
    "_process_start_attempts",
    "_request_id",
    "_scheduler",
    "_stages",
    "_state",
    "_worker",
}


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    new_name: str
    path: str
    lineno: int
    col_offset: int
    qualname: str
    class_qualname: str | None
    is_method: bool
    collision: str | None = None


@dataclass(frozen=True)
class Site:
    lineno: int
    col: int
    old: str
    new: str


def is_vendor(path: Path) -> bool:
    return path.is_relative_to(VENDOR_PREFIX)


def iter_python_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == SKIP_SCRIPT:
                continue
            files.append(path)
    return sorted(files)


def should_rename(name: str) -> bool:
    if name == "_" or name.startswith("__"):
        return False
    return name.startswith("_")


def public_name(name: str) -> str:
    return name[1:] if name.startswith("_") else name


def module_name_for(path: Path) -> str:
    rel = path.resolve().relative_to(REPO_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_from_module(
    current_mod: str, is_package: bool, module: str | None, level: int
) -> str | None:
    if level == 0:
        return module
    parts = current_mod.split(".")
    if not is_package:
        parts = parts[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    if module:
        parts = parts + module.split(".")
    return ".".join(parts) if parts else None


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.function_depth = 0
        self.class_stack: list[str] = []
        self.symbols: list[Symbol] = []
        self.module_names: set[str] = set()
        self.class_names: dict[str, set[str]] = defaultdict(set)
        self.attribute_stores: dict[str, set[str]] = defaultdict(set)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join([*self.class_stack, node.name])
        if self.function_depth == 0:
            if self.class_stack:
                self.class_names[".".join(self.class_stack)].add(node.name)
            else:
                self.module_names.add(node.name)
            if should_rename(node.name):
                self.symbols.append(
                    self._symbol("class", node, qualname, is_method=False)
                )
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0:
            owner = ".".join(self.class_stack) or None
            if owner is None:
                self.module_names.add(node.name)
            else:
                self.class_names[owner].add(node.name)
            if should_rename(node.name):
                kind = (
                    "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                self.symbols.append(
                    self._symbol(
                        kind,
                        node,
                        ".".join([*self.class_stack, node.name]),
                        is_method=owner is not None,
                    )
                )
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        owner = ".".join(self.class_stack) or None
        if owner is not None and self.function_depth > 0:
            for target in node.targets:
                self._record_self_store(target, owner)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        owner = ".".join(self.class_stack) or None
        if owner is not None:
            self._record_self_store(node.target, owner)
        self.generic_visit(node)

    def _record_self_store(self, target: ast.expr, owner: str) -> None:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id in {"self", "cls"} and should_rename(target.attr):
                self.attribute_stores[owner].add(target.attr)

    def _symbol(
        self,
        kind: str,
        node: ast.AST,
        qualname: str,
        *,
        is_method: bool,
    ) -> Symbol:
        return Symbol(
            kind=kind,
            name=node.name,  # type: ignore[attr-defined]
            new_name=public_name(node.name),  # type: ignore[attr-defined]
            path=str(self.path),
            lineno=node.lineno,
            col_offset=node.col_offset,
            qualname=qualname,
            class_qualname=".".join(self.class_stack) or None,
            is_method=is_method,
        )


def collect_file(path: Path) -> tuple[list[Symbol], DefinitionCollector]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collector = DefinitionCollector(path)
    collector.visit(tree)
    return collector.symbols, collector


def annotate_collisions(
    symbols: list[Symbol], collector: DefinitionCollector
) -> list[Symbol]:
    annotated: list[Symbol] = []
    for symbol in symbols:
        collision = None
        if symbol.class_qualname is None:
            if symbol.new_name in collector.module_names:
                collision = f"module already has {symbol.new_name}"
        else:
            existing = collector.class_names.get(symbol.class_qualname, set())
            if symbol.new_name in existing:
                collision = f"class {symbol.class_qualname} already has {symbol.new_name}"
            elif symbol.is_method and symbol.name in collector.attribute_stores.get(
                symbol.class_qualname, set()
            ):
                collision = (
                    f"class {symbol.class_qualname} also stores attribute {symbol.name}"
                )
        annotated.append(
            Symbol(
                kind=symbol.kind,
                name=symbol.name,
                new_name=symbol.new_name,
                path=symbol.path,
                lineno=symbol.lineno,
                col_offset=symbol.col_offset,
                qualname=symbol.qualname,
                class_qualname=symbol.class_qualname,
                is_method=symbol.is_method,
                collision=collision,
            )
        )
    return annotated


def scan_definitions() -> list[Symbol]:
    symbols: list[Symbol] = []
    for path in iter_python_files(SOURCE_ROOTS):
        if is_vendor(path):
            continue
        try:
            file_symbols, collector = collect_file(path)
        except SyntaxError as exc:
            print(f"skip unreadable {path}: {exc}", file=sys.stderr)
            continue
        symbols.extend(annotate_collisions(file_symbols, collector))
    return symbols


def write_report(symbols: list[Symbol], output: Path) -> None:
    payload = {
        "total": len(symbols),
        "collisions": sum(1 for s in symbols if s.collision),
        "by_kind": {
            "class": sum(1 for s in symbols if s.kind == "class"),
            "function": sum(1 for s in symbols if s.kind == "function"),
            "async_function": sum(1 for s in symbols if s.kind == "async_function"),
            "method": sum(1 for s in symbols if s.is_method),
            "module_level": sum(1 for s in symbols if not s.is_method),
        },
        "symbols": [asdict(s) for s in symbols],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def definition_name_col(node: ast.AST, line: str) -> int:
    keyword = "class" if isinstance(node, ast.ClassDef) else "def"
    idx = line.find(keyword, node.col_offset)
    if idx < 0:
        idx = line.find(keyword)
    idx += len(keyword)
    while idx < len(line) and line[idx].isspace():
        idx += 1
    return idx


def attr_span(node: ast.Attribute) -> tuple[int, int]:
    return node.end_lineno or node.lineno, (node.end_col_offset or 0) - len(node.attr)


class RenameTables:
    def __init__(self, symbols: list[Symbol]) -> None:
        self.module_funcs: dict[str, dict[str, str]] = defaultdict(dict)
        self.class_methods: dict[str, dict[str, dict[str, str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.class_simple: dict[str, dict[str, str]] = defaultdict(dict)
        self.renamed_names: dict[str, str] = {}
        self.method_names: dict[str, str] = {}
        for symbol in symbols:
            if symbol.collision:
                continue
            module = module_name_for(Path(symbol.path))
            self.renamed_names[symbol.name] = symbol.new_name
            if symbol.class_qualname is None:
                self.module_funcs[module][symbol.name] = symbol.new_name
            else:
                simple = symbol.class_qualname.split(".")[-1]
                self.class_methods[module][symbol.class_qualname][symbol.name] = (
                    symbol.new_name
                )
                self.class_simple[simple][symbol.name] = symbol.new_name
                if symbol.is_method:
                    self.method_names[symbol.name] = symbol.new_name

    def method_in_class(self, module: str, class_qualname: str, name: str) -> str | None:
        current = class_qualname
        while current:
            mapping = self.class_methods.get(module, {}).get(current)
            if mapping and name in mapping:
                return mapping[name]
            if "." not in current:
                break
            current = current.rsplit(".", 1)[0]
        simple = class_qualname.split(".")[-1]
        return self.class_simple.get(simple, {}).get(name)


class SiteCollector(ast.NodeVisitor):
    def __init__(
        self,
        path: Path,
        source_lines: list[str],
        tables: RenameTables,
    ) -> None:
        self.path = path
        self.lines = source_lines
        self.tables = tables
        self.module = module_name_for(path)
        self.is_package = path.name == "__init__.py"
        self.function_depth = 0
        self.class_stack: list[str] = []
        self.nested_defs: set[str] = set()
        self.local_name_renames: dict[str, str] = dict(
            tables.module_funcs.get(self.module, {})
        )
        self.imported_modules: dict[str, str] = {}
        self.sites: list[Site] = []

    def add(self, lineno: int, col: int, old: str, new: str) -> None:
        if old == new:
            return
        self.sites.append(Site(lineno, col, old, new))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.function_depth == 0:
            new = self.local_name_renames.get(node.name)
            if new:
                self.add(node.lineno, definition_name_col(node, self.lines[node.lineno - 1]), node.name, new)
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0:
            owner = ".".join(self.class_stack) or None
            if owner is None:
                new = self.local_name_renames.get(node.name)
            else:
                new = self.tables.method_in_class(self.module, owner, node.name)
            if new:
                self.add(
                    node.lineno,
                    definition_name_col(node, self.lines[node.lineno - 1]),
                    node.name,
                    new,
                )
        else:
            self.nested_defs.add(node.name)
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        if self.function_depth == 0:
            self.nested_defs.clear()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[-1]
            self.imported_modules[bound] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = resolve_from_module(self.module, self.is_package, node.module, node.level)
        for alias in node.names:
            if alias.name == "*":
                continue
            new = None
            if target:
                new = self.tables.module_funcs.get(target, {}).get(alias.name)
                if new is None:
                    new = self.tables.class_simple.get(alias.name, {}).get(alias.name)
                    if alias.name in self.tables.class_simple or any(
                        alias.name == qual.split(".")[-1]
                        for qual_map in self.tables.class_methods.get(target, {})
                        for qual in [qual_map]
                    ):
                        pass
                if new is None:
                    # Imported class name itself.
                    class_renames = self.tables.module_funcs.get(target, {})
                    new = class_renames.get(alias.name)
            if new:
                lineno = getattr(alias, "lineno", node.lineno)
                col = getattr(alias, "col_offset", None)
                if col is None:
                    line = self.lines[lineno - 1]
                    col = line.find(alias.name)
                else:
                    # col_offset points at the imported name.
                    pass
                if col >= 0:
                    self.add(lineno, col, alias.name, new)
                bound = alias.asname or alias.name
                if alias.asname is None:
                    self.local_name_renames[bound] = new
                    self.local_name_renames.pop(alias.name, None)
                    self.local_name_renames[new] = new
                    # After rewrite, remaining references still use the old
                    # identifier until we rename them too.
                    self.local_name_renames[alias.name] = new
                if target:
                    self.imported_modules[alias.asname or new] = f"{target}.{alias.name}"
            elif target:
                bound = alias.asname or alias.name
                self.imported_modules[bound] = f"{target}.{alias.name}"
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.nested_defs:
            return
        new = self.local_name_renames.get(node.id)
        if new and new != node.id:
            self.add(node.lineno, node.col_offset, node.id, new)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        new = None
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            owner = ".".join(self.class_stack)
            if owner:
                new = self.tables.method_in_class(self.module, owner, node.attr)
        elif isinstance(node.value, ast.Name):
            simple = node.value.id
            new = self.tables.class_simple.get(simple, {}).get(node.attr)
            if new is None:
                imported = self.imported_modules.get(simple)
                if imported:
                    new = self.tables.module_funcs.get(imported, {}).get(node.attr)
                    if new is None:
                        # imported may be a module, not module.symbol
                        new = self.tables.module_funcs.get(imported, {}).get(node.attr)
                        # try parent module if imported is module.Class
                        if "." in imported:
                            parent, _, cls = imported.rpartition(".")
                            new = self.tables.method_in_class(parent, cls, node.attr)
                            if new is None:
                                new = self.tables.module_funcs.get(parent, {}).get(
                                    node.attr
                                )
        if new:
            lineno, col = attr_span(node)
            if col >= 0:
                self.add(lineno, col, node.attr, new)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_attr_helper = isinstance(func, ast.Name) and func.id in {
            "getattr",
            "setattr",
            "hasattr",
            "delattr",
        }
        if (
            is_attr_helper
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            key = node.args[1].value
            new = self.local_name_renames.get(key) or self.tables.method_names.get(key)
            if new and key not in ATTRIBUTE_COLLISION_NAMES:
                const = node.args[1]
                self.add(const.lineno, const.col_offset + 1, key, new)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                self._rewrite_all(node.value)
        self.generic_visit(node)

    def _rewrite_all(self, value: ast.expr) -> None:
        elts: list[ast.expr] = []
        if isinstance(value, (ast.List, ast.Tuple)):
            elts = list(value.elts)
        for elt in elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                new = self.local_name_renames.get(elt.value)
                if new:
                    self.add(elt.lineno, elt.col_offset + 1, elt.value, new)


def apply_sites(source: str, sites: list[Site]) -> str:
    if not sites:
        return source
    lines = source.splitlines(keepends=True)
    # Last site on a line first so columns stay valid.
    ordered = sorted(sites, key=lambda s: (s.lineno, s.col), reverse=True)
    for site in ordered:
        if site.lineno < 1 or site.lineno > len(lines):
            raise RuntimeError(f"bad site line {site}")
        line = lines[site.lineno - 1]
        actual = line[site.col : site.col + len(site.old)]
        if actual != site.old:
            raise RuntimeError(
                f"site mismatch line {site.lineno} col {site.col}: "
                f"expected {site.old!r} found {actual!r} in {line!r}"
            )
        lines[site.lineno - 1] = (
            line[: site.col] + site.new + line[site.col + len(site.old) :]
        )
    return "".join(lines)


def collect_sites(path: Path, tables: RenameTables) -> list[Site]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    collector = SiteCollector(path, lines, tables)
    collector.visit(tree)
    return collector.sites


def rewrite_dotted_names(path: Path, mapping: dict[str, str]) -> list[Site]:
    if not mapping:
        return []
    sites: list[Site] = []
    readline = path.open("rb").readline
    tokens = tokenize.tokenize(readline)
    prev: tokenize.TokenInfo | None = None
    for token in tokens:
        if (
            prev is not None
            and prev.string == "."
            and token.type == tokenize.NAME
            and token.string in mapping
        ):
            sites.append(
                Site(token.start[0], token.start[1], token.string, mapping[token.string])
            )
        prev = token
    return sites


def apply_renames(symbols: list[Symbol]) -> dict[str, int]:
    tables = RenameTables(symbols)
    dotted_mapping = {
        old: new
        for old, new in tables.method_names.items()
        if old not in ATTRIBUTE_COLLISION_NAMES
    }
    stats = {"files": 0, "sites": 0}
    for path in iter_python_files(REFERENCE_ROOTS):
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        sites = collect_sites(path, tables)
        if not is_vendor(path):
            sites.extend(rewrite_dotted_names(path, dotted_mapping))
        # Dedup exact same spans.
        unique: dict[tuple[int, int, str], Site] = {}
        for site in sites:
            unique[(site.lineno, site.col, site.old)] = site
        sites = list(unique.values())
        if not sites:
            continue
        rewritten = apply_sites(source, sites)
        if rewritten != source:
            path.write_text(rewritten, encoding="utf-8")
            stats["files"] += 1
            stats["sites"] += len(sites)
    return stats


def cmd_scan(args: argparse.Namespace) -> int:
    symbols = scan_definitions()
    write_report(symbols, Path(args.output))
    collisions = [s for s in symbols if s.collision]
    print(f"found {len(symbols)} leading-underscore class/function names")
    print(f"collisions skipped: {len(collisions)}")
    print(f"report: {args.output}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    if args.from_report:
        payload = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
        symbols = [Symbol(**item) for item in payload["symbols"]]
    else:
        symbols = scan_definitions()
    stats = apply_renames(symbols)
    print(f"rewrote {stats['files']} files ({stats['sites']} identifier sites)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="Write a JSON report of rename candidates")
    scan.add_argument(
        "--output",
        default=str(REPO_ROOT / "scripts" / "leading_underscore_report.json"),
    )
    scan.set_defaults(func=cmd_scan)
    apply = sub.add_parser("apply", help="Apply identifier renames")
    apply.add_argument("--from-report", default="")
    apply.set_defaults(func=cmd_apply)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
