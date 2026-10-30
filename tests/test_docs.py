"""The Markdown docs point at things that exist: links, module commands and script paths."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)\)")
MODULE_COMMAND = re.compile(r"python -m (chess_engine(?:\.\w+)+)")
SCRIPT_PATH = re.compile(r"\bscripts/[\w./-]+\.(?:py|sh)\b")


def checked_docs():
    """README and every doc except the internal planning records in ``docs/superpowers/``.

    :return: Paths of the Markdown files to check.
    """
    docs = ROOT / "docs"
    others = sorted(
        path for path in docs.rglob("*.md") if path.relative_to(docs).parts[0] != "superpowers"
    )
    return [ROOT / "README.md", *others]


def test_relative_links_resolve():
    broken = []
    for doc in checked_docs():
        for target in LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = (doc.parent / target.split("#", 1)[0]).resolve()
            if path != ROOT and ROOT not in path.parents:
                continue  # GitHub-relative link (the CI badge), not a file of this checkout
            if not path.exists():
                broken.append(f"{doc.relative_to(ROOT)}: {target}")
    assert not broken, "\n".join(broken)


def test_quoted_module_commands_import():
    missing = []
    for doc in checked_docs():
        for module in MODULE_COMMAND.findall(doc.read_text(encoding="utf-8")):
            if importlib.util.find_spec(module) is None:
                missing.append(f"{doc.relative_to(ROOT)}: {module}")
    assert not missing, "\n".join(missing)


def test_quoted_script_paths_exist():
    missing = []
    for doc in checked_docs():
        for path in set(SCRIPT_PATH.findall(doc.read_text(encoding="utf-8"))):
            if not (ROOT / path).exists():
                missing.append(f"{doc.relative_to(ROOT)}: {path}")
    assert not missing, "\n".join(missing)
