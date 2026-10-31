"""The Markdown docs point at things that exist: links, module commands and script paths."""

import importlib.util
import re

from tests.paths import ROOT

LINK = re.compile(r"\]\(([^)\s]+)\)|\b(?:src|srcset)=\"([^\"]+)\"")
TECHNIQUE_SECTIONS = [
    "Problem",
    "Idea",
    "How it works",
    "Implementation",
    "Measured effect",
    "Status",
]
MODULE_COMMAND = re.compile(r"python -m ((?:chess_engine|chess_lab)(?:\.\w+)+)")
SCRIPT_PATH = re.compile(r"\bsrc/chess_lab/scripts/[\w./-]+\.(?:py|sh)\b")


def checked_docs():
    """README and every page under ``docs/``.

    :return: Paths of the Markdown files to check.
    """
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def links(doc):
    """Targets of the Markdown links and of the ``src``/``srcset`` attributes of a page.

    :param doc: Path of the Markdown file.
    :return: Link targets, as written.
    """
    return [a or b for a, b in LINK.findall(doc.read_text(encoding="utf-8"))]


def linked_from(index):
    """Files that an index page links to.

    :param index: Path of the index page.
    :return: Resolved paths of its relative links.
    """
    targets = links(index)
    return {(index.parent / t.split("#", 1)[0]).resolve() for t in targets if "://" not in t}


def test_relative_links_resolve():
    broken = []
    for doc in checked_docs():
        for target in links(doc):
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


def test_every_page_is_in_an_index():
    docs = ROOT / "docs"
    listed = linked_from(docs / "README.md") | linked_from(docs / "techniques" / "README.md")
    indexes = {docs / "README.md", docs / "techniques" / "README.md"}
    orphans = [
        str(page.relative_to(ROOT))
        for page in docs.rglob("*.md")
        if page not in indexes and page.resolve() not in listed
    ]
    assert not orphans, "\n".join(orphans)


def test_technique_pages_follow_the_template():
    folder = ROOT / "docs" / "techniques"
    wrong = []
    for page in sorted(folder.glob("*.md")):
        if page.name == "README.md":
            continue
        sections = re.findall(r"^## (.+)$", page.read_text(encoding="utf-8"), re.MULTILINE)
        if sections != TECHNIQUE_SECTIONS:
            wrong.append(f"{page.name}: {sections}")
    assert not wrong, "\n".join(wrong)
