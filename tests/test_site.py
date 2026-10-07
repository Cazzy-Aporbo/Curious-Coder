from pathlib import Path

import pytest

from scripts.build_site import build_site, rewrite_link, validate_links


def test_link_rewriting_preserves_external_urls_and_fragments():
    assert rewrite_link("README.md#start") == "index.html#start"
    assert rewrite_link("notes.md#question") == "notes.html#question"
    assert rewrite_link("../notes.md") == "../notes.html"
    assert rewrite_link("https://example.org/notes.md") == "https://example.org/notes.md"
    assert rewrite_link("lesson.py") == "lesson.py"


def test_site_build_is_repeatable_and_only_copies_learning_content(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "README.md").write_text('# A learning lab\n\n[Note](note.md)\n\n## Question\n\nTry it.')
    (root / "note.md").write_text('# A note\n\n[Home](README.md)')
    (root / ".env").write_text("PRIVATE=not-for-publication")
    output = tmp_path / "site"
    assert build_site(root, output) == 2
    first = (output / "index.html").read_bytes()
    assert build_site(root, output) == 2
    assert first == (output / "index.html").read_bytes()
    assert not (output / ".env").exists()
    assert 'href="note.html"' in first.decode()
    assert 'lang="en"' in first.decode()


def test_site_rejects_broken_local_links(tmp_path):
    (tmp_path / "index.html").write_text('<a href="missing.html">Missing</a>')
    with pytest.raises(ValueError, match="missing.html"):
        validate_links(tmp_path)


def test_site_rejects_links_escaping_artifact(tmp_path):
    (tmp_path / "index.html").write_text('<a href="../">Outside</a>')
    with pytest.raises(ValueError, match="missing local target"):
        validate_links(tmp_path)


def test_site_refuses_to_overwrite_existing_content(tmp_path):
    (tmp_path / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="Refusing"):
        build_site(Path(__file__).parents[1], tmp_path)
    assert (tmp_path / "important.txt").read_text() == "keep"
