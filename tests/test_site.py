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


def test_image_headings_keep_accessible_titles_and_unique_navigation(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "heading.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (tmp_path / "README.md").write_text('# ![Curious Coder](assets/heading.svg)\n\n## ![Evidence](assets/heading.svg)\n\n## ![Evidence](assets/heading.svg)')
    output = tmp_path / "site"
    build_site(tmp_path, output)
    html = (output / "index.html").read_text()
    assert '<title>Curious Coder · Curious Coder</title>' in html
    assert 'id="curious-coder"' in html
    assert 'href="#evidence"' in html and 'href="#evidence-1"' in html
    assert 'alt="Curious Coder"' in html


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
