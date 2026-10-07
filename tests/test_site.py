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


def test_published_site_links_are_built_as_checked_local_links(tmp_path):
    from scripts.build_site import SITE_URL
    (tmp_path / "studies").mkdir()
    (tmp_path / "studies" / "note.md").write_text("# Note")
    (tmp_path / "README.md").write_text(f"# Home\n\n[Note]({SITE_URL}studies/note.html#part)")
    output = tmp_path / "site"
    build_site(tmp_path, output)
    assert 'href="studies/note.html#part"' in (output / "index.html").read_text()
    (tmp_path / "README.md").write_text(f"# Home\n\n[Missing]({SITE_URL}studies/missing.html)")
    with pytest.raises(ValueError, match="missing"):
        build_site(tmp_path, output)


def test_site_copies_only_named_public_policy_files_and_links_credit(tmp_path):
    (tmp_path / "README.md").write_text("# Home")
    (tmp_path / "studies").mkdir()
    (tmp_path / "studies/note.md").write_text("# Note")
    (tmp_path / "COPYRIGHT.md").write_text("# Attribution and reuse")
    for name in ("robots.txt", "CITATION.cff"):
        (tmp_path / name).write_text("public policy fixture")
    (tmp_path / "private.txt").write_text("not a publication input")
    output = tmp_path / "site"
    build_site(tmp_path, output)
    for name in ("robots.txt", "CITATION.cff"):
        assert (output / name).read_text() == "public policy fixture"
    assert not (output / "private.txt").exists()
    html = (output / "studies/note.html").read_text()
    assert 'href="../COPYRIGHT.html"' in html
    assert 'href="../CITATION.cff"' in html
    assert '<meta name="author" content="Cazandra Aporbo">' in html
    assert '<meta name="robots"' not in html


def test_public_policy_symlinks_are_not_copied(tmp_path):
    from scripts.build_site import content_files
    private = tmp_path / "private.txt"
    private.write_text("not for publication")
    for name in ("robots.txt", "CITATION.cff"):
        (tmp_path / name).symlink_to(private)
    assert not any(path.name in {"robots.txt", "CITATION.cff"} for path in content_files(tmp_path))


@pytest.mark.parametrize("agent", ["GPTBot", "ClaudeBot", "Google-Extended", "CCBot"])
def test_ai_crawler_policy_excludes_only_this_project(agent):
    from urllib.robotparser import RobotFileParser
    policy = RobotFileParser()
    policy.parse((Path(__file__).parents[1] / "robots.txt").read_text().splitlines())
    assert not policy.can_fetch(agent, "https://cazzy-aporbo.github.io/Curious-Coder/index.html")
    assert not policy.can_fetch(agent, "https://cazzy-aporbo.github.io/Curious-Coder/studies/learning_signals.html")
    assert policy.can_fetch(agent, "https://cazzy-aporbo.github.io/Other-Project/index.html")


@pytest.mark.parametrize("agent", ["Googlebot", "bingbot", "OAI-SearchBot", "Claude-SearchBot"])
def test_crawler_policy_preserves_search_discovery(agent):
    from urllib.robotparser import RobotFileParser
    policy = RobotFileParser()
    policy.parse((Path(__file__).parents[1] / "robots.txt").read_text().splitlines())
    assert policy.can_fetch(agent, "https://cazzy-aporbo.github.io/Curious-Coder/index.html")


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
