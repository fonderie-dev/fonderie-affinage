"""Check internal links and fragment anchors in a built Hugo site.

Usage: python3 docs/tools/check_links.py docs/public /fonderie-<mot>/
Exits non-zero if any internal href or src points to a missing file or anchor.
"""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id":
                self.ids.add(value)
            elif name in ("href", "src") and not (
                tag == "link" and name == "href" and value.startswith("http")
            ):
                self.refs.append(value)


def main() -> int:
    root = Path(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else "/"
    pages: dict[Path, Collector] = {}
    for html in root.rglob("*.html"):
        c = Collector()
        c.feed(html.read_text(encoding="utf-8"))
        pages[html] = c

    def resolve(url_path: str) -> Path | None:
        if not url_path.startswith(base):
            return None
        rel = unquote(url_path[len(base) :])
        target = root / rel
        if url_path.endswith("/") or target.is_dir():
            target = target / "index.html"
        return target

    errors = []
    for html, c in pages.items():
        page_url = base + str(html.relative_to(root)).replace("index.html", "")
        for ref in c.refs:
            parsed = urlparse(ref)
            if parsed.scheme in ("http", "https", "mailto") or ref.startswith("//"):
                continue
            absolute = urlparse(urljoin(page_url, ref))
            target = resolve(absolute.path) if absolute.path else html
            if target is None or not target.exists():
                errors.append(f"{html.relative_to(root)}: missing {ref}")
                continue
            if absolute.fragment and target.suffix == ".html":
                ids = pages[target].ids if target in pages else set()
                if absolute.fragment not in ids:
                    errors.append(f"{html.relative_to(root)}: missing anchor {ref}")

    if not pages:
        errors.append(f"no HTML pages found under {root}")
    for e in errors:
        print(e)
    print(f"{len(pages)} pages checked, {len(errors)} broken internal links")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
