# Project site

Source of this atelier's project site on GitHub Pages: a static Hugo site with
its own templates and no theme module. It is an outline of the repository;
every section that describes functionality is stamped *coming soon* until the
code exists.

## Build locally

Requires Hugo extended ≥ 0.146 (CI pins 0.169.0) and Python 3. From the
repository root:

```sh
hugo server --source docs                          # live preview
hugo --source docs --gc --minify --panicOnWarning  # production build into docs/public/
python3 docs/tools/check_links.py docs/public /<repository>/
```

Pushes to `main` that touch `docs/` build, check internal links and deploy;
pull requests build and check without deploying
(`.github/workflows/pages.yaml`).

## Layout

```
hugo.toml             base URL, site parameters, menu
content/              guide, reference and journal pages; Impressum and Datenschutz
data/repo.yaml        this atelier's status line, scope and milestones
data/ateliers.yaml    the three ateliers in chain order (same file in every atelier)
layouts/              templates (home, page, 404 and partials)
assets/css/main.css   the Fonderie stylesheet plus a block for project sites
static/images/        mark, favicons and the social preview image
tools/check_links.py  internal link and anchor check for the built site
```

A page's planned sections come from `outline` in its front matter; each one is
rendered with a *coming soon* stamp. When a section describes code that
exists, move it into the page body and remove it from `outline`.

The stylesheet, mark, `baseof.html`, the `ingot` and `stamp` partials, the
link checker, the Impressum and the Datenschutz notice are copied from
[fonderie-dev/website](https://github.com/fonderie-dev/website); keep them in
step with it. `data/ateliers.yaml` and the layouts are the same in all three
ateliers.
