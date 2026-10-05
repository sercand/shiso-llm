//! Model cards: each model's README, fetched once per pinned revision and
//! rendered into its hub page.
//!
//! The README is read at the revision a recipe pins and cached in the
//! registry repository under `cards/`, so a rebuild reads the same bytes
//! without the network and a card can only change when a recipe's revision
//! does. A revision with no README leaves an empty `.missing` marker, so it is
//! not asked for again.
//!
//! The rendered HTML is third-party content served under the registry's
//! domain, so it is sanitized before it reaches a page; nothing in a README can
//! add a script, a style or an event handler to the site.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use shiso_hub::recipe::{Recipe, Role};

/// Where a model's card comes from: a Hugging Face repository at a commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CardRef {
    /// `owner/name`.
    pub(crate) repo: String,
    /// 40-hex commit.
    pub(crate) revision: String,
}

impl CardRef {
    /// The card a recipe names: `info.card` when set, else the first
    /// variant's `base` component, else its first Hugging Face component (an
    /// ASR or image model has no `base`).
    ///
    /// `None` when there is no such repository, or its name could not be a
    /// cache path.
    pub(crate) fn of(recipe: &Recipe) -> Option<Self> {
        if let Some(card) = recipe.info.as_ref().and_then(|i| i.card.as_ref()) {
            return Some(Self {
                repo: card.repo.clone(),
                revision: card.revision.to_string(),
            });
        }
        let components = &recipe.variants.first()?.components;
        let comp = components
            .get(&Role::Base)
            .or_else(|| components.values().find(|c| c.provider == "huggingface"))?;
        let safe = comp.repo.split('/').count() == 2
            && comp
                .repo
                .split('/')
                .all(|seg| !seg.is_empty() && seg != "." && seg != "..");
        (comp.provider == "huggingface" && safe).then(|| Self {
            repo: comp.repo.clone(),
            revision: comp.revision.to_string(),
        })
    }

    fn cache_stem(&self, repo_path: &Path) -> PathBuf {
        // `owner/name` maps onto two directory levels. `of` and the recipe
        // validation both refuse an empty, `.` or `..` half, so the path
        // stays under `cards/`.
        repo_path
            .join("cards")
            .join(&self.repo)
            .join(&self.revision)
    }

    fn markdown_path(&self, repo_path: &Path) -> PathBuf {
        self.cache_stem(repo_path).with_extension("md")
    }

    fn missing_path(&self, repo_path: &Path) -> PathBuf {
        self.cache_stem(repo_path).with_extension("missing")
    }

    /// The README's page on Hugging Face.
    pub(crate) fn page_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/blob/{}/README.md",
            self.repo, self.revision
        )
    }

    fn raw_base(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/",
            self.repo, self.revision
        )
    }
}

/// What a cache refresh did.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct RefreshStats {
    /// READMEs downloaded.
    pub(crate) fetched: usize,
    /// Revisions found to have no README.
    pub(crate) missing: usize,
    /// Cards already cached.
    pub(crate) cached: usize,
}

/// Downloads every card the recipes name that is not cached yet.
///
/// # Errors
/// When a download fails for any reason other than the file not existing: a
/// build that silently dropped a card it could not reach would publish a page
/// that differs from the last build for no reason the maintainer can see.
pub(crate) async fn refresh_cache(
    repo_path: &Path,
    recipes: &[Recipe],
) -> anyhow::Result<RefreshStats> {
    let client = reqwest::Client::new();
    let mut stats = RefreshStats::default();
    for card in recipes.iter().filter_map(CardRef::of) {
        let md = card.markdown_path(repo_path);
        let missing = card.missing_path(repo_path);
        if md.is_file() || missing.is_file() {
            stats.cached += 1;
            continue;
        }
        let url = format!("{}README.md", card.raw_base());
        let resp = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("fetching {url}"))?;
        let dest = if resp.status() == reqwest::StatusCode::NOT_FOUND {
            stats.missing += 1;
            (missing, Vec::new())
        } else if resp.status().is_success() {
            stats.fetched += 1;
            let bytes = resp
                .bytes()
                .await
                .with_context(|| format!("reading {url}"))?;
            (md, bytes.to_vec())
        } else {
            bail!(
                "fetching {url} answered {}; gated repositories need --offline and a hand-placed card",
                resp.status()
            );
        };
        if let Some(parent) = dest.0.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&dest.0, &dest.1).with_context(|| format!("writing {}", dest.0.display()))?;
    }
    Ok(stats)
}

/// A card ready for a page.
#[derive(Debug, Clone)]
pub(crate) struct RenderedCard {
    /// Sanitized HTML.
    pub(crate) html: String,
    /// The card's Hugging Face page.
    pub(crate) source_url: String,
    /// `license:` from the README's front matter, when it has one.
    pub(crate) license: Option<String>,
}

/// Reads a cached card and renders it.
///
/// `Ok(None)` means the card is not cached, or its revision has no README.
///
/// # Errors
/// When the cached file exists but cannot be read or is not UTF-8.
pub(crate) fn load(repo_path: &Path, card: &CardRef) -> anyhow::Result<Option<RenderedCard>> {
    let path = card.markdown_path(repo_path);
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let (front, body) = split_front_matter(&text);
    Ok(Some(RenderedCard {
        html: render_markdown(body, &card.raw_base()),
        source_url: card.page_url(),
        license: front.and_then(front_matter_license),
    }))
}

/// Splits a leading `---` YAML block off a README.
fn split_front_matter(text: &str) -> (Option<&str>, &str) {
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (None, text);
    };
    for marker in ["\n---\n", "\n---\r\n"] {
        if let Some(end) = rest.find(marker) {
            return (Some(&rest[..end]), &rest[end + marker.len()..]);
        }
    }
    // An unterminated block is not front matter; render it as written.
    (None, text)
}

/// The top-level `license:` value of a front-matter block.
///
/// Read by hand instead of through a YAML parser: this one scalar is all the
/// site takes from the block, and a value that is anything but a plain
/// scalar is skipped rather than guessed at.
fn front_matter_license(front: &str) -> Option<String> {
    front.lines().find_map(|line| {
        let value = line.strip_prefix("license:")?.trim();
        let value = value.trim_matches(|c| c == '"' || c == '\'');
        (!value.is_empty() && !value.starts_with(['[', '{', '|', '>'])).then(|| value.to_string())
    })
}

/// `CommonMark` (with tables, footnotes, strikethrough and task lists) to
/// sanitized HTML. Relative links and images resolve against the README's own
/// revision.
pub(crate) fn render_markdown(markdown: &str, base: &str) -> String {
    use pulldown_cmark::{Options, Parser, html};
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS;
    let mut raw = String::with_capacity(markdown.len() * 2);
    html::push_html(&mut raw, Parser::new_ext(markdown, options));

    let mut builder = ammonia::Builder::default();
    builder.link_rel(Some("noopener noreferrer nofollow"));
    if let Ok(url) = url::Url::parse(base) {
        builder.url_relative(ammonia::UrlRelative::RewriteWithBase(url));
    } else {
        builder.url_relative(ammonia::UrlRelative::Deny);
    }
    builder.clean(&raw).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str =
        "https://huggingface.co/o/n/resolve/0123456789abcdef0123456789abcdef01234567/";

    #[test]
    fn a_readme_cannot_bring_script_into_the_page() {
        let html = render_markdown(
            "# Title\n\n<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n\n[x](javascript:void) <a href=\"javascript:void\">y</a>\n",
            BASE,
        );
        assert!(!html.contains("<script"), "{html}");
        assert!(!html.contains("onerror"), "{html}");
        assert!(!html.contains("href=\"javascript"), "{html}");
        assert!(html.contains("<h1>Title</h1>"), "{html}");
    }

    #[test]
    fn relative_links_resolve_against_the_pinned_revision() {
        let html = render_markdown(
            "![chart](assets/chart.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
            BASE,
        );
        assert!(html.contains(&format!("{BASE}assets/chart.png")), "{html}");
        assert!(html.contains("<table>"), "{html}");
    }

    #[test]
    fn front_matter_is_split_off_and_its_license_read() {
        let (front, body) =
            split_front_matter("---\nlicense: apache-2.0\ntags:\n- x\n---\n# Card\n");
        assert_eq!(body, "# Card\n");
        assert_eq!(
            front.and_then(front_matter_license).as_deref(),
            Some("apache-2.0")
        );
        assert_eq!(split_front_matter("# No front\n").0, None);
        assert_eq!(front_matter_license("license: [a, b]"), None);
    }
}
