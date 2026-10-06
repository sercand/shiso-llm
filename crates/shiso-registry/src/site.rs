//! The hub website: a static catalog generated from the compiled recipes.
//!
//! Every build rewrites the pages from the same inputs as the manifests, so
//! the site can never describe a model differently from what `shiso pull`
//! installs. The layout, relative to the site root:
//!
//! ```text
//! index.html                 catalog with offline search
//! 404.html                   served by the host for any unknown path
//! models/<name>/index.html   one page per model
//! search-index.json          the documents the search indexes
//! assets/                    stylesheet, script, search library
//! ```
//!
//! Search runs in the browser over documents embedded in the catalog page,
//! so it needs no server and works from a saved copy. Pages link relative to
//! their own depth (`root` in the templates), so the same output serves at a
//! host's root and under the editor's `/hub/` preview. Only `404.html` links
//! root-absolute, because the host serves it at whatever depth the missing URL
//! had.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::Context;
use minijinja::{Environment, context};
use serde::Serialize;
use shiso_hub::recipe::{QualityScope, Recipe, Variant};

use crate::bench::{BenchKind, BenchResult, PublishedRecord};
use crate::card::RenderedCard;

const BASE: &str = include_str!("site/base.html");
const HOME: &str = include_str!("site/home.html");
const MODEL: &str = include_str!("site/model.html");
const NOT_FOUND: &str = include_str!("site/404.html");
const CSS: &str = include_str!("site/site.css");
const JS: &str = include_str!("site/site.js");
const SEARCH_LIB: &str = include_str!("site/minisearch.js");

/// Everything one model's page is built from.
pub(crate) struct ModelInput<'a> {
    /// The compiled recipe.
    pub(crate) recipe: &'a Recipe,
    /// Its rendered card, when it has one.
    pub(crate) card: Option<RenderedCard>,
    /// Its throughput records.
    pub(crate) benchmarks: &'a [PublishedRecord],
}

/// Writes the whole site under `root`, returning the files written relative
/// to it.
///
/// # Errors
/// When a template fails to render or a file cannot be written.
pub(crate) fn write(root: &Path, models: &[ModelInput<'_>]) -> anyhow::Result<Vec<String>> {
    let env = environment()?;
    let views: Vec<ModelView> = models.iter().map(ModelView::new).collect();
    let docs: Vec<SearchDoc> = views.iter().map(SearchDoc::new).collect();
    let docs_json = serde_json::to_string(&docs)?;
    let facets = Facets::new(&views);

    let mut written = Vec::new();
    let mut put = |rel: String, body: &[u8]| -> anyhow::Result<()> {
        crate::editor::atomic_write(&root.join(&rel), body)?;
        written.push(rel);
        Ok(())
    };

    let home = env.get_template("home.html")?.render(context! {
        root => "",
        models => &views,
        facets => &facets,
        // `</` would end the embedding script element early; JSON allows the
        // escaped spelling.
        search_docs => docs_json.replace("</", "<\\/"),
    })?;
    put("index.html".to_string(), home.as_bytes())?;
    put(
        "404.html".to_string(),
        env.get_template("404.html")?
            .render(context! { root => "/" })?
            .as_bytes(),
    )?;
    for view in &views {
        let page = env
            .get_template("model.html")?
            .render(context! { root => "../../", m => view })
            .with_context(|| format!("rendering the page of {}", view.name))?;
        put(format!("models/{}/index.html", view.name), page.as_bytes())?;
    }
    put("search-index.json".to_string(), docs_json.as_bytes())?;
    put("assets/site.css".to_string(), CSS.as_bytes())?;
    put("assets/site.js".to_string(), JS.as_bytes())?;
    put("assets/minisearch.js".to_string(), SEARCH_LIB.as_bytes())?;
    Ok(written)
}

fn environment() -> anyhow::Result<Environment<'static>> {
    let mut env = Environment::new();
    // The default HTML escape also rewrites `/` as `&#x2f;`, which is valid
    // but turns every URL in the page source unreadable; these five are the
    // characters that can end an element or an attribute.
    env.set_formatter(|out, state, value| match value.as_str() {
        Some(text) if !value.is_safe() && state.auto_escape() != minijinja::AutoEscape::None => {
            let mut escaped = String::with_capacity(text.len());
            for c in text.chars() {
                match c {
                    '&' => escaped.push_str("&amp;"),
                    '<' => escaped.push_str("&lt;"),
                    '>' => escaped.push_str("&gt;"),
                    '"' => escaped.push_str("&quot;"),
                    '\'' => escaped.push_str("&#39;"),
                    c => escaped.push(c),
                }
            }
            out.write_str(&escaped).map_err(minijinja::Error::from)
        }
        _ => minijinja::escape_formatter(out, state, value),
    });
    env.add_template("base.html", BASE)?;
    env.add_template("home.html", HOME)?;
    env.add_template("model.html", MODEL)?;
    env.add_template("404.html", NOT_FOUND)?;
    Ok(env)
}

/// Binary gibibytes, the unit RAM is sold in: `34359738368` is `32 GiB`.
fn gib(bytes: u64) -> String {
    #[expect(clippy::cast_precision_loss, reason = "a display value")]
    let g = bytes as f64 / f64::from(1u32 << 30);
    if g >= 100.0 || (g - g.round()).abs() < 0.05 {
        format!("{g:.0} GiB")
    } else {
        format!("{g:.1} GiB")
    }
}

/// `27000000000` is `27B`, `4500000000` is `4.5B`, `600000000` is `600M`.
fn count_label(n: u64) -> String {
    #[expect(clippy::cast_precision_loss, reason = "a display value")]
    let x = n as f64;
    let (v, unit) = if x >= 1e12 {
        (x / 1e12, "T")
    } else if x >= 1e9 {
        (x / 1e9, "B")
    } else if x >= 1e6 {
        (x / 1e6, "M")
    } else {
        return n.to_string();
    };
    let s = format!("{v:.1}");
    format!("{}{unit}", s.strip_suffix(".0").unwrap_or(&s))
}

/// Context lengths are powers of two in practice: `262144` is `256K`.
fn context_label(n: u64) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}K", n / 1024)
    } else {
        n.to_string()
    }
}

/// The date part of an RFC 3339 timestamp.
fn date(ts: Option<&str>) -> String {
    ts.and_then(|t| t.get(..10)).unwrap_or("").to_string()
}

fn label(s: &str) -> String {
    let s = s.replace('_', " ");
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

fn short(s: &str, n: usize) -> String {
    s.get(..n).unwrap_or(s).to_string()
}

#[derive(Serialize)]
struct FileView {
    name: String,
    size: String,
    sha: String,
    sha_short: String,
    binds: Option<String>,
}

#[derive(Serialize)]
struct ComponentView {
    role: String,
    /// `owner/name@rev10` for Hugging Face, `hosted on the hub` for the hub.
    label: String,
    url: String,
    entry: Option<String>,
    files: Vec<FileView>,
}

#[derive(Serialize)]
struct VariantView {
    key: String,
    quant: String,
    backends: Vec<&'static str>,
    min_ram: String,
    recommended_ram: Option<String>,
    disk: String,
    requires: Vec<String>,
    pull: String,
    components: Vec<ComponentView>,
    server_defaults: Vec<String>,
}

impl VariantView {
    fn new(model: &str, first: bool, v: &Variant) -> Self {
        let mut requires = Vec::new();
        if let Some(r) = &v.requires {
            if let Some(sm) = r.min_cuda_sm {
                requires.push(format!("CUDA sm{sm}+"));
            }
            if let Some(f) = r.min_metal_family {
                let chip = match f {
                    7 => "M1".to_string(),
                    8 => "M2".to_string(),
                    9 => "M3".to_string(),
                    10 => "M5".to_string(),
                    other => format!("Apple GPU family {other}"),
                };
                requires.push(format!("{chip} or newer"));
            }
        }
        if let Some(vram) = v.requirements.min_vram_bytes {
            requires.push(format!("{} GPU memory", gib(vram)));
        }
        let server_defaults = v
            .server_defaults
            .as_ref()
            .and_then(|d| serde_json::to_value(d).ok())
            .and_then(|d| d.as_object().cloned())
            .map(|o| o.iter().map(|(k, val)| format!("{k} = {val}")).collect())
            .unwrap_or_default();
        Self {
            key: v.key.to_string(),
            quant: v.quant.clone(),
            backends: v.backends.iter().map(|b| b.as_str()).collect(),
            min_ram: gib(v.requirements.min_ram_bytes),
            recommended_ram: v.requirements.recommended_ram_bytes.map(gib),
            disk: gib(v.requirements.disk_bytes),
            requires,
            // The first variant is what a bare pull picks on a host it fits.
            pull: if first {
                format!("shiso pull {model}")
            } else {
                format!("shiso pull {model}:{}", v.key)
            },
            components: v
                .components
                .iter()
                .map(|(role, c)| {
                    let (label, url) = match c.hf_source() {
                        Some((repo, rev)) => (
                            format!("{repo}@{}", short(rev.as_str(), 10)),
                            format!("https://huggingface.co/{repo}/tree/{rev}"),
                        ),
                        None => (
                            "hosted on the hub".to_string(),
                            // Relative to the model page at models/<name>/.
                            format!(
                                "../../registry/v1/models/{model}/{}",
                                c.entry
                                    .as_deref()
                                    .or(c.files.first().map(|f| f.name.as_str()))
                                    .unwrap_or("")
                            ),
                        ),
                    };
                    ComponentView {
                        role: role.to_string(),
                        label,
                        url,
                        entry: c.entry.clone(),
                        files: c
                            .files
                            .iter()
                            .map(|f| FileView {
                                name: f.name.clone(),
                                size: gib(f.size),
                                sha: f.sha256.to_string(),
                                sha_short: short(f.sha256.as_str(), 12),
                                binds: f.binds.map(|b| b.to_string()),
                            })
                            .collect(),
                    }
                })
                .collect(),
            server_defaults,
        }
    }
}

/// One record's rows, in its kind's columns.
#[derive(Serialize)]
struct BenchView {
    device: String,
    chip: String,
    backend: String,
    variant: String,
    ram: String,
    os: String,
    shiso_version: String,
    measured: String,
    runs: u32,
    context: Option<String>,
    stale: bool,
    notes: Option<String>,
    rows: Vec<Vec<String>>,
}

/// The records of one kind, which share a table.
#[derive(Serialize)]
struct BenchGroup {
    title: &'static str,
    hint: &'static str,
    columns: &'static [&'static str],
    /// The column a reader compares first, set in bold.
    lead: usize,
    records: Vec<BenchView>,
}

/// Each kind's table: title, explanation, columns and the lead column.
const fn layout(kind: BenchKind) -> (&'static str, &'static str, &'static [&'static str], usize) {
    match kind {
        BenchKind::Chat => (
            "Chat",
            "Batch 1, recipe defaults. Prefill comes from the time to first token; decode is tokens a second after it.",
            &["Prompt", "Prefill tok/s", "Decode tok/s", "TTFT ms"],
            2,
        ),
        BenchKind::Decision => (
            "Decisions",
            "Wall time of one /v1/decisions request with one yes/no question, batch 1.",
            &["Prompt", "Latency ms", "Prompt tok/s"],
            1,
        ),
        BenchKind::Image => (
            "Images",
            "Wall time of one single-image request at the server's step count.",
            &["Size", "Steps", "s / image"],
            2,
        ),
        BenchKind::Transcription => (
            "Transcription",
            "One streamed /v1/audio/transcriptions request. × real time is audio seconds per wall second.",
            &["Clip", "Audio s", "× real time", "First text ms"],
            2,
        ),
    }
}

fn result_row(r: &BenchResult) -> Vec<String> {
    match r {
        BenchResult::Chat(x) => vec![
            x.prompt_tokens.to_string(),
            format!("{:.0}", x.prefill_tokens_per_second),
            format!("{:.1}", x.decode_tokens_per_second),
            format!("{:.0}", x.time_to_first_token_ms),
        ],
        BenchResult::Decision(x) => vec![
            x.prompt_tokens.to_string(),
            format!("{:.0}", x.latency_ms),
            format!("{:.0}", x.prompt_tokens_per_second),
        ],
        BenchResult::Image(x) => vec![
            format!("{}×{}", x.width, x.height),
            x.steps.to_string(),
            format!("{:.2}", x.seconds_per_image),
        ],
        BenchResult::Transcription(x) => vec![
            x.clip.clone(),
            format!("{:.1}", x.audio_seconds),
            format!("{:.1}", x.realtime_multiple),
            format!("{:.0}", x.first_text_ms),
        ],
    }
}

/// The records grouped by kind, in the kinds' declared order.
fn bench_groups(records: &[PublishedRecord]) -> Vec<BenchGroup> {
    let mut by_kind: BTreeMap<BenchKind, Vec<BenchView>> = BTreeMap::new();
    for p in records {
        if let Some(first) = p.record.results.first() {
            by_kind
                .entry(first.kind())
                .or_default()
                .push(BenchView::new(p));
        }
    }
    by_kind
        .into_iter()
        .map(|(kind, records)| {
            let (title, hint, columns, lead) = layout(kind);
            BenchGroup {
                title,
                hint,
                columns,
                lead,
                records,
            }
        })
        .collect()
}

#[derive(Serialize)]
struct QualityView {
    source: String,
    metric: String,
    value: String,
    scope: String,
    as_of: String,
    url: String,
    subject: Option<String>,
}

impl BenchView {
    fn new(p: &PublishedRecord) -> Self {
        let b = &p.record;
        Self {
            device: b.device.to_string(),
            chip: b.chip.clone(),
            backend: b.backend.to_string(),
            variant: b.variant.to_string(),
            ram: gib(b.ram_bytes),
            os: b.os.clone(),
            shiso_version: b.shiso_version.clone(),
            measured: date(Some(&b.measured_at)),
            runs: b.runs,
            context: b.context_length.map(context_label),
            stale: p.stale,
            notes: b.notes.clone(),
            rows: b.results.iter().map(result_row).collect(),
        }
    }
}

/// The catalog card's one number: the best result on current weights, for
/// the kind the model was measured as (decode rate, decision latency, image
/// time, real-time multiple). Higher is better for rates, lower for times.
fn headline(records: &[PublishedRecord]) -> Option<String> {
    let mut best: Option<(f64, String)> = None;
    for p in records.iter().filter(|p| !p.stale) {
        let device = &p.record.device;
        for r in &p.record.results {
            let (score, text) = match r {
                BenchResult::Chat(x) => (
                    x.decode_tokens_per_second,
                    format!("{:.0} tok/s on {device}", x.decode_tokens_per_second),
                ),
                BenchResult::Decision(x) => (
                    -x.latency_ms,
                    format!("{:.0} ms/decision on {device}", x.latency_ms),
                ),
                BenchResult::Image(x) => (
                    -x.seconds_per_image,
                    format!("{:.1} s/image on {device}", x.seconds_per_image),
                ),
                BenchResult::Transcription(x) => (
                    x.realtime_multiple,
                    format!("{:.0}× real time on {device}", x.realtime_multiple),
                ),
            };
            if best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, text));
            }
        }
    }
    best.map(|(_, text)| text)
}

impl QualityView {
    fn new(q: &shiso_hub::recipe::QualityScore) -> Self {
        Self {
            source: q.source.clone(),
            metric: q.metric.clone(),
            value: match q.max {
                Some(max) => format!("{} / {}", trim_num(q.value), trim_num(max)),
                None => trim_num(q.value),
            },
            scope: match (q.scope, &q.variant) {
                (QualityScope::Variant, Some(k)) => format!("variant {k}"),
                _ => "base model (unquantized)".to_string(),
            },
            as_of: q.as_of.clone(),
            url: q.url.clone(),
            subject: q.subject.clone(),
        }
    }
}

#[derive(Serialize)]
struct ModelView {
    name: String,
    aliases: Vec<String>,
    description: String,
    family: String,
    tags: Vec<String>,
    tasks: Vec<String>,
    task_keys: Vec<&'static str>,
    inputs: Vec<&'static str>,
    outputs: Vec<&'static str>,
    features: Vec<&'static str>,
    reasoning_efforts: Vec<String>,
    updated: String,
    published: String,
    publisher: Option<String>,
    license: Option<String>,
    homepage: Option<String>,
    parameters: Option<String>,
    context: Option<String>,
    backends: Vec<&'static str>,
    quants: Vec<String>,
    min_ram: String,
    min_disk: String,
    variants: Vec<VariantView>,
    bench_groups: Vec<BenchGroup>,
    headline: Option<String>,
    quality: Vec<QualityView>,
    card_html: Option<String>,
    card_source: Option<String>,
    manifest: String,
    page: String,
}

impl ModelView {
    fn new(input: &ModelInput<'_>) -> Self {
        let r = input.recipe;
        let name = r.name.to_string();
        let info = r.info.clone().unwrap_or_default();
        let backends: BTreeSet<&'static str> = r
            .variants
            .iter()
            .flat_map(|v| v.backends.iter().map(|b| b.as_str()))
            .collect();
        let mut features = Vec::new();
        if r.features.tools {
            features.push("Tool calls");
        }
        if r.features.reasoning {
            features.push("Reasoning");
        }
        if r.features.structured_output {
            features.push("Structured output");
        }
        let parameters = info.parameters.map(|p| match info.active_parameters {
            Some(a) if a < p => format!("{} total · {} active", count_label(p), count_label(a)),
            _ => count_label(p),
        });
        let bench_groups = bench_groups(input.benchmarks);
        let headline = headline(input.benchmarks);
        let quality = r.quality.iter().map(QualityView::new).collect();
        let license = info
            .license
            .clone()
            .or_else(|| input.card.as_ref().and_then(|c| c.license.clone()));
        Self {
            aliases: r.aliases.iter().map(ToString::to_string).collect(),
            description: r.description.clone(),
            family: r.family.clone(),
            tags: r.tags.clone(),
            tasks: r.tasks.iter().map(|t| label(t.as_str())).collect(),
            task_keys: r.tasks.iter().map(|t| t.as_str()).collect(),
            inputs: r.modalities.inputs.iter().map(|m| m.as_str()).collect(),
            outputs: r.modalities.outputs.iter().map(|m| m.as_str()).collect(),
            features,
            reasoning_efforts: r.features.reasoning_efforts.clone(),
            updated: date(r.updated_at.as_deref()),
            published: date(r.published_at.as_deref()),
            publisher: info.publisher.clone(),
            license,
            homepage: info.homepage.clone(),
            parameters,
            context: info.context_length.map(context_label),
            backends: backends.into_iter().collect(),
            quants: r.variants.iter().map(|v| v.quant.clone()).collect(),
            min_ram: r
                .variants
                .iter()
                .map(|v| v.requirements.min_ram_bytes)
                .min()
                .map(gib)
                .unwrap_or_default(),
            min_disk: r
                .variants
                .iter()
                .map(|v| v.requirements.disk_bytes)
                .min()
                .map(gib)
                .unwrap_or_default(),
            variants: r
                .variants
                .iter()
                .enumerate()
                .map(|(i, v)| VariantView::new(&name, i == 0, v))
                .collect(),
            bench_groups,
            headline,
            quality,
            card_html: input.card.as_ref().map(|c| c.html.clone()),
            card_source: input.card.as_ref().map(|c| c.source_url.clone()),
            manifest: format!("registry/v1/{}", crate::editor::manifest_rel(&r.name)),
            page: format!("models/{name}/"),
            name,
        }
    }
}

fn trim_num(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// One searchable document per model.
#[derive(Serialize)]
struct SearchDoc {
    id: String,
    name: String,
    aliases: String,
    description: String,
    tags: String,
    family: String,
    tasks: String,
    inputs: String,
    backends: String,
    quants: String,
    publisher: String,
    url: String,
}

impl SearchDoc {
    fn new(v: &ModelView) -> Self {
        Self {
            id: v.name.clone(),
            name: v.name.clone(),
            aliases: v.aliases.join(" "),
            description: v.description.clone(),
            tags: v.tags.join(" "),
            family: v.family.clone(),
            tasks: v.tasks.join(" "),
            inputs: v.inputs.join(" "),
            backends: v.backends.join(" "),
            quants: v.quants.join(" "),
            publisher: v.publisher.clone().unwrap_or_default(),
            url: v.page.clone(),
        }
    }
}

/// The filter chips: every value some model has, with how many have it.
#[derive(Serialize)]
struct Facets {
    tasks: Vec<(String, String, usize)>,
    backends: Vec<(String, usize)>,
    inputs: Vec<(String, usize)>,
}

impl Facets {
    fn new(views: &[ModelView]) -> Self {
        let mut tasks: BTreeMap<String, (String, usize)> = BTreeMap::new();
        let mut backends: BTreeMap<String, usize> = BTreeMap::new();
        let mut inputs: BTreeMap<String, usize> = BTreeMap::new();
        for v in views {
            for (key, lab) in v.task_keys.iter().zip(&v.tasks) {
                tasks
                    .entry((*key).to_string())
                    .or_insert((lab.clone(), 0))
                    .1 += 1;
            }
            for b in &v.backends {
                *backends.entry((*b).to_string()).or_default() += 1;
            }
            for i in &v.inputs {
                *inputs.entry((*i).to_string()).or_default() += 1;
            }
        }
        Self {
            tasks: tasks.into_iter().map(|(k, (l, n))| (k, l, n)).collect(),
            backends: backends.into_iter().collect(),
            inputs: inputs.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_read_the_way_people_write_them() {
        assert_eq!(gib(34_359_738_368), "32 GiB");
        assert_eq!(gib(19_252_222_358), "17.9 GiB");
        assert_eq!(count_label(27_000_000_000), "27B");
        assert_eq!(count_label(4_500_000_000), "4.5B");
        assert_eq!(count_label(600_000_000), "600M");
        assert_eq!(context_label(262_144), "256K");
        assert_eq!(trim_num(41.0), "41");
        assert_eq!(trim_num(41.25), "41.25");
        assert_eq!(label("image_generation"), "Image generation");
    }
}
