// shiso hub: copy buttons on every page; search and filters on the catalog.
// Works without this file too: every card is in the HTML, search only hides
// and reorders them.
(function () {
  "use strict";

  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-copy]");
    if (!btn) return;
    e.preventDefault();
    var text = btn.getAttribute("data-copy");
    var done = function () {
      var label = btn.textContent;
      btn.textContent = "Copied";
      btn.classList.add("done");
      setTimeout(function () { btn.textContent = label; btn.classList.remove("done"); }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () {});
    }
  });

  var docsEl = document.getElementById("search-docs");
  var input = document.getElementById("q");
  var grid = document.getElementById("models");
  if (!docsEl || !input || !grid || typeof MiniSearch === "undefined") return;

  var docs = JSON.parse(docsEl.textContent);
  var index = new MiniSearch({
    fields: ["name", "aliases", "description", "tags", "family", "tasks", "inputs", "backends", "quants", "publisher"],
    searchOptions: {
      boost: { name: 4, aliases: 3, tags: 2, family: 1.5 },
      prefix: true,
      // A fraction of a short word allows no edit at all, so a three-letter
      // typo ("qwn") would find nothing; one edit until words get longer.
      fuzzy: function (term) { return term.length < 3 ? 0 : term.length < 6 ? 1 : 0.2; },
      combineWith: "AND"
    }
  });
  index.addAll(docs);

  var cards = new Map();
  grid.querySelectorAll("[data-model]").forEach(function (el) { cards.set(el.dataset.model, el); });
  var order = docs.map(function (d) { return d.id; });
  var count = document.getElementById("count");
  var empty = document.getElementById("empty");
  var active = { task: new Set(), backend: new Set(), input: new Set() };

  function passes(el) {
    return Object.keys(active).every(function (k) {
      if (active[k].size === 0) return true;
      var have = (el.dataset[k] || "").split(" ");
      // Within one group a card needs any of the chosen values.
      return have.some(function (v) { return active[k].has(v); });
    });
  }

  function apply() {
    var q = input.value.trim();
    var ids = q ? index.search(q).map(function (r) { return r.id; }) : order;
    var shown = new Set();
    ids.forEach(function (id) {
      var el = cards.get(id);
      if (el && passes(el)) { shown.add(id); grid.appendChild(el); }
    });
    cards.forEach(function (el, id) { el.hidden = !shown.has(id); });
    count.textContent = shown.size + (shown.size === 1 ? " model" : " models");
    empty.hidden = shown.size > 0;
    var url = new URL(location.href);
    if (q) url.searchParams.set("q", q); else url.searchParams.delete("q");
    history.replaceState(null, "", url);
  }

  document.querySelectorAll(".chip[data-filter]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var set = active[chip.dataset.filter];
      var on = chip.getAttribute("aria-pressed") !== "true";
      chip.setAttribute("aria-pressed", on ? "true" : "false");
      if (on) set.add(chip.dataset.value); else set.delete(chip.dataset.value);
      apply();
    });
  });

  input.addEventListener("input", apply);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var first = grid.querySelector("[data-model]:not([hidden])");
      if (first) location.href = first.getAttribute("href");
    } else if (e.key === "Escape") {
      input.value = "";
      apply();
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
  });

  var initial = new URL(location.href).searchParams.get("q");
  if (initial) { input.value = initial; apply(); }
})();
