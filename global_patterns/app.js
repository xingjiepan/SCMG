const PAGE_SIZE = 96;
const HF_DATA_BASE = "https://huggingface.co/datasets/xingjiepan/SCMG_data/resolve/main/data";
const IMAGE_DIR = `${HF_DATA_BASE}/global_gene_exp_plots_all`;
const UMAP_URL = `${HF_DATA_BASE}/global_cell_type_umap.png`;

const state = {
  plots: [],
  filtered: [],
  shown: PAGE_SIZE,
  query: "",
};

const els = {
  counter: document.querySelector("#counter"),
  search: document.querySelector("#searchInput"),
  clear: document.querySelector("#clearButton"),
  grid: document.querySelector("#resultsGrid"),
  title: document.querySelector("#resultsTitle"),
  meta: document.querySelector("#resultsMeta"),
  loadMore: document.querySelector("#loadMoreButton"),
  dialog: document.querySelector("#imageDialog"),
  dialogImage: document.querySelector("#dialogImage"),
  dialogCaption: document.querySelector("#dialogCaption"),
  closeDialog: document.querySelector("#closeDialogButton"),
  umapZoom: document.querySelector("#umapZoomButton"),
};

function imageUrl(file) {
  const prefix = file[0].toUpperCase();
  return `${IMAGE_DIR}/${prefix}/${encodeURIComponent(file)}`;
}

function normalize(value) {
  return value.trim().toLowerCase();
}

function rankPlot(plot, query) {
  const name = plot.nameLower;
  if (name === query) return 0;
  if (name.startsWith(query)) return 1;
  if (name.includes(query)) return 2;
  return 3;
}

function filterPlots() {
  const query = normalize(state.query);

  if (!query) {
    state.filtered = state.plots;
    return;
  }

  state.filtered = state.plots
    .filter((plot) => plot.nameLower.includes(query))
    .sort((a, b) => {
      const rankDiff = rankPlot(a, query) - rankPlot(b, query);
      return rankDiff || a.name.localeCompare(b.name);
    });
}

function updateUrl() {
  const url = new URL(window.location);
  if (state.query) {
    url.searchParams.set("q", state.query);
  } else {
    url.searchParams.delete("q");
  }
  history.replaceState(null, "", url);
}

function renderMeta() {
  const total = state.plots.length.toLocaleString();
  const matched = state.filtered.length.toLocaleString();
  const visible = Math.min(state.shown, state.filtered.length).toLocaleString();

  els.counter.textContent = `${total} plots`;
  els.title.textContent = state.query ? `Results for "${state.query}"` : "Plots";
  els.meta.textContent = state.filtered.length
    ? `${visible} of ${matched}`
    : "0 results";
}

function createCard(plot) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "plot-card";
  card.setAttribute("aria-label", `Open ${plot.name}`);

  const thumbWrap = document.createElement("span");
  thumbWrap.className = "thumb-wrap";

  const img = document.createElement("img");
  img.src = imageUrl(plot.file);
  img.alt = `${plot.name} expression plot`;
  img.loading = "lazy";

  const name = document.createElement("span");
  name.className = "plot-name";
  name.textContent = plot.name;
  name.title = plot.name;

  thumbWrap.append(img);
  card.append(thumbWrap, name);
  card.addEventListener("click", () => openDialog(plot));

  return card;
}

function renderGrid() {
  els.grid.replaceChildren();

  const visiblePlots = state.filtered.slice(0, state.shown);
  if (!visiblePlots.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No matching plots found.";
    els.grid.append(empty);
    els.loadMore.hidden = true;
    renderMeta();
    return;
  }

  const fragment = document.createDocumentFragment();
  visiblePlots.forEach((plot) => fragment.append(createCard(plot)));
  els.grid.append(fragment);

  els.loadMore.hidden = state.shown >= state.filtered.length;
  renderMeta();
}

function render() {
  filterPlots();
  renderGrid();
  updateUrl();
}

function openImageDialog({ src, alt, caption }) {
  els.dialogImage.src = src;
  els.dialogImage.alt = alt;
  els.dialogCaption.textContent = caption;

  if (typeof els.dialog.showModal === "function") {
    els.dialog.showModal();
  } else {
    window.open(src, "_blank", "noopener");
  }
}

function openDialog(plot) {
  openImageDialog({
    src: imageUrl(plot.file),
    alt: `${plot.name} expression plot`,
    caption: plot.name,
  });
}

function openUmapDialog() {
  openImageDialog({
    src: UMAP_URL,
    alt: "Global cell type UMAP",
    caption: "Global Cell Type UMAP",
  });
}

function closeDialog() {
  els.dialog.close();
  els.dialogImage.removeAttribute("src");
}

async function loadManifest() {
  const response = await fetch("manifest.json");
  if (!response.ok) {
    throw new Error(`Could not load manifest.json (${response.status})`);
  }
  const files = await response.json();
  state.plots = files.map((file) => {
    const name = file.replace(/\.png$/i, "");
    return {
      file,
      name,
      nameLower: name.toLowerCase(),
    };
  });
  state.filtered = state.plots;
}

function bindEvents() {
  els.umapZoom.querySelector("img").src = UMAP_URL;

  els.search.addEventListener("input", (event) => {
    state.query = event.target.value;
    state.shown = PAGE_SIZE;
    render();
  });

  els.clear.addEventListener("click", () => {
    state.query = "";
    els.search.value = "";
    els.search.focus();
    state.shown = PAGE_SIZE;
    render();
  });

  els.loadMore.addEventListener("click", () => {
    state.shown += PAGE_SIZE;
    renderGrid();
  });

  els.umapZoom.addEventListener("click", openUmapDialog);
  els.closeDialog.addEventListener("click", closeDialog);
  els.dialog.addEventListener("click", (event) => {
    if (event.target === els.dialog) closeDialog();
  });
}

async function init() {
  bindEvents();
  state.query = new URLSearchParams(window.location.search).get("q") || "";
  els.search.value = state.query;

  try {
    await loadManifest();
    render();
  } catch (error) {
    els.counter.textContent = "Manifest unavailable";
    els.grid.innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
}

init();
