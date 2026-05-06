const PAGE_SIZE = 96;
const HF_DATA_BASE = "https://huggingface.co/datasets/xingjiepan/SCMG_data/resolve/main/data";
const UMAP_URL = `${HF_DATA_BASE}/global_cell_type_umap.png`;
const COLLECTIONS = {
  genes: {
    label: "Genes",
    itemLabel: "gene",
    manifest: "manifest.json",
    imageDir: `${HF_DATA_BASE}/global_gene_exp_plots_all`,
    searchLabel: "Search by gene name",
    searchNote: "",
    placeholder: "MYC, A1BG, PTGS2",
    title: "Gene plots",
    empty: "No matching gene plots found.",
    alt: "gene expression plot",
  },
  cellTypes: {
    label: "Cell types",
    itemLabel: "cell type",
    manifest: "cell_type_manifest.json",
    imageDir: `${HF_DATA_BASE}/global_cell_type_plots_all`,
    searchLabel: "Search by cell type name",
    searchNote: "Cell type names are taken directly from author annotations in the original published datasets. Discrepancy may come from naming convention differences and annotation errors.",
    placeholder: "macrophage, neuron, epithelial",
    title: "Cell type plots",
    empty: "No matching cell type plots found.",
    alt: "cell type global pattern plot",
  },
};

const state = {
  collection: "genes",
  collections: {},
  filtered: [],
  shown: PAGE_SIZE,
  query: "",
};

const els = {
  counter: document.querySelector("#counter"),
  search: document.querySelector("#searchInput"),
  searchLabel: document.querySelector("#searchLabel"),
  searchNote: document.querySelector("#searchNote"),
  clear: document.querySelector("#clearButton"),
  modeButtons: document.querySelectorAll(".mode-button"),
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

function activeCollection() {
  return COLLECTIONS[state.collection];
}

function imageUrl(file) {
  const prefix = file[0].toUpperCase();
  return `${activeCollection().imageDir}/${prefix}/${encodeURIComponent(file)}`;
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
  const plots = state.collections[state.collection] || [];

  if (!query) {
    state.filtered = plots;
    return;
  }

  state.filtered = plots
    .filter((plot) => plot.nameLower.includes(query))
    .sort((a, b) => {
      const rankDiff = rankPlot(a, query) - rankPlot(b, query);
      return rankDiff || a.name.localeCompare(b.name);
    });
}

function updateUrl() {
  const url = new URL(window.location);
  if (state.collection === "cellTypes") {
    url.searchParams.set("type", "cell-types");
  } else {
    url.searchParams.delete("type");
  }

  if (state.query) {
    url.searchParams.set("q", state.query);
  } else {
    url.searchParams.delete("q");
  }
  history.replaceState(null, "", url);
}

function renderMeta() {
  const collection = activeCollection();
  const total = (state.collections[state.collection] || []).length.toLocaleString();
  const matched = state.filtered.length.toLocaleString();
  const visible = Math.min(state.shown, state.filtered.length).toLocaleString();

  els.counter.textContent = `${total} ${collection.itemLabel} plots`;
  els.title.textContent = state.query ? `Results for "${state.query}"` : collection.title;
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
  img.alt = `${plot.name} ${activeCollection().alt}`;
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
    empty.textContent = activeCollection().empty;
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
    alt: `${plot.name} ${activeCollection().alt}`,
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

async function loadManifest(collectionKey = state.collection) {
  if (state.collections[collectionKey]) return;

  const collection = COLLECTIONS[collectionKey];
  const response = await fetch(collection.manifest);
  if (!response.ok) {
    throw new Error(`Could not load ${collection.manifest} (${response.status})`);
  }
  const files = await response.json();
  state.collections[collectionKey] = files.map((file) => {
    const name = file.replace(/\.png$/i, "");
    return {
      file,
      name,
      nameLower: name.toLowerCase(),
    };
  });
}

function updateCollectionControls() {
  const collection = activeCollection();
  document.title = `Global Pattern Browser - ${collection.label}`;
  els.searchLabel.textContent = collection.searchLabel;
  els.searchNote.textContent = collection.searchNote;
  els.searchNote.hidden = !collection.searchNote;
  els.search.placeholder = collection.placeholder;
  els.modeButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.collection));
  });
}

async function setCollection(collectionKey) {
  if (!COLLECTIONS[collectionKey] || collectionKey === state.collection) return;

  state.collection = collectionKey;
  state.query = "";
  state.shown = PAGE_SIZE;
  els.search.value = "";
  updateCollectionControls();
  try {
    await loadManifest();
    render();
  } catch (error) {
    els.counter.textContent = "Manifest unavailable";
    els.grid.innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
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

  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setCollection(button.dataset.mode));
  });

  els.umapZoom.addEventListener("click", openUmapDialog);
  els.closeDialog.addEventListener("click", closeDialog);
  els.dialog.addEventListener("click", (event) => {
    if (event.target === els.dialog) closeDialog();
  });
}

async function init() {
  bindEvents();
  const params = new URLSearchParams(window.location.search);
  state.collection = params.get("type") === "cell-types" ? "cellTypes" : "genes";
  state.query = params.get("q") || "";
  els.search.value = state.query;
  updateCollectionControls();

  try {
    await loadManifest();
    render();
  } catch (error) {
    els.counter.textContent = "Manifest unavailable";
    els.grid.innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
}

init();
