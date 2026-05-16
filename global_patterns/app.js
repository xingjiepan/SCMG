const PAGE_SIZE = 96;
const LUCKY_SIZE = 48;
const HF_DATA_BASE = "https://huggingface.co/datasets/xingjiepan/SCMG_data/resolve/main/data";
const UMAP_URL = `${HF_DATA_BASE}/global_cell_type_umap.png`;
const MARKER_GENES_URL = "global_marker_genes.json";
const GENE_TOP_CELL_TYPES_URL = "gene_top_cell_types.json";
const GENE_ANNOTATIONS_URL = "gene_annotations.json";
const DEFAULT_HIDDEN_GENE_NAMES = new Set(["5s_rrna", "5_8s_rrna", "7sk"]);
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
  markerGenes: null,
  geneTopCellTypes: null,
  geneAnnotations: null,
  geneFilesByName: null,
  cellTypeFilesByName: null,
  activeGeneName: "",
  filtered: [],
  luckyItems: [],
  isLucky: false,
  shown: PAGE_SIZE,
  query: "",
};

const els = {
  counter: document.querySelector("#counter"),
  search: document.querySelector("#searchInput"),
  searchLabel: document.querySelector("#searchLabel"),
  searchNote: document.querySelector("#searchNote"),
  clear: document.querySelector("#clearButton"),
  lucky: document.querySelector("#luckyButton"),
  modeButtons: document.querySelectorAll(".mode-button"),
  grid: document.querySelector("#resultsGrid"),
  title: document.querySelector("#resultsTitle"),
  meta: document.querySelector("#resultsMeta"),
  loadMore: document.querySelector("#loadMoreButton"),
  dialog: document.querySelector("#imageDialog"),
  dialogFigures: document.querySelector("#dialogFigures"),
  dialogImage: document.querySelector("#dialogImage"),
  dialogCaption: document.querySelector("#dialogCaption"),
  comparisonFigure: document.querySelector("#comparisonFigure"),
  comparisonDialogImage: document.querySelector("#comparisonDialogImage"),
  comparisonDialogCaption: document.querySelector("#comparisonDialogCaption"),
  annotationBox: document.querySelector("#annotationBox"),
  relatedPanel: document.querySelector("#relatedPanel"),
  relatedPanelTitle: document.querySelector("#relatedPanelTitle"),
  relatedList: document.querySelector("#relatedList"),
  closeDialog: document.querySelector("#closeDialogButton"),
  umapZoom: document.querySelector("#umapZoomButton"),
};

function activeCollection() {
  return COLLECTIONS[state.collection];
}

function imageUrl(file) {
  return collectionImageUrl(state.collection, file);
}

function collectionImageUrl(collectionKey, file) {
  const prefix = file[0].toUpperCase();
  return `${COLLECTIONS[collectionKey].imageDir}/${prefix}/${encodeURIComponent(file)}`;
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

  if (state.isLucky) {
    state.filtered = state.luckyItems;
    return;
  }

  if (!query) {
    state.filtered = state.collection === "genes"
      ? plots.filter((plot) => !DEFAULT_HIDDEN_GENE_NAMES.has(plot.nameLower))
      : plots;
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
  if (state.isLucky) {
    els.title.textContent = `Random ${collection.itemLabel} plots`;
  } else {
    els.title.textContent = state.query ? `Results for "${state.query}"` : collection.title;
  }
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

function visibleDefaultPlots(collectionKey = state.collection) {
  const plots = state.collections[collectionKey] || [];
  if (collectionKey !== "genes") return plots;

  return plots.filter((plot) => !DEFAULT_HIDDEN_GENE_NAMES.has(plot.nameLower));
}

function samplePlots(plots, sampleSize) {
  const pool = [...plots];
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [pool[index], pool[swapIndex]] = [pool[swapIndex], pool[index]];
  }
  return pool.slice(0, Math.min(sampleSize, pool.length));
}

async function showLuckyPlots() {
  await loadManifest();
  state.query = "";
  state.isLucky = true;
  state.luckyItems = samplePlots(visibleDefaultPlots(), LUCKY_SIZE);
  state.shown = PAGE_SIZE;
  els.search.value = "";
  render();
}

function resetDialog() {
  els.dialogFigures.classList.remove("has-comparison");
  els.comparisonFigure.hidden = true;
  els.comparisonDialogImage.hidden = false;
  els.comparisonDialogImage.removeAttribute("src");
  els.comparisonDialogImage.alt = "";
  els.comparisonDialogCaption.textContent = "";
  els.annotationBox.hidden = true;
  els.annotationBox.replaceChildren();
  els.relatedPanel.hidden = true;
  els.relatedPanelTitle.textContent = "Related plots";
  els.relatedList.replaceChildren();
  state.activeGeneName = "";
}

function openImageDialog({ src, alt, caption }) {
  resetDialog();
  els.dialogImage.src = src;
  els.dialogImage.alt = alt;
  els.dialogCaption.textContent = caption;

  if (typeof els.dialog.showModal === "function") {
    els.dialog.showModal();
  } else {
    window.open(src, "_blank", "noopener");
  }
}

async function loadMarkerGenes() {
  if (state.markerGenes) return;

  const response = await fetch(MARKER_GENES_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${MARKER_GENES_URL} (${response.status})`);
  }
  state.markerGenes = await response.json();
}

async function loadGeneTopCellTypes() {
  if (state.geneTopCellTypes) return;

  const response = await fetch(GENE_TOP_CELL_TYPES_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${GENE_TOP_CELL_TYPES_URL} (${response.status})`);
  }
  state.geneTopCellTypes = await response.json();
}

async function loadGeneAnnotations() {
  if (state.geneAnnotations) return;

  const response = await fetch(GENE_ANNOTATIONS_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${GENE_ANNOTATIONS_URL} (${response.status})`);
  }
  state.geneAnnotations = await response.json();
}

function addAnnotationField(parent, label, value, href = "") {
  if (!value) return;

  const block = document.createElement("p");
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  block.append(strong);

  if (href) {
    const link = document.createElement("a");
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = value;
    block.append(link);
  } else {
    block.append(document.createTextNode(value));
  }
  parent.append(block);
}

function addAnnotationSection(parent, title, source) {
  const section = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = title;
  section.append(heading);

  if (!source) {
    const empty = document.createElement("p");
    empty.className = "annotation-empty";
    empty.textContent = "No annotation available.";
    section.append(empty);
    parent.append(section);
    return;
  }

  if (source.geneId) addAnnotationField(section, "Gene ID", source.geneId, source.url);
  if (source.accession) addAnnotationField(section, "UniProt", source.accession, source.url);
  addAnnotationField(section, "Description", source.description);
  addAnnotationField(section, "Protein", source.proteinName);
  addAnnotationField(section, "Function", source.function);
  addAnnotationField(section, "Type", source.type);
  if (source.aliases?.length) addAnnotationField(section, "Aliases", source.aliases.join(", "));
  if (source.otherDesignations?.length) {
    addAnnotationField(section, "Other names", source.otherDesignations.join("; "));
  }
  parent.append(section);
}

function renderGeneAnnotation(geneName) {
  els.comparisonDialogImage.hidden = true;
  els.comparisonDialogImage.removeAttribute("src");
  els.comparisonDialogImage.alt = "";
  els.comparisonDialogCaption.textContent = "Gene annotations";
  els.annotationBox.hidden = false;
  els.annotationBox.replaceChildren();
  els.comparisonFigure.hidden = false;
  els.dialogFigures.classList.add("has-comparison");

  const title = document.createElement("h2");
  title.textContent = geneName;
  els.annotationBox.append(title);

  const annotation = state.geneAnnotations?.[geneName];
  if (!state.geneAnnotations) {
    const loading = document.createElement("p");
    loading.className = "annotation-empty";
    loading.textContent = "Loading annotations...";
    els.annotationBox.append(loading);
    return;
  }

  if (!annotation) {
    const empty = document.createElement("p");
    empty.className = "annotation-empty";
    empty.textContent = "No NCBI or UniProt annotation available for this gene.";
    els.annotationBox.append(empty);
    return;
  }

  addAnnotationSection(els.annotationBox, "NCBI Gene", annotation.ncbi);
  addAnnotationSection(els.annotationBox, "UniProt", annotation.uniprot);
}

function showComparisonImage(src, alt, caption) {
  els.annotationBox.hidden = true;
  els.annotationBox.replaceChildren();
  els.comparisonDialogImage.hidden = false;
  els.comparisonDialogImage.src = src;
  els.comparisonDialogImage.alt = alt;
  els.comparisonDialogCaption.textContent = caption;
  els.comparisonFigure.hidden = false;
  els.dialogFigures.classList.add("has-comparison");
}

async function ensureGeneLookup() {
  await loadManifest("genes");
  if (state.geneFilesByName) return;

  state.geneFilesByName = new Map();
  state.collections.genes.forEach((plot) => {
    state.geneFilesByName.set(plot.nameLower, plot.file);
  });
}

async function ensureCellTypeLookup() {
  await loadManifest("cellTypes");
  if (state.cellTypeFilesByName) return;

  state.cellTypeFilesByName = new Map();
  state.collections.cellTypes.forEach((plot) => {
    state.cellTypeFilesByName.set(plot.nameLower, plot.file);
  });
}

async function showMarkerGene(gene, button) {
  await ensureGeneLookup();
  const file = state.geneFilesByName.get(gene.toLowerCase());
  if (!file) return;

  els.relatedList.querySelectorAll(".related-item").forEach((node) => {
    node.setAttribute("aria-pressed", String(node === button));
  });
  showComparisonImage(collectionImageUrl("genes", file), `${gene} gene expression plot`, gene);
}

async function showTopCellType(cellType, button) {
  await ensureCellTypeLookup();
  const file = state.cellTypeFilesByName.get(cellType.toLowerCase());
  if (!file) return;

  if (button.getAttribute("aria-pressed") === "true") {
    button.setAttribute("aria-pressed", "false");
    renderGeneAnnotation(state.activeGeneName);
    return;
  }

  els.relatedList.querySelectorAll(".related-item").forEach((node) => {
    node.setAttribute("aria-pressed", String(node === button));
  });
  showComparisonImage(
    collectionImageUrl("cellTypes", file),
    `${cellType} cell type global pattern plot`,
    cellType,
  );
}

function renderMarkerGenes(cellTypeName) {
  const markers = state.markerGenes?.[cellTypeName] || [];
  els.relatedPanelTitle.textContent = "Global marker genes";
  els.relatedPanel.hidden = false;
  els.relatedList.replaceChildren();

  if (!markers.length) {
    const empty = document.createElement("p");
    empty.className = "related-empty";
    empty.textContent = "No marker genes available.";
    els.relatedList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  markers.forEach((gene) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "related-item";
    button.textContent = gene;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => showMarkerGene(gene, button));
    fragment.append(button);
  });
  els.relatedList.append(fragment);
}

function renderTopCellTypes(geneName) {
  const cellTypes = state.geneTopCellTypes?.[geneName] || [];
  els.relatedPanelTitle.textContent = "Highest expression cell types";
  els.relatedPanel.hidden = false;
  els.relatedList.replaceChildren();

  if (!cellTypes.length) {
    const empty = document.createElement("p");
    empty.className = "related-empty";
    empty.textContent = "No highest expression cell types available.";
    els.relatedList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  cellTypes.forEach((cellType) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "related-item";
    button.textContent = cellType;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => showTopCellType(cellType, button));
    fragment.append(button);
  });
  els.relatedList.append(fragment);
}

async function openCellTypeDialog(plot) {
  openImageDialog({
    src: imageUrl(plot.file),
    alt: `${plot.name} ${activeCollection().alt}`,
    caption: plot.name,
  });

  try {
    await loadMarkerGenes();
    renderMarkerGenes(plot.name);
  } catch (error) {
    els.relatedPanel.hidden = false;
    els.relatedList.innerHTML = `<p class="related-empty">${error.message}</p>`;
  }
}

async function openGeneDialog(plot) {
  openImageDialog({
    src: imageUrl(plot.file),
    alt: `${plot.name} ${activeCollection().alt}`,
    caption: plot.name,
  });
  state.activeGeneName = plot.name;
  renderGeneAnnotation(plot.name);

  try {
    await Promise.all([loadGeneTopCellTypes(), loadGeneAnnotations()]);
    renderGeneAnnotation(plot.name);
    renderTopCellTypes(plot.name);
  } catch (error) {
    els.relatedPanel.hidden = false;
    els.relatedList.innerHTML = `<p class="related-empty">${error.message}</p>`;
  }
}

function openDialog(plot) {
  if (state.collection === "cellTypes") {
    openCellTypeDialog(plot);
    return;
  }

  openGeneDialog(plot);
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
  resetDialog();
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
  state.isLucky = false;
  state.luckyItems = [];
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
    state.isLucky = false;
    state.luckyItems = [];
    state.shown = PAGE_SIZE;
    render();
  });

  els.clear.addEventListener("click", () => {
    state.query = "";
    state.isLucky = false;
    state.luckyItems = [];
    els.search.value = "";
    els.search.focus();
    state.shown = PAGE_SIZE;
    render();
  });

  els.lucky.addEventListener("click", () => {
    showLuckyPlots();
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
