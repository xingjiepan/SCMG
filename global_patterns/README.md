# Global Gene Expression Plot Browser

This directory is a static website for browsing gene-expression plot PNGs.
The search manifest is stored locally, while the images are loaded from the
`xingjiepan/SCMG_data` Hugging Face dataset. The global cell type UMAP is also
loaded from Hugging Face as a persistent reference panel.

## Run locally

From this directory:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Update the image manifest

Run this from the repository root whenever images are added or removed:

```bash
python -c 'import json, pathlib; p=pathlib.Path("global_patterns/global_gene_exp_plots_all"); files=sorted(x.name for x in p.glob("*.png") if x.is_file()); pathlib.Path("global_patterns/manifest.json").write_text(json.dumps(files, indent=2)+"\n")'
```

The website expects plot images in the Hugging Face dataset under prefix
folders like:

```text
data/global_gene_exp_plots_all/M/MYC.png
data/global_gene_exp_plots_all/S/snoZ196.png
data/global_cell_type_umap.png
```

## Free hosting options

### GitHub Pages

The workflow in `.github/workflows/gene-plots-pages.yml` publishes this
directory as a GitHub Pages site. In the repository settings, set Pages to
deploy from GitHub Actions.

### Hugging Face Spaces

Create a new Space using the Static SDK, then upload the contents of this
directory. The site entry point is `index.html`.
