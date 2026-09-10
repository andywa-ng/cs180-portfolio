# CS180 Project 1 — Images of the Russian Empire

Project webpage: https://andywa-ng.github.io/cs180-portfolio/proj1.html

The implementation splits grayscale B/G/R glass plates, aligns green and red
to blue, and saves full-resolution RGB JPEGs. It includes exhaustive
single-scale NCC/L2 alignment, a manually built coarse-to-fine pyramid, and an
optional gradient-magnitude NCC extension. NumPy and Pillow are the only dependencies.

## Run everything

Use Python 3.10 or newer, preferably in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_examples.py
python -m unittest -v test_proj1
python run_project1.py
```

Before the last command, place the 14 supplied image files in `proj1_data/`:
cathedral, church, emir, harvesters, icon, ilemselga, melons, monastery,
religous_painting, self_portrait, siren, three_generations, tobolsk, and wharf.
The misspelling `religous_painting.tif` is retained to match the provided filename.
Download the course data from the assignment page; raw inputs are not in Git.

`download_examples.py` downloads three additional original LoC scans using
the URLs in `additional_sources.json`. That manifest includes catalog titles,
source links, attribution, and checksums. Full-resolution TIFF downloads are large.

`run_project1.py` runs the experiments sequentially and produces:

- `outputs/single/`: the three supplied JPEGs, exhaustive raw-intensity NCC.
- `outputs/pyramid/`: all 17 inputs, raw-intensity NCC pyramid.
- `outputs/gradient/`: all 17 inputs, gradient-magnitude NCC pyramid.
- `results.json` in each directory: shifts, pyramid traces, parameters, timings.
- `outputs/run_metadata.json`: environment and SHA-256 checksums of the inputs.
- `proj1.html` and `assets/proj1/`: the static report, thumbnails, full-resolution
  JPEGs, CSV offsets, and JSON records generated from the actual results.

To rebuild only the webpage from existing results:

```sh
python build_project1_page.py
```

## Run individual images or select a metric

```sh
python proj1.py proj1_data/cathedral.jpg --method single --metric ncc --output-dir outputs/example
python proj1.py proj1_data/cathedral.jpg --method single --metric l2 --output-dir outputs/l2
python proj1.py proj1_data/emir.tif --feature gradient --output-dir outputs/emir_edges
python proj1.py proj1_data/*.tif --output-dir outputs/tiffs
```

Each invocation writes one JPEG per image and a `results.json` manifest for
that invocation; use separate output folders for separate experiments.

The CLI prints **(x, y)** offsets applied to G and R relative to B. Positive
x moves right and positive y moves down. `np.roll` receives `(y, x)` because
NumPy indexes rows before columns. Remainder rows after dividing the plate by
three are discarded. Unsigned integer inputs are normalized by their dtype
maximum, preserving the proper scale for 8-bit and 16-bit data.

## Fixed parameters and algorithm

- Raw-pixel NCC is the baseline; L2 is also implemented. For L2, minimizing MSE
  is equivalent to minimizing the Euclidean norm on the same fixed support.
- Single-scale/coarsest search: inclusive ±15 pixels per axis (961 candidates).
- Pyramid: separable `[1, 4, 6, 4, 1]/16` smoothing, followed by 2× decimation;
  stop at longest side ≤350 (or minimum side ≤64).
- Finer-level refinement: double the previous offset, then search ±3 pixels.
- Scoring: exclude 12% on each side and use a common valid support for all
  candidates. Neither circular wrapping nor changing support sizes affects NCC.
- Every valid interior pixel is scored by default at each level. Optional
  `--max-samples 650` uses a regular grid for faster scoring on slower machines;
  the submitted results use `--max-samples 0` (all pixels).
- Compose with `np.roll`, then crop to the intersection of all three valid
  image domains. Original photographic borders are not automatically detected.
- Gradient extension: central differences, gradient magnitude, then the same
  NCC pyramid. All images share the same configuration.

No high-level pyramid, registration, phase-correlation, or automatic alignment
functions are used. Tests cover signed shifts, large displacements on odd image
sizes, brightness invariance, normalization, splitting, anti-aliasing, and
valid-overlap composition.

## Interpretation

Raw NCC has a clear failure on Emir: red/blue intensity patterns differ because
of the subject's saturated clothing. The gradient extension corrects the large
misalignment. The webpage preserves both outputs and reports all baseline
offsets. Small residual fringes may remain because an integer translation does
not model motion, rotation, scale, or local deformation. No automatic border
detection, contrast adjustment, white balance, or color remapping is claimed.

## Gradescope and gallery

Run `python package_submission.py` to create `submission/project1-code.zip`.
It contains the Python source, webpage HTML/CSS, requirements, source manifest,
README, measured JSON/CSV records, and `website_url.txt`. It intentionally excludes every image,
raw-data directory, virtual environment, and Git metadata. Reproduction downloads
or uses local input images; the archive itself does not include them.

The project webpage URL belongs in both Gradescope and the class-gallery Google
Form. Review the explanation and results before submitting through those portals.

## Sources

Photographs: Sergei Mikhailovich Prokudin-Gorskii. Prokudin-Gorskii photograph
collection, Library of Congress, Prints and Photographs Division.
Additional sources are listed individually in `additional_sources.json`.
The original starter script and assignment supplied the BGR splitting and
image-stacking outline; the alignment, pyramid, tests, and report generation
are implemented in this project.
