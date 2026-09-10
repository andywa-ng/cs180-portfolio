"""Generate static Project 1 HTML and website JPEGs from measured results.

Run after run_project1.py. Text, offsets, timings and image paths share the same
experiment records, so the webpage cannot silently drift from the computation.
"""
import csv
from html import escape
import json
from pathlib import Path
import shutil
from statistics import median
import numpy as np
from PIL import Image
from proj1 import read_channels, save_jpeg

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/proj1"
TITLES = {"religous_painting": "Religious painting", "ilemselga": "Ilemselga",
          "tiflis": "Tiflis (?)", "adobe_buildings": "Adobe buildings and yurts",
          "cotton_mill": "Cotton textile mill, probably in Tashkent"}


def title(name):
    return TITLES.get(name, name.replace("_", " ").capitalize())


def xy(values):
    return f"({values[0]}, {values[1]})"


def export_image(record, experiment):
    destination = ASSETS / experiment
    destination.mkdir(parents=True, exist_ok=True)
    name = record["name"]
    shutil.copyfile(ROOT / record["output"], destination / f"{name}-full.jpg")
    with Image.open(ROOT / record["output"]) as image:
        image.thumbnail((1050, 1050), Image.Resampling.LANCZOS)
        image.save(destination / f"{name}.jpg", quality=90, subsampling=0, optimize=True)


def photo(name, experiment, label=None, eager=False):
    path = f"assets/proj1/{experiment}/{name}.jpg"
    with Image.open(ROOT / path) as im:
        width, height = im.size
    full = f"assets/proj1/{experiment}/{name}-full.jpg"
    return (f'<a href="{full}" aria-label="Open full-resolution {escape(title(name))}, {experiment}">'
            f'<img src="{path}" alt="{escape(label or title(name))}" width="{width}" height="{height}" '
            f'loading="{"eager" if eager else "lazy"}" decoding="async"></a>')


def card(record, experiment, source=None):
    name = record["name"]
    note = '<br><a href="#emir">Raw NCC fails here; see the edge-based correction.</a>' if name == "emir" and experiment == "pyramid" else ""
    credit = (f'<br><a href="{escape(source["item_url"])}">Library of Congress record</a> · '
              f'<a href="{escape(source["plate_url"])}">Original three-frame scan</a>') if source else ""
    return (f'<figure class="result" id="{experiment}-{name}">{photo(name, experiment)}'
            f'<figcaption><strong>{escape(title(name))}</strong><span class="offsets">'
            f'G {xy(record["green_xy"])} · R {xy(record["red_xy"])}</span>{note}{credit}</figcaption></figure>')


def result_table(records, label, experiment, include_links=True):
    rows = []
    for r in records:
        h, w = r["channel_shape"]
        name = escape(title(r["name"]))
        if include_links:
            name = f'<a href="assets/proj1/{experiment}/{r["name"]}-full.jpg">{name}</a>'
        rows.append(f'<tr><th scope="row">{name}</th><td>{w} × {h}</td><td>{xy(r["green_xy"])}</td>'
                    f'<td>{xy(r["red_xy"])}</td><td>{r["alignment_seconds"]:.2f}</td><td>{r["total_seconds"]:.2f}</td></tr>')
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="{escape(label)}">'
            f'<table><caption>{escape(label)}</caption><thead><tr><th scope="col">Image</th><th scope="col">Frame W × H</th>'
            '<th scope="col">G (x, y)</th><th scope="col">R (x, y)</th><th scope="col">Align (s)</th><th scope="col">Total (s)</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    records = {kind: json.loads((ROOT / f"outputs/{kind}/results.json").read_text()) for kind in ("single", "pyramid", "gradient")}
    sources = {s["name"]: s for s in json.loads((ROOT / "additional_sources.json").read_text())}
    for kind, rows in records.items():
        for record in rows:
            export_image(record, kind)
        (ASSETS / f"{kind}-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    shutil.copyfile(ROOT / "outputs/run_metadata.json", ASSETS / "run-metadata.json")
    with (ASSETS / "offsets.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["method", "image", "green_x", "green_y", "red_x", "red_y", "alignment_seconds", "total_seconds"])
        for kind, rows in records.items():
            for r in rows:
                writer.writerow([kind, r["name"], *r["green_xy"], *r["red_xy"], r["alignment_seconds"], r["total_seconds"]])
    blue, green, red = read_channels(ROOT / "proj1_data/cathedral.jpg")
    save_jpeg(np.dstack((red, green, blue)), ASSETS / "cathedral-unaligned.jpg")
    provided = [r for r in records["pyramid"] if r["name"] not in sources]
    additional = [r for r in records["pyramid"] if r["name"] in sources]
    raw_emir = next(r for r in provided if r["name"] == "emir")
    edge_emir = next(r for r in records["gradient"] if r["name"] == "emir")
    melons = next(r for r in provided if r["name"] == "melons")
    trace_rows = "".join(f'<tr><td>{g["level"]}</td><td>{g["shape"][1]} × {g["shape"][0]}</td>'
                         f'<td>{xy(g["shift_xy"])}</td><td>{xy(r["shift_xy"])}</td></tr>'
                         for g, r in zip(melons["green_trace"], melons["red_trace"]))
    times = [r["total_seconds"] for r in records["pyramid"]]
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Colorizing Prokudin-Gorskii glass plates with exhaustive NCC alignment, a coarse-to-fine image pyramid, and gradient features. CS180 Project 1 by Andy Wang.">
  <title>Project 1 · Images of the Russian Empire · Andy Wang</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="proj1.css">
</head>
<body>
<header><h1>CS 180 Portfolio</h1><p>Computer Vision and Computational Photography</p></header>
<nav class="nav-tabs" aria-label="Projects">
  <a href="index.html">Home</a><a href="proj0.html">Project 0</a><a href="proj1.html" class="active" aria-current="page">Project 1</a>
</nav>
<main class="content-box project-one">
  <p class="eyebrow">Project 1 · Andy Wang · Fall 2026</p>
  <h2>Images of the Russian Empire</h2>
  <p class="intro">One glass plate, three exposures, and a missing color photograph. This project reconstructs color images from Prokudin-Gorskii’s photographs by finding the translations that bring their blue, green, and red records into alignment.</p>
  <div class="comparison">
    <figure><img src="assets/proj1/cathedral-unaligned.jpg" alt="Cathedral with the red, green, and blue frames stacked without alignment" width="390" height="341"><figcaption><strong>Before:</strong> stack the three channels directly.</figcaption></figure>
    <figure>{photo("cathedral", "single", "Cathedral after exhaustive NCC alignment", eager=True)}<figcaption><strong>After:</strong> align G and R to the blue reference.</figcaption></figure>
  </div>
  <nav class="page-links" aria-label="On this page"><a href="#single">Single scale</a><a href="#pyramid">Image pyramid</a><a href="#results">14 supplied images</a><a href="#additional">3 additional images</a><a href="#emir">Better features</a><a href="#reproduce">Code &amp; data</a></nav>

  <section id="single">
    <h3><span class="section-number">01</span> Finding a translation</h3>
    <p>The input is a grayscale scan with the <strong>blue, green, and red exposures arranged from top to bottom</strong>. I split the scan into three equal-height arrays, discarding at most two leftover rows. Integer intensities are converted to floating point in [0, 1] by dividing by the data type’s maximum: 255 for 8-bit JPEGs and 65,535 for 16-bit TIFFs.</p>
    <p>The blue frame stays fixed. For green and then red, two nested loops try every integer translation in <strong>[−15, 15] × [−15, 15]</strong>: 961 candidates per channel. Each candidate is scored with normalized cross-correlation (NCC):</p>
    <div class="equation" role="math" aria-label="NCC is the dot product of mean-subtracted image vectors divided by the product of their Euclidean norms">NCC(A, B) = Σ[(A − mean(A))(B − mean(B))] / (‖A − mean(A)‖₂ ‖B − mean(B)‖₂)</div>
    <p>I select the displacement with the highest score. Subtracting the mean removes a global brightness offset; dividing by each vector’s norm removes a positive global contrast scale. The code also supports L2 matching, minimizing mean squared pixel error. All displayed baseline results use <strong>NCC on raw channel intensities</strong>.</p>
    <p>Glass-plate borders are strong edges but do not represent the scene. For scoring, I exclude 12% of each side and further restrict the support so every candidate samples valid interior pixels in both frames. Every shift uses exactly the same blue-frame coordinates and number of pixels. This avoids rewarding a candidate for having a smaller overlap or matching wrapped pixels.</p>
    <p>After estimating the shifts, I apply <code>np.roll</code> to green and red and stack the channels in RGB order. The saved image contains only the three shifted frames’ valid intersection, so circularly wrapped pixels never appear. This intersection crop does not detect the original photographic border; some colored frame edges remain visible.</p>
    <p><strong>Offset convention:</strong> all offsets below are <strong>(x, y)</strong> in original-resolution pixels, applied to the moving channel relative to blue. Positive x moves right; positive y moves down. NumPy receives <code>(y, x)</code> because its axes are row, then column.</p>
    {result_table(records["single"], "Single-scale NCC · all three supplied JPEGs", "single")}
    <div class="gallery">{"".join(card(r, "single") for r in records["single"])}</div>
    <p>The cathedral’s architecture, monastery’s towers, and Tobolsk’s rooflines line up without the large color-separated edges of the unaligned stack. These offsets fit inside the single-scale search window.</p>
  </section>

  <section id="pyramid">
    <h3><span class="section-number">02</span> Searching coarse to fine</h3>
    <p>A ±15-pixel search is too small for the full-resolution TIFFs. For example, the melons image needs a red-channel displacement of {xy(melons["red_xy"])} pixels. Enlarging the exhaustive window would multiply the number of comparisons while each comparison still touches millions of pixels.</p>
    <ol>
      <li><strong>Build the pyramid.</strong> Apply the separable binomial filter [1, 4, 6, 4, 1] / 16 horizontally and vertically with reflected padding, then keep every second row and column. Smoothing suppresses aliasing before downsampling. Repeat until the longest dimension is at most 350 pixels, with a 64-pixel minimum-side guard for narrow inputs.</li>
      <li><strong>Search the coarsest level.</strong> Run the same exhaustive NCC search over ±15 pixels in each direction.</li>
      <li><strong>Refine at the next level.</strong> Double the estimated displacement and search ±3 pixels around it, evaluating 49 candidates per channel.</li>
      <li><strong>Continue to full resolution.</strong> Repeat the doubling and local search until the original image is reached, then compose the valid RGB overlap.</li>
    </ol>
    <p>The pyramid and searches are implemented directly with NumPy; no automatic alignment or pyramid-building routine is used. The final runs score <strong>every pixel in the valid interior</strong> at each level, including the original-resolution level. The same parameters are used for every image, with no per-image offset adjustments.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Melons pyramid trace"><table><caption>Melons: measured coarse-to-fine estimates</caption><thead><tr><th scope="col">Level</th><th scope="col">Frame W × H</th><th scope="col">G (x, y)</th><th scope="col">R (x, y)</th></tr></thead><tbody>{trace_rows}</tbody></table></div>
    <p>Across all 17 baseline inputs, the median measured total time was {median(times):.2f} seconds and the slowest was {max(times):.2f} seconds on this run. “Align” measures the two channel searches, including pyramid construction; “Total” also includes loading, composition, and full-resolution JPEG writing. These are individual local measurements, not cross-machine benchmarks. The <a href="assets/proj1/run-metadata.json">run metadata</a> records the environment and input hashes.</p>
  </section>

  <section id="results">
    <h3><span class="section-number">03</span> All 14 supplied images</h3>
    <p>These are the raw-intensity NCC pyramid results, including the three JPEGs. Click a photograph to inspect the full-resolution output. The thumbnails are resized for the webpage; the TIFFs were processed at their original resolution.</p>
    <div class="failure"><p><strong>One clear baseline failure: Emir.</strong> Its red channel is substantially misaligned. The other supplied scenes have coherent alignment on stationary structures, though small fringes, moving subjects, and damaged borders remain. The <a href="#emir">feature extension below</a> addresses Emir’s failure.</p></div>
    {result_table(provided, "Raw-intensity NCC pyramid · supplied images", "pyramid")}
    <div class="gallery">{"".join(card(r, "pyramid") for r in provided)}</div>
  </section>

  <section id="additional">
    <h3><span class="section-number">04</span> Three more glass plates</h3>
    <p>To test the same fixed parameters beyond the supplied set, I used three additional full-resolution, three-frame TIFF scans from the Library of Congress: an industrial interior, adobe buildings and yurts, and a city view cataloged as Tiflis (?). These results use the same raw-intensity NCC pyramid. The source links lead to the original records and grayscale plates, rather than the Library’s restored color composites. The cotton-mill scan is 16-bit; the other two are the Library’s 8-bit full-resolution TIFF versions.</p>
    {result_table(additional, "Raw-intensity NCC pyramid · additional LoC scans", "pyramid")}
    <div class="gallery">{"".join(card(r, "pyramid", sources[r["name"]]) for r in additional)}</div>
    <p>The mill’s ducts and machinery provide repeated structural edges; the distant buildings and tree line anchor the adobe scene; and Tiflis’s dense rooflines provide many correspondences. All three form coherent color images. Fine fringes are still visible on some mill machinery, and the adobe scene retains a colored patch near the bottom. Alignment alone does not remove these artifacts.</p>
  </section>

  <section id="emir">
    <h3><span class="section-number">05</span> Better features: rescuing Emir</h3>
    <p>NCC compensates for a global brightness change, but different color filters can change <em>which objects</em> are bright. Emir’s blue clothing has very different intensity in the red and blue records. Raw-pixel NCC therefore favors the wrong red-channel correspondence, and the pyramid propagates that mistake to finer levels.</p>
    <p>For an optional extension, I replace each pyramid level’s intensity image with its gradient magnitude. Central differences estimate horizontal and vertical changes:</p>
    <div class="equation" role="math" aria-label="Gradient magnitude is the square root of squared horizontal and vertical central differences">DₓI = (I[y, x+1] − I[y, x−1]) / 2<br>DᵧI = (I[y+1, x] − I[y−1, x]) / 2<br>feature(I) = √(DₓI² + DᵧI²)</div>
    <p>The same NCC search now matches structural boundaries. Gradient magnitude also treats a dark-to-light edge like a light-to-dark edge, making it less sensitive to contrast reversal between color channels. Gradients are computed anew at each smoothed pyramid level. The scoring window, blue reference, and translation model are unchanged.</p>
    <div class="comparison">
      <figure>{photo("emir", "pyramid", "Emir with severe red-channel displacement under raw-intensity NCC")}<figcaption><strong>Before: raw-intensity NCC</strong><span class="offsets">G {xy(raw_emir["green_xy"])} · R {xy(raw_emir["red_xy"])}</span></figcaption></figure>
      <figure>{photo("emir", "gradient", "Emir with face and robe aligned using gradient-magnitude NCC")}<figcaption><strong>After: gradient-magnitude NCC</strong><span class="offsets">G {xy(edge_emir["green_xy"])} · R {xy(edge_emir["red_xy"])}</span></figcaption></figure>
    </div>
    <p>The face, robe pattern, and background now register far more closely. This extension changes the alignment features; it does not recolor or retouch the photograph. I also ran the gradient method on all 17 inputs with the same parameters. Their offsets and full-resolution outputs are available below.</p>
    <details><summary>All gradient-NCC offsets and full-resolution results</summary>{result_table(records["gradient"], "Gradient-magnitude NCC pyramid · all 17 images", "gradient")}</details>
    <p><strong>Limitations.</strong> A single integer translation cannot correct subject motion between exposures, rotation, scale changes, or local film deformation. The remaining small fringes around people and foliage should not be confused with Emir’s large global alignment failure. Scratches and colored photographic borders also remain: no border-detection, white balance, contrast stretching, or color remapping is claimed here.</p>
  </section>

  <section id="reproduce">
    <h3><span class="section-number">06</span> Code and reproducibility</h3>
    <p>The implementation uses NumPy for array operations and Pillow for reading and writing images. <a href="proj1.py">Alignment code</a> · <a href="README.md">Reproduction instructions</a> · <a href="additional_sources.json">Additional-image sources</a> · <a href="assets/proj1/offsets.csv">All offsets (CSV)</a></p>
    <pre><code>python -m pip install -r requirements.txt
python download_examples.py
python -m unittest -v test_proj1
python run_project1.py</code></pre>
    <p class="small">Place the 14 supplied input images in <code>proj1_data/</code> before running. The batch script produces all three experiments, records measured offsets and runtimes, and regenerates this page from those records. Tests check known signed translations, large pyramid displacements, NCC brightness invariance, 8/16-bit normalization, BGR splitting, anti-aliasing, and valid-overlap composition.</p>
    <p class="small">Machine-readable results: <a href="assets/proj1/single-results.json">single-scale</a>, <a href="assets/proj1/pyramid-results.json">raw pyramid</a>, and <a href="assets/proj1/gradient-results.json">gradient pyramid</a>.</p>
  </section>
  <footer class="photo-credit"><p class="small">Photographs: Sergei Mikhailovich Prokudin-Gorskii. Prokudin-Gorskii photograph collection, Library of Congress, Prints and Photographs Division. Additional photographs are cataloged with no known restrictions on publication. <a href="https://www.loc.gov/collections/prokudin-gorskii/about-this-collection/">About the collection</a> · <a href="https://www.loc.gov/pictures/collection/prok/digitizing.html">About the original scans</a>.</p><a href="index.html">← Back to Home</a></footer>
</main>
</body>
</html>
'''
    (ROOT / "proj1.html").write_text(html)
    print(f"Built proj1.html: {len(provided)} supplied + {len(additional)} additional images.")


if __name__ == "__main__":
    main()
