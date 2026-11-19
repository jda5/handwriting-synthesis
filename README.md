# Handwriting Synthesis

A model that generates realistic handwriting given a line of text.

![Generated handwriting: "hello, how are you today?"](assets/sample.png)

It's the architecture from Graves, [*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850) (2013), built in TensorFlow/Keras with a few fixes from the decade since. The model's the small part, though. Most of this repo is the work around it: cleaning two datasets of recorded handwriting, and generating synthetic handwriting for the characters those datasets barely have.

🤖 Claude wrote the rest of this readme...

## How it Works

### The Model

The text goes in one character at a time and gets read through a Gaussian window. Three LSTM layers with skip connections feed a mixture density output - twenty bivariate Gaussians for where the pen goes next, plus whether the pen is down, lifted or finished. You sample it one pen step at a time, and there's a bias setting that trades variety for neatness.

It isn't pure Graves. The window is parameterised the way Battenberg et al. do it in [*Location-Relative Attention Mechanisms for Robust Long-Form Speech Synthesis*](https://arxiv.org/abs/1910.10288) (2020), and the loss is computed in log space. The train, validation and test sets are split by a hash of each line's text, so the same line never turns up in two of them in different people's handwriting. `notebooks/04-model-improved.ipynb` goes through the reasons.

### The Data

There are about 16,400 lines of recorded pen data, in two sets that both come from the [DeepWriting](https://github.com/emreaksan/deepwriting) project. IAM-OnDB was written on a whiteboard by 200 people, and DeepWriting on an iPad by 94.

Every recording device leaves artefacts behind: stray taps, a pixel grid, a repeated point each time the pen lifts. A model will learn the artefacts instead of the handwriting if you let it, so the `preprocessing` package takes them out. Each module's docstring says what it found and why it matters.

### Synthetic Data

The recordings are mostly lower-case. There aren't many capitals, there's almost no `Q`, `X` or `J`, and there's no `$`, `@`, `£` or `%` at all.

The `synthetic` package fills the gap. It picks texts that lean on the rare characters (UK postcodes, prices, email addresses, dates, lines from Wikipedia) and renders them in handwriting fonts, with each character shifted slightly so no two look the same. Then Google's [InkSight](https://github.com/google-research/inksight) traces the images back into pen strokes. The defective tracings are removed, and the points are re-spaced the way a real pen would have sampled them.

The synthetic set stays at about a tenth of the size of the real one. [`synthetic/synthetic-data.md`](synthetic/synthetic-data.md) has the full plan.

## What's in here

`notebooks/` has the interactive steps, numbered in the order they were written. `preprocessing/` builds and audits the dataset the model trains on. `synthetic/` makes the synthetic texts and images, and re-times the traced strokes. `scripts/` is for one-off experiments. Right now that's only `jev_playground.py`, which asks a vision model how legible a rendered sample is and needs a `TYPESAFE_API_KEY` in `.env`.

The notebooks:

| Notebook | What it does |
| --- | --- |
| `01-clean-data` | Explores and cleans one dataset at a time. `03` replaced it, but it has more plots |
| `02-model` | A first attempt at an encoder-decoder model. `04` replaced it |
| `03-combine-data` | Cleans IAM-OnDB and DeepWriting, puts them on one scale and combines them |
| `04-model-improved` | Trains the model |
| `04a-model` | `04` for Colab. It can pick a run back up from its checkpoint |
| `05-generate` | Writes text with a trained model |
| `06-inksight-stroke-generation` | Turns a screenshot of handwriting into training samples with InkSight |
| `07-inksight-synthetic-strokes` | Runs InkSight over the whole synthetic set |
| `08-clean-synthetic-strokes` | Removes the defective tracings from the synthetic set |

## Getting started

You'll need Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
mkdir -p data/raw data/cleaned checkpoints
```

There's no data, fonts or model weights in this repo, so you'll need to supply your own.

For the recordings, download the IAM-OnDB and DeepWriting JSON files from the [DeepWriting](https://github.com/emreaksan/deepwriting) project and unpack them to `data/raw/iamondb` and `data/raw/deepwriting`.

The other two are only for the synthetic data. Put some handwriting fonts (`.ttf` or `.otf`) in `synthetic/fonts`, and save the postcode CSV from [doogal.co.uk](https://www.doogal.co.uk/UKPostcodes) as `synthetic/postcodes.csv`. Note that `synthetic/fonts.py` has a list of glyphs that were rejected by eye, keyed by font file name. It only covers the fonts used here, so it won't cover yours.

Each notebook sets a `ROOT` path in one of its first cells. Point it at wherever you cloned the repo, or at a Google Drive folder if you're on Colab. Training and InkSight need a GPU to run in reasonable time. Everything else runs fine on a laptop.

## Running the pipeline

Each step reads the newest dataset of the kind it needs from `data/cleaned` and writes a new timestamped folder next to it. The order matters.

1. **Combine the recordings.** Run `notebooks/03-combine-data.ipynb`. It writes
   `data/cleaned/combined-<timestamp>`.

2. **Generate the synthetic texts and images.** They land in `data/synthetic`.

   ```bash
   uv run python -m synthetic.generate_text --seed 42
   ```

   ```bash
   uv run python -m synthetic.render_images --seed 42
   ```

3. **Trace them and remove the defective ones.** Run
   `notebooks/07-inksight-synthetic-strokes.ipynb`, then
   `notebooks/08-clean-synthetic-strokes.ipynb`. The first one writes
   `data/cleaned/synthetic-<timestamp>`.

4. **Add them to the combined dataset.** There's no script for this yet. The last cell
   of notebook 07 shows how to append the samples. You'll also need to write a `sources.json`
   that marks each sample as `"real"` or `"synthetic"`, because the next steps rely on it.

5. **Re-space the synthetic strokes.** This writes a new `combined-<timestamp>`.

   ```bash
   uv run python -m synthetic.retime_strokes --seed 42
   ```

6. **Build the training set and check it.** It ends up in `data/cleaned/final-<timestamp>`.

   ```bash
   uv run python -m preprocessing.build --seed 42
   ```

   ```bash
   uv run python -m preprocessing.audit
   ```

7. **Train.** Run `notebooks/04-model-improved.ipynb`. Each run gets its own folder in
   `checkpoints` with the weights and a `config.json` that has everything you need to rebuild
   the model.

8. **Generate handwriting.** Run `notebooks/05-generate.ipynb`. It loads the newest run.

   ```python
   strokes, window = generate("hello, how are you today?", bias=1.5)
   ```

Step 6 puts back marks that the synthetic samples lost in tracing, so it needs the images and labels from step 2 in `data/synthetic` even if your dataset has no synthetic samples in it.

## Dataset format

A dataset is a folder in `data/cleaned` with three files in it. `strokes_data.npz` holds one float32 array of shape `(N, 5)` per sample, with rows of `[dx, dy, pen_down, pen_up, eos]`. `texts.json` has the text each sample writes, and `sources.json` says whether each one is `"real"` or `"synthetic"`.

`dx` and `dy` are how far the pen moved from its last position, in units of roughly one lower-case letter's height. Every sample starts with `[0, 0, 1, 0, 0]` and ends with `[0, 0, 0, 0, 1]`, and the point where the pen lifts has `pen_up = 1`.

## Data and fonts

The datasets, fonts and postcode list don't belong to this project, and each one comes with its own terms. IAM-OnDB in particular is for non-commercial research only. Read the terms before you use any of it, and again before you share a model trained on it.

## References

The model comes from Graves, [*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850) (2013), with the attention window from Battenberg et al., [*Location-Relative Attention Mechanisms for Robust Long-Form Speech Synthesis*](https://arxiv.org/abs/1910.10288) (2020). The pen-state encoding is from Ha and Eck, [*A Neural Representation of Sketch Drawings*](https://arxiv.org/abs/1704.03477) (2017).

The recordings are the [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database) and Aksan et al., [*DeepWriting: Making Digital Ink Editable via Deep Generative Modeling*](https://arxiv.org/abs/1801.08379) (2018). The tracing is Mitrevski et al.,
[*InkSight: Offline-to-Online Handwriting Conversion by Teaching Vision-Language Models to Read and Write*](https://arxiv.org abs/2402.05804) (2024).
