# Strategy for Creating Synthetic Data

## Observations

1. This is an extremely skewed dataset. There are many more examples of lower-case letters than uppercase letters and almost no examples of more niche punctuation or rare letters (such as Q, X, J U, V, K, etc.).
2. The model is bad at generating postcodes and acronyms (QR, AI, PS, etc.).
3. There are no examples any of the following symbols in our dataset: $, @, £, %. These are likely needed in B2B copy.

## Strategy

Generate synthetic data by sampling text snippets and using TrueType fonts (ttf), that mimic human handwriting, to render them as images. Then use Google's [InkSight](https://github.com/google-research/inksight) model to transform these images into online strokes, that we can use to fine-tune our handwriting synthesis model.

The number of synthetic examples must be significantly smaller than the number of ground truth examples - approximately 10% of the total dataset, which at the time of writing is about 16,000 examples.

This synthetic dataset should be composed of:

1. UK postcodes. Take a list of all UK postcodes (from here: https://www.doogal.co.uk/UKPostcodes), shuffle them, then sample them based on the distributions of upper-case characters and numbers in our groundtruth dataset, with the less likely a character is to appear the higher the probability of it being sampled.
2. Text for symbols: generate some plausible snippets such as prices, email address, percentages, dates.
3. Corpus snippets: pull sentences from a real corpus, such as Wikipedia or news (there are a few news API's we could tap into). Keep windows containing at least one rare character. Put rare characters in many contexts: “X” at the start of words, after spaces, next to digits. Keep lines around the median of the groundtruth dataset.
4. Letter snippets: pull sentences from the InkBound corpus. Use a similar strategy to the news corpus sampling. This gives the model examples of the kind of outputs it's expected to produce in production. What might be good is to focus on words with a higher concentration of upper-case characters. Words such as acronyms or company names.

When rendering the texts as images we should add per-character jitters, small rotations, scale shifts, baseline shifts and spacing changes, to break the repetition of fonts.

Not all fonts have glyphs for every character, and having looked at some renderings some fonts don't draw certain characters well - "E" seems to be particularly problematic. Perhaps we could get Jev to look at the images and give an assessment of how legible each one is.

## Implementation

The `synthetic` package holds one module per source of text, each offering a pool of
`candidates()`, and `sampling.py`, which draws from a pool with probability proportional to
how rare a candidate's characters are in the ground truth. Unseen characters (`$`, `@`, `£`,
`%`) score highest.

| Module | Role |
| --- | --- |
| `sampling.py` | Ground-truth character counts, rarity scores and weighted sampling without replacement |
| `postcodes.py` | UK postcodes from `postcodes.csv` with no repeated characters |
| `emails.py` | Addresses built from the short names in `words.py` |
| `symbols.py` | Templated prices, percentages, handles, dates, times and acronyms |
| `corpus.py` | Lines cut from Wikipedia introductions (cached at `data/synthetic/wikipedia.txt`) that hold a rare character |
| `fonts.py` | Which characters each font draws faithfully (mapped to an inked glyph of its own, not rejected by eye) and whether it joins its letters. Scrawly fonts live in `fonts/dropped` |
| `render.py` | Renders a text one character at a time with per-character rotation, scale, baseline and spacing jitter, lifting the pen around symbols in joined-up fonts |
| `generate_text.py` | Builds the dataset: 10% of the ground truth's size, split across the sources |
| `render_images.py` | Renders each text in a random font that draws all of its characters |

```bash
uv run python -m synthetic.generate_text --seed 42
uv run python -m synthetic.render_images --seed 42
```

Images land in `data/synthetic/images` and their labels (text, source, font and the box around
each word) in `data/synthetic/labels.json`. `notebooks/07-inksight-synthetic-strokes.ipynb` then
derenders every word with InkSight, using those boxes instead of OCR, and saves the strokes to
`data/cleaned/synthetic-<timestamp>`. The fonts' rejected glyphs live in `fonts.REJECTED`; it is the
natural place to feed in Jev's legibility assessments.

The texts land in `data/synthetic/texts.json`, keyed by source, and the script prints how many
examples of each rare character the synthetic set adds. Letter snippets from the InkBound
corpus are not yet included: `corpus.candidates` takes any text, so they can be added the same
way once the corpus is to hand.
