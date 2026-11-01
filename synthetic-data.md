# Strategy for Creating Synthetic Data

## Observations

1. This is an extremely skewed dataset. There are many more examples of lower-case letters than uppercase letters and almost no examples of more niche punctuation or rare letters (such as Q, X, J U, V, K, etc.).
2. The model is bad at generating postcodes and acronyms (QR, AI, PS, etc.).
3. There are no examples any of the following symbols in our dataset: $, @, £, %. These are likely needed in B2B copy.

## Strategy

Generate synthetic data by sampling text snippets and using TrueType fonts (ttf), that mimic human handwriting, to render them as images. Then use Google's [InkSight](https://github.com/google-research/inksight) model to transform these images into online stokes, that we can use to fine-tune our handwriting synthesis model.

The number of synthetic examples must be signficantly smaller than the number of ground truth examples - approximately 10% of the total dataset, which at the time of writing is about 16,000 examples.

This synthetic dataset should be composed of:

1. UK postcodes. Take a list of all UK postcodes (from here: https://www.doogal.co.uk/UKPostcodes), shuffle them, then sample them based on the distributions of upper-case characters and numbers in our groundtruth dataset, with the less likely a character is to appear the higher the probability of it being sampled.
2. Text for symbols: generate some plausible snippets such as prices, email address, percentages, dates.
3. Corpus snippets: pull sentences from a real corpus, such as Wikipedia or news (there are a few news API's we could tap into). Keep windows containing at least one rare character. Put rare characters in many contexts: “X” at the start of words, after spaces, next to digits. Keep lines around the median of the groundtruth dataset.
4. Letter snippets: pull sentences from the InkBound corpus. Use a similar strategy to the news corpus sampling. This gives the model examples of the kind of outputs it's expected to produce in production. What might be good is to focus on words with a higher concentration of upper-case characters. Words such as acronyms or company names.

When rendering the texts as images we should add per-character jitters, small rotations, scale shifts, baseline shifts and spacing changes, to break the repetition of fonts.

Not all fonts have glyphs for every character, and having looked at some renderings some fonts don't draw certain characters well - "E" seems to be particularly problematic. Perhaps we could get Jev to look at the images and give an assessment based on 