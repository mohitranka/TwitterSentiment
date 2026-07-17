# TwitterSentiment

> **Historical project** — early NLP / sentiment-analysis experiments (circa 2010–2011). Kept public for reference. Not actively maintained; Python packaging and Twitter APIs have moved on.

A small collection of algorithms for **sentiment analysis of tweets**, primarily a Naive Bayes pipeline with train/test pickles.

## What's inside

| Path | Purpose |
|------|---------|
| `naivebayes.py` | Run classification |
| `naivebayes_train.py` | Train the classifier |
| `recreate_pickles.py` | Rebuild pickle artifacts from corpora |
| `pickles/` | Pre-built train/test data and model |
| `results/` | Sample run outputs |

## Corpora

Pickled objects were built from corpora in [`mohitranka/TwitterSentimentCorpora`](https://github.com/mohitranka/TwitterSentimentCorpora).

## Status

- Educational / archival
- Expect older Python conventions
- Fork freely if you want to modernize it

## License

See [LICENSE](LICENSE).
