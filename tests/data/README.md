# Test data

`tiny-sample.cwa.gz` is a real Axivity recording used by the model-backed
numerical regression test. `tiny-sample-baseline.json` pins its expected RF and
SSL results to the model versions declared in `stepcount.__model_version__`.

Run the regression locally with:

```console
STEPCOUNT_RUN_MODEL_REGRESSION=1 python -m pytest tests/test_cwa_regression.py
```

The SSL case is forced onto CPU so its execution path is consistent across CI
platforms. The test compares exact discrete predictions and timestamps, while
allowing explicit narrow tolerances for continuous accelerometer summaries.
Every CSV export is also compared through a canonical, type-tagged signature.
Strings and integers remain exact; floating values use six-decimal comparison
precision, and input paths are normalized to their basenames.

Only update the baseline after an intentional model or algorithm change. Review
the generated `Info.json` and CSV results independently before accepting new
values, then update the model identity, summary values, row counts, and canonical
step-series digest together. A fixture change also requires a new fixture SHA-256.
