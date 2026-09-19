# Reddit and AI research

This project studies Reddit participation and post content in relation to AI availability. It includes an ACSI annotation and analysis pipeline and a semantic pilot comparing methods for grouping post text.

- [Semantic pilot: methods, code, and results](semantic_pilot/README.md)
- [Numerical comparison](semantic_pilot/reports/RESULTS.md)
- [Saved embeddings](data/embeddings/README.md)
- [Other project data](semantic_pilot/data-access.md)

Raw data are held separately. The [data guide](data/README.md) describes the local files used by the existing [analysis scripts](scripts/).

## Existing annotation and analysis pipeline

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

With the required input files available, run the analysis or consult the annotation commands:

```bash
.venv/bin/python scripts/run.py
.venv/bin/python scripts/annotate.py --help
```

The older [annotation_code](annotation_code/) scripts are historical material; their original utility module and sampling configuration are not included.
