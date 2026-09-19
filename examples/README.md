# Connecting to UNC Azure

This example uses the same endpoint and authentication as the project's embedding run. It sends one example sentence to `text-embedding-3-large` and reports the number of dimensions returned. It makes one API request and does not process the Reddit dataset.

You need Python 3 and an active UNC API key with access to the model. No additional Python packages are required.

From the repository folder, run:

```bash
python3 examples/azure_embeddings.py
```

Paste your key when prompted and press Enter. It will not appear on screen or be saved. If `UNC_AI_API_KEY` is already set in your environment, the script uses that value.

A successful request should report an embedding with 3,072 dimensions. If a request fails, the script prints an error without displaying the key. Share the error message when asking for help, keeping the key private.

If you do not yet have a key, ask the person managing the project's UNC API access. The script tests existing access; it does not create an account or subscription.

The embeddings for all 60,000 reference posts already exist. See [data access](../semantic_pilot/data-access.md) to use those files for the mapping comparison.
