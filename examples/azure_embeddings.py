"""Check UNC Azure access by embedding one example sentence."""

import getpass
import http.client
import json
import math
import os
import sys
import urllib.error
import urllib.request
import warnings


ENDPOINT = "https://azureaiapi.cloud.unc.edu/openai/v1/embeddings"
MODEL = "text-embedding-3-large"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    key = os.environ.get("UNC_AI_API_KEY", "").strip()
    if not key:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("UNC API key (input is hidden): ").strip()
        except getpass.GetPassWarning:
            print("Run in a terminal or set UNC_AI_API_KEY in your environment.", file=sys.stderr)
            return 1
    if not key or key in {"123", "YOUR_KEY", "YOUR_API_KEY"}:
        print("Enter your actual UNC API key.", file=sys.stderr)
        return 1
    if any(ord(char) < 32 or ord(char) > 126 for char in key):
        print("The key contains unexpected characters; check the copied value.", file=sys.stderr)
        return 1

    payload = {
        "model": MODEL,
        "input": ["This is a test of the embedding service."],
        "encoding_format": "float",
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(request, timeout=60) as response:
            result = json.load(response)
        vector = result["data"][0]["embedding"]
        if (not isinstance(vector, list) or not vector
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
            raise ValueError("Missing embedding")
    except urllib.error.HTTPError as error:
        hints = {
            401: "Check that your UNC API key is correct and active.",
            403: "Check whether your account has access to this service.",
            404: "Check whether the embedding model is available to your account.",
            429: "The service limit was reached; try again later.",
        }
        print(f"HTTP {error.code}. " + hints.get(error.code, "The request was unsuccessful."), file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        print("Connection failed. Check your network and Python certificate setup.", file=sys.stderr)
        return 1
    except (ValueError, KeyError, IndexError, TypeError):
        print("The service returned an unexpected response.", file=sys.stderr)
        return 1

    print(f"Connection successful: received an embedding with {len(vector)} dimensions.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)
