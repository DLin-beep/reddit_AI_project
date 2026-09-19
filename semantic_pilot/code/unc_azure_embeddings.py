"""Small, credential-safe client for an explicitly selected Azure embedding model.

This module never reads credentials, writes files, or makes a request on import.
The caller chooses the deployment, tokenizer, batching, and persistence policy.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import errno
import http.client
import json
import math
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from numbers import Integral
from typing import Sequence

import numpy as np


DEFAULT_ENDPOINT = "https://azureaiapi.cloud.unc.edu/openai/v1/embeddings"
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ERROR_BODY_BYTES = 16_384
SAFE_ERROR_GUIDANCE = {
    "DeploymentNotFound": "Check the exact embedding deployment name in your Azure portal.",
    "model_not_found": "Check that the embedding deployment name is correct and available to your account.",
    "ResourceNotFound": "Check the embedding endpoint and deployment name.",
    "invalid_api_key": "Check the UNC_AI_API_KEY configured locally.",
}


class AzureEmbeddingError(RuntimeError):
    """An error safe to display without response text, credentials, or post text."""

    def __init__(self, message, *, category=None, attempts=None):
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # An authenticated POST must never forward its bearer key elsewhere.
        return None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _retry_delay(headers, attempt):
    """Honor Retry-After when valid, with an unconditional 60-second bound."""
    value = headers.get("Retry-After") if headers is not None else None
    delay = min(2 ** attempt, 60)
    if value is not None:
        try:
            proposed = float(value)
        except (TypeError, ValueError):
            try:
                when = email.utils.parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=dt.timezone.utc)
                proposed = (when - dt.datetime.now(dt.timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                proposed = delay
        if math.isfinite(proposed):
            delay = max(0, min(proposed, 60))
    return delay


def _http_error_message(error):
    """Only expose allowlisted codes; never include server messages or raw JSON."""
    message = f"Embedding API returned HTTP {error.code}"
    code = None
    try:
        raw = error.read(MAX_ERROR_BODY_BYTES + 1)
        if len(raw) <= MAX_ERROR_BODY_BYTES:
            document = json.loads(raw)
            if isinstance(document, dict):
                detail = document.get("error", document)
                if isinstance(detail, dict):
                    candidate = detail.get("code")
                    if isinstance(candidate, str) and candidate in SAFE_ERROR_GUIDANCE:
                        code = candidate
    except Exception:
        # Reading/parsing an error can fail and can itself include sensitive text.
        pass
    if code is not None:
        return f"{message} ({code}). {SAFE_ERROR_GUIDANCE[code]}"
    if error.code == 404:
        return f"{message}. Check the embedding endpoint and deployment name."
    return message


def _transport_error(error):
    """Classify only known exception types/codes, never their untrusted text.

    urllib wraps socket and TLS exceptions in URLError.reason. String reasons
    and unknown exceptions stay nonretryable; we do not infer from messages.
    """
    for _ in range(4):
        if not isinstance(error, urllib.error.URLError):
            break
        reason = error.reason
        if not isinstance(reason, BaseException) or reason is error:
            break
        error = reason
    if isinstance(error, ssl.SSLCertVerificationError):
        return ("tls_verification", False,
                "Embedding API TLS certificate verification failed. "
                "Check the local certificate setup; TLS verification remains enabled.")
    if isinstance(error, ssl.SSLError):
        return ("tls_failure", False,
                "Embedding API TLS connection failed. Check the local TLS or network setup.")
    if isinstance(error, socket.gaierror):
        if error.errno == socket.EAI_AGAIN:
            return ("dns_temporary", True,
                    "Embedding API hostname lookup temporarily failed")
        return ("dns_resolution", False,
                "Embedding API hostname could not be resolved. Check the endpoint and network connection.")
    if (isinstance(error, (TimeoutError, socket.timeout))
            or isinstance(error, OSError) and error.errno == errno.ETIMEDOUT):
        return ("timeout", True, "Embedding API request timed out before a complete response")
    if (isinstance(error, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError,
                           http.client.RemoteDisconnected, http.client.IncompleteRead))
            or isinstance(error, OSError)
            and error.errno in {errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE}):
        return ("connection_interrupted", True,
                "Embedding API connection was interrupted before a complete response")
    return ("transport_unknown", False,
            "Embedding API request failed before a complete response (unclassified transport error)")


class AzureEmbeddingClient:
    def __init__(self, endpoint, model, api_key, dimensions=None, timeout=60,
                 max_retries=3):
        try:
            parsed = urllib.parse.urlsplit(endpoint)
            valid_endpoint = (parsed.scheme == "https" and parsed.hostname
                              and not parsed.username and not parsed.password
                              and not parsed.query and not parsed.fragment)
            # Accessing the port validates malformed authority values as well.
            parsed.port
        except (TypeError, ValueError):
            valid_endpoint = False
        if not valid_endpoint:
            raise ValueError("endpoint must be an HTTPS URL without credentials, query, or fragment")
        if (not isinstance(model, str) or not model.strip()
                or any(ord(c) < 32 for c in model)):
            raise ValueError("An explicit nonempty embedding model/deployment name is required")
        if (not isinstance(api_key, str) or not api_key.strip()
                or any(ord(c) < 32 for c in api_key)):
            raise ValueError("A nonempty API key without control characters is required")
        if dimensions is not None:
            dimensions = _positive_integer(dimensions, "dimensions")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        if (isinstance(max_retries, bool) or not isinstance(max_retries, Integral)
                or not 0 <= max_retries <= 10):
            raise ValueError("max_retries must be an integer between 0 and 10")
        self.endpoint = endpoint
        self.model = model.strip()
        self._api_key = api_key.strip()
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = int(max_retries)
        self._opener = urllib.request.build_opener(_NoRedirect())

    @staticmethod
    def _validate_inputs(inputs):
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a nonempty batch of strings or token lists")
        if not 1 <= len(inputs) <= 2048:
            raise ValueError("inputs batch must contain 1 to 2048 items")
        if all(isinstance(item, str) for item in inputs):
            if any(not item for item in inputs):
                raise ValueError("Embedding input strings must not be empty")
            return list(inputs)
        if not all(isinstance(item, (list, tuple, np.ndarray)) for item in inputs):
            raise ValueError("Embedding inputs must use one consistent input type")
        result = []
        for item in inputs:
            if not 1 <= len(item) <= 8192:
                raise ValueError("Each token input must contain 1 to 8192 tokens")
            if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in item):
                raise ValueError("Token IDs must be nonnegative integers")
            result.append([int(x) for x in item])
        if sum(map(len, result)) > 300_000:
            raise ValueError("The embedding batch exceeds 300000 tokens")
        return result

    def embed(self, inputs):
        batch = self._validate_inputs(inputs)
        payload = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=body, method="POST",
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json", "Accept": "application/json"})
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    raw = response.read()
                    request_id = (response.headers.get("x-request-id")
                                  or response.headers.get("apim-request-id"))
            except urllib.error.HTTPError as error:
                status = error.code
                delay = _retry_delay(error.headers, attempt)
                if status in RETRYABLE_STATUSES and attempt < self.max_retries:
                    error.close()
                    time.sleep(delay)
                    continue
                try:
                    message = _http_error_message(error)
                finally:
                    error.close()
                raise AzureEmbeddingError(message) from None
            except Exception as error:
                # Retrying a request whose response was lost can repeat server
                # work. Retry only known transient failures, with a fixed bound.
                category, retryable, message = _transport_error(error)
                if retryable and attempt < self.max_retries:
                    time.sleep(_retry_delay(None, attempt))
                    continue
                if retryable:
                    count = attempt + 1
                    message += f" ({count} {'attempt' if count == 1 else 'attempts'})."
                raise AzureEmbeddingError(message, category=category,
                                          attempts=attempt + 1) from None
            try:
                document = json.loads(raw)
            except (ValueError, TypeError, UnicodeError):
                raise AzureEmbeddingError("Embedding API returned invalid JSON") from None
            return self._parse_response(document, len(batch), request_id)
        raise AzureEmbeddingError("Embedding API retry limit exceeded")

    def _parse_response(self, document, count, request_id):
        if not isinstance(document, dict):
            raise AzureEmbeddingError("Embedding API response must be an object")
        model = document.get("model")
        if (not isinstance(model, str) or not model.strip()
                or any(ord(c) < 32 for c in model)):
            raise AzureEmbeddingError("Embedding API response is missing a valid model name")
        rows = document.get("data")
        if not isinstance(rows, list) or len(rows) != count:
            raise AzureEmbeddingError("Embedding API returned an unexpected number of embeddings")
        ordered = [None] * count
        dimension = self.dimensions
        for row in rows:
            if not isinstance(row, dict):
                raise AzureEmbeddingError("Embedding API returned an invalid embedding record")
            index = row.get("index")
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < count or ordered[index] is not None):
                raise AzureEmbeddingError("Embedding API indices are incomplete, duplicated, or invalid")
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise AzureEmbeddingError("Embedding API returned an invalid embedding vector")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in embedding):
                raise AzureEmbeddingError("Embedding API returned nonnumeric vector values")
            try:
                vector = np.asarray(embedding, dtype=np.float64)
            except (ValueError, TypeError, OverflowError):
                raise AzureEmbeddingError("Embedding API returned an invalid embedding vector") from None
            if dimension is None:
                dimension = len(vector)
            if vector.ndim != 1 or len(vector) != dimension:
                raise AzureEmbeddingError("Embedding API returned inconsistent or unexpected dimensions")
            with np.errstate(over="ignore", invalid="ignore"):
                norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
                raise AzureEmbeddingError("Embedding API returned a nonfinite or zero embedding vector")
            ordered[index] = vector
        if any(x is None for x in ordered):
            raise AzureEmbeddingError("Embedding API returned incomplete indices")
        usage = document.get("usage", {})
        if not isinstance(usage, dict):
            raise AzureEmbeddingError("Embedding API returned invalid usage metadata")
        validated_usage = {}
        for key in ("prompt_tokens", "total_tokens"):
            if key in usage:
                value = usage[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AzureEmbeddingError("Embedding API returned invalid usage metadata")
                validated_usage[key] = value
        if request_id is not None and not isinstance(request_id, str):
            request_id = None
        return {"embeddings": np.stack(ordered), "model": model.strip(),
                "usage": validated_usage, "request_id": request_id}


def load_tokenizer(encoding_name="cl100k_base"):
    """Load the explicitly configured tokenizer; importing this module is lazy."""
    if not isinstance(encoding_name, str) or not encoding_name.strip():
        raise ValueError("An explicit tokenizer encoding name is required")
    try:
        import tiktoken
    except ImportError:
        raise RuntimeError("Install tiktoken to tokenize embedding inputs") from None
    return tiktoken.get_encoding(encoding_name)


def chunk_tokens(tokens: Sequence[int], max_tokens=8191, overlap=128):
    """Return overlapping token lists covering every input token without truncation."""
    max_tokens = _positive_integer(max_tokens, "max_tokens")
    if max_tokens > 8192:
        raise ValueError("max_tokens must not exceed the API limit of 8192")
    if (isinstance(overlap, bool) or not isinstance(overlap, Integral)
            or not 0 <= overlap < max_tokens):
        raise ValueError("overlap must be an integer from zero to max_tokens minus one")
    try:
        values = list(tokens)
    except TypeError:
        raise ValueError("tokens must be a sequence of nonnegative integer IDs") from None
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError("tokens must be a sequence of nonnegative integer IDs")
    values = [int(x) for x in values]
    chunks = []
    start = 0
    while start < len(values):
        end = min(start + max_tokens, len(values))
        chunks.append(values[start:end])
        if end == len(values):
            break
        start = end - overlap
    return chunks


def pool_post_embeddings(vectors, weights):
    """Normalize chunks, take their token-weighted mean, then normalize the result."""
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
        mass = np.asarray(weights, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Chunk embeddings and weights must be numeric") from None
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("Chunk embeddings must be a nonempty two-dimensional array")
    if mass.ndim != 1 or len(mass) != len(matrix):
        raise ValueError("A weight is required for every chunk embedding")
    if not np.isfinite(matrix).all() or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Chunk embeddings must be finite and weights finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_norms = np.linalg.norm(matrix, axis=1)
        total = mass.sum()
    if (not np.isfinite(row_norms).all() or np.any(row_norms <= 0)
            or not np.isfinite(total) or total <= 0):
        raise ValueError("Chunk embeddings cannot be pooled into a finite nonzero vector")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        unit_vectors = matrix / row_norms[:, None]
        pooled = np.sum(unit_vectors * (mass / total)[:, None], axis=0)
        norm = np.linalg.norm(pooled)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Chunk embeddings cannot be pooled into a finite nonzero vector")
    return pooled / norm
