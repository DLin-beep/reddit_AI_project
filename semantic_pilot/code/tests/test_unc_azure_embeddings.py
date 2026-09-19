"""Offline contract checks: mocked HTTP only, no credentials or live API calls."""
import copy
import errno
import http.client
import io
import json
from pathlib import Path
import socket
import ssl
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unc_azure_embeddings import (AzureEmbeddingClient, AzureEmbeddingError,
                                  DEFAULT_ENDPOINT, MAX_ERROR_BODY_BYTES,
                                  _NoRedirect, _retry_delay,
                                  chunk_tokens, load_tokenizer, pool_post_embeddings)


def document():
    return {"model": "actual-model-version",
            "data": [{"index": 1, "embedding": [0.0, 2.0]},
                     {"index": 0, "embedding": [1.0, 0.0]}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8}}


def response(payload=None, raw=None):
    result = MagicMock()
    result.__enter__.return_value = result
    result.read.return_value = json.dumps(payload if payload is not None else document()).encode() if raw is None else raw
    result.headers = {"apim-request-id": "test-request-id"}
    return result


def client(**kwargs):
    instance = AzureEmbeddingClient(DEFAULT_ENDPOINT, "deployment-alias",
                                    "test-credential-must-not-leak", **kwargs)
    instance._opener = MagicMock()
    instance._opener.open.return_value = response()
    return instance


def http_error(status, retry_after=None, raw=b"sensitive response body"):
    return urllib.error.HTTPError("https://example.invalid/secret-url", status,
                                  "sensitive error detail", {"Retry-After": retry_after},
                                  io.BytesIO(raw))


class ClientTests(unittest.TestCase):
    def test_request_auth_and_ordered_response_without_dimensions(self):
        api = client()
        actual = api.embed(["first text", "second text"])
        request = api._opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.headers["Authorization"], "Bearer test-credential-must-not-leak")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(payload, {"input": ["first text", "second text"],
                                   "model": "deployment-alias", "encoding_format": "float"})
        self.assertEqual(api._opener.open.call_args.kwargs["timeout"], 60)
        np.testing.assert_array_equal(actual["embeddings"], [[1, 0], [0, 2]])
        self.assertEqual(actual["model"], "actual-model-version")
        self.assertEqual(actual["usage"], {"prompt_tokens": 8, "total_tokens": 8})
        self.assertEqual(actual["request_id"], "test-request-id")

    def test_token_requests_and_explicit_dimensions(self):
        api = client(dimensions=2)
        api.embed([[1, np.int64(2)], [3]])
        payload = json.loads(api._opener.open.call_args.args[0].data)
        self.assertEqual(payload["input"], [[1, 2], [3]])
        self.assertEqual(payload["dimensions"], 2)

    def test_invalid_endpoints_and_required_model(self):
        for endpoint in ["http://example.com/embeddings", "https://key:secret@example.com/",
                         "https://example.com/?key=secret", "https://example.com/#secret",
                         "https://example.com:bad/", "", None]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                AzureEmbeddingClient(endpoint, "model", "key")
        for model in [None, "", "  ", "model\nname"]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                AzureEmbeddingClient(DEFAULT_ENDPOINT, model, "key")

    def test_invalid_configuration(self):
        for options in [{"dimensions": 0}, {"dimensions": 2.5}, {"timeout": float("nan")},
                        {"timeout": 0}, {"max_retries": -1}, {"max_retries": 11}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                client(**options)
        with self.assertRaises(ValueError):
            AzureEmbeddingClient(DEFAULT_ENDPOINT, "model", "key\nheader")

    def test_invalid_batches_do_not_call_transport(self):
        for batch in [[], "text", [""], ["a", [1]], [[]], [[-1]], [[True]],
                      [[1.5]], [[1] * 8193], ["x"] * 2049, [[1] * 8000] * 38]:
            api = client()
            with self.subTest(size=len(batch)), self.assertRaises(ValueError):
                api.embed(batch)
            api._opener.open.assert_not_called()

    def test_429_and_503_retry_with_bounded_sleep(self):
        api = client(max_retries=3)
        api._opener.open.side_effect = [http_error(429, "999999"), http_error(503, "2"), response()]
        with patch("unc_azure_embeddings.time.sleep") as sleep:
            api.embed(["a", "b"])
        self.assertEqual(api._opener.open.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 2])

    def test_retry_exhaustion_is_bounded_and_sanitized(self):
        api = client(max_retries=1)
        api._opener.open.side_effect = [http_error(500), http_error(500)]
        with patch("unc_azure_embeddings.time.sleep") as sleep:
            with self.assertRaises(AzureEmbeddingError) as caught:
                api.embed(["private post text", "b"])
        self.assertEqual(str(caught.exception), "Embedding API returned HTTP 500")
        self.assertEqual(api._opener.open.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_nonretryable_and_redirect_statuses_are_not_retried(self):
        for status in [400, 401, 403, 404, 301, 302, 307, 308]:
            api = client()
            api._opener.open.side_effect = http_error(status)
            with self.subTest(status=status), patch("unc_azure_embeddings.time.sleep") as sleep:
                with self.assertRaisesRegex(AzureEmbeddingError, f"HTTP {status}"):
                    api.embed(["private post text", "b"])
                api._opener.open.assert_called_once()
                sleep.assert_not_called()
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid"))

    def test_recognized_error_codes_have_safe_actionable_guidance(self):
        for code, status, guidance in [
            ("DeploymentNotFound", 404, "exact embedding deployment name"),
            ("model_not_found", 404, "available to your account"),
            ("ResourceNotFound", 404, "endpoint and deployment name"),
            ("invalid_api_key", 401, "UNC_AI_API_KEY configured locally"),
        ]:
            api = client()
            raw = json.dumps({"error": {"code": code,
                              "message": "private post text test-credential-must-not-leak"}}).encode()
            api._opener.open.side_effect = http_error(status, raw=raw)
            with self.subTest(code=code), self.assertRaises(AzureEmbeddingError) as caught:
                api.embed(["private post text", "b"])
            message = str(caught.exception)
            self.assertIn(f"HTTP {status} ({code})", message)
            self.assertIn(guidance, message)
            self.assertNotIn("private", message)
            self.assertNotIn("test-credential", message)
            self.assertTrue(caught.exception.__suppress_context__)
            api._opener.open.assert_called_once()

    def test_untrusted_error_bodies_and_codes_never_leak(self):
        for raw in [b"private post text test-credential-must-not-leak",
                    b'{"error":{"code":"private post text","message":"test-credential-must-not-leak"}}',
                    b'{"error":{"code":["DeploymentNotFound"]}}',
                    b'{"error":"private post text"}', b'[]',
                    b'{"error":{"code":"DeploymentNotFound"}}' + b" " * MAX_ERROR_BODY_BYTES]:
            api = client()
            error = http_error(404, raw=raw)
            error.read = MagicMock(wraps=error.read)
            api._opener.open.side_effect = error
            with self.subTest(raw=raw[:80]), self.assertRaises(AzureEmbeddingError) as caught:
                api.embed(["private post text", "b"])
            self.assertEqual(str(caught.exception),
                             "Embedding API returned HTTP 404. Check the embedding endpoint and deployment name.")
            error.read.assert_called_once_with(MAX_ERROR_BODY_BYTES + 1)

    def test_recognized_error_code_does_not_change_retry_policy(self):
        api = client(max_retries=1)
        error = http_error(429, "2", raw=b'{"error":{"code":"model_not_found"}}')
        api._opener.open.side_effect = [error, response()]
        with patch("unc_azure_embeddings.time.sleep") as sleep:
            api.embed(["a", "b"])
        self.assertEqual(api._opener.open.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_generic_transport_errors_are_sanitized_and_not_retried(self):
        api = client()
        api._opener.open.side_effect = urllib.error.URLError("test-credential-must-not-leak and private post text")
        with self.assertRaises(AzureEmbeddingError) as caught:
            api.embed(["private post text", "b"])
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("credential", str(caught.exception))
        self.assertEqual(caught.exception.category, "transport_unknown")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertTrue(caught.exception.__suppress_context__)
        api._opener.open.assert_called_once()

    def test_transient_transport_errors_recover_without_exposing_error_text(self):
        errors = [TimeoutError("private post text test-credential-must-not-leak"),
                  ConnectionResetError(errno.ECONNRESET, "private post text"),
                  ConnectionAbortedError(errno.ECONNABORTED, "private post text"),
                  BrokenPipeError(errno.EPIPE, "private post text"),
                  http.client.RemoteDisconnected("private post text"),
                  socket.gaierror(socket.EAI_AGAIN, "private post text")]
        for error in errors:
            for wrapped in [False, True]:
                api = client(max_retries=1)
                failure = urllib.error.URLError(error) if wrapped else error
                api._opener.open.side_effect = [failure, response()]
                with self.subTest(error=type(error).__name__, wrapped=wrapped), \
                        patch("unc_azure_embeddings.time.sleep") as sleep:
                    actual = api.embed(["private post text", "b"])
                np.testing.assert_array_equal(actual["embeddings"], [[1, 0], [0, 2]])
                self.assertEqual(api._opener.open.call_count, 2)
                sleep.assert_called_once_with(1)

    def test_transport_retry_exhaustion_has_safe_category_and_attempt_count(self):
        for error, category, text in [
            (TimeoutError("private post text test-credential-must-not-leak"), "timeout", "timed out"),
            (http.client.RemoteDisconnected("private post text"), "connection_interrupted", "interrupted"),
            (socket.gaierror(socket.EAI_AGAIN, "private post text"), "dns_temporary", "hostname lookup"),
        ]:
            api = client(max_retries=2)
            api._opener.open.side_effect = urllib.error.URLError(error)
            with self.subTest(category=category), patch("unc_azure_embeddings.time.sleep") as sleep:
                with self.assertRaises(AzureEmbeddingError) as caught:
                    api.embed(["private post text", "b"])
            self.assertEqual(api._opener.open.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
            self.assertEqual(caught.exception.category, category)
            self.assertEqual(caught.exception.attempts, 3)
            self.assertIn(text, str(caught.exception))
            self.assertIn("3 attempts", str(caught.exception))
            self.assertNotIn("private", str(caught.exception))
            self.assertNotIn("credential", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_read_interruption_retries_entire_response_and_closes_failed_response(self):
        for error in [http.client.IncompleteRead(b"private post text test-credential-must-not-leak", 500),
                      socket.timeout("private post text")]:
            api = client(max_retries=1)
            broken = response()
            broken.read.side_effect = error
            api._opener.open.side_effect = [broken, response()]
            with self.subTest(error=type(error).__name__), patch("unc_azure_embeddings.time.sleep") as sleep:
                actual = api.embed(["private post text", "b"])
            np.testing.assert_array_equal(actual["embeddings"], [[1, 0], [0, 2]])
            broken.__exit__.assert_called_once()
            self.assertEqual(api._opener.open.call_count, 2)
            sleep.assert_called_once_with(1)

    def test_read_interruption_exhaustion_never_exposes_partial_body(self):
        api = client(max_retries=0)
        broken = response()
        broken.read.side_effect = http.client.IncompleteRead(
            b"private post text test-credential-must-not-leak", 500)
        api._opener.open.return_value = broken
        with patch("unc_azure_embeddings.time.sleep") as sleep, self.assertRaises(AzureEmbeddingError) as caught:
            api.embed(["private post text", "b"])
        self.assertEqual(caught.exception.category, "connection_interrupted")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertIn("1 attempt", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("credential", str(caught.exception))
        sleep.assert_not_called()
        api._opener.open.assert_called_once()
        broken.__exit__.assert_called_once()

    def test_certificates_unknown_errors_and_permanent_dns_are_not_retried(self):
        failures = [
            (ssl.SSLCertVerificationError(1, "private post text test-credential-must-not-leak"), "tls_verification"),
            (ssl.SSLError(1, "private post text"), "tls_failure"),
            (socket.gaierror(socket.EAI_NONAME, "private post text"), "dns_resolution"),
            (RuntimeError("private post text test-credential-must-not-leak"), "transport_unknown"),
            (OSError(errno.EINVAL, "private post text"), "transport_unknown"),
        ]
        for error, category in failures:
            for wrapped in [False, True]:
                api = client(max_retries=3)
                api._opener.open.side_effect = urllib.error.URLError(error) if wrapped else error
                with self.subTest(category=category, wrapped=wrapped), \
                        patch("unc_azure_embeddings.time.sleep") as sleep:
                    with self.assertRaises(AzureEmbeddingError) as caught:
                        api.embed(["private post text", "b"])
                self.assertEqual(caught.exception.category, category)
                self.assertEqual(caught.exception.attempts, 1)
                self.assertNotIn("private", str(caught.exception))
                self.assertNotIn("credential", str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)
                api._opener.open.assert_called_once()
                sleep.assert_not_called()

    def test_invalid_json_does_not_expose_body(self):
        api = client()
        api._opener.open.return_value = response(raw=b"sensitive invalid response")
        with self.assertRaisesRegex(AzureEmbeddingError, "invalid JSON") as caught:
            api.embed(["a", "b"])
        self.assertNotIn("sensitive", str(caught.exception))

    def test_invalid_response_records(self):
        cases = []
        for change in [lambda p: p.pop("model"),
                       lambda p: p.update(model=""),
                       lambda p: p["data"][0].update(index=0),
                       lambda p: p["data"][0].update(index=2),
                       lambda p: p["data"][0].update(index=True),
                       lambda p: p["data"].pop(),
                       lambda p: p["data"][0].update(embedding=[0, 0]),
                       lambda p: p["data"][0].update(embedding=[float("nan"), 1]),
                       lambda p: p["data"][0].update(embedding=[float("inf"), 1]),
                       lambda p: p["data"][0].update(embedding=[1, 2, 3]),
                       lambda p: p["data"][0].update(embedding=["1", 2]),
                       lambda p: p["data"][0].update(embedding=[True, 2]),
                       lambda p: p.update(usage={"total_tokens": -1})]:
            value = copy.deepcopy(document())
            change(value)
            cases.append(value)
        cases.extend([[], {}, {"model": "model", "data": [None, None]}])
        for value in cases:
            api = client()
            api._opener.open.return_value = response(value)
            with self.subTest(value=value), self.assertRaises(AzureEmbeddingError):
                api.embed(["a", "b"])

    def test_requested_dimension_mismatch_rejected(self):
        api = client(dimensions=3)
        with self.assertRaisesRegex(AzureEmbeddingError, "dimensions"):
            api.embed(["a", "b"])

    def test_retry_after_fallback_and_bounds(self):
        self.assertEqual(_retry_delay({"Retry-After": "garbage"}, 2), 4)
        self.assertEqual(_retry_delay({"Retry-After": "-10"}, 2), 0)
        self.assertEqual(_retry_delay({"Retry-After": "NaN"}, 2), 4)
        self.assertEqual(_retry_delay({"Retry-After": "Wed, 01 Jan 2099 00:00:00 GMT"}, 2), 60)
        self.assertEqual(_retry_delay(None, 2), 4)


class ChunkAndPoolTests(unittest.TestCase):
    def test_chunks_cover_input_without_truncation_or_extra_tail(self):
        for length in [0, 1, 8, 9, 15, 16, 17, 100]:
            tokens = list(range(length))
            for overlap in [0, 1, 7]:
                chunks = chunk_tokens(tokens, max_tokens=8, overlap=overlap)
                rebuilt = chunks[0][:] if chunks else []
                for previous, following in zip(chunks, chunks[1:]):
                    if overlap:
                        self.assertEqual(previous[-overlap:], following[:overlap])
                    rebuilt.extend(following[overlap:])
                self.assertEqual(rebuilt, tokens)
                self.assertTrue(all(1 <= len(c) <= 8 for c in chunks))
                if len(chunks) > 1:
                    self.assertGreater(len(chunks[-1]), overlap)

    def test_invalid_chunk_parameters(self):
        for kwargs in [{"max_tokens": 0}, {"max_tokens": 8193}, {"max_tokens": 2, "overlap": 2},
                       {"overlap": -1}, {"overlap": 0.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chunk_tokens([1], **kwargs)
        with self.assertRaises(ValueError):
            chunk_tokens([-1])

    def test_token_weighted_pooling_and_normalization(self):
        actual = pool_post_embeddings([[1, 0], [0, 1]], [3, 4])
        np.testing.assert_allclose(actual, [.6, .8])
        self.assertAlmostEqual(np.linalg.norm(actual), 1)
        np.testing.assert_allclose(pool_post_embeddings([[0, 2]], [12]), [0, 1])

    def test_pooling_normalizes_each_chunk_before_weighting(self):
        actual = pool_post_embeddings([[10, 0], [0, 1]], [1, 1])
        np.testing.assert_allclose(actual, np.array([1, 1]) / np.sqrt(2))
        np.testing.assert_allclose(
            pool_post_embeddings([[10, 0], [0, .5]], [3, 4]), [.6, .8])

    def test_invalid_pooling_inputs_and_cancellation(self):
        for vectors, weights in [([], []), ([[1, 0]], []), ([[1, 0]], [0]),
                                  ([[1, 0]], [-1]), ([[1, 0]], [float("nan")]),
                                  ([[0, 0]], [1]), ([[float("inf"), 0]], [1]),
                                  ([[1, 0], [-1, 0]], [1, 1]),
                                  ([[1, 0], [1]], [1, 1])]:
            with self.subTest(vectors=vectors, weights=weights), self.assertRaises(ValueError):
                pool_post_embeddings(vectors, weights)

    def test_explicit_tokenizer_name_is_used(self):
        module = MagicMock()
        with patch.dict(sys.modules, {"tiktoken": module}):
            self.assertIs(load_tokenizer("explicit-encoding"), module.get_encoding.return_value)
        module.get_encoding.assert_called_once_with("explicit-encoding")


if __name__ == "__main__":
    unittest.main()
