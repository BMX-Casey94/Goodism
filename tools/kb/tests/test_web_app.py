"""Tests for the client review WebApp (server/web_app.py + web_ui.html).

The web layer is exercised through ``starlette.testclient.TestClient``. The
SQLite index is never built: retrieval and neighbour reads are replaced with
small fakes so the tests stay fast and hermetic, while the real
``tools.kb.retrieval`` merge/render path still runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from server import web_app
from tools.kb import config, indexer, retrieval
from tools.kb.hashing import sha256_text

TOKEN = "t" * 40
ORIGIN = "http://testserver"
CHUNK_TEXT = "Verbatim text from the corpus."


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path_factory):
    """Every test starts from a known environment.

    The feedback log is redirected unconditionally so no test can append to
    the operator's real ``review/client-feedback.jsonl``.
    """
    for name in (
        "KB_WEB_HOST",
        "KB_WEB_PORT",
        "KB_WEB_AUTH_TOKEN",
        "KB_WEB_SECURE_COOKIE",
        "KB_WEB_MAX_CONCURRENCY",
        "KB_WEB_FEEDBACK_PATH",
        "KB_LLM_API_KEY",
        "OPENAI_API_KEY",
        "KB_MODEL_STRONG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv_if_present", lambda: None)
    sandbox = tmp_path_factory.mktemp("review")
    monkeypatch.setattr(web_app, "FEEDBACK_PATH", sandbox / "client-feedback.jsonl")


@pytest.fixture
def client():
    with TestClient(web_app.create_app()) as test_client:
        yield test_client


def _hit(
    *,
    doc_id: str = "doc-a",
    ordinal: int = 0,
    score: float = 6.5,
    status: str = "published",
    provenance: dict | None = None,
) -> indexer.SearchHit:
    return indexer.SearchHit(
        doc_id=doc_id,
        title="Doc A",
        heading_path="Doc A / One",
        text=CHUNK_TEXT,
        score=score,
        authority="supporting",
        precedence=40,
        status=status,
        source="both",
        provenance=provenance,
        ordinal=ordinal,
    )


@pytest.fixture
def fake_retrieval(monkeypatch):
    """Patch the two retrieval entry points the web layer calls."""
    state = {"hits": [_hit()], "top_scores": [6.5], "calls": []}

    def fake_select_hits(queries, *, per_query=8, max_hits=16, per_doc_cap=6, guarantee=2, domain=None):
        state["calls"].append({"queries": list(queries), "max_hits": max_hits, "domain": domain})
        hits = state["hits"]
        matched = {(h.doc_id, h.ordinal): [0] for h in hits}
        fused = {(h.doc_id, h.ordinal): 1.0 / (config.RRF_K + 1) for h in hits}
        return hits, matched, fused, state["top_scores"]

    def fake_neighbours(doc_id, ordinal, before=1, after=1, db_path=None):
        lo = max(0, ordinal - max(0, before))
        hi = ordinal + max(0, after)
        return [
            {"ordinal": n, "heading_path": f"Doc A / Part {n}", "text": CHUNK_TEXT}
            for n in range(lo, hi + 1)
        ]

    monkeypatch.setattr(retrieval, "select_hits", fake_select_hits)
    monkeypatch.setattr(retrieval, "get_neighbours", fake_neighbours)
    return state


# ---------------------------------------------------------------------------
# Security headers and the shell page
# ---------------------------------------------------------------------------


def test_index_serves_html_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    body = response.text
    assert "Evidence Mode" in body and "Q&amp;A Mode" in body
    assert "Accuracy" in body


def test_security_headers_on_api_and_no_store_cache(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"]


def test_ui_has_no_external_dependencies_and_no_innerhtml():
    html = (Path(web_app.__file__).with_name("web_ui.html")).read_text(encoding="utf-8")
    lower = html.lower()
    assert "cdn." not in lower
    assert "unpkg" not in lower and "jsdelivr" not in lower
    assert "<script src" not in lower
    assert "<link rel=\"stylesheet\"" not in lower
    # Untrusted API text must never reach the DOM as HTML.
    assert "innerHTML" not in html
    assert "outerHTML" not in html
    assert "insertAdjacentHTML" not in html


# ---------------------------------------------------------------------------
# Health, status, documents, conflicts
# ---------------------------------------------------------------------------


def test_health_reports_no_qa_without_api_key(client):
    payload = client.get("/api/health").json()
    assert payload == {"ok": True, "auth_required": False, "qa_enabled": False}


def test_health_reports_qa_enabled_when_key_present(monkeypatch):
    monkeypatch.setenv("KB_LLM_API_KEY", "sk-not-a-real-key")
    with TestClient(web_app.create_app()) as test_client:
        payload = test_client.get("/api/health").json()
    assert payload["qa_enabled"] is True
    assert payload["auth_required"] is False


def test_status_reports_snapshot_and_documents(monkeypatch, client):
    monkeypatch.setattr(
        indexer,
        "index_status",
        lambda db_path=None: {
            "pinned_snapshot": 7,
            "created_at": "2026-11-20T14:06:21Z",
            "embedding_model": "local:test-model",
            "configured_embedding_model": "local:test-model",
            "documents": 3,
            "chunks": 99,
            "note": None,
        },
    )
    payload = client.get("/api/status").json()
    assert payload["pinned_snapshot"] == 7
    assert payload["documents"] == 3
    assert payload["qa_enabled"] is False
    assert "llm_model" not in payload


def test_documents_and_conflicts_endpoints(monkeypatch, client):
    monkeypatch.setattr(
        indexer,
        "list_indexed_documents",
        lambda db_path=None: [
            {"doc_id": "doc-a", "title": "Doc A", "domain": "goodism",
             "authority": "supporting", "summary": "s", "topics": [], "sections": 4}
        ],
    )
    monkeypatch.setattr(
        web_app,
        "confirmed_conflicts",
        lambda topic=None: [
            {"doc_id": "doc-a", "conflicts_with": "doc-b", "prevails": "doc-a",
             "reason": "r", "notes": ["n"]}
        ],
    )
    documents = client.get("/api/documents").json()["documents"]
    assert documents[0]["doc_id"] == "doc-a"
    conflicts = client.get("/api/conflicts", params={"topic": "goodism"}).json()["conflicts"]
    assert conflicts[0]["conflicts_with"] == "doc-b"


def test_status_failure_does_not_leak_paths(monkeypatch, client):
    def explode(db_path=None):
        raise RuntimeError(r"unable to open database file: E:\Goodism\index\kb.sqlite")

    monkeypatch.setattr(indexer, "index_status", explode)
    response = client.get("/api/status")
    assert response.status_code == 503
    body = response.text
    assert "kb.sqlite" not in body
    assert "E:\\" not in body
    assert response.json()["error"] == "the knowledge base index is unavailable"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_non_loopback_host_without_token_raises(monkeypatch):
    monkeypatch.setenv("KB_WEB_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError):
        web_app.create_app()


def test_non_loopback_host_with_short_token_raises(monkeypatch):
    monkeypatch.setenv("KB_WEB_HOST", "192.168.1.10")
    monkeypatch.setenv("KB_WEB_AUTH_TOKEN", "x" * 31)
    with pytest.raises(RuntimeError):
        web_app.create_app()


def test_loopback_host_allows_tokenless_local_use(monkeypatch):
    monkeypatch.setenv("KB_WEB_HOST", "127.0.0.1")
    app = web_app.create_app()
    assert app is not None


def test_loopback_ipv6_and_localhost_are_recognised():
    for host in ("127.0.0.1", "::1", "[::1]", "localhost", "127.0.0.5", "LOCALHOST"):
        assert web_app._is_loopback(host) is True
    for host in ("0.0.0.0", "::", "192.168.1.4", "kb.example.com", ""):
        assert web_app._is_loopback(host) is False


@pytest.fixture
def protected(monkeypatch):
    monkeypatch.setenv("KB_WEB_HOST", "0.0.0.0")
    monkeypatch.setenv("KB_WEB_AUTH_TOKEN", TOKEN)
    monkeypatch.setattr(
        indexer,
        "index_status",
        lambda db_path=None: {
            "pinned_snapshot": 1, "created_at": "2026-11-20T00:00:00Z",
            "embedding_model": "m", "configured_embedding_model": "m",
            "documents": 1, "chunks": 2, "note": None,
        },
    )
    with TestClient(web_app.create_app()) as test_client:
        yield test_client


def test_protected_status_requires_auth_then_accepts_bearer(protected):
    assert protected.get("/api/status").status_code == 401
    assert protected.get("/api/health").status_code == 200
    assert protected.get("/api/health").json()["auth_required"] is True

    ok = protected.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200 and ok.json()["pinned_snapshot"] == 1

    wrong = protected.get("/api/status", headers={"Authorization": "Bearer " + "z" * 40})
    assert wrong.status_code == 401


def test_login_sets_httponly_cookie_that_authenticates(protected):
    response = protected.post("/api/auth/login", json={"token": TOKEN})
    assert response.status_code == 200
    cookie_header = response.headers["set-cookie"]
    assert "HttpOnly" in cookie_header
    assert "SameSite=strict" in cookie_header.replace("samesite", "SameSite")
    assert "kb_session=" in cookie_header
    # The raw token must never be the cookie value.
    assert f"kb_session={TOKEN}" not in cookie_header

    follow = protected.get("/api/status")
    assert follow.status_code == 200

    bad = protected.post("/api/auth/login", json={"token": "wrong-token-value"})
    assert bad.status_code == 401


def test_secure_cookie_flag_follows_environment(monkeypatch, protected):
    monkeypatch.setenv("KB_WEB_SECURE_COOKIE", "1")
    with TestClient(web_app.create_app()) as test_client:
        response = test_client.post("/api/auth/login", json={"token": TOKEN})
    assert "Secure" in response.headers["set-cookie"]


def test_logout_clears_cookie(protected):
    protected.post("/api/auth/login", json={"token": TOKEN})
    assert protected.get("/api/status").status_code == 200
    logout = protected.post("/api/auth/logout")
    assert logout.status_code == 200
    assert protected.get("/api/status").status_code == 401


def test_cross_origin_cookie_post_is_refused(protected, feedback_log):
    protected.post("/api/auth/login", json={"token": TOKEN})
    response = protected.post(
        "/api/feedback",
        json={"question": "q", "mode": "evidence", "rating": "accurate",
              "comment": "", "doc_ids": []},
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403
    assert not feedback_log.exists()
    same = protected.post(
        "/api/feedback",
        json={"question": "q", "mode": "evidence", "rating": "accurate",
              "comment": "", "doc_ids": []},
        headers={"Origin": ORIGIN},
    )
    assert same.status_code == 200
    assert len(feedback_log.read_text(encoding="utf-8").splitlines()) == 1


# ---------------------------------------------------------------------------
# Evidence Mode
# ---------------------------------------------------------------------------


def test_evidence_returns_structured_passages_and_markdown(fake_retrieval, client):
    record = {
        "doc_id": "doc-a",
        "version": 1,
        "section_path": "Doc A / One",
        "txid": "a" * 64,
        "vout": 0,
        "section_sha256": sha256_text(CHUNK_TEXT),
        "published_at": "2026-11-01T00:00:00Z",
    }
    fake_retrieval["hits"] = [_hit(provenance=record)]
    fake_retrieval["top_scores"] = [6.5]

    response = client.post("/api/evidence", json={"queries": ["cestui que vie"], "context": 0})
    assert response.status_code == 200
    payload = response.json()

    assert payload["queries"] == ["cestui que vie"]
    assert payload["weak_match"] is False
    assert retrieval.DELIM_START in payload["markdown"]
    assert payload["markdown"].count(retrieval.DELIM_START) == 1

    passage = payload["passages"][0]
    for field in ("doc_id", "title", "authority", "precedence", "status", "caveat",
                  "sections", "start", "end", "text", "score", "weak_match",
                  "queries", "provenance"):
        assert field in passage, field
    assert passage["doc_id"] == "doc-a"
    assert passage["text"] == CHUNK_TEXT
    assert passage["weak_match"] is False
    assert passage["caveat"] is None
    assert passage["provenance"][0]["txid"] == "a" * 64

    assert fake_retrieval["calls"][0]["max_hits"] == 12
    assert fake_retrieval["calls"][0]["domain"] is None


def test_evidence_single_query_convenience_and_weak_match(fake_retrieval, client):
    fake_retrieval["hits"] = [_hit(score=-9.5)]
    fake_retrieval["top_scores"] = [-9.5]
    response = client.post("/api/evidence", json={"query": "weather in manchester"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["queries"] == ["weather in manchester"]
    assert payload["weak_match"] is True
    assert payload["passages"][0]["weak_match"] is True
    assert payload["query_scores"][0]["best_score"] == -9.5
    assert payload["low_relevance_threshold"] == retrieval.LOW_RELEVANCE_THRESHOLD


def test_evidence_surfaces_conflicts_for_matched_documents(fake_retrieval, client, monkeypatch):
    monkeypatch.setattr(
        web_app,
        "confirmed_conflicts",
        lambda topic=None: [
            {"doc_id": "doc-a", "conflicts_with": "doc-b", "prevails": "doc-a",
             "reason": "precedence 40 vs 60", "notes": ["checked by reviewer"]},
            {"doc_id": "doc-x", "conflicts_with": "doc-y", "prevails": "doc-x",
             "reason": "unrelated", "notes": []},
        ],
    )
    payload = client.post("/api/evidence", json={"queries": ["q"]}).json()
    assert [c["conflicts_with"] for c in payload["conflicts"]] == ["doc-b"]


def test_evidence_rejects_oversized_query(fake_retrieval, client):
    response = client.post("/api/evidence", json={"queries": ["x" * 513]})
    assert response.status_code == 400
    assert "512" in response.json()["error"]


def test_evidence_rejects_too_many_queries(fake_retrieval, client):
    response = client.post("/api/evidence", json={"queries": [f"q{i}" for i in range(9)]})
    assert response.status_code == 400
    assert "8" in response.json()["error"]

    ok = client.post("/api/evidence", json={"queries": [f"q{i}" for i in range(8)]})
    assert ok.status_code == 200


def test_evidence_rejects_empty_and_wrong_types(fake_retrieval, client):
    assert client.post("/api/evidence", json={"queries": []}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["   "]}).status_code == 400
    assert client.post("/api/evidence", json={"queries": "one string"}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["q"], "context": 5}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["q"], "context": True}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["q"], "max_hits": 0}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["q"], "max_hits": 21}).status_code == 400
    assert client.post("/api/evidence", json={"queries": ["q"], "domain": "../etc"}).status_code == 400


def test_body_larger_than_64kib_is_rejected(fake_retrieval, client):
    oversized = b"x" * (65 * 1024)
    response = client.post(
        "/api/evidence", content=oversized, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    # The rejection still carries the hardening headers.
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_malformed_json_is_rejected_without_traceback(client):
    response = client.post(
        "/api/evidence", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json() == {"error": "request body must be valid JSON"}


def test_retrieval_failure_is_generic(monkeypatch, client):
    def explode(*args, **kwargs):
        raise RuntimeError("embedding model missing at C:/models/x")

    monkeypatch.setattr(retrieval, "select_hits", explode)
    response = client.post("/api/evidence", json={"queries": ["q"]})
    assert response.status_code == 503
    assert "C:/models" not in response.text


# ---------------------------------------------------------------------------
# Read-around
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [-1, 10**30, 1_000_001])
def test_read_around_rejects_invalid_chunk(monkeypatch, client, chunk):
    monkeypatch.setattr(
        indexer, "get_neighbours", lambda *a, **k: pytest.fail("must not be called")
    )
    response = client.post("/api/read-around", json={"doc_id": "doc-a", "chunk": chunk})
    assert response.status_code == 400


def test_read_around_rejects_bad_doc_id_and_window(monkeypatch, client):
    for body in (
        {"doc_id": "../secret", "chunk": 0},
        {"doc_id": "doc-a", "chunk": 0, "before": 11},
        {"doc_id": "doc-a", "chunk": 0, "after": -1},
        {"doc_id": "doc-a"},
    ):
        assert client.post("/api/read-around", json=body).status_code == 400


def test_read_around_returns_neighbouring_chunks(monkeypatch, client):
    calls = {}

    def fake_neighbours(doc_id, ordinal, before=1, after=1, db_path=None):
        calls.update({"doc_id": doc_id, "ordinal": ordinal, "before": before, "after": after})
        return [
            {"ordinal": 3, "heading_path": "Doc A / One", "text": "first"},
            {"ordinal": 4, "heading_path": "Doc A / Two", "text": "second"},
        ]

    monkeypatch.setattr(indexer, "get_neighbours", fake_neighbours)
    response = client.post(
        "/api/read-around", json={"doc_id": "doc-a", "chunk": 4, "before": 1, "after": 0}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "doc-a"
    assert [c["ordinal"] for c in payload["chunks"]] == [3, 4]
    assert payload["chunks"][1]["heading_path"] == "Doc A / Two"
    assert calls == {"doc_id": "doc-a", "ordinal": 4, "before": 1, "after": 0}


def test_read_around_defaults_to_two_each_side(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(
        indexer,
        "get_neighbours",
        lambda doc_id, ordinal, before=1, after=1, db_path=None: seen.update(
            {"before": before, "after": after}
        ) or [],
    )
    assert client.post("/api/read-around", json={"doc_id": "doc-a", "chunk": 2}).status_code == 200
    assert seen == {"before": 2, "after": 2}


# ---------------------------------------------------------------------------
# Q&A Mode
# ---------------------------------------------------------------------------


def test_ask_returns_503_without_api_key(client):
    response = client.post("/api/ask", json={"question": "What is a trust?"})
    assert response.status_code == 503
    assert response.json() == {
        "error": (
            "Q&A mode requires KB_LLM_API_KEY in .env; "
            "use Evidence Mode to inspect verbatim passages."
        )
    }


def test_ask_returns_grounded_answer_with_mock_llm(monkeypatch, fake_retrieval):
    monkeypatch.setenv("KB_LLM_API_KEY", "sk-test-key-not-real")
    captured = {}

    class FakeLLM:
        def model_for(self, role):
            return "fake-strong-model"

        def chat(self, *, role, system, user, images=None, json_mode=False, model=None,
                 max_tokens=4096):
            captured["role"] = role
            captured["system"] = system
            captured["user"] = user
            return "## Direct answer\n\nA trust separates legal and beneficial ownership."

    monkeypatch.setattr(web_app, "LLMClient", FakeLLM)

    with TestClient(web_app.create_app()) as test_client:
        response = test_client.post("/api/ask", json={"question": "What is a trust?", "context": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("## Direct answer")
    assert payload["model"] == "fake-strong-model"
    assert payload["passages"][0]["doc_id"] == "doc-a"
    assert payload["weak_match"] is False
    assert "conflicts" in payload

    # The bot prompt and the fenced evidence pack both reach the model.
    assert "Goodism knowledge base" in captured["system"]
    assert captured["role"] == "strong"
    assert "What is a trust?" in captured["user"]
    assert retrieval.DELIM_START in captured["user"]
    assert "```" in captured["user"]


def test_ask_validates_question(monkeypatch, fake_retrieval):
    monkeypatch.setenv("KB_LLM_API_KEY", "sk-test-key-not-real")
    monkeypatch.setattr(web_app, "LLMClient", type("F", (), {
        "model_for": lambda self, role: "m",
        "chat": lambda self, **kwargs: "answer",
    }))
    with TestClient(web_app.create_app()) as test_client:
        assert test_client.post("/api/ask", json={"question": ""}).status_code == 400
        assert test_client.post("/api/ask", json={"question": "x" * 513}).status_code == 400
        assert test_client.post("/api/ask", json={"question": "ok", "queries": []}).status_code == 400


def test_ask_reports_llm_failure_as_bad_gateway(monkeypatch, fake_retrieval):
    monkeypatch.setenv("KB_LLM_API_KEY", "sk-test-key-not-real")

    class FailingLLM:
        def model_for(self, role):
            return "m"

        def chat(self, **kwargs):
            raise web_app.LLMError("LLM call failed for role=strong model=m: 401")

    monkeypatch.setattr(web_app, "LLMClient", FailingLLM)
    with TestClient(web_app.create_app()) as test_client:
        response = test_client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 502
    assert "sk-" not in response.text


# ---------------------------------------------------------------------------
# Accuracy review
# ---------------------------------------------------------------------------


@pytest.fixture
def feedback_log(tmp_path, monkeypatch):
    path = tmp_path / "review" / "client-feedback.jsonl"
    monkeypatch.setattr(web_app, "FEEDBACK_PATH", path)
    return path


def _feedback(**overrides) -> dict:
    body = {
        "question": "What is a cestui que vie trust?",
        "mode": "evidence",
        "rating": "accurate",
        "comment": "Passage matches the book.",
        "doc_ids": ["doc-a", "doc-b"],
    }
    body.update(overrides)
    return body


def test_feedback_writes_one_jsonl_line(feedback_log, client):
    response = client.post("/api/feedback", json=_feedback())
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    lines = feedback_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["rating"] == "accurate"
    assert record["mode"] == "evidence"
    assert record["doc_ids"] == ["doc-a", "doc-b"]
    assert record["comment"] == "Passage matches the book."
    assert record["at"].endswith("Z") and "T" in record["at"]

    client.post("/api/feedback", json=_feedback(rating="inaccurate", mode="qa"))
    assert len(feedback_log.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("rating", ["Accurate", "great", "", None, 5])
def test_feedback_rejects_rating_outside_allowlist(feedback_log, client, rating):
    response = client.post("/api/feedback", json=_feedback(rating=rating))
    assert response.status_code == 400
    assert "rating" in response.json()["error"]
    assert not feedback_log.exists()


@pytest.mark.parametrize("mode", ["Evidence", "chat", "", None])
def test_feedback_rejects_unknown_mode(feedback_log, client, mode):
    assert client.post("/api/feedback", json=_feedback(mode=mode)).status_code == 400


def test_feedback_rejects_oversized_fields(feedback_log, client):
    assert client.post("/api/feedback", json=_feedback(comment="c" * 2001)).status_code == 400
    assert client.post("/api/feedback", json=_feedback(question="q" * 513)).status_code == 400
    assert client.post(
        "/api/feedback", json=_feedback(doc_ids=["d"] * 21)
    ).status_code == 400
    assert client.post(
        "/api/feedback", json=_feedback(doc_ids=["d" * 101])
    ).status_code == 400
    assert client.post("/api/feedback", json=_feedback(question="   ")).status_code == 400
    assert not feedback_log.exists()


def test_feedback_deduplicates_doc_ids_and_allows_empty(feedback_log, client):
    assert client.post(
        "/api/feedback", json=_feedback(doc_ids=["doc-a", "doc-a", "doc-b"])
    ).status_code == 200
    assert client.post("/api/feedback", json=_feedback(doc_ids=[])).status_code == 200
    records = [json.loads(line) for line in feedback_log.read_text("utf-8").splitlines()]
    assert records[0]["doc_ids"] == ["doc-a", "doc-b"]
    assert records[1]["doc_ids"] == []


def test_feedback_log_failure_is_generic(monkeypatch, client):
    monkeypatch.setattr(
        web_app, "_append_feedback", lambda record: (_ for _ in ()).throw(OSError("E:\\denied"))
    )
    response = client.post("/api/feedback", json=_feedback())
    assert response.status_code == 503
    assert "denied" not in response.text
