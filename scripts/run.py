import sys
import json
import gc
import os
import random
import re
import hashlib
import webbrowser
import argparse
import shutil
import socket
from pathlib import Path
from datetime import datetime
from threading import Timer
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from flask import Flask, render_template_string, jsonify

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_ROOT = ROOT / "output"
OUTPUT_DIR = OUTPUT_ROOT / "latest"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
ACSI_SCORE_PATH = DATA_DIR / "acsi_scores.csv"
LEGACY_GSE_SCORE_PATH = DATA_DIR / "gse_scores.csv"
GSE_PATH = ACSI_SCORE_PATH  # Backward-compatible alias for older notebooks/scripts.
POSTS_ECOSYSTEM_PATH = OUTPUT_DIR / "posts_clean_ecosystem.parquet"
POSTS_CREATOR_PATH = OUTPUT_DIR / "posts_clean_all.parquet"
POSTS_SURVIVOR_PATH = OUTPUT_DIR / "posts_clean_surv.parquet"
PANEL_PATH = OUTPUT_DIR / "panel_all.parquet"
SUBMONTH_PANEL_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.parquet"
CREATORS_PATH = OUTPUT_DIR / "creators.parquet"
SUBMONTH_PANEL_META_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.meta.json"
POST_MONTHLY_AGG_PATH = CACHE_DIR / "post_monthly_aggregates.parquet"
POST_MONTHLY_AGG_META_PATH = CACHE_DIR / "post_monthly_aggregates.meta.json"
COMMENT_AUTHOR_MONTHLY_PATH = CACHE_DIR / "comment_author_monthly.parquet"
COMMENT_AUTHOR_MONTHLY_META_PATH = CACHE_DIR / "comment_author_monthly.meta.json"
COMMENT_KEYWORD_MONTHLY_PATH = CACHE_DIR / "comment_keyword_monthly.parquet"
COMMENT_KEYWORD_MONTHLY_META_PATH = CACHE_DIR / "comment_keyword_monthly.meta.json"
DEFAULT_DASHBOARD_PORT = 8000
SUBMONTH_PANEL_CACHE_VERSION = 1
POST_MONTHLY_AGG_CACHE_VERSION = 1
COMMENT_AUTHOR_MONTHLY_CACHE_VERSION = 1
COMMENT_KEYWORD_MONTHLY_CACHE_VERSION = 1

ACTIVE_SUBREDDITS = None
MAX_LINES_PER_FILE = None

def validate_generated_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_root = OUTPUT_ROOT.resolve()
    resolved = output_dir.resolve()
    if resolved == output_root or output_root not in resolved.parents:
        raise ValueError(
            f"Refusing unsafe output directory: {output_dir}. "
            f"Use a subdirectory under {OUTPUT_ROOT}."
        )
    return output_dir

def configure_output_dir(output_dir):
    global OUTPUT_DIR, TABLES_DIR, FIGURES_DIR
    global POSTS_ECOSYSTEM_PATH, POSTS_CREATOR_PATH, POSTS_SURVIVOR_PATH
    global PANEL_PATH, SUBMONTH_PANEL_PATH, CREATORS_PATH, SUBMONTH_PANEL_META_PATH

    OUTPUT_DIR = validate_generated_output_dir(output_dir)
    TABLES_DIR = OUTPUT_DIR / "tables"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    POSTS_ECOSYSTEM_PATH = OUTPUT_DIR / "posts_clean_ecosystem.parquet"
    POSTS_CREATOR_PATH = OUTPUT_DIR / "posts_clean_all.parquet"
    POSTS_SURVIVOR_PATH = OUTPUT_DIR / "posts_clean_surv.parquet"
    PANEL_PATH = OUTPUT_DIR / "panel_all.parquet"
    SUBMONTH_PANEL_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.parquet"
    SUBMONTH_PANEL_META_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.meta.json"
    CREATORS_PATH = OUTPUT_DIR / "creators.parquet"

    for _d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, DATA_DIR]:
        _d.mkdir(exist_ok=True, parents=True)

configure_output_dir(OUTPUT_DIR)

def ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

def clean_generated_output(output_dir):
    output_dir = validate_generated_output_dir(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    # Older versions wrote run artifacts directly under output/. Remove only
    # those known generated paths so output/latest is the visible current run.
    if output_dir.resolve() == (OUTPUT_ROOT / "latest").resolve():
        for path in [
            OUTPUT_ROOT / "dashboard.html",
            OUTPUT_ROOT / "subreddit_month_gse_panel.parquet",
            OUTPUT_ROOT / "subreddit_month_gse_panel.meta.json",
            OUTPUT_ROOT / "posts_clean_ecosystem.parquet",
            OUTPUT_ROOT / "posts_clean_all.parquet",
            OUTPUT_ROOT / "posts_clean_surv.parquet",
            OUTPUT_ROOT / "panel_all.parquet",
            OUTPUT_ROOT / "creators.parquet",
            OUTPUT_ROOT / "tables",
            OUTPUT_ROOT / "figures",
        ]:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    configure_output_dir(output_dir)

def clean_analysis_artifacts():
    validate_generated_output_dir(OUTPUT_DIR)
    for generated_dir in [TABLES_DIR, FIGURES_DIR]:
        if generated_dir.exists():
            shutil.rmtree(generated_dir, ignore_errors=True)
        generated_dir.mkdir(exist_ok=True, parents=True)

def find_open_dashboard_port(start_port=DEFAULT_DASHBOARD_PORT, max_tries=1000):
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except PermissionError:
                raise
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"No open localhost dashboard port found from {start_port} "
        f"to {start_port + max_tries - 1}."
    )

# --- Configuration & Hyperparameters ---
QUICK_MODE = False
N_RANDOMIZATION_PERMS = 250 if QUICK_MODE else 1000

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)
END_DATE_EXCLUSIVE = datetime(2025, 1, 1)

EXACT_SHOCK_DATE = datetime(2022, 11, 30)
SHOCK_MONTH = pd.Timestamp("2022-12-01")

MIN_PRE_POSTS = 5
MAX_POSTS_PER_DAY = 50
RANDOM_SEED = 42
ACSI_RELIABILITY_TARGET = 50

INDEX_LABEL = "AI Content Substitutability Index"
INDEX_SHORT = "ACSI"
H1_LABEL = "H1: Uniform productivity enhancement"
H2_LABEL = "H2: Substitutability-shaped transformation"
H1_PREDICTION = "ACSI x Post coefficient is near zero with tight uncertainty"
H2_PREDICTION = "ACSI x Post coefficient is negative for log monthly posts"

ACSI_COMPONENT_COLUMNS = [
    "direct_gen", "usefulness", "quality_comp", "physical_req", "personal_req",
]
ACSI_SCORE_METADATA_COLUMNS = [
    "n_coded", "n_used", "n_ai_related_excluded", "n_hard_cases",
    "score_reliability", "low_n_flag", "hard_case_policy",
]
ACSI_MERGE_COLUMNS = [
    "subreddit", "gse", "raw_gse", *ACSI_COMPONENT_COLUMNS,
    "physical_free", "non_personal", *ACSI_SCORE_METADATA_COLUMNS,
]
ACSI_DIMENSION_SPECS = [
    {
        "source": "direct_gen",
        "norm": "direct_gen_norm",
        "post": "direct_gen_post",
        "label": "Direct generation",
        "description": "GenAI can directly produce the requested answer or artifact.",
    },
    {
        "source": "usefulness",
        "norm": "usefulness_norm",
        "post": "usefulness_post",
        "label": "Usefulness",
        "description": "GenAI would be materially useful for the poster's task.",
    },
    {
        "source": "quality_comp",
        "norm": "quality_comp_norm",
        "post": "quality_comp_post",
        "label": "Quality competitiveness",
        "description": "GenAI output is competitive with a knowledgeable human response.",
    },
    {
        "source": "physical_free",
        "norm": "physical_free_norm",
        "post": "physical_free_post",
        "label": "Low physical constraint",
        "description": "The task is less tied to physical materials, inspection, body, or place.",
    },
    {
        "source": "non_personal",
        "norm": "non_personal_norm",
        "post": "non_personal_post",
        "label": "Low personal-context need",
        "description": "The task depends less on the poster's personal situation or lived context.",
    },
]
ACSI_MECHANISM_SPECS = [
    {
        "source": "generation_capability",
        "norm": "generation_capability_norm",
        "post": "generation_capability_post",
        "label": "Generation capability",
        "description": "Average of direct generation, usefulness, and quality competitiveness.",
    },
    {
        "source": "physical_free",
        "norm": "physical_free_norm",
        "post": "physical_free_post",
        "label": "Low physical constraint",
        "description": "The task is less tied to physical materials, inspection, body, or place.",
    },
    {
        "source": "non_personal",
        "norm": "non_personal_norm",
        "post": "non_personal_post",
        "label": "Low personal-context need",
        "description": "The task depends less on the poster's personal situation or lived context.",
    },
]

EXCLUDED_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

AI_BROAD_PATTERN = re.compile(
    r"(?:\bchatgpt\b|\bgpt\b|\bgpt-4\b|\bopenai\b|\bmidjourney\b|"
    r"\bstable diffusion\b|\bdall[- ]?e\b|\bdalle\b|"
    r"\bllm\b|\blanguage model\b|\bgenerative ai\b|"
    r"\bai[- ]?generated\b|\bai art\b|\bprompt\b|\bbot\b|\bai\b)",
    re.IGNORECASE,
)
AI_TOOL_PATTERN = re.compile(
    r"(?:\bchatgpt\b|\bgpt\b|\bgpt-4\b|\bopenai\b|\bmidjourney\b|"
    r"\bstable diffusion\b|\bdall[- ]?e\b|\bdalle\b|\bllm\b)",
    re.IGNORECASE,
)

SUBREDDITS = {
    "writing": "treatment", "worldbuilding": "treatment", "shortstories": "treatment",
    "screenwriting": "treatment", "poetry": "treatment", "fanfiction": "treatment",
    "songwriting": "treatment", "art": "treatment", "illustration": "treatment",
    "conceptart": "treatment", "comics": "treatment", "digitalart": "treatment",
    "graphic_design": "treatment", "gamedev": "treatment", "applyingtocollege": "treatment",
    "gre": "treatment", "lsat": "treatment", "mcat": "treatment", "sat": "treatment",
    "woodworking": "control", "pottery": "control", "sewing": "control",
    "baking": "control", "cooking": "control", "knitting": "control",
    "breadit": "control", "carpentry": "control", "leathercraft": "control",
    "quilting": "control", "ceramics": "control", "photography": "control",
    "askphotography": "control", "learnart": "control", "learntodraw": "control",
    "chanceme": "control", "college": "control", "gradschool": "control",
    "lawschool": "control", "medicalschool": "control", "phd": "control",
    "premed": "control", "machinelearning": "ambiguous", "learnprogramming": "ambiguous",
    "learnmath": "ambiguous", "cscareerquestions": "ambiguous", "askacademia": "ambiguous",
    "fantasywriters": "treatment", "scifiwriting": "treatment", "fiction": "treatment",
    "books": "control", "3Dmodeling": "treatment",
    "personalstatement": "treatment", "resume": "treatment", "devops": "treatment",
    "fermentation": "control", "gardening": "control", "homebrewing": "control",
    "plants": "control", "chess": "control", "programminghumor": "ambiguous",
    "rowing": "ambiguous", "running": "ambiguous",
    "swimming": "ambiguous", "solotravel": "ambiguous", "travel": "ambiguous",
}

MU_K = {
    "art": 22.4, "cooking": 6.1, "woodworking": 6.0, "photography": 5.6,
    "learnprogramming": 4.4, "baking": 4.4, "comics": 3.8, "writing": 3.4,
    "learntodraw": 3.1, "machinelearning": 3.0, "graphic_design": 2.9,
    "gamedev": 2.0, "college": 2.9, "cscareerquestions": 2.3, "sewing": 2.2,
    "poetry": 2.2, "askacademia": 2.1, "worldbuilding": 1.9, "screenwriting": 1.8,
    "learnart": 1.7, "applyingtocollege": 1.3, "breadit": 1.2, "songwriting": 1.0,
    "illustration": 0.9, "gradschool": 0.9, "lawschool": 0.9, "premed": 0.9,
    "leathercraft": 0.87, "medicalschool": 0.8, "digitalart": 0.76,
    "askphotography": 0.72, "sat": 0.66, "knitting": 0.6, "carpentry": 0.6,
    "fanfiction": 0.4, "learnmath": 0.44, "pottery": 0.26, "phd": 0.26,
    "quilting": 0.29, "mcat": 0.3, "conceptart": 0.15, "shortstories": 0.11,
    "ceramics": 0.16, "gre": 0.12, "chanceme": 0.12, "lsat": 0.2,
    "books": 27.1, "travel": 14.3, "gardening": 8.8,
    "programminghumor": 4.7, "solotravel": 4.5,
    "running": 4.2, "fantasywriters": 3.9,
    "3Dmodeling": 1.5, "resume": 1.3, "homebrewing": 1.2, "chess": 1.8,
    "scifiwriting": 0.1, "swimming": 0.6, "plants": 0.54, "devops": 0.45,
    "fermentation": 0.3, "rowing": 0.13, "fiction": 0.014, "personalstatement": 0.003,
}

COMMUNITY_TYPES = {
    "text": {"writing", "worldbuilding", "shortstories", "screenwriting", "poetry",
             "fanfiction", "songwriting", "fantasywriters", "scifiwriting", "fiction"},
    "image": {"art", "illustration", "conceptart", "comics", "digitalart",
              "graphic_design", "gamedev", "3Dmodeling"},
    "academia": {"applyingtocollege", "gre", "lsat", "mcat", "sat",
                 "personalstatement", "resume", "devops"},
}

MATCHED_CONTROLS = {
    "text": {"chanceme", "college", "gradschool", "lawschool", "medicalschool",
             "phd", "premed", "books"},
    "image": {"woodworking", "pottery", "sewing", "baking", "cooking", "knitting",
              "breadit", "carpentry", "leathercraft", "quilting", "ceramics",
              "fermentation", "gardening", "homebrewing", "plants", "photography",
              "askphotography", "learnart", "learntodraw"},
    "academia": {"chanceme", "college", "gradschool", "lawschool", "medicalschool",
                 "phd", "premed"},
}

TREATMENT_SUBS = {s for s, r in SUBREDDITS.items() if r == "treatment"}
CONTROL_SUBS   = {s for s, r in SUBREDDITS.items() if r == "control"}
AMBIGUOUS_SUBS = {s for s, r in SUBREDDITS.items() if r == "ambiguous"}

SMOKE_SUBREDDITS = [
    "personalstatement", "fiction", "scifiwriting", "conceptart",
    "ceramics", "fermentation", "programminghumor", "rowing",
]

ALL_MONTHS  = pd.date_range(
    pd.Timestamp(START_DATE).replace(day=1),
    pd.Timestamp(END_DATE).replace(day=1),
    freq="MS",
)
PRE_MONTHS  = ALL_MONTHS[ALL_MONTHS < SHOCK_MONTH]
POST_MONTHS = ALL_MONTHS[ALL_MONTHS >= SHOCK_MONTH]
N_PRE_MONTHS  = int(len(PRE_MONTHS))
N_POST_MONTHS = int(len(POST_MONTHS))

# --- Utils ---
def iter_subreddits():
    if ACTIVE_SUBREDDITS is None:
        return list(SUBREDDITS.keys())
    return [s for s in ACTIVE_SUBREDDITS if s in SUBREDDITS]

def iter_post_records(sub, desc):
    path = DATA_DIR / f"r_{sub}_posts.jsonl"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=desc)):
            if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                break
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except Exception:
                continue

            ts = p.get("created_utc")
            if ts is None:
                continue
            try:
                dt = datetime.utcfromtimestamp(int(ts))
            except Exception:
                continue
            if dt < START_DATE or dt >= END_DATE_EXCLUSIVE:
                continue

            author = str(p.get("author") or "")
            if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                continue

            score = p.get("score")
            try:
                score = int(score) if score is not None else 0
            except Exception:
                score = 0

            yield author, dt, score, str(p.get("id") or "")

def post_file_signature(sub):
    path = DATA_DIR / f"r_{sub}_posts.jsonl"
    st = path.stat()
    return {
        "subreddit": sub,
        "path": str(path.relative_to(ROOT)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

def comment_file_signature(sub):
    path = DATA_DIR / f"r_{sub}_comments.jsonl"
    st = path.stat()
    return {
        "subreddit": sub,
        "path": str(path.relative_to(ROOT)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

def file_signature(path):
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

def hash_values(values):
    h = hashlib.sha256()
    n = 0
    for value in sorted(values):
        h.update(str(value).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        n += 1
    return h.hexdigest(), n

def available_post_subreddits():
    return [sub for sub in iter_subreddits() if (DATA_DIR / f"r_{sub}_posts.jsonl").exists()]

def available_comment_subreddits():
    return [sub for sub in iter_subreddits() if (DATA_DIR / f"r_{sub}_comments.jsonl").exists()]

def use_persistent_raw_cache():
    return ACTIVE_SUBREDDITS is None and MAX_LINES_PER_FILE is None

def resolve_acsi_score_path():
    """Prefer the clearer ACSI filename, but keep the old score file readable."""
    if ACSI_SCORE_PATH.exists():
        return ACSI_SCORE_PATH
    if LEGACY_GSE_SCORE_PATH.exists():
        return LEGACY_GSE_SCORE_PATH
    return ACSI_SCORE_PATH

def submonth_panel_cache_metadata(apply_author_cap):
    return {
        "cache_version": SUBMONTH_PANEL_CACHE_VERSION,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date_exclusive": END_DATE_EXCLUSIVE.strftime("%Y-%m-%d"),
        "shock_month": SHOCK_MONTH.strftime("%Y-%m-%d"),
        "max_lines_per_file": MAX_LINES_PER_FILE,
        "active_subreddits": None if ACTIVE_SUBREDDITS is None else list(ACTIVE_SUBREDDITS),
        "apply_author_cap": bool(apply_author_cap),
        "post_files": [post_file_signature(sub) for sub in available_post_subreddits()],
    }

def submonth_panel_cache_is_current(apply_author_cap):
    if not SUBMONTH_PANEL_PATH.exists() or not SUBMONTH_PANEL_META_PATH.exists():
        return False
    try:
        old = json.loads(SUBMONTH_PANEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        return old == submonth_panel_cache_metadata(apply_author_cap)
    except FileNotFoundError:
        return False

def write_submonth_panel_cache_metadata(apply_author_cap):
    metadata = submonth_panel_cache_metadata(apply_author_cap)
    SUBMONTH_PANEL_META_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

def post_monthly_agg_cache_metadata(apply_author_cap, n_ecosystem_authors=None):
    metadata = {
        "cache_version": POST_MONTHLY_AGG_CACHE_VERSION,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date_exclusive": END_DATE_EXCLUSIVE.strftime("%Y-%m-%d"),
        "shock_month": SHOCK_MONTH.strftime("%Y-%m-%d"),
        "max_lines_per_file": MAX_LINES_PER_FILE,
        "active_subreddits": None if ACTIVE_SUBREDDITS is None else list(ACTIVE_SUBREDDITS),
        "apply_author_cap": bool(apply_author_cap),
        "post_files": [post_file_signature(sub) for sub in available_post_subreddits()],
    }
    if n_ecosystem_authors is not None:
        metadata["n_ecosystem_authors"] = int(n_ecosystem_authors)
    return metadata

def post_monthly_agg_cache_is_current(apply_author_cap):
    if not POST_MONTHLY_AGG_PATH.exists() or not POST_MONTHLY_AGG_META_PATH.exists():
        return False
    try:
        old = json.loads(POST_MONTHLY_AGG_META_PATH.read_text(encoding="utf-8"))
        expected = post_monthly_agg_cache_metadata(apply_author_cap)
    except Exception:
        return False
    old_without_runtime = {k: v for k, v in old.items() if k != "n_ecosystem_authors"}
    return old_without_runtime == expected

def write_post_monthly_agg_cache_metadata(apply_author_cap, n_ecosystem_authors):
    ensure_cache_dir()
    metadata = post_monthly_agg_cache_metadata(apply_author_cap, n_ecosystem_authors)
    POST_MONTHLY_AGG_META_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

def read_cached_n_ecosystem_authors():
    try:
        metadata = json.loads(POST_MONTHLY_AGG_META_PATH.read_text(encoding="utf-8"))
        return safe_int(metadata.get("n_ecosystem_authors"))
    except Exception:
        return None

def comment_keyword_cache_metadata():
    return {
        "cache_version": COMMENT_KEYWORD_MONTHLY_CACHE_VERSION,
        "start_date": "2022-09-01",
        "end_date_exclusive": "2023-03-01",
        "max_lines_per_file": MAX_LINES_PER_FILE,
        "active_subreddits": None if ACTIVE_SUBREDDITS is None else list(ACTIVE_SUBREDDITS),
        "broad_pattern": AI_BROAD_PATTERN.pattern,
        "tool_pattern": AI_TOOL_PATTERN.pattern,
        "comment_files": [comment_file_signature(sub) for sub in available_comment_subreddits()],
    }

def comment_keyword_cache_is_current():
    if not COMMENT_KEYWORD_MONTHLY_PATH.exists() or not COMMENT_KEYWORD_MONTHLY_META_PATH.exists():
        return False
    try:
        old = json.loads(COMMENT_KEYWORD_MONTHLY_META_PATH.read_text(encoding="utf-8"))
        return old == comment_keyword_cache_metadata()
    except Exception:
        return False

def write_comment_keyword_cache_metadata():
    ensure_cache_dir()
    COMMENT_KEYWORD_MONTHLY_META_PATH.write_text(
        json.dumps(comment_keyword_cache_metadata(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def comment_author_cache_metadata(valid_authors):
    author_hash, n_authors = hash_values(valid_authors)
    return {
        "cache_version": COMMENT_AUTHOR_MONTHLY_CACHE_VERSION,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date_exclusive": END_DATE_EXCLUSIVE.strftime("%Y-%m-%d"),
        "shock_month": SHOCK_MONTH.strftime("%Y-%m-%d"),
        "max_lines_per_file": MAX_LINES_PER_FILE,
        "active_subreddits": None if ACTIVE_SUBREDDITS is None else list(ACTIVE_SUBREDDITS),
        "valid_author_count": n_authors,
        "valid_author_hash": author_hash,
        "comment_files": [comment_file_signature(sub) for sub in available_comment_subreddits()],
    }

def comment_author_cache_is_current(valid_authors):
    if not COMMENT_AUTHOR_MONTHLY_PATH.exists() or not COMMENT_AUTHOR_MONTHLY_META_PATH.exists():
        return False
    try:
        old = json.loads(COMMENT_AUTHOR_MONTHLY_META_PATH.read_text(encoding="utf-8"))
        return old == comment_author_cache_metadata(valid_authors)
    except Exception:
        return False

def write_comment_author_cache_metadata(valid_authors):
    ensure_cache_dir()
    COMMENT_AUTHOR_MONTHLY_META_PATH.write_text(
        json.dumps(comment_author_cache_metadata(valid_authors), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def safe_float(x):
    try:
        if x is None: return None
        v = float(x)
        if np.isnan(v) or np.isinf(v): return None
        return v
    except Exception:
        return None

def safe_int(x):
    try:
        if x is None: return None
        if isinstance(x, float) and np.isnan(x): return None
        return int(x)
    except Exception:
        return None

def fmt4(x):
    return "NA" if x is None else f"{x:.4f}"

def fmt_signed4(x):
    return "NA" if x is None else f"{x:+.4f}"

def fmt_signed1(x):
    return "NA" if x is None else f"{x:+.1f}"

def pct_effect_from_coef(coef):
    if coef is None:
        return None
    return safe_float(100 * (np.exp(coef) - 1))

def classify_substitutability_hypothesis(result, model_name, alpha=0.05):
    coef = None if result is None else result.get("coef")
    pvalue = None if result is None else result.get("pvalue")
    effect = None if result is None else result.get("percent_effect_full_exposure")

    if coef is None or pvalue is None:
        conclusion = "Inconclusive"
        interpretation = (
            "The model did not return a usable ACSI x Post estimate, so it cannot "
            "adjudicate H1 versus H2."
        )
        supported = "Inconclusive"
    elif pvalue < alpha and coef < 0:
        conclusion = "H2-consistent"
        interpretation = (
            "Higher-substitutability communities contract relative to lower-"
            "substitutability communities after ChatGPT, matching the predicted "
            "substitutability-shaped transformation."
        )
        supported = H2_LABEL
    elif pvalue < alpha:
        conclusion = "Opposite gradient"
        interpretation = (
            "Higher-substitutability communities expand relative to lower-"
            "substitutability communities after ChatGPT, which is evidence against "
            "the predicted displacement direction."
        )
        supported = "Neither H1 nor predicted H2"
    elif coef < 0:
        conclusion = "Negative but imprecise"
        interpretation = (
            "The point estimate is in the predicted H2 direction, but it is not "
            "statistically distinguishable from zero at the chosen threshold. The "
            "current model does not provide clear evidence for a simple ACSI "
            "displacement gradient."
        )
        supported = "No clear support for H1 or H2"
    elif coef > 0:
        conclusion = "Positive but imprecise"
        interpretation = (
            "The point estimate is opposite the predicted H2 direction, but it is "
            "not statistically distinguishable from zero at the chosen threshold. "
            "The current model does not provide clear evidence for a simple ACSI "
            "gradient."
        )
        supported = "No clear support for H1 or H2"
    else:
        conclusion = "No detectable gradient"
        interpretation = (
            "The point estimate is exactly zero in this model, but H1 should still "
            "be interpreted cautiously because a null result is not affirmative "
            "evidence of equivalence without tight uncertainty."
        )
        supported = "No clear support for H1 or H2"

    return {
        "model": model_name,
        "index_label": INDEX_LABEL,
        "index_short": INDEX_SHORT,
        "h1": H1_LABEL,
        "h1_prediction": H1_PREDICTION,
        "h2": H2_LABEL,
        "h2_prediction": H2_PREDICTION,
        "predicted_sign": "negative",
        "alpha": alpha,
        "coef": coef,
        "pvalue": pvalue,
        "percent_effect_full_exposure": effect,
        "conclusion": conclusion,
        "supported_hypothesis": supported,
        "interpretation": interpretation,
    }

def json_safe(obj):
    if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, pd.Timestamp): return str(obj.date())
    try:
        if pd.isna(obj): return None
    except Exception:
        pass
    return obj

def reg_result(model, term):
    if model is None: return None
    return {
        "coef":   safe_float(model.params.get(term, np.nan)),
        "se":     safe_float(model.bse.get(term, np.nan)),
        "pvalue": safe_float(model.pvalues.get(term, np.nan)),
        "n_obs":  safe_int(model.nobs),
    }

def winsor(s, lo=0.01, hi=0.99):
    s = s.copy()
    nm = s.dropna()
    if len(nm) < 10: return s
    return s.clip(lower=nm.quantile(lo), upper=nm.quantile(hi))

def fit_ols(formula, data, cov_type="HC3", cluster_col=None):
    if data is None or data.empty: return None
    try:
        m = smf.ols(formula, data=data)
        if cluster_col is not None:
            return m.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        return m.fit(cov_type=cov_type)
    except Exception as e:
        print(f"  OLS failed: {e}")
        return None

def residualize_two_way(values, fe_a, fe_b):
    s = pd.Series(values, index=fe_a.index, dtype="float64")
    return (
        s
        - s.groupby(fe_a).transform("mean")
        - s.groupby(fe_b).transform("mean")
        + float(s.mean())
    )

def two_way_fe_coef_from_residualized_y(y_resid, x_values, fe_a, fe_b):
    x_resid = residualize_two_way(x_values, fe_a, fe_b)
    denom = float(np.dot(x_resid, x_resid))
    if denom <= 0:
        return None
    return safe_float(float(np.dot(x_resid, y_resid)) / denom)

def is_balanced_two_way_panel(data, fe_a_col, fe_b_col):
    if data is None or data.empty:
        return False
    counts = data.groupby([fe_a_col, fe_b_col]).size()
    expected = data[fe_a_col].nunique() * data[fe_b_col].nunique()
    return len(counts) == expected and bool(counts.eq(1).all())

def safe_corr(x, y, method="pearson"):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None, None
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)
    return safe_float(r), safe_float(p)

def load_acsi_scores():
    score_path = resolve_acsi_score_path()
    if not score_path.exists():
        print(f"\nERROR: {ACSI_SCORE_PATH} not found.")
        print(f"Creating blank {INDEX_SHORT} template. Fill it manually before running analysis.")
        score_template = pd.DataFrame({
            "subreddit": sorted(SUBREDDITS.keys()),
            "direct_gen": np.nan, "usefulness": np.nan, "quality_comp": np.nan,
            "physical_req": np.nan, "personal_req": np.nan,
        })
        score_template.to_csv(ACSI_SCORE_PATH, index=False)
        raise FileNotFoundError(f"Created template at {ACSI_SCORE_PATH}. Fill in 1-5 scores before rerunning.")

    score_table = pd.read_csv(score_path)
    
    required_columns = {"subreddit", "direct_gen", "usefulness", "quality_comp", "physical_req", "personal_req"}
    missing_columns = required_columns - set(score_table.columns)
    if missing_columns:
        raise ValueError(f"{INDEX_SHORT} file missing columns: {missing_columns}")

    score_table["subreddit"] = score_table["subreddit"].astype(str).str.strip()

    expected_subreddits = set(SUBREDDITS.keys())
    actual_subreddits = set(score_table["subreddit"])
    missing_subreddits = sorted(expected_subreddits - actual_subreddits)
    extra_subreddits = sorted(actual_subreddits - expected_subreddits)
    duplicate_subreddits = sorted(score_table.loc[score_table["subreddit"].duplicated(), "subreddit"].unique())

    if missing_subreddits:
        raise ValueError(f"{INDEX_SHORT} file missing subreddits: {missing_subreddits}")
    if duplicate_subreddits:
        raise ValueError(f"{INDEX_SHORT} file has duplicate subreddit rows: {duplicate_subreddits}")
    if extra_subreddits:
        print(f"WARNING: {INDEX_SHORT} file contains extra subreddits not in SUBREDDITS: {extra_subreddits}")

    score_columns = ACSI_COMPONENT_COLUMNS
    rows_with_missing_scores = score_table[score_table[score_columns].isna().any(axis=1)]["subreddit"].tolist()
    if rows_with_missing_scores:
        raise ValueError(f"Missing {INDEX_SHORT} component scores for: {rows_with_missing_scores}")

    for score_column in score_columns:
        invalid_score_rows = score_table.loc[
            ~score_table[score_column].between(1, 5),
            ["subreddit", score_column],
        ]
        if not invalid_score_rows.empty:
            raise ValueError(f"{INDEX_SHORT} column {score_column} has values outside 1-5:\n{invalid_score_rows}")

    score_table["physical_free"] = 6 - score_table["physical_req"]
    score_table["non_personal"] = 6 - score_table["personal_req"]
    for metadata_column in ACSI_SCORE_METADATA_COLUMNS:
        if metadata_column not in score_table.columns:
            score_table[metadata_column] = np.nan
    if "n_used" in score_table.columns:
        n_used_numeric = pd.to_numeric(score_table["n_used"], errors="coerce")
        if n_used_numeric.notna().any():
            score_table["low_n_flag"] = (n_used_numeric < ACSI_RELIABILITY_TARGET).fillna(False).astype(int)
            score_table["score_reliability"] = np.select(
                [n_used_numeric < ACSI_RELIABILITY_TARGET, n_used_numeric < 100],
                ["low", "medium"],
                default="high",
            )
    score_table["raw_gse"] = (
        score_table["direct_gen"]
        + score_table["usefulness"]
        + score_table["quality_comp"]
        + score_table["physical_free"]
        + score_table["non_personal"]
    )
    score_table["gse"] = ((score_table["raw_gse"] - 5) / 20).clip(0, 1)

    for dimension_spec in ACSI_DIMENSION_SPECS:
        score_table[dimension_spec["norm"]] = ((score_table[dimension_spec["source"]] - 1) / 4).clip(0, 1)
    score_table["generation_capability_norm"] = score_table[
        ["direct_gen_norm", "usefulness_norm", "quality_comp_norm"]
    ].mean(axis=1)
    score_table["generation_capability"] = 1 + 4 * score_table["generation_capability_norm"]

    return score_table

load_gse_scores = load_acsi_scores

# --- Parsing & Cleaning ---
def parse_posts():
    print("\n=== Step 1a: parse posts ===")
    frames = []
    for sub in iter_subreddits():
        rows = []
        for author, dt, score, post_id in iter_post_records(sub, f"  r/{sub} posts"):
            rows.append({
                "author":       author,
                "subreddit":    sub,
                "date":         dt,
                "year_month":   dt.strftime("%Y-%m"),
                "year_month_dt": pd.Timestamp(dt.strftime("%Y-%m-01")),
                "score":        score,
                "post_id":      post_id,
            })
        if rows:
            frames.append(pd.DataFrame.from_records(rows))
            del rows
            gc.collect()
    if not frames:
        print("ERROR: no posts parsed")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  parsed: {len(df):,} posts, {df['author'].nunique():,} authors")
    return df

def parse_comments(valid_authors=None, restrict_to_valid_authors=False, start_date=None, end_date=None, keep_body=True):
    print(f"\n=== Step 1b: parse comments (keep_body={keep_body}) ===")
    frames = []
    for sub in iter_subreddits():
        path = DATA_DIR / f"r_{sub}_comments.jsonl"
        if not path.exists(): continue
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc=f"  r/{sub} comments")):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line: continue
                try: p = json.loads(line)
                except Exception: continue
                
                ts = p.get("created_utc")
                if ts is None: continue
                try: dt = datetime.utcfromtimestamp(int(ts))
                except Exception: continue
                
                if start_date is not None and dt < start_date: continue
                if end_date is not None and dt >= end_date: continue
                
                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                    continue
                if restrict_to_valid_authors and valid_authors is not None:
                    if author not in valid_authors: continue
                
                month = pd.Timestamp(dt.strftime("%Y-%m-01"))
                row = {
                    "author":       author,
                    "subreddit":    sub,
                    "year_month":   dt.strftime("%Y-%m"),
                    "year_month_dt": month,
                    "comment_id":   str(p.get("id") or ""),
                    "post_shock":   int(month >= SHOCK_MONTH),
                    "post_shock_exact": int(dt >= EXACT_SHOCK_DATE),
                }
                if keep_body:
                    row["body"] = str(p.get("body") or "")
                rows.append(row)
        if rows:
            frames.append(pd.DataFrame.from_records(rows))
            del rows
            gc.collect()

    if not frames:
        print("  no comment files found")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"  parsed: {len(df):,} comments")
    return df

def build_comment_keyword_monthly():
    print("\n=== Step 1b-cache: build comment keyword monthly aggregates ===")
    start_date = datetime(2022, 9, 1)
    end_date = datetime(2023, 3, 1)
    cells = defaultdict(lambda: {"comment_count": 0, "ai_kw_count": 0, "tool_kw_count": 0})

    for sub in available_comment_subreddits():
        path = DATA_DIR / f"r_{sub}_comments.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc=f"  r/{sub} keyword comments")):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue

                ts = p.get("created_utc")
                if ts is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(ts))
                except Exception:
                    continue
                if dt < start_date or dt >= end_date:
                    continue

                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                    continue

                body = str(p.get("body") or "")
                month = pd.Timestamp(dt.strftime("%Y-%m-01"))
                cell = cells[(sub, month)]
                cell["comment_count"] += 1
                cell["ai_kw_count"] += int(bool(AI_BROAD_PATTERN.search(body)))
                cell["tool_kw_count"] += int(bool(AI_TOOL_PATTERN.search(body)))

    rows = []
    for (sub, month), cell in cells.items():
        n = cell["comment_count"]
        rows.append({
            "subreddit": sub,
            "year_month_dt": month,
            "year_month": month.strftime("%Y-%m"),
            "comment_count": n,
            "ai_kw_count": cell["ai_kw_count"],
            "tool_kw_count": cell["tool_kw_count"],
            "ai_kw_rate": cell["ai_kw_count"] / n if n else 0.0,
            "tool_kw_rate": cell["tool_kw_count"] / n if n else 0.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "subreddit", "year_month_dt", "year_month", "comment_count",
            "ai_kw_count", "tool_kw_count", "ai_kw_rate", "tool_kw_rate",
        ])
    print(f"  keyword aggregate rows: {len(df):,}")
    return df

def load_or_build_comment_keyword_monthly(force_rebuild=False):
    if not use_persistent_raw_cache():
        return build_comment_keyword_monthly()

    if not force_rebuild and comment_keyword_cache_is_current():
        print("\n=== Step 1b-cache: load cached comment keyword aggregates ===")
        df = pd.read_parquet(COMMENT_KEYWORD_MONTHLY_PATH)
        print(f"  loaded: {len(df):,} rows from {COMMENT_KEYWORD_MONTHLY_PATH}")
        return df

    df = build_comment_keyword_monthly()
    ensure_cache_dir()
    df.to_parquet(COMMENT_KEYWORD_MONTHLY_PATH, index=False)
    write_comment_keyword_cache_metadata()
    print(f"  cached comment keyword aggregates -> {COMMENT_KEYWORD_MONTHLY_PATH}")
    return df

def build_comment_author_monthly(valid_authors):
    print("\n=== Step 1b-cache: build valid-author comment monthly aggregates ===")
    valid_authors = set(valid_authors)
    cells = defaultdict(int)

    for sub in available_comment_subreddits():
        path = DATA_DIR / f"r_{sub}_comments.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc=f"  r/{sub} author comments")):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue

                ts = p.get("created_utc")
                if ts is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(ts))
                except Exception:
                    continue
                if dt < START_DATE or dt >= END_DATE_EXCLUSIVE:
                    continue

                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                    continue
                if author not in valid_authors:
                    continue

                month = pd.Timestamp(dt.strftime("%Y-%m-01"))
                cells[(author, sub, month)] += 1

    rows = []
    for (author, sub, month), count in cells.items():
        rows.append({
            "author": author,
            "subreddit": sub,
            "year_month_dt": month,
            "year_month": month.strftime("%Y-%m"),
            "post_shock": int(month >= SHOCK_MONTH),
            "comment_count": int(count),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "author", "subreddit", "year_month_dt", "year_month",
            "post_shock", "comment_count",
        ])
    print(f"  valid-author comment aggregate rows: {len(df):,}")
    return df

def build_comment_caches_combined(valid_authors):
    print("\n=== Step 1b-cache: build comment keyword + valid-author aggregates in one pass ===")
    valid_authors = set(valid_authors)
    keyword_start = datetime(2022, 9, 1)
    keyword_end = datetime(2023, 3, 1)
    keyword_cells = defaultdict(lambda: {"comment_count": 0, "ai_kw_count": 0, "tool_kw_count": 0})
    author_cells = defaultdict(int)

    for sub in available_comment_subreddits():
        path = DATA_DIR / f"r_{sub}_comments.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc=f"  r/{sub} comments combined")):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except Exception:
                    continue

                ts = p.get("created_utc")
                if ts is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(ts))
                except Exception:
                    continue

                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                    continue

                if START_DATE <= dt < END_DATE_EXCLUSIVE:
                    month = pd.Timestamp(dt.strftime("%Y-%m-01"))
                    if author in valid_authors:
                        author_cells[(author, sub, month)] += 1

                    if keyword_start <= dt < keyword_end:
                        body = str(p.get("body") or "")
                        cell = keyword_cells[(sub, month)]
                        cell["comment_count"] += 1
                        cell["ai_kw_count"] += int(bool(AI_BROAD_PATTERN.search(body)))
                        cell["tool_kw_count"] += int(bool(AI_TOOL_PATTERN.search(body)))

    keyword_rows = []
    for (sub, month), cell in keyword_cells.items():
        n = cell["comment_count"]
        keyword_rows.append({
            "subreddit": sub,
            "year_month_dt": month,
            "year_month": month.strftime("%Y-%m"),
            "comment_count": n,
            "ai_kw_count": cell["ai_kw_count"],
            "tool_kw_count": cell["tool_kw_count"],
            "ai_kw_rate": cell["ai_kw_count"] / n if n else 0.0,
            "tool_kw_rate": cell["tool_kw_count"] / n if n else 0.0,
        })

    author_rows = []
    for (author, sub, month), count in author_cells.items():
        author_rows.append({
            "author": author,
            "subreddit": sub,
            "year_month_dt": month,
            "year_month": month.strftime("%Y-%m"),
            "post_shock": int(month >= SHOCK_MONTH),
            "comment_count": int(count),
        })

    keyword_df = pd.DataFrame(keyword_rows)
    if keyword_df.empty:
        keyword_df = pd.DataFrame(columns=[
            "subreddit", "year_month_dt", "year_month", "comment_count",
            "ai_kw_count", "tool_kw_count", "ai_kw_rate", "tool_kw_rate",
        ])

    author_df = pd.DataFrame(author_rows)
    if author_df.empty:
        author_df = pd.DataFrame(columns=[
            "author", "subreddit", "year_month_dt", "year_month",
            "post_shock", "comment_count",
        ])

    print(f"  keyword aggregate rows: {len(keyword_df):,}")
    print(f"  valid-author comment aggregate rows: {len(author_df):,}")
    return keyword_df, author_df

def load_or_build_comment_author_monthly(valid_authors, force_rebuild=False):
    if not use_persistent_raw_cache():
        return build_comment_author_monthly(valid_authors)

    if not force_rebuild and comment_author_cache_is_current(valid_authors):
        print("\n=== Step 1b-cache: load cached valid-author comment aggregates ===")
        df = pd.read_parquet(COMMENT_AUTHOR_MONTHLY_PATH)
        print(f"  loaded: {len(df):,} rows from {COMMENT_AUTHOR_MONTHLY_PATH}")
        return df

    df = build_comment_author_monthly(valid_authors)
    ensure_cache_dir()
    df.to_parquet(COMMENT_AUTHOR_MONTHLY_PATH, index=False)
    write_comment_author_cache_metadata(valid_authors)
    print(f"  cached valid-author comment aggregates -> {COMMENT_AUTHOR_MONTHLY_PATH}")
    return df

def load_or_build_comment_caches(valid_authors, force_rebuild=False):
    if not use_persistent_raw_cache():
        return build_comment_caches_combined(valid_authors)

    keyword_current = (not force_rebuild) and comment_keyword_cache_is_current()
    author_current = (not force_rebuild) and comment_author_cache_is_current(valid_authors)

    if keyword_current and author_current:
        return (
            load_or_build_comment_keyword_monthly(force_rebuild=False),
            load_or_build_comment_author_monthly(valid_authors, force_rebuild=False),
        )

    if not keyword_current and not author_current:
        keyword_df, author_df = build_comment_caches_combined(valid_authors)
        ensure_cache_dir()
        keyword_df.to_parquet(COMMENT_KEYWORD_MONTHLY_PATH, index=False)
        write_comment_keyword_cache_metadata()
        author_df.to_parquet(COMMENT_AUTHOR_MONTHLY_PATH, index=False)
        write_comment_author_cache_metadata(valid_authors)
        print(f"  cached comment keyword aggregates -> {COMMENT_KEYWORD_MONTHLY_PATH}")
        print(f"  cached valid-author comment aggregates -> {COMMENT_AUTHOR_MONTHLY_PATH}")
        return keyword_df, author_df

    keyword_df = load_or_build_comment_keyword_monthly(force_rebuild=not keyword_current)
    author_df = load_or_build_comment_author_monthly(valid_authors, force_rebuild=not author_current)
    return keyword_df, author_df

def aggregate_comment_counts(df_comments, group_cols, count_name):
    if df_comments is None or df_comments.empty:
        return pd.DataFrame(columns=[*group_cols, count_name])
    if "comment_count" in df_comments.columns:
        return (
            df_comments.groupby(group_cols, as_index=False)
            .agg(**{count_name: ("comment_count", "sum")})
        )
    return (
        df_comments.groupby(group_cols, as_index=False)
        .agg(**{count_name: ("comment_id", "count")})
    )

def clean_posts_ecosystem(df):
    print("\n=== Step 2a: clean_posts_ecosystem (no pre-activity restriction) ===")
    df = df.copy()
    df["author"] = df["author"].fillna("").astype(str)
    df["sub_role"] = df["subreddit"].map(SUBREDDITS)
    df["post_shock_exact"] = (df["date"] >= EXACT_SHOCK_DATE).astype(int)
    df["post_shock"] = (df["year_month_dt"] >= SHOCK_MONTH).astype(int)

    df = df[~df["author"].isin(EXCLUDED_AUTHORS)].copy()
    df = df[~df["author"].str.lower().str.endswith("bot", na=False)].copy()

    days = max((END_DATE_EXCLUSIVE - START_DATE).days, 1)
    cap = MAX_POSTS_PER_DAY * days
    counts = df.groupby("author")["post_id"].count()
    df = df[df["author"].isin(counts[counts <= cap].index)].copy()

    if df.empty:
        print("ERROR: no posts after ecosystem cleaning")
        sys.exit(1)
    print(f"  ecosystem clean: {len(df):,} posts, {df['author'].nunique():,} authors")
    return df

def clean_posts_creator_sample(df):
    print("\n=== Step 2b: clean_posts_creator_sample (keeps complete exits) ===")
    df = df.copy()
    df["author"]         = df["author"].fillna("").astype(str)
    df["sub_role"]       = df["subreddit"].map(SUBREDDITS)
    df["post_shock_exact"] = (df["date"] >= EXACT_SHOCK_DATE).astype(int)
    df["post_shock"]     = (df["year_month_dt"] >= SHOCK_MONTH).astype(int)

    df = df[~df["author"].isin(EXCLUDED_AUTHORS)].copy()
    df = df[~df["author"].str.lower().str.endswith("bot", na=False)].copy()

    days = max((END_DATE_EXCLUSIVE - START_DATE).days, 1)
    cap  = MAX_POSTS_PER_DAY * days
    counts = df.groupby("author")["post_id"].count()
    df = df[df["author"].isin(counts[counts <= cap].index)].copy()

    pre_counts = df[df["post_shock"] == 0].groupby("author")["post_id"].count()
    active = pre_counts[pre_counts >= MIN_PRE_POSTS].index
    df = df[df["author"].isin(active)].copy()

    if df.empty:
        print("ERROR: no posts after cleaning")
        sys.exit(1)
    print(f"  creator clean: {len(df):,} posts, {df['author'].nunique():,} authors")
    return df

def clean_post_samples(df):
    print("\n=== Step 2: clean post samples (shared pass) ===")
    df["author"] = df["author"].fillna("").astype(str)
    df["sub_role"] = df["subreddit"].map(SUBREDDITS)
    df["post_shock_exact"] = (df["date"] >= EXACT_SHOCK_DATE).astype(int)
    df["post_shock"] = (df["year_month_dt"] >= SHOCK_MONTH).astype(int)

    mask = ~df["author"].isin(EXCLUDED_AUTHORS)
    mask &= ~df["author"].str.lower().str.endswith("bot", na=False)
    df_ecosystem = df.loc[mask].copy()

    days = max((END_DATE_EXCLUSIVE - START_DATE).days, 1)
    cap = MAX_POSTS_PER_DAY * days
    counts = df_ecosystem.groupby("author")["post_id"].count()
    valid_authors = counts[counts <= cap].index
    df_ecosystem = df_ecosystem[df_ecosystem["author"].isin(valid_authors)].copy()

    if df_ecosystem.empty:
        print("ERROR: no posts after ecosystem cleaning")
        sys.exit(1)

    pre_counts = df_ecosystem[df_ecosystem["post_shock"] == 0].groupby("author")["post_id"].count()
    active = pre_counts[pre_counts >= MIN_PRE_POSTS].index
    df_creator = df_ecosystem[df_ecosystem["author"].isin(active)].copy()

    if df_creator.empty:
        print("ERROR: no posts after creator cleaning")
        sys.exit(1)

    print(f"  ecosystem clean: {len(df_ecosystem):,} posts, {df_ecosystem['author'].nunique():,} authors")
    print(f"  creator clean: {len(df_creator):,} posts, {df_creator['author'].nunique():,} authors")
    return df_ecosystem, df_creator

def restrict_to_survivors(df):
    pre       = set(df.loc[df["post_shock"] == 0, "author"].unique())
    post      = set(df.loc[df["post_shock"] == 1, "author"].unique())
    survivors = pre & post
    out       = df[df["author"].isin(survivors)].copy()
    n_exits   = df["author"].nunique() - out["author"].nunique()
    print(f"  survivors: {out['author'].nunique():,}  complete exits removed: {n_exits:,}")
    return out

# --- Panel Generation ---
def build_panel(df):
    print("\n=== Step 3a: build base creator-level panel ===")
    df    = df.copy()
    panel = (
        df.groupby(
            ["author", "subreddit", "year_month", "year_month_dt", "post_shock"],
            as_index=False,
        ).agg(post_count=("post_id", "count"), avg_score=("score", "mean"))
    )
    panel["x_ikt"]         = panel["post_count"] * (1.0 + np.log1p(panel["avg_score"].clip(lower=0)))
    panel["sub_role"]      = panel["subreddit"].map(SUBREDDITS)
    panel["sub_role_main"] = panel["sub_role"].replace("ambiguous", "treatment")
    print(f"  panel: {len(panel):,} rows")
    return panel

def attach_acsi_scores(panel, acsi_scores):
    drop_cols = [
        "gse", "raw_gse", "gse_post",
        *ACSI_COMPONENT_COLUMNS, "physical_free", "non_personal",
        *ACSI_SCORE_METADATA_COLUMNS,
        "generation_capability", "generation_capability_norm", "generation_capability_post",
        *[spec["norm"] for spec in ACSI_DIMENSION_SPECS],
        *[spec["post"] for spec in ACSI_DIMENSION_SPECS],
    ]
    panel = panel.drop(columns=drop_cols, errors="ignore").copy()
    merge_columns = [
        column_name
        for column_name in (
            ACSI_MERGE_COLUMNS
            + ["generation_capability", "generation_capability_norm"]
            + [spec["norm"] for spec in ACSI_DIMENSION_SPECS]
        )
        if column_name in acsi_scores.columns
    ]
    panel = panel.merge(acsi_scores[merge_columns], on="subreddit", how="left")
    if panel["gse"].isna().any():
        missing_subreddits = panel.loc[panel["gse"].isna(), "subreddit"].unique()
        print(f"  WARNING: Missing {INDEX_SHORT} scores for subreddits: {missing_subreddits}")
        panel = panel.dropna(subset=["gse"])
    panel["gse_post"] = panel["gse"] * panel["post_shock"]
    for dimension_spec in ACSI_DIMENSION_SPECS:
        if dimension_spec["norm"] not in panel.columns:
            panel[dimension_spec["norm"]] = ((panel[dimension_spec["source"]] - 1) / 4).clip(0, 1)
        panel[dimension_spec["post"]] = panel[dimension_spec["norm"]] * panel["post_shock"]
    if "generation_capability_norm" not in panel.columns:
        panel["generation_capability_norm"] = panel[
            ["direct_gen_norm", "usefulness_norm", "quality_comp_norm"]
        ].mean(axis=1)
    panel["generation_capability_post"] = panel["generation_capability_norm"] * panel["post_shock"]
    return panel

attach_gse_scores = attach_acsi_scores

def build_subreddit_month_panel(df_all, acsi_scores):
    print(f"\n=== Step 3b: Build ecosystem subreddit-month panel for {INDEX_SHORT} dose-response ===")
    df = df_all.copy()
    df["score_pos"] = df["score"].clip(lower=0)
    df["post_value"] = 1.0 + np.log1p(df["score_pos"])
    
    agg = (
        df.groupby(["subreddit", "year_month_dt"], as_index=False)
        .agg(
            posts=("post_id", "count"),
            active_creators=("author", "nunique"),
            avg_score=("score", "mean"),
            calibrated_output=("post_value", "sum"),
        )
    )
    
    subs = sorted(df["subreddit"].unique())
    grid = pd.MultiIndex.from_product(
        [subs, ALL_MONTHS],
        names=["subreddit", "year_month_dt"]
    ).to_frame(index=False)
    
    panel = grid.merge(agg, on=["subreddit", "year_month_dt"], how="left")
    
    for col in ["posts", "active_creators", "calibrated_output"]:
        panel[col] = panel[col].fillna(0)
    panel["avg_score"] = panel["avg_score"].fillna(0)
    
    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    panel["post_shock"] = (panel["year_month_dt"] >= SHOCK_MONTH).astype(int)
    
    panel["log_posts"] = np.log1p(panel["posts"])
    panel["log_active_creators"] = np.log1p(panel["active_creators"])
    panel["log_calibrated_output"] = np.log1p(panel["calibrated_output"])
    panel["log_avg_score"] = np.log1p(panel["avg_score"].clip(lower=0))
    if use_persistent_raw_cache():
        try:
            ensure_cache_dir()
            panel.to_parquet(POST_MONTHLY_AGG_PATH, index=False)
            write_post_monthly_agg_cache_metadata(apply_author_cap=True, n_ecosystem_authors=df_all["author"].nunique())
            print(f"  cached post aggregates -> {POST_MONTHLY_AGG_PATH}")
        except Exception as e:
            print(f"  post aggregate cache write skipped: {e}")
    else:
        print("  post aggregate cache write skipped: non-full raw scan")
    panel = attach_acsi_scores(panel, acsi_scores)
    print(f"  subreddit-month panel: {len(panel):,} rows")
    return panel

def build_post_monthly_aggregate_streaming(apply_author_cap=True):
    print("\n=== Step 3b-cache: Build post subreddit-month aggregates (streaming) ===")
    days = max((END_DATE_EXCLUSIVE - START_DATE).days, 1)
    cap = MAX_POSTS_PER_DAY * days

    available_subs = available_post_subreddits()

    valid_authors = None
    if apply_author_cap:
        author_counts = Counter()
        for sub in available_subs:
            for author, _dt, _score, _post_id in iter_post_records(sub, f"  r/{sub} posts pass 1"):
                author_counts[author] += 1

        valid_authors = {author for author, n_posts in author_counts.items() if n_posts <= cap}
        del author_counts
        gc.collect()
    else:
        print("  author cap skipped: one-pass streaming mode")

    author_ids = {}
    cells = defaultdict(lambda: {"posts": 0, "author_ids": set(), "score_sum": 0.0, "calibrated_output": 0.0})
    for sub in available_subs:
        pass_label = "pass 2" if apply_author_cap else "one pass"
        for author, dt, score, _post_id in iter_post_records(sub, f"  r/{sub} posts {pass_label}"):
            if valid_authors is not None and author not in valid_authors:
                continue
            author_id = author_ids.setdefault(author, len(author_ids))
            month = pd.Timestamp(dt.strftime("%Y-%m-01"))
            cell = cells[(sub, month)]
            cell["posts"] += 1
            cell["author_ids"].add(author_id)
            cell["score_sum"] += score
            cell["calibrated_output"] += 1.0 + np.log1p(max(score, 0))

    rows = []
    for (sub, month), cell in cells.items():
        rows.append({
            "subreddit": sub,
            "year_month_dt": month,
            "posts": cell["posts"],
            "active_creators": len(cell["author_ids"]),
            "avg_score": cell["score_sum"] / cell["posts"] if cell["posts"] else 0.0,
            "calibrated_output": cell["calibrated_output"],
        })

    if rows:
        agg = pd.DataFrame(rows)
    else:
        agg = pd.DataFrame(columns=["subreddit", "year_month_dt", "posts", "active_creators", "avg_score", "calibrated_output"])

    grid = pd.MultiIndex.from_product(
        [sorted(available_subs), ALL_MONTHS],
        names=["subreddit", "year_month_dt"]
    ).to_frame(index=False)

    panel = grid.merge(agg, on=["subreddit", "year_month_dt"], how="left")
    for col in ["posts", "active_creators", "calibrated_output"]:
        panel[col] = panel[col].fillna(0)
    panel["avg_score"] = panel["avg_score"].fillna(0)

    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    panel["post_shock"] = (panel["year_month_dt"] >= SHOCK_MONTH).astype(int)
    panel["log_posts"] = np.log1p(panel["posts"])
    panel["log_active_creators"] = np.log1p(panel["active_creators"])
    panel["log_calibrated_output"] = np.log1p(panel["calibrated_output"])
    panel["log_avg_score"] = np.log1p(panel["avg_score"].clip(lower=0))
    print(f"  post aggregate panel: {len(panel):,} rows")
    return panel, len(author_ids)

def load_or_build_post_monthly_aggregate(apply_author_cap=True, force_rebuild=False):
    if not use_persistent_raw_cache():
        return build_post_monthly_aggregate_streaming(apply_author_cap=apply_author_cap)

    if not force_rebuild and post_monthly_agg_cache_is_current(apply_author_cap):
        print("\n=== Step 3b-cache: load cached post subreddit-month aggregates ===")
        panel = pd.read_parquet(POST_MONTHLY_AGG_PATH)
        n_ecosystem_authors = read_cached_n_ecosystem_authors()
        print(f"  loaded: {len(panel):,} rows from {POST_MONTHLY_AGG_PATH}")
        return panel, n_ecosystem_authors

    legacy_panel_path = OUTPUT_ROOT / "subreddit_month_gse_panel.parquet"
    legacy_meta_path = OUTPUT_ROOT / "subreddit_month_gse_panel.meta.json"
    if not force_rebuild and legacy_panel_path.exists() and legacy_meta_path.exists():
        try:
            old = json.loads(legacy_meta_path.read_text(encoding="utf-8"))
            if old == submonth_panel_cache_metadata(apply_author_cap):
                print("\n=== Step 3b-cache: seed post aggregates from legacy output panel ===")
                panel = pd.read_parquet(legacy_panel_path)
                drop_cols = [
                    "gse", "raw_gse", "gse_post",
                    *ACSI_COMPONENT_COLUMNS, "physical_free", "non_personal",
                    *[spec["norm"] for spec in ACSI_DIMENSION_SPECS],
                    *[spec["post"] for spec in ACSI_DIMENSION_SPECS],
                ]
                panel = panel.drop(columns=drop_cols, errors="ignore")
                ensure_cache_dir()
                panel.to_parquet(POST_MONTHLY_AGG_PATH, index=False)
                write_post_monthly_agg_cache_metadata(apply_author_cap, n_ecosystem_authors=None)
                print(f"  cached post aggregates -> {POST_MONTHLY_AGG_PATH}")
                return panel, None
        except Exception as e:
            print(f"  legacy panel seed skipped: {e}")

    panel, n_ecosystem_authors = build_post_monthly_aggregate_streaming(apply_author_cap=apply_author_cap)
    ensure_cache_dir()
    panel.to_parquet(POST_MONTHLY_AGG_PATH, index=False)
    write_post_monthly_agg_cache_metadata(apply_author_cap, n_ecosystem_authors)
    print(f"  cached post aggregates -> {POST_MONTHLY_AGG_PATH}")
    return panel, n_ecosystem_authors

def build_subreddit_month_panel_streaming(acsi_scores, apply_author_cap=True, force_rebuild=False):
    print(f"\n=== Step 3b: Build ecosystem subreddit-month panel for {INDEX_SHORT} dose-response ===")
    base_panel, n_ecosystem_authors = load_or_build_post_monthly_aggregate(
        apply_author_cap=apply_author_cap,
        force_rebuild=force_rebuild,
    )
    panel = attach_acsi_scores(base_panel, acsi_scores)
    print(f"  subreddit-month panel: {len(panel):,} rows")
    return panel, n_ecosystem_authors

def add_pre_covariates(panel):
    pre = panel[panel["post_shock"] == 0].copy()
    pre_avg = pre.groupby("subreddit")["log_posts"].mean().rename("pre_avg_log_posts")

    min_month = pre["year_month_dt"].min()
    pre["t"] = (
        (pre["year_month_dt"].dt.year - min_month.year) * 12
        + (pre["year_month_dt"].dt.month - min_month.month)
    )

    trend_rows = []
    for sub, g in pre.groupby("subreddit"):
        if len(g) >= 6 and g["log_posts"].nunique() > 1:
            slope = np.polyfit(g["t"], g["log_posts"], 1)[0]
        else:
            slope = 0.0
        trend_rows.append({"subreddit": sub, "pre_trend": slope})

    pre_covariates = pre_avg.reset_index().merge(pd.DataFrame(trend_rows), on="subreddit", how="left")
    pre_covariates["mu_k"] = pre_covariates["subreddit"].map(MU_K).fillna(0.5)
    pre_covariates["log_mu_k"] = np.log1p(pre_covariates["mu_k"])
    return pre_covariates

def build_creators_v3(df_all, df_comments=None):
    print("\n=== Step 4: build_creators_v3 (calendar rates, includes exits) ===")

    pre = (
        df_all[df_all["post_shock"] == 0]
        .groupby("author")
        .agg(
            pre_count=("post_id", "count"),
            pre_score=("score", "median"),
            pre_active_months=("year_month", "nunique"),
            pre_subs=("subreddit", "nunique"),
        )
        .reset_index()
    )

    post = (
        df_all[df_all["post_shock"] == 1]
        .groupby("author")
        .agg(
            post_count=("post_id", "count"),
            post_score=("score", "median"),
            post_active_months=("year_month", "nunique"),
        )
        .reset_index()
    )

    creators = pre.merge(post, on="author", how="left")
    creators["post_count"]         = creators["post_count"].fillna(0).astype(int)
    creators["post_active_months"] = creators["post_active_months"].fillna(0).astype(int)
    creators["post_score"]         = creators["post_score"].fillna(0)
    creators["survived"]           = (creators["post_count"] > 0).astype(int)

    creators["pre_rate_cal"]    = creators["pre_count"] / max(N_PRE_MONTHS, 1)
    creators["post_rate_cal"]   = creators["post_count"] / max(N_POST_MONTHS, 1)
    creators["pre_rate_active"] = creators["pre_count"] / creators["pre_active_months"].replace(0, np.nan)

    creators["c_i"]     = 1.0 / creators["pre_rate_cal"].replace(0, np.nan)
    creators["log_c_i"] = np.log(creators["c_i"])

    safe_pre = creators["pre_rate_cal"].replace(0, np.nan)
    creators["delta_rate_cal"] = (creators["post_rate_cal"] - creators["pre_rate_cal"]) / safe_pre
    creators["delta_score"]    = creators["post_score"] - creators["pre_score"]

    if df_comments is not None and not df_comments.empty:
        pre_c = aggregate_comment_counts(
            df_comments[df_comments["post_shock"] == 0],
            ["author"],
            "pre_comment_count",
        )
        post_c = aggregate_comment_counts(
            df_comments[df_comments["post_shock"] == 1],
            ["author"],
            "post_comment_count",
        )
        creators = creators.merge(pre_c,  on="author", how="left")
        creators = creators.merge(post_c, on="author", how="left")
        creators["pre_comment_count"]  = creators["pre_comment_count"].fillna(0)
        creators["post_comment_count"] = creators["post_comment_count"].fillna(0)
        creators["pre_comment_rate"]   = creators["pre_comment_count"]  / max(N_PRE_MONTHS, 1)
        creators["post_comment_rate"]  = creators["post_comment_count"] / max(N_POST_MONTHS, 1)
        safe_pc = creators["pre_comment_rate"].replace(0, np.nan)
        creators["delta_comment_rate"] = (
            creators["post_comment_rate"] - creators["pre_comment_rate"]
        ) / safe_pc
    else:
        creators["delta_comment_rate"] = np.nan

    ci_thresh = creators.loc[creators["survived"] == 1, "c_i"].quantile(0.20)
    creators["is_stable"] = (
        (creators["c_i"] <= ci_thresh) & (creators["survived"] == 1)
    ).astype(int)

    print(f"  total eligible: {len(creators):,}")
    print(f"  survived: {creators['survived'].sum():,}  exits: {(creators['survived']==0).sum():,}")
    print(f"  stable (top quintile eff, survived): {creators['is_stable'].sum():,}")
    return creators

def compute_content_validation_sample(top_subs=None):
    print("\n=== content validation sample ===")
    if top_subs is None:
        top_subs = ["art", "writing", "applyingtocollege", "poetry", "fanfiction"]

    rows = []
    rng_cv = random.Random(RANDOM_SEED)
    for sub in top_subs:
        path = DATA_DIR / f"r_{sub}_posts.jsonl"
        if not path.exists(): continue
        pre_pool, post_pool = [], []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line: continue
                try: p = json.loads(line)
                except Exception: continue
                
                ts = p.get("created_utc")
                if ts is None: continue
                try: dt = datetime.utcfromtimestamp(int(ts))
                except Exception: continue
                if dt < START_DATE or dt >= END_DATE_EXCLUSIVE: continue
                
                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"): continue
                
                title    = str(p.get("title") or "")
                selftext = str(p.get("selftext") or "")
                if selftext in {"[removed]", "[deleted]"}: selftext = ""
                full_text = (title + " " + selftext).strip()
                if len(full_text) < 10: continue
                
                words = full_text.split()
                ttr   = len(set(w.lower() for w in words)) / max(len(words), 1)
                month = pd.Timestamp(dt.strftime("%Y-%m-01"))
                rec   = {
                    "subreddit":     sub,
                    "author":        author,
                    "date":          dt.strftime("%Y-%m-%d"),
                    "post_shock":    int(month >= SHOCK_MONTH),
                    "post_shock_exact": int(dt >= EXACT_SHOCK_DATE),
                    "word_count":    len(words),
                    "ttr":           round(ttr, 4),
                    "title_preview": title[:120],
                }
                if month < SHOCK_MONTH: pre_pool.append(rec)
                else: post_pool.append(rec)

        sampled = rng_cv.sample(pre_pool,  min(60, len(pre_pool))) + \
                  rng_cv.sample(post_pool, min(60, len(post_pool)))
        rows.extend(sampled)

    if not rows: return {"n_sampled": 0, "caution": "No text posts found."}

    df_cv     = pd.DataFrame(rows)
    out_path  = TABLES_DIR / "content_validation_sample.csv"
    df_cv.to_csv(out_path, index=False)

    result = {"n_sampled": len(df_cv), "caution": "Heuristic only — not diagnostic of AI use."}
    for label, col in [("word_count", "word_count"), ("ttr", "ttr")]:
        pre_v  = df_cv[df_cv["post_shock"] == 0][col].values
        post_v = df_cv[df_cv["post_shock"] == 1][col].values
        if len(pre_v) > 1 and len(post_v) > 1:
            t, p = stats.ttest_ind(pre_v, post_v, equal_var=False)
            result[f"pre_mean_{label}"]  = float(np.mean(pre_v))
            result[f"post_mean_{label}"] = float(np.mean(post_v))
            result[f"pvalue_{label}"]    = float(p)
    print(f"  saved {len(df_cv)} sampled posts -> {out_path}")
    return result

def acsi_model_panel(panel, model_label):
    if "n_used" in panel.columns:
        n_used = pd.to_numeric(panel["n_used"], errors="coerce")
        low_coverage_subreddits = sorted(panel.loc[n_used < ACSI_RELIABILITY_TARGET, "subreddit"].astype(str).unique())
        if low_coverage_subreddits:
            print(
                f"  {model_label}: retaining {len(low_coverage_subreddits)} low-coverage "
                f"{INDEX_SHORT} score(s), n_used < {ACSI_RELIABILITY_TARGET}; "
                "code more posts for: "
                + ", ".join(low_coverage_subreddits[:12])
                + ("..." if len(low_coverage_subreddits) > 12 else "")
            )
    return panel.copy()

def compute_acsi_component_correlation(score_table):
    if score_table is None or score_table.empty:
        return None
    component_columns = [
        "direct_gen_norm",
        "usefulness_norm",
        "quality_comp_norm",
        "physical_free_norm",
        "non_personal_norm",
        "gse",
    ]
    available_columns = [column_name for column_name in component_columns if column_name in score_table.columns]
    if len(available_columns) < 2:
        return None
    correlation_table = score_table[available_columns].astype(float).corr()
    correlation_table.to_csv(TABLES_DIR / "acsi_component_correlations.csv")
    return correlation_table.round(3).reset_index().rename(columns={"index": "component"}).to_dict("records")

def fit_three_dimensional_acsi_model(acsi_panel, outcome="log_posts"):
    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    model_data = acsi_panel.dropna(subset=terms + [outcome]).copy()
    model = fit_ols(
        f"{outcome} ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
        model_data,
        cluster_col="subreddit",
    )
    if not model:
        return None, []

    results = []
    spec_by_term = {spec["post"]: spec for spec in ACSI_MECHANISM_SPECS}
    for term in terms:
        spec = spec_by_term[term]
        model_result = reg_result(model, term)
        model_result["source"] = spec["source"]
        model_result["term"] = term
        model_result["label"] = spec["label"]
        model_result["description"] = spec["description"]
        model_result["outcome"] = outcome
        model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
        results.append(model_result)
    return model, results

def run_analysis(df_all, df_surv, panel_all, creators, df_comments_valid=None, df_comments_all=None, acsi_scores=None, submonth_panel=None, run_legacy=True):
    print("\n=== Step 5: analysis ===")
    clean_analysis_artifacts()

    results = {
        "gse_main": None, "gse_dimensions": None, "gse_secondary": None, "gse_quartiles": None, 
        "gse_event_study": None, "gse_construct_validity": None,
        "gse_permutation": None, "gse_covariate_adj": None,
        "acsi_three_dimensional": None,
        "acsi_mechanisms": None, "acsi_mechanisms_joint": None,
        "acsi_component_correlations": None, "binary_did_consistency": None,
        "substitutability_hypothesis": None,
        "substitutability_hypotheses": None,
        "q1a": None, "q1a_robust": None, "q1b": None, "q1c": None,
        "q1d": None, "q1e": None, "q1e_step_slope": None, "q1e_loo": None,
        "q1a_het": None, "q1a_het_loo": None,
        "q1a_no_covid": None, "q1a_no_api": None,
        "q2_attrition": None,
        "q2_rate": None, "q2_score": None, "q2_comment": None, "q2_joint": None,
        "q2_bonferroni": None, "q2_robust_quartile": None,
        "q2_robust_score": None,            
        "q2_score_quartile_table": None,
        "q3": None, "q4": None,
        "attrition_bounds": None,
        "predeparture_engagement": None,
        "classification_sensitivity": None,
        "placebo_rolling": None,
        "randomization_inference": None,
        "event_study": None,
        "content_validation": None,
        "q1f": None,                        
        "q1g": None,                        
    }

    # ------------------------------------------------------------------
    # MAIN: AI Content Substitutability dose-response models
    # ------------------------------------------------------------------
    results["acsi_component_correlations"] = compute_acsi_component_correlation(acsi_scores)
    if results["acsi_component_correlations"]:
        print(f"\n--- {INDEX_SHORT} measurement diagnostics: component correlations saved ---")

    print(f"\n--- MAIN: Three-dimensional {INDEX_SHORT} dose-response DiD ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Three-dimensional main dose-response")
        three_dimensional_model, three_dimensional_results = fit_three_dimensional_acsi_model(acsi_panel)
        if three_dimensional_results:
            pd.DataFrame(three_dimensional_results).to_csv(
                TABLES_DIR / "acsi_three_dimensional_main_model.csv",
                index=False,
            )
            (TABLES_DIR / "acsi_three_dimensional_main_model.tex").write_text(
                three_dimensional_model.summary().as_latex()
            )
            results["acsi_three_dimensional"] = three_dimensional_results
            for model_result in three_dimensional_results:
                print(
                    f"  {model_result['label']} x Post: coef={fmt_signed4(model_result['coef'])} "
                    f"SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])} "
                    f"effect={fmt_signed1(model_result['percent_effect_full_exposure'])}%"
                )
    except Exception as e:
        print(f"  Three-dimensional {INDEX_SHORT} main failed: {e}")

    print(f"\n--- Supporting: aggregate {INDEX_LABEL} dose-response DiD ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Main dose-response")
        main_acsi_model = fit_ols(
            "log_posts ~ gse_post + C(subreddit) + C(year_month)",
            acsi_panel,
            cluster_col="subreddit",
        )

        if main_acsi_model:
            model_result = reg_result(main_acsi_model, "gse_post")
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            results["gse_main"] = model_result
            print(f"  {INDEX_SHORT} x Post coef={fmt_signed4(model_result['coef'])} SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])} "
                  f"full exposure effect={fmt_signed1(model_result['percent_effect_full_exposure'])}%")
            (TABLES_DIR / "gse_main_dose_response.tex").write_text(main_acsi_model.summary().as_latex())
    except Exception as e:
        print(f"  {INDEX_SHORT} main failed: {e}")

    print(f"\n--- {INDEX_SHORT} dose-response: covariate adjusted model ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Covariate-adjusted dose-response")
        pre_covariates = add_pre_covariates(acsi_panel)
        adjusted_panel = acsi_panel.merge(pre_covariates, on="subreddit", how="left")
        adjusted_panel["pre_avg_post"] = adjusted_panel["pre_avg_log_posts"] * adjusted_panel["post_shock"]
        adjusted_panel["pre_trend_post"] = adjusted_panel["pre_trend"] * adjusted_panel["post_shock"]
        adjusted_panel["log_mu_post"] = adjusted_panel["log_mu_k"] * adjusted_panel["post_shock"]

        adjusted_acsi_model = fit_ols(
            "log_posts ~ gse_post + pre_avg_post + pre_trend_post + log_mu_post + C(subreddit) + C(year_month)",
            adjusted_panel, cluster_col="subreddit"
        )
        if adjusted_acsi_model:
            model_result = reg_result(adjusted_acsi_model, "gse_post")
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            results["gse_covariate_adj"] = model_result
            print(f"  Adjusted {INDEX_SHORT} x Post coef={fmt_signed4(model_result['coef'])} SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])}")
            (TABLES_DIR / "gse_covariate_adj.tex").write_text(adjusted_acsi_model.summary().as_latex())
    except Exception as e:
        print(f"  {INDEX_SHORT} covariate adjusted failed: {e}")

    print(f"\n--- {INDEX_SHORT} dimension-specific dose-response models ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Dimension-specific dose-response")
        dimension_model_results = []
        for spec in ACSI_DIMENSION_SPECS:
            term = spec["post"]
            model_data = acsi_panel.dropna(subset=[term, "log_posts"]).copy()
            dimension_model = fit_ols(
                f"log_posts ~ {term} + C(subreddit) + C(year_month)",
                model_data,
                cluster_col="subreddit",
            )
            if not dimension_model:
                continue
            model_result = reg_result(dimension_model, term)
            model_result["dimension"] = spec["source"]
            model_result["term"] = term
            model_result["label"] = spec["label"]
            model_result["description"] = spec["description"]
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            dimension_model_results.append(model_result)
            print(
                f"  {spec['label']}: coef={fmt_signed4(model_result['coef'])} "
                f"p={fmt4(model_result['pvalue'])} effect={fmt_signed1(model_result['percent_effect_full_exposure'])}%"
            )

        if dimension_model_results:
            dimension_results_table = pd.DataFrame(dimension_model_results)
            dimension_results_table.to_csv(TABLES_DIR / "gse_dimension_models.csv", index=False)
            results["gse_dimensions"] = dimension_model_results

            try:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                x = np.arange(len(dimension_results_table))
                y = dimension_results_table["coef"].astype(float).values
                se = dimension_results_table["se"].astype(float).values
                colors = ["#f85149" if v < 0 else "#56d364" for v in y]
                ax.bar(x, y, color=colors, alpha=0.75)
                ax.errorbar(x, y, yerr=1.96 * se, fmt="none", color="#f0f6fc", capsize=5, linewidth=1)
                ax.axhline(0, color="#c9d1d9", linestyle="--", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(dimension_results_table["label"], rotation=25, ha="right")
                ax.set_ylabel("Post-shock effect on log monthly posts")
                ax.set_title(f"{INDEX_SHORT} component-wise dose-response")
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "gse_dimension_models.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as fe:
                print(f"  {INDEX_SHORT} dimension figure failed: {fe}")
    except Exception as e:
        print(f"  {INDEX_SHORT} dimension-specific models failed: {e}")

    print(f"\n--- {INDEX_SHORT} mechanism models: generation, physical, personal ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Mechanism dose-response")
        mechanism_model_results = []
        for spec in ACSI_MECHANISM_SPECS:
            term = spec["post"]
            model_data = acsi_panel.dropna(subset=[term, "log_posts"]).copy()
            mechanism_model = fit_ols(
                f"log_posts ~ {term} + C(subreddit) + C(year_month)",
                model_data,
                cluster_col="subreddit",
            )
            if not mechanism_model:
                continue
            model_result = reg_result(mechanism_model, term)
            model_result["dimension"] = spec["source"]
            model_result["term"] = term
            model_result["label"] = spec["label"]
            model_result["description"] = spec["description"]
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            mechanism_model_results.append(model_result)
            print(
                f"  {spec['label']}: coef={fmt_signed4(model_result['coef'])} "
                f"p={fmt4(model_result['pvalue'])} effect={fmt_signed1(model_result['percent_effect_full_exposure'])}%"
            )

        if mechanism_model_results:
            pd.DataFrame(mechanism_model_results).to_csv(TABLES_DIR / "acsi_mechanism_models.csv", index=False)
            results["acsi_mechanisms"] = mechanism_model_results

        joint_terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
        joint_model_data = acsi_panel.dropna(subset=joint_terms + ["log_posts"]).copy()
        joint_model = fit_ols(
            "log_posts ~ "
            + " + ".join(joint_terms)
            + " + C(subreddit) + C(year_month)",
            joint_model_data,
            cluster_col="subreddit",
        )
        if joint_model:
            joint_results = []
            label_by_term = {spec["post"]: spec["label"] for spec in ACSI_MECHANISM_SPECS}
            for term in joint_terms:
                model_result = reg_result(joint_model, term)
                model_result["term"] = term
                model_result["label"] = label_by_term.get(term, term)
                model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
                joint_results.append(model_result)
                print(
                    f"  Joint {label_by_term.get(term, term)}: coef={fmt_signed4(model_result['coef'])} "
                    f"p={fmt4(model_result['pvalue'])}"
                )
            pd.DataFrame(joint_results).to_csv(TABLES_DIR / "acsi_mechanism_joint_model.csv", index=False)
            (TABLES_DIR / "acsi_mechanism_joint_model.tex").write_text(joint_model.summary().as_latex())
            results["acsi_mechanisms_joint"] = joint_results
    except Exception as e:
        print(f"  {INDEX_SHORT} mechanism models failed: {e}")

    print("\n--- Binary DiD consistency check ---")
    try:
        binary_panel = submonth_panel.copy()
        binary_panel["sub_role"] = binary_panel["subreddit"].map(SUBREDDITS)
        binary_panel = binary_panel[binary_panel["sub_role"].isin(["treatment", "control"])].copy()
        binary_panel = acsi_model_panel(binary_panel, "Binary DiD consistency check")
        binary_panel["treated"] = (binary_panel["sub_role"] == "treatment").astype(int)
        binary_panel["treated_post"] = binary_panel["treated"] * binary_panel["post_shock"]
        binary_model = fit_ols(
            "log_posts ~ treated_post + C(subreddit) + C(year_month)",
            binary_panel,
            cluster_col="subreddit",
        )
        if binary_model:
            model_result = reg_result(binary_model, "treated_post")
            model_result["percent_effect"] = pct_effect_from_coef(model_result["coef"])
            model_result["label"] = "Treatment x Post"
            results["binary_did_consistency"] = model_result
            (TABLES_DIR / "binary_did_consistency.tex").write_text(binary_model.summary().as_latex())
            print(
                f"  Treatment x Post coef={fmt_signed4(model_result['coef'])} "
                f"p={fmt4(model_result['pvalue'])} effect={fmt_signed1(model_result['percent_effect'])}%"
            )
    except Exception as e:
        print(f"  Binary DiD consistency check failed: {e}")

    print("\n--- H1 vs H2 hypothesis adjudication ---")
    try:
        hypothesis_rows = []
        if results.get("gse_main") is not None:
            hypothesis_rows.append(classify_substitutability_hypothesis(results["gse_main"], "Main dose-response"))
        if results.get("gse_covariate_adj") is not None:
            hypothesis_rows.append(classify_substitutability_hypothesis(results["gse_covariate_adj"], "Covariate-adjusted dose-response"))

        if hypothesis_rows:
            results["substitutability_hypothesis"] = hypothesis_rows[0]
            results["substitutability_hypotheses"] = hypothesis_rows
            pd.DataFrame(hypothesis_rows).to_csv(TABLES_DIR / "substitutability_hypothesis_test.csv", index=False)
            for row in hypothesis_rows:
                print(
                    f"  {row['model']}: {row['conclusion']} "
                    f"(coef={fmt_signed4(row['coef'])}, p={fmt4(row['pvalue'])})"
                )
        else:
            print("  No usable dose-response estimates for H1/H2 adjudication.")
    except Exception as e:
        print(f"  H1/H2 adjudication failed: {e}")

    print(f"\n--- {INDEX_SHORT} permutation inference ---")
    try:
        if not results.get("gse_main") or results["gse_main"].get("coef") is None:
            print(f"  Skipping {INDEX_SHORT} permutation: no observed {INDEX_SHORT} coefficient.")
        else:
            observed_coef = results["gse_main"]["coef"]
            permutation_panel = acsi_model_panel(submonth_panel.copy(), "Permutation inference")
            use_fast_permutation = is_balanced_two_way_panel(permutation_panel, "subreddit", "year_month")
            if use_fast_permutation:
                y_resid = residualize_two_way(permutation_panel["log_posts"], permutation_panel["subreddit"], permutation_panel["year_month"])
                post_shock_values = permutation_panel["post_shock"].astype(float).to_numpy()
            else:
                print("  Panel is not balanced; using exact OLS fallback for permutations.")

            subreddit_permutation_frame = permutation_panel[["subreddit", "gse"]].drop_duplicates()
            subreddit_permutation_frame["mu_k"] = subreddit_permutation_frame["subreddit"].map(MU_K).fillna(0.5)
            subreddit_permutation_frame["size_bin"] = pd.qcut(subreddit_permutation_frame["mu_k"].rank(method="first"), q=4, labels=False)

            rng = np.random.default_rng(RANDOM_SEED)
            permuted_coefficients = []

            for _ in tqdm(range(N_RANDOMIZATION_PERMS), desc=f"  {INDEX_SHORT} perms"):
                permuted_scores = []
                for _, group in subreddit_permutation_frame.groupby("size_bin"):
                    shuffled = group["gse"].sample(frac=1, replace=False, random_state=int(rng.integers(1e9))).values
                    permuted_subreddit_scores = group[["subreddit"]].copy()
                    permuted_subreddit_scores["permuted_acsi_scores"] = shuffled
                    permuted_scores.append(permuted_subreddit_scores)

                permuted_scores = pd.concat(permuted_scores, ignore_index=True)
                if use_fast_permutation:
                    permuted_acsi_scores = permutation_panel["subreddit"].map(permuted_scores.set_index("subreddit")["permuted_acsi_scores"])
                    permuted_coefficient = two_way_fe_coef_from_residualized_y(
                        y_resid,
                        permuted_acsi_scores.astype(float).to_numpy() * post_shock_values,
                        permutation_panel["subreddit"],
                        permutation_panel["year_month"],
                    )
                else:
                    exact_permutation_panel = permutation_panel.drop(
                        columns=["permuted_acsi_scores"], errors="ignore"
                    ).merge(permuted_scores, on="subreddit", how="left")
                    exact_permutation_panel["permuted_acsi_post"] = (
                        exact_permutation_panel["permuted_acsi_scores"]
                        * exact_permutation_panel["post_shock"]
                    )
                    exact_permutation_model = fit_ols(
                        "log_posts ~ permuted_acsi_post + C(subreddit) + C(year_month)",
                        exact_permutation_panel, cluster_col="subreddit"
                    )
                    permuted_coefficient = (
                        None
                        if not exact_permutation_model
                        else safe_float(exact_permutation_model.params.get("permuted_acsi_post", np.nan))
                    )
                if permuted_coefficient is not None:
                    permuted_coefficients.append(permuted_coefficient)

            if permuted_coefficients:
                left_tail_permutation_pvalue = float(np.mean([coef <= observed_coef for coef in permuted_coefficients]))
                two_sided_permutation_pvalue = float(np.mean([abs(coef) >= abs(observed_coef) for coef in permuted_coefficients]))
                results["gse_permutation"] = {
                    "observed_coef": observed_coef,
                    "perm_pvalue_left": left_tail_permutation_pvalue,
                    "perm_pvalue_two_sided": two_sided_permutation_pvalue,
                    "perm_mean": safe_float(np.mean(permuted_coefficients)),
                    "perm_sd": safe_float(np.std(permuted_coefficients)),
                    "n_perms": len(permuted_coefficients),
                }
                pd.DataFrame({"coef": permuted_coefficients}).to_csv(TABLES_DIR / "gse_permutation_coefs.csv", index=False)
                print(f"  {INDEX_SHORT} permutation p={left_tail_permutation_pvalue:.4f} (left-sided)")
    except Exception as e:
        print(f"  {INDEX_SHORT} permutation failed: {e}")

    print(f"\n--- {INDEX_SHORT} dose-response: secondary outcomes ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Secondary outcome dose-response")
        secondary_model_results = []
        outcome_labels = {
            "log_active_creators": "Active creators",
            "log_calibrated_output": "Calibrated output",
            "log_avg_score": "Average score per post",
        }

        for outcome in ["log_active_creators", "log_calibrated_output", "log_avg_score"]:
            model_data = acsi_panel if outcome != "log_avg_score" else acsi_panel[acsi_panel["posts"] > 0].copy()
            secondary_model = fit_ols(f"{outcome} ~ gse_post + C(subreddit) + C(year_month)", model_data, cluster_col="subreddit")
            if secondary_model:
                model_result = reg_result(secondary_model, "gse_post")
                model_result["outcome"] = outcome
                model_result["label"] = outcome_labels[outcome]
                model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
                secondary_model_results.append(model_result)
                print(f"  {outcome}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])}")

        if secondary_model_results:
            pd.DataFrame(secondary_model_results).to_csv(TABLES_DIR / "gse_secondary_outcomes.csv", index=False)
            results["gse_secondary"] = secondary_model_results
    except Exception as e:
        print(f"  {INDEX_SHORT} secondary outcomes failed: {e}")

    print(f"\n--- {INDEX_SHORT} quartile nonlinearity check ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Quartile nonlinearity check")
        sub_scores = acsi_panel[["subreddit", "gse"]].drop_duplicates()
        sub_scores["gse_rank"] = sub_scores["gse"].rank(method="first")
        sub_scores["gse_q"] = pd.qcut(
            sub_scores["gse_rank"], q=4,
            labels=["Q1_low", "Q2_midlow", "Q3_midhigh", "Q4_high"]
        )

        acsi_panel = acsi_panel.merge(sub_scores[["subreddit", "gse_q"]], on="subreddit", how="left")

        for q in ["Q2_midlow", "Q3_midhigh", "Q4_high"]:
            acsi_panel[f"{q}_post"] = ((acsi_panel["gse_q"] == q).astype(int) * acsi_panel["post_shock"])

        quartile_model = fit_ols(
            "log_posts ~ Q2_midlow_post + Q3_midhigh_post + Q4_high_post + C(subreddit) + C(year_month)",
            acsi_panel, cluster_col="subreddit",
        )

        if quartile_model:
            quartile_model_results = []
            quartile_label_map = {
                "Q2_midlow_post": "Medium-low exposure",
                "Q3_midhigh_post": "Medium-high exposure",
                "Q4_high_post": "High exposure",
            }
            for term in ["Q2_midlow_post", "Q3_midhigh_post", "Q4_high_post"]:
                model_result = reg_result(quartile_model, term)
                model_result["term"] = term
                model_result["label"] = quartile_label_map.get(term, term)
                model_result["percent_effect"] = pct_effect_from_coef(model_result["coef"])
                quartile_model_results.append(model_result)
                print(f"  {term}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])} effect={fmt_signed1(model_result['percent_effect'])}%")

            pd.DataFrame(quartile_model_results).to_csv(TABLES_DIR / "gse_quartile_check.csv", index=False)
            results["gse_quartiles"] = quartile_model_results
            
            try:
                quartile_results_table = pd.DataFrame(quartile_model_results)
                fig, ax = plt.subplots(figsize=(7, 4))
                x = np.arange(len(quartile_results_table))
                y = quartile_results_table["coef"].values
                se = quartile_results_table["se"].values

                ax.bar(x, y, color="#4c8bf5", alpha=0.8)
                ax.errorbar(x, y, yerr=1.96 * se, fmt="none", color="black", capsize=5)
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(quartile_results_table["label"])
                ax.set_ylabel("Post-shock effect on log monthly posts")
                ax.set_title(f"{INDEX_SHORT} quartile dose-response check\nReference: lowest-exposure quartile")
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "gse_quartile_check.png", dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as fe:
                print(f"  {INDEX_SHORT} quartile figure failed: {fe}")

    except Exception as e:
        print(f"  {INDEX_SHORT} quartile check failed: {e}")

    print(f"\n--- {INDEX_SHORT} event study: dose-response pre-trends ---")
    try:
        event_study_panel = acsi_model_panel(submonth_panel.copy(), "Event-study dose-response")
        event_study_panel = event_study_panel[event_study_panel["year_month_dt"] >= SHOCK_MONTH - pd.DateOffset(months=24)].copy()

        def get_bin(dt):
            mb = (SHOCK_MONTH.year - dt.year) * 12 + (SHOCK_MONTH.month - dt.month)
            if mb <= 0: return "post"
            elif mb <= 6: return "pre_6"
            elif mb <= 12: return "pre_12"
            elif mb <= 18: return "pre_18"
            else: return "pre_24"

        event_study_panel["bin"] = event_study_panel["year_month_dt"].apply(get_bin)

        for b in ["pre_24", "pre_18", "pre_12", "post"]:
            event_study_panel[f"gse_{b}"] = event_study_panel["gse"] * (event_study_panel["bin"] == b).astype(int)

        event_study_model = fit_ols(
            "log_posts ~ gse_pre_24 + gse_pre_18 + gse_pre_12 + gse_post + C(subreddit) + C(year_month)",
            event_study_panel, cluster_col="subreddit",
        )

        event_rows = []
        if event_study_model:
            for b in ["pre_24", "pre_18", "pre_12", "post"]:
                term = f"gse_{b}"
                model_result = reg_result(event_study_model, term)
                model_result["bin"] = b
                event_rows.append(model_result)
                print(f"  {b}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])}")

            pre_terms = ["gse_pre_24", "gse_pre_18", "gse_pre_12"]
            try:
                restriction_matrix = np.zeros((len(pre_terms), len(event_study_model.params)))
                param_names = list(event_study_model.params.index)
                valid_pre = []
                for i, pretrend_term in enumerate(pre_terms):
                    if pretrend_term in param_names:
                        restriction_matrix[i, param_names.index(pretrend_term)] = 1.0
                        valid_pre.append(pretrend_term)
                if valid_pre:
                    restriction_matrix = restriction_matrix[:len(valid_pre), :]
                    f_test = event_study_model.f_test(restriction_matrix)
                    pretrend_pvalue = safe_float(f_test.pvalue)
                else:
                    pretrend_pvalue = None
            except Exception:
                pretrend_pvalue = None

            results["gse_event_study"] = {
                "rows": event_rows,
                "pretrend_pvalue": pretrend_pvalue,
            }
            pd.DataFrame(event_rows).to_csv(TABLES_DIR / "gse_event_study.csv", index=False)
            
            try:
                plot_rows = pd.DataFrame(event_rows)
                if not plot_rows.empty:
                    order = ["pre_24", "pre_18", "pre_12", "post"]
                    plot_rows["bin"] = pd.Categorical(plot_rows["bin"], categories=order, ordered=True)
                    plot_rows = plot_rows.sort_values("bin")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    x = np.arange(len(plot_rows))
                    y = plot_rows["coef"].values
                    se = plot_rows["se"].values

                    colors = ["#4c8bf5" if b != "post" else "#e63946" for b in plot_rows["bin"]]

                    ax.errorbar(x, y, yerr=1.96 * se, fmt="o", capsize=5, color="black", ecolor="gray")
                    ax.scatter(x, y, c=colors, s=60, zorder=3)
                    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels(["24–18 mo pre", "18–12 mo pre", "12–6 mo pre", "Post"])
                    ax.set_ylabel(f"{INDEX_SHORT} x period coefficient")
                    ax.set_title(f"{INDEX_SHORT} dose-response event study")

                    if pretrend_pvalue is not None:
                        ax.text(0.02, 0.95, f"Joint pretrend p={pretrend_pvalue:.3f}", transform=ax.transAxes, va="top", fontsize=10)

                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / "gse_event_study.png", dpi=150, bbox_inches="tight")
                    plt.close()
            except Exception as fe:
                print(f"  {INDEX_SHORT} event-study figure failed: {fe}")
                
    except Exception as e:
        print(f"  {INDEX_SHORT} event study failed: {e}")

    print(f"\n--- {INDEX_SHORT} construct validity ---")
    try:
        if df_comments_all is not None and not df_comments_all.empty and acsi_scores is not None:
            pre_months_cv = ["2022-09", "2022-10", "2022-11"]
            post_months_cv = ["2022-12", "2023-01", "2023-02"]

            df_cv = df_comments_all[df_comments_all["year_month"].isin(pre_months_cv + post_months_cv)].copy()

            if not df_cv.empty:
                df_cv["cv_period"] = np.where(df_cv["year_month"].isin(post_months_cv), "post", "pre")

                if {"ai_kw_count", "tool_kw_count", "comment_count"}.issubset(df_cv.columns):
                    sub_period = df_cv.groupby(["subreddit", "cv_period"], as_index=False).agg(
                        ai_kw_count=("ai_kw_count", "sum"),
                        tool_kw_count=("tool_kw_count", "sum"),
                        n_comments=("comment_count", "sum"),
                    )
                    sub_period["ai_kw_rate"] = sub_period["ai_kw_count"] / sub_period["n_comments"].replace(0, np.nan)
                    sub_period["tool_kw_rate"] = sub_period["tool_kw_count"] / sub_period["n_comments"].replace(0, np.nan)
                else:
                    df_cv["has_ai_kw"] = (
                        df_cv["body"].fillna("")
                        .str.contains(AI_BROAD_PATTERN.pattern, flags=re.IGNORECASE, na=False, regex=True)
                        .astype(int)
                    )
                    df_cv["has_tool_kw"] = (
                        df_cv["body"].fillna("")
                        .str.contains(AI_TOOL_PATTERN.pattern, flags=re.IGNORECASE, na=False, regex=True)
                        .astype(int)
                    )
                    sub_period = df_cv.groupby(["subreddit", "cv_period"]).agg(
                        ai_kw_rate=("has_ai_kw", "mean"),
                        tool_kw_rate=("has_tool_kw", "mean"),
                        n_comments=("comment_id", "count"),
                    ).reset_index()

                wide_ai = sub_period.pivot(index="subreddit", columns="cv_period", values="ai_kw_rate").reset_index().rename(columns={"pre": "ai_pre", "post": "ai_post"})
                wide_tool = sub_period.pivot(index="subreddit", columns="cv_period", values="tool_kw_rate").reset_index().rename(columns={"pre": "tool_pre", "post": "tool_post"})
                counts = sub_period.pivot(index="subreddit", columns="cv_period", values="n_comments").reset_index().rename(columns={"pre": "n_pre", "post": "n_post"})

                wide = wide_ai.merge(wide_tool, on="subreddit", how="inner").merge(counts, on="subreddit", how="inner")

                needed = {"ai_pre", "ai_post", "tool_pre", "tool_post", "n_pre", "n_post"}
                if needed.issubset(wide.columns):
                    wide["delta_ai_kw_rate"] = wide["ai_post"] - wide["ai_pre"]
                    wide["delta_tool_kw_rate"] = wide["tool_post"] - wide["tool_pre"]

                    wide = wide[(wide["n_pre"] >= 50) & (wide["n_post"] >= 50)]
                    wide = wide.merge(acsi_scores[["subreddit", "gse"]], on="subreddit", how="inner")

                    if len(wide) > 5:
                        p_r_ai, p_p_ai = safe_corr(wide["gse"], wide["delta_ai_kw_rate"], "pearson")
                        s_r_ai, s_p_ai = safe_corr(wide["gse"], wide["delta_ai_kw_rate"], "spearman")
                        p_r_tool, p_p_tool = safe_corr(wide["gse"], wide["delta_tool_kw_rate"], "pearson")
                        s_r_tool, s_p_tool = safe_corr(wide["gse"], wide["delta_tool_kw_rate"], "spearman")

                        print(f"  {INDEX_SHORT} vs Δ Broad AI keyword rate: Pearson r={fmt4(p_r_ai)} p={fmt4(p_p_ai)}")
                        print(f"  {INDEX_SHORT} vs Δ Narrow Tool keyword rate: Pearson r={fmt4(p_r_tool)} p={fmt4(p_p_tool)}")

                        wide.to_csv(TABLES_DIR / "gse_construct_validity_keywords.csv", index=False)

                        results["gse_construct_validity"] = {
                            "broad_pearson_r": p_r_ai,
                            "broad_pearson_pvalue": p_p_ai,
                            "narrow_pearson_r": p_r_tool,
                            "narrow_pearson_pvalue": p_p_tool,
                            "n_subreddits": safe_int(len(wide)),
                        }
    except Exception as e:
        print(f"  Construct validity failed: {e}")

    if not run_legacy:
        print("\n--- Legacy Q1-Q4 skipped (--gse-only) ---")
        print(f"\n  Tables -> {TABLES_DIR}")
        print(f"  Figures -> {FIGURES_DIR}")
        return results

    # ------------------------------------------------------------------
    # Old Binary Legacy code below
    # ------------------------------------------------------------------
    stable_authors = set(creators[creators["is_stable"] == 1]["author"].unique())
    df_did = panel_all[panel_all["sub_role"].isin(["treatment", "control"])].copy()
    df_did["treated"] = (df_did["sub_role"] == "treatment").astype(int)
    df_did["did"]     = df_did["treated"] * df_did["post_shock"]

    df_did_rob = panel_all[panel_all["sub_role_main"].isin(["treatment", "control"])].copy()
    df_did_rob["treated"] = (df_did_rob["sub_role_main"] == "treatment").astype(int)
    df_did_rob["did"]     = df_did_rob["treated"] * df_did_rob["post_shock"]
    df_did_rob["sub_role"] = df_did_rob["sub_role_main"]

    def sub_agg_did(df_panel, stable_only=True):
        df_s = df_panel[df_panel["author"].isin(stable_authors)].copy() if stable_only else df_panel.copy()
        role_col = "sub_role" if "sub_role" in df_s.columns else "sub_role_main"
        agg  = (
            df_s.groupby(["subreddit", "year_month", "post_shock", role_col, "treated"])["x_ikt"]
            .sum().reset_index()
        )
        if role_col != "sub_role":
            agg = agg.rename(columns={role_col: "sub_role"})
        agg["did"] = agg["treated"] * agg["post_shock"]
        return agg

    print("\n--- Legacy Q1a: DiD primary spec ---")
    try:
        agg_q1a = sub_agg_did(df_did)
        m_q1a   = fit_ols("x_ikt ~ did + C(subreddit) + C(year_month)", agg_q1a, cluster_col="subreddit")
        if m_q1a:
            results["q1a"] = reg_result(m_q1a, "did")
            (TABLES_DIR / "q1a_did.tex").write_text(m_q1a.summary().as_latex())
    except Exception as e:
        print(f"  Q1a failed: {e}")

    print("\n--- Legacy Q1b + attrition scenario bounds ---")
    all_treat_pre = set(df_all[(df_all["sub_role"] == "treatment") & (df_all["post_shock"] == 0)]["author"].unique())
    survived_treat = set(df_all[(df_all["sub_role"] == "treatment") & (df_all["post_shock"] == 1)]["author"].unique())
    all_ctrl_pre   = set(df_all[(df_all["sub_role"] == "control") & (df_all["post_shock"] == 0)]["author"].unique())
    survived_ctrl  = set(df_all[(df_all["sub_role"] == "control") & (df_all["post_shock"] == 1)]["author"].unique())

    stable_treat_pre      = stable_authors & all_treat_pre
    stable_treat_survived = stable_authors & survived_treat
    stable_treat_dropout  = stable_treat_pre - stable_treat_survived

    dropout_n    = len(stable_treat_dropout)
    dropout_rate = dropout_n / max(len(stable_treat_pre), 1)

    panel_treat = panel_all[(panel_all["sub_role"] == "treatment") & (panel_all["author"].isin(stable_authors))].copy()
    panel_ctrl  = panel_all[(panel_all["sub_role"] == "control") & (panel_all["author"].isin(stable_authors))].copy()

    try:
        pre_t   = panel_treat[panel_treat["post_shock"]==0].groupby("author")["x_ikt"].mean()
        post_t  = panel_treat[panel_treat["post_shock"]==1].groupby("author")["x_ikt"].mean()
        q1b     = pd.DataFrame({"pre": pre_t, "post": post_t}).dropna()
        pre_c   = panel_ctrl[panel_ctrl["post_shock"]==0].groupby("author")["x_ikt"].mean()
        post_c  = panel_ctrl[panel_ctrl["post_shock"]==1].groupby("author")["x_ikt"].mean()
        q1b_ctrl= pd.DataFrame({"pre": pre_c, "post": post_c}).dropna()

        if len(q1b) >= 2:
            t_stat, p_two = stats.ttest_rel(q1b["pre"], q1b["post"])
            mean_pre  = q1b["pre"].mean()
            mean_post = q1b["post"].mean()
            pct_obs   = (mean_pre - mean_post) / mean_pre * 100

            ctrl_drop = float("nan")
            if len(q1b_ctrl) >= 2:
                cp        = q1b_ctrl["pre"].mean()
                cpo       = q1b_ctrl["post"].mean()
                ctrl_drop = (cp - cpo) / cp * 100

            n_surv   = len(q1b)
            n_tot    = n_surv + dropout_n
            post_lb  = (mean_post * n_surv + mean_pre * dropout_n) / n_tot
            drop_lb  = (mean_pre - post_lb) / mean_pre * 100
            post_ub  = (mean_post * n_surv) / n_tot
            drop_ub  = (mean_pre - post_ub) / mean_pre * 100

            results["attrition_bounds"] = {
                "lower_bound":  safe_float(drop_lb), "observed":     safe_float(pct_obs),
                "upper_bound":  safe_float(drop_ub), "dropout_rate": safe_float(dropout_rate),
                "dropout_n":    safe_int(dropout_n), "control_drop": safe_float(ctrl_drop),
            }
    except Exception as e:
        print(f"  Q1b/attrition failed: {e}")

    print("\n--- Legacy Q2 Attrition: survival ~ log(c_i) ---")
    try:
        from statsmodels.formula.api import logit as smf_logit
        df_treat_pre = df_all[(df_all["sub_role"] == "treatment") & (df_all["post_shock"] == 0)].copy()
        t_attr = df_treat_pre.groupby("author").agg(tp_count=("post_id","count")).reset_index()
        t_attr["tp_rate"]    = t_attr["tp_count"] / max(N_PRE_MONTHS, 1)
        t_attr["treat_c_i"]  = 1.0 / t_attr["tp_rate"].replace(0, np.nan)
        t_attr["log_c_i"]    = np.log(t_attr["treat_c_i"])

        df_treat_post    = df_all[(df_all["sub_role"] == "treatment") & (df_all["post_shock"] == 1)].copy()
        survivors_treat  = set(df_treat_post["author"].unique())
        t_attr["survived_post_shock"] = t_attr["author"].isin(survivors_treat).astype(int)

        t_attr = t_attr.dropna(subset=["log_c_i"])
        if not t_attr.empty:
            m_attr = smf_logit("survived_post_shock ~ log_c_i", data=t_attr).fit(disp=0)
            results["q2_attrition"] = reg_result(m_attr, "log_c_i")
    except Exception as e:
        print(f"  Q2 attrition failed: {e}")

    print("\n--- Legacy Q2: behavioral shift ~ log(c_i) ---")
    df_treat_posts = df_all[df_all["sub_role"] == "treatment"].copy()
    t_pre  = df_treat_posts[df_treat_posts["post_shock"]==0].groupby("author").agg(
        tp_count=("post_id","count"), tp_score=("score","median"),
    ).reset_index()
    t_post = df_treat_posts[df_treat_posts["post_shock"]==1].groupby("author").agg(
        tpo_count=("post_id","count"), tpo_score=("score","median"),
    ).reset_index()
    t_beh = t_pre.merge(t_post, on="author", how="left")
    t_beh["tpo_count"]     = t_beh["tpo_count"].fillna(0)
    t_beh["tpo_score"]     = t_beh["tpo_score"].fillna(0)
    t_beh["tp_rate"]       = t_beh["tp_count"]  / max(N_PRE_MONTHS, 1)
    t_beh["tpo_rate"]      = t_beh["tpo_count"] / max(N_POST_MONTHS, 1)
    safe_tr                = t_beh["tp_rate"].replace(0, np.nan)
    t_beh["t_delta_rate"]  = (t_beh["tpo_rate"] - t_beh["tp_rate"]) / safe_tr
    t_beh["t_delta_score"] = t_beh["tpo_score"] - t_beh["tp_score"]
    t_beh["treat_c_i"]     = 1.0 / t_beh["tp_rate"].replace(0, np.nan)
    t_beh["treat_log_c_i"] = np.log(t_beh["treat_c_i"])
    for col in ["t_delta_rate", "t_delta_score"]:
        t_beh[col] = winsor(t_beh[col])

    df_q2 = creators[["author","delta_comment_rate"]].merge(
        t_beh, on="author", how="inner"
    ).dropna(subset=["treat_c_i","t_delta_rate","t_delta_score"])
    df_q2 = df_q2[df_q2["treat_c_i"] > 0].copy()
    df_q2["c_i"]       = df_q2["treat_c_i"]
    df_q2["log_c_i"]   = df_q2["treat_log_c_i"]
    df_q2["delta_rate"]  = df_q2["t_delta_rate"]
    df_q2["delta_score"] = df_q2["t_delta_score"]

    for outcome, label, key in [("delta_rate",  "delta posting rate", "q2_rate"), ("delta_score", "delta avg score",    "q2_score")]:
        m = fit_ols(f"{outcome} ~ log_c_i", df_q2)
        if m: results[key] = reg_result(m, "log_c_i")

    df_q2_comment    = pd.DataFrame()
    try:
        if df_comments_valid is not None and not df_comments_valid.empty:
            tc = df_comments_valid[df_comments_valid["subreddit"].map(SUBREDDITS) == "treatment"].copy()
            if not tc.empty:
                tc_pre  = aggregate_comment_counts(tc[tc["post_shock"]==0], ["author"], "tc_pre")
                tc_post = aggregate_comment_counts(tc[tc["post_shock"]==1], ["author"], "tc_post")
                tcb               = tc_pre.merge(tc_post, on="author", how="left")
                tcb["tc_post"]    = tcb["tc_post"].fillna(0)
                tcb["tc_pre_rate"]  = tcb["tc_pre"]  / max(N_PRE_MONTHS, 1)
                tcb["tc_post_rate"] = tcb["tc_post"] / max(N_POST_MONTHS, 1)
                safe_tc           = tcb["tc_pre_rate"].replace(0, np.nan)
                tcb["t_dcr"]      = (tcb["tc_post_rate"] - tcb["tc_pre_rate"]) / safe_tc
                tcb["t_dcr"]      = winsor(tcb["t_dcr"])
                df_q2_comment     = df_q2.merge(tcb[["author","t_dcr"]], on="author", how="inner")
                df_q2_comment     = df_q2_comment.dropna(subset=["t_dcr"])
    except Exception as ce:
        print(f"  comment rate setup failed: {ce}")

    if not df_q2_comment.empty and "t_dcr" in df_q2_comment.columns and df_q2_comment["t_dcr"].nunique() > 1:
        m = fit_ols("t_dcr ~ log_c_i", df_q2_comment)
        if m:
            results["q2_comment"] = reg_result(m, "log_c_i")

    print("\n--- Legacy Q3: stable creator engagement DiD ---")
    try:
        df_q3 = df_surv[df_surv["author"].isin(stable_authors)].copy()
        df_q3 = df_q3[df_q3["sub_role"].isin(["treatment","control"])]
        if not df_q3.empty:
            q3p = (
                df_q3.groupby(["author","subreddit","year_month","post_shock"], as_index=False)
                .agg(avg_score=("score","mean"))
            )
            q3p["sub_role"]      = q3p["subreddit"].map(SUBREDDITS)
            q3p["treated"]       = (q3p["sub_role"] == "treatment").astype(int)
            q3p["did"]           = q3p["treated"] * q3p["post_shock"]
            q3p["log_avg_score"] = np.log1p(q3p["avg_score"].clip(lower=0))
            m3 = fit_ols("log_avg_score ~ did + C(subreddit) + C(year_month)", q3p, cluster_col="author")
            if m3: results["q3"] = reg_result(m3, "did")
    except Exception as e:
        print(f"  Q3 failed: {e}")

    print("\n--- Legacy Q4: topic reallocation ---")
    try:
        df_q4 = df_all[df_all["author"].isin(stable_authors)].copy()
        df_q4["is_control"] = (df_q4["sub_role"] == "control").astype(float)
        pre4  = df_q4[df_q4["post_shock"]==0]
        has_t = set(pre4[pre4["sub_role"]=="treatment"]["author"])
        has_c = set(pre4[pre4["sub_role"]=="control"]["author"])
        cross = has_t & has_c
        df_cross = df_q4[df_q4["author"].isin(cross)].copy()
        if df_cross["author"].nunique() >= 10:
            tot = (df_cross.groupby(["author","post_shock"])["post_id"]
                   .count().reset_index().rename(columns={"post_id":"total"}))
            ctrl_cnt = (df_cross[df_cross["is_control"]==1]
                        .groupby(["author","post_shock"])["post_id"]
                        .count().reset_index().rename(columns={"post_id":"ctrl_cnt"}))
            q4s             = tot.merge(ctrl_cnt, on=["author","post_shock"], how="left")
            q4s["ctrl_cnt"] = q4s["ctrl_cnt"].fillna(0)
            q4s["ctrl_share"]= q4s["ctrl_cnt"] / q4s["total"]
            q4s             = q4s[q4s["total"] >= 3]
            both4           = q4s.groupby("author")["post_shock"].nunique()
            q4s             = q4s[q4s["author"].isin(both4[both4==2].index)]
            if not q4s.empty:
                q4_wide = q4s.pivot(index="author", columns="post_shock", values="ctrl_share").reset_index()
                q4_wide.columns = ["author","pre_cs","post_cs"]
                q4_wide["delta"]= q4_wide["post_cs"] - q4_wide["pre_cs"]
                q4_wide = q4_wide.dropna(subset=["delta"])
                if not q4_wide.empty:
                    mean_d   = q4_wide["delta"].mean()
                    t_q4, p_q4 = stats.ttest_1samp(q4_wide["delta"], 0)
                    p1_q4    = p_q4/2 if mean_d > 0 else 1.0
                    results["q4"] = {
                        "coef":               safe_float(mean_d),
                        "pvalue":             safe_float(p_q4),
                        "pvalue_one":         safe_float(p1_q4),
                        "n_obs":              safe_int(len(q4_wide)),
                        "genuine_reallocation": bool(mean_d > 0 and (p_q4/2) < 0.05),
                    }
    except Exception as e:
        print(f"  Q4 failed: {e}")

    print(f"\n  Tables -> {TABLES_DIR}")
    print(f"  Figures -> {FIGURES_DIR}")
    return results

def require_file(path, validation_errors, label=None):
    required_path = Path(path)
    display_label = label or str(required_path)
    if not required_path.exists():
        validation_errors.append(f"Missing required output: {display_label}")
        return
    if required_path.is_file() and required_path.stat().st_size == 0:
        validation_errors.append(f"Required output is empty: {display_label}")

def validate_numeric_range(table, column_name, minimum_value, maximum_value, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = (
        numeric_values.isna()
        | ~np.isfinite(numeric_values)
        | (numeric_values < minimum_value)
        | (numeric_values > maximum_value)
    )
    if bool(invalid_value_mask.any()):
        example_columns = ["subreddit", column_name] if "subreddit" in table.columns else [column_name]
        examples = table.loc[invalid_value_mask, example_columns].head(10).to_dict("records")
        validation_errors.append(
            f"{table_label} column {column_name} has values outside "
            f"[{minimum_value}, {maximum_value}]; examples={examples}"
        )

def validate_nonnegative_numeric(table, column_name, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = numeric_values.isna() | ~np.isfinite(numeric_values) | (numeric_values < 0)
    if bool(invalid_value_mask.any()):
        validation_errors.append(f"{table_label} column {column_name} has missing, infinite, or negative values")

def validate_finite_numeric(table, column_name, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = numeric_values.isna() | ~np.isfinite(numeric_values)
    if bool(invalid_value_mask.any()):
        validation_errors.append(f"{table_label} column {column_name} has missing, infinite, or nonnumeric values")

def expected_panel_subreddits():
    return sorted(available_post_subreddits())

def validate_subreddit_month_panel(panel, score_table, validation_errors, panel_label="subreddit-month panel"):
    if panel is None or panel.empty:
        validation_errors.append(f"{panel_label} is empty")
        return

    required_columns = [
        "subreddit", "year_month", "year_month_dt", "post_shock",
        "posts", "active_creators", "calibrated_output", "avg_score",
        "log_posts", "log_active_creators", "log_calibrated_output", "log_avg_score",
        "gse", "raw_gse", "gse_post",
        *ACSI_COMPONENT_COLUMNS,
        "generation_capability", "generation_capability_norm", "generation_capability_post",
        "physical_free", "non_personal",
        *[spec["norm"] for spec in ACSI_DIMENSION_SPECS],
        *[spec["post"] for spec in ACSI_DIMENSION_SPECS],
    ]
    missing_columns = [column_name for column_name in required_columns if column_name not in panel.columns]
    if missing_columns:
        validation_errors.append(f"{panel_label} missing required columns: {missing_columns}")
        return

    duplicate_row_count = int(panel.duplicated(["subreddit", "year_month"]).sum())
    if duplicate_row_count:
        validation_errors.append(f"{panel_label} has {duplicate_row_count:,} duplicate subreddit-month rows")

    expected_subreddits = expected_panel_subreddits()
    actual_subreddits = sorted(panel["subreddit"].astype(str).unique())
    if actual_subreddits != expected_subreddits:
        missing_subreddits = sorted(set(expected_subreddits) - set(actual_subreddits))
        extra_subreddits = sorted(set(actual_subreddits) - set(expected_subreddits))
        validation_errors.append(
            f"{panel_label} subreddit coverage mismatch; "
            f"missing={missing_subreddits}, extra={extra_subreddits}"
        )

    expected_row_count = len(expected_subreddits) * len(ALL_MONTHS)
    if len(panel) != expected_row_count:
        validation_errors.append(
            f"{panel_label} row count is {len(panel):,}; expected {expected_row_count:,} "
            f"({len(expected_subreddits)} subreddits x {len(ALL_MONTHS)} months)"
        )

    month_counts_by_subreddit = panel.groupby("subreddit")["year_month"].nunique()
    incomplete_subreddit_month_counts = month_counts_by_subreddit[month_counts_by_subreddit != len(ALL_MONTHS)]
    if not incomplete_subreddit_month_counts.empty:
        validation_errors.append(
            f"{panel_label} has subreddits without all {len(ALL_MONTHS)} months: "
            f"{incomplete_subreddit_month_counts.head(10).to_dict()}"
        )

    parsed_month_dates = pd.to_datetime(panel["year_month_dt"], errors="coerce")
    if parsed_month_dates.isna().any():
        validation_errors.append(f"{panel_label} has invalid year_month_dt values")
    else:
        expected_month_labels = set(pd.Series(ALL_MONTHS).dt.strftime("%Y-%m"))
        actual_month_labels = set(parsed_month_dates.dt.strftime("%Y-%m"))
        if actual_month_labels != expected_month_labels:
            validation_errors.append(
                f"{panel_label} month coverage mismatch; "
                f"missing={sorted(expected_month_labels - actual_month_labels)}, "
                f"extra={sorted(actual_month_labels - expected_month_labels)}"
            )
        month_label_mismatch = panel["year_month"].astype(str) != parsed_month_dates.dt.strftime("%Y-%m")
        if bool(month_label_mismatch.any()):
            validation_errors.append(f"{panel_label} has year_month values that do not match year_month_dt")

    observed_shock_values = set(pd.to_numeric(panel["post_shock"], errors="coerce").dropna().astype(int).unique())
    if not observed_shock_values.issubset({0, 1}):
        validation_errors.append(f"{panel_label} post_shock has values outside {{0, 1}}: {sorted(observed_shock_values)}")

    nonnegative_columns = [
        "posts", "active_creators", "calibrated_output",
        "log_posts", "log_active_creators", "log_calibrated_output", "log_avg_score",
    ]
    for column_name in nonnegative_columns:
        validate_nonnegative_numeric(panel, column_name, validation_errors, panel_label)
    validate_finite_numeric(panel, "avg_score", validation_errors, panel_label)

    component_score_columns = ACSI_COMPONENT_COLUMNS + ["generation_capability", "physical_free", "non_personal"]
    normalized_score_columns = (
        ["gse", "gse_post", "generation_capability_norm", "generation_capability_post"]
        + [spec["norm"] for spec in ACSI_DIMENSION_SPECS]
        + [spec["post"] for spec in ACSI_DIMENSION_SPECS]
    )
    for column_name in component_score_columns:
        validate_numeric_range(panel, column_name, 1, 5, validation_errors, panel_label)
    for column_name in normalized_score_columns:
        validate_numeric_range(panel, column_name, 0, 1, validation_errors, panel_label)
    validate_numeric_range(panel, "raw_gse", 5, 25, validation_errors, panel_label)

    if score_table is not None and not score_table.empty:
        duplicate_score_subreddits = sorted(score_table.loc[score_table["subreddit"].duplicated(), "subreddit"].unique())
        if duplicate_score_subreddits:
            validation_errors.append(f"{INDEX_SHORT} score data has duplicate subreddit rows: {duplicate_score_subreddits}")
        missing_scores = sorted(set(actual_subreddits) - set(score_table["subreddit"].astype(str)))
        if missing_scores:
            validation_errors.append(f"{INDEX_SHORT} score data missing panel subreddits: {missing_scores}")
        if "n_used" in score_table.columns:
            n_used = pd.to_numeric(score_table["n_used"], errors="coerce")
            if n_used.isna().any():
                bad_subreddits = score_table.loc[n_used.isna(), "subreddit"].astype(str).tolist()
                validation_errors.append(f"{INDEX_SHORT} score data has invalid n_used for: {bad_subreddits}")

def validate_required_artifacts(analysis_results, run_legacy_models, include_content_validation, validation_errors):
    core_tables = [
        "acsi_scores_computed.csv",
        "acsi_three_dimensional_main_model.csv",
        "acsi_three_dimensional_main_model.tex",
        "gse_main_dose_response.tex",
        "gse_covariate_adj.tex",
        "gse_dimension_models.csv",
        "acsi_component_correlations.csv",
        "acsi_mechanism_models.csv",
        "acsi_mechanism_joint_model.csv",
        "acsi_mechanism_joint_model.tex",
        "binary_did_consistency.tex",
        "substitutability_hypothesis_test.csv",
        "gse_permutation_coefs.csv",
        "gse_secondary_outcomes.csv",
        "gse_quartile_check.csv",
        "gse_event_study.csv",
    ]
    core_figures = [
        "gse_dimension_models.png",
        "gse_quartile_check.png",
        "gse_event_study.png",
    ]
    for name in core_tables:
        require_file(TABLES_DIR / name, validation_errors)
    for name in core_figures:
        require_file(FIGURES_DIR / name, validation_errors)
    for required_path in [SUBMONTH_PANEL_PATH, SUBMONTH_PANEL_META_PATH]:
        require_file(required_path, validation_errors)

    required_results = [
        "acsi_three_dimensional",
        "gse_main",
        "gse_covariate_adj",
        "gse_dimensions",
        "acsi_component_correlations",
        "acsi_mechanisms",
        "acsi_mechanisms_joint",
        "binary_did_consistency",
        "substitutability_hypothesis",
        "gse_permutation",
        "gse_secondary",
        "gse_quartiles",
        "gse_event_study",
    ]
    missing_results = [result_key for result_key in required_results if not analysis_results.get(result_key)]
    if missing_results:
        validation_errors.append(f"Analysis results missing required entries: {missing_results}")

    if analysis_results.get("gse_dimensions") and len(analysis_results["gse_dimensions"]) != len(ACSI_DIMENSION_SPECS):
        validation_errors.append(f"Expected {len(ACSI_DIMENSION_SPECS)} dimension model results; got {len(analysis_results['gse_dimensions'])}")
    if analysis_results.get("acsi_three_dimensional") and len(analysis_results["acsi_three_dimensional"]) != len(ACSI_MECHANISM_SPECS):
        validation_errors.append(f"Expected {len(ACSI_MECHANISM_SPECS)} three-dimensional model results; got {len(analysis_results['acsi_three_dimensional'])}")
    if analysis_results.get("acsi_mechanisms") and len(analysis_results["acsi_mechanisms"]) != len(ACSI_MECHANISM_SPECS):
        validation_errors.append(f"Expected {len(ACSI_MECHANISM_SPECS)} mechanism model results; got {len(analysis_results['acsi_mechanisms'])}")
    if analysis_results.get("acsi_mechanisms_joint") and len(analysis_results["acsi_mechanisms_joint"]) != len(ACSI_MECHANISM_SPECS):
        validation_errors.append(f"Expected {len(ACSI_MECHANISM_SPECS)} joint mechanism results; got {len(analysis_results['acsi_mechanisms_joint'])}")
    if analysis_results.get("gse_secondary") and len(analysis_results["gse_secondary"]) != 3:
        validation_errors.append(f"Expected 3 secondary outcome results; got {len(analysis_results['gse_secondary'])}")
    if analysis_results.get("gse_quartiles") and len(analysis_results["gse_quartiles"]) != 3:
        validation_errors.append(f"Expected 3 quartile results; got {len(analysis_results['gse_quartiles'])}")

    if include_content_validation and analysis_results.get("content_validation", {}).get("n_sampled", 0) > 0:
        require_file(TABLES_DIR / "content_validation_sample.csv", validation_errors)

    if run_legacy_models:
        for required_path in [POSTS_ECOSYSTEM_PATH, POSTS_CREATOR_PATH, POSTS_SURVIVOR_PATH, PANEL_PATH, CREATORS_PATH]:
            require_file(required_path, validation_errors)
        require_file(TABLES_DIR / "q1a_did.tex", validation_errors)

def validate_cache_metadata(author_cap_enabled, valid_author_ids, require_comment_cache, validation_errors):
    try:
        if not submonth_panel_cache_is_current(author_cap_enabled):
            validation_errors.append("Subreddit-month panel metadata is missing or does not match current raw files/settings")
    except Exception as exc:
        validation_errors.append(f"Could not validate subreddit-month panel metadata: {exc}")

    if use_persistent_raw_cache() and POST_MONTHLY_AGG_PATH.exists():
        try:
            if not post_monthly_agg_cache_is_current(author_cap_enabled):
                validation_errors.append("Post monthly aggregate cache metadata does not match current raw files/settings")
        except Exception as exc:
            validation_errors.append(f"Could not validate post monthly aggregate cache metadata: {exc}")

    if require_comment_cache:
        if not COMMENT_KEYWORD_MONTHLY_PATH.exists() or not COMMENT_AUTHOR_MONTHLY_PATH.exists():
            validation_errors.append("Comment cache required but one or both comment aggregate cache files are missing")
        if valid_author_ids is not None:
            try:
                if COMMENT_KEYWORD_MONTHLY_PATH.exists() and not comment_keyword_cache_is_current():
                    validation_errors.append("Comment keyword cache metadata does not match current raw files/settings")
                if COMMENT_AUTHOR_MONTHLY_PATH.exists() and not comment_author_cache_is_current(valid_author_ids):
                    validation_errors.append("Comment author cache metadata does not match current raw files/settings")
            except Exception as exc:
                validation_errors.append(f"Could not validate comment cache metadata: {exc}")

def validate_run_outputs(
    submonth_panel,
    score_table,
    analysis_results,
    run_legacy_models,
    author_cap_enabled,
    valid_author_ids=None,
    require_comment_cache=False,
    include_content_validation=False,
):
    print("\n=== Final validation gate ===")
    validation_errors = []

    validate_subreddit_month_panel(submonth_panel, score_table, validation_errors)
    if SUBMONTH_PANEL_PATH.exists():
        try:
            saved_panel = pd.read_parquet(SUBMONTH_PANEL_PATH)
            validate_subreddit_month_panel(saved_panel, score_table, validation_errors, panel_label="saved subreddit-month panel")
            if submonth_panel is not None and len(saved_panel) != len(submonth_panel):
                validation_errors.append(
                    f"Saved panel row count {len(saved_panel):,} does not match "
                    f"in-memory panel row count {len(submonth_panel):,}"
                )
        except Exception as exc:
            validation_errors.append(f"Could not read saved subreddit-month panel: {exc}")

    validate_required_artifacts(analysis_results, run_legacy_models, include_content_validation, validation_errors)
    validate_cache_metadata(author_cap_enabled, valid_author_ids, require_comment_cache, validation_errors)

    if validation_errors:
        print("  VALIDATION FAILED:")
        for validation_error in validation_errors:
            print(f"  - {validation_error}")
        raise RuntimeError(f"Final validation failed with {len(validation_errors)} issue(s).")

    print(
        f"  validation passed: {len(submonth_panel):,} panel rows, "
        f"{submonth_panel['subreddit'].nunique():,} subreddits, "
        f"{submonth_panel['year_month'].nunique():,} months"
    )

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Reddit GenAI Pipeline — AI Content Substitutability</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 2rem; background: #0d1117; color: #c9d1d9; }
    h1   { color: #58a6ff; margin-top: 0; }
    h2   { color: #79c0ff; border-bottom: 1px solid #30363d;
           padding-bottom: .5rem; margin-top: 2rem; }
    h3   { color: #79c0ff; margin-top: 1.2rem; margin-bottom: .3rem; font-size: 1rem; }
    .container { max-width: 1150px; margin: 0 auto; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
             gap: 1rem; margin: 1rem 0 2rem; }
    .stat  { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem; }
    .stat .label { color: #8b949e; font-size: .8rem; text-transform: uppercase; }
    .stat .value { font-size: 1.6rem; font-weight: 600; color: #f0f6fc; margin-top: .3rem; }
    table  { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { text-align: left; padding: .45rem .75rem;
             border-bottom: 1px solid #30363d; font-size: .88rem; }
    th     { color: #8b949e; font-weight: 500; }
    .reg   { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
             padding: .9rem 1.2rem; font-family: ui-monospace, monospace;
             font-size: .88rem; line-height: 1.9; }
    .fig   { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
             padding: 1rem; margin: 1rem 0; }
    .fig img { max-width: 100%; height: auto; display: block; }
    .sig   { color: #56d364; font-weight: 600; }
    .neg   { color: #f85149; font-weight: 600; }
    .nsig  { color: #d29922; }
    .desc  { color: #8b949e; font-size: .88rem; margin-bottom: .5rem; line-height: 1.6; }
    .note  { color: #8b949e; font-size: .8rem; font-style: italic; margin-top: .4rem; }
    footer { color: #484f58; font-size: .8rem; text-align: center; margin-top: 3rem; }
  </style>
</head>
<body>
<div class="container">
  <h1>Reddit GenAI Pipeline — AI Content Substitutability</h1>
  <p class="desc">Empirical validation &mdash; window: {{ start }} to {{ end }}
  &mdash; shock: Nov 2022 (ChatGPT)</p>

  <div class="stats">
    <div class="stat"><div class="label">{{ posts_label }}</div><div class="value">{{ n_posts }}</div></div>
    <div class="stat"><div class="label">Subreddits</div><div class="value">{{ n_subs }}</div></div>
    {% if show_creator_stats %}
    <div class="stat"><div class="label">Creators (Ecosystem)</div><div class="value">{{ n_authors }}</div></div>
    <div class="stat"><div class="label">Stable creators</div><div class="value">{{ n_stable }}</div></div>
    {% endif %}
  </div>

  <h2>H1 vs H2: Substitutability Exposure Test</h2>

  <div class="reg">
    <strong>H1:</strong> Uniform productivity enhancement; ACSI x Post should be near zero with tight uncertainty.<br>
    <strong>H2:</strong> Substitutability-shaped transformation; ACSI x Post should be negative for log monthly posts.
  </div>

  <h3>Single-Index Readout (Supporting)</h3>
  <div class="reg">{% if hypothesis %}
    Model: {{ hypothesis.model }}<br>
    Conclusion: <strong>{{ hypothesis.conclusion }}</strong><br>
    Evidence readout: {{ hypothesis.supported_hypothesis }}<br>
    {{ hypothesis.interpretation }}
  {% else %}Not estimated.{% endif %}</div>

  <h2>Main Three-Dimensional Dose-Response Model</h2>
  <p class="desc">Uses each subreddit's measured profile directly: generation capability, low physical constraint, and low personal-context need interacted with the post-ChatGPT period, plus subreddit and month fixed effects. This avoids binary categories and avoids collapsing the three dimensions into one aggregate treatment.</p>
  {% if acsi_three_dimensional %}
  <table>
    <tr><th>Dimension</th><th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th></tr>
    {% for r in acsi_three_dimensional %}
    <tr>
      <td>{{ r.label }}</td>
      <td><span class="{{ 'sig' if r.pvalue is not none and r.pvalue < 0.05 else 'nsig' }}">{% if r.coef is not none %}{{ "%.4f"|format(r.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.se is not none %}{{ "%.4f"|format(r.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pvalue is not none %}{{ "%.4f"|format(r.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.percent_effect_full_exposure) if r.percent_effect_full_exposure is not none else '' }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  <h2>Aggregate ACSI Model (Supporting Check)</h2>
  <p class="desc">This keeps the older single-index treatment as a diagnostic for overcollapsed effects, not the primary specification.</p>
  
  <h3>Main Effect (log_posts ~ ACSI x Post)</h3>
  <div class="reg">{% if gse_main %}
    ACSI x Post coef = <span class="{{ gse_main_c }}">{% if gse_main.coef is not none %}{{ "%.4f"|format(gse_main.coef) }}{% else %}N/A{% endif %}</span> 
    SE {% if gse_main.se is not none %}{{ "%.4f"|format(gse_main.se) }}{% else %}N/A{% endif %} p={% if gse_main.pvalue is not none %}{{ "%.4f"|format(gse_main.pvalue) }}{% else %}N/A{% endif %}<br>
    Full exposure (0 to 1) effect: <span class="{{ gse_main_c }}">{% if gse_main.percent_effect_full_exposure is not none %}{{ "%.1f"|format(gse_main.percent_effect_full_exposure) }}%{% else %}N/A{% endif %}</span> 
  {% else %}Not estimated.{% endif %}</div>

  <h3>Covariate-Adjusted Model</h3>
  <div class="reg">{% if gse_adj %}
    ACSI x Post coef = <span class="{{ gse_adj_c }}">{% if gse_adj.coef is not none %}{{ "%.4f"|format(gse_adj.coef) }}{% else %}N/A{% endif %}</span> 
    SE {% if gse_adj.se is not none %}{{ "%.4f"|format(gse_adj.se) }}{% else %}N/A{% endif %} p={% if gse_adj.pvalue is not none %}{{ "%.4f"|format(gse_adj.pvalue) }}{% else %}N/A{% endif %}<br>
    <span class="note">Controls for pre-shock size, linear trends, and subscriber count.</span>
  {% else %}Not estimated.{% endif %}</div>

  <h3>Component-wise Dose-Response</h3>
  {% if gse_dimensions %}
  <table>
    <tr><th>Dimension</th><th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th></tr>
    {% for r in gse_dimensions %}
    <tr>
      <td>{{ r.label }}</td>
      <td><span class="{{ 'sig' if r.pvalue is not none and r.pvalue < 0.05 else 'nsig' }}">{% if r.coef is not none %}{{ "%.4f"|format(r.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.se is not none %}{{ "%.4f"|format(r.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pvalue is not none %}{{ "%.4f"|format(r.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.percent_effect_full_exposure) if r.percent_effect_full_exposure is not none else '' }}</td>
    </tr>
    {% endfor %}
  </table>
  <div class="fig"><img src="/figures/gse_dimension_models.png" alt="ACSI component-wise dose response"></div>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  <h3>Measurement Correlations</h3>
  {% if acsi_component_correlations %}
  <table>
    <tr>
      <th>Component</th><th>Direct</th><th>Useful</th><th>Quality</th><th>Low Physical</th><th>Low Personal</th><th>ACSI</th>
    </tr>
    {% for r in acsi_component_correlations %}
    <tr>
      <td>{{ r.component }}</td>
      <td>{{ "%.3f"|format(r.direct_gen_norm) if r.direct_gen_norm is not none else "" }}</td>
      <td>{{ "%.3f"|format(r.usefulness_norm) if r.usefulness_norm is not none else "" }}</td>
      <td>{{ "%.3f"|format(r.quality_comp_norm) if r.quality_comp_norm is not none else "" }}</td>
      <td>{{ "%.3f"|format(r.physical_free_norm) if r.physical_free_norm is not none else "" }}</td>
      <td>{{ "%.3f"|format(r.non_personal_norm) if r.non_personal_norm is not none else "" }}</td>
      <td>{{ "%.3f"|format(r.gse) if r.gse is not none else "" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <h3>Mechanism Models</h3>
  {% if acsi_mechanisms %}
  <table>
    <tr><th>Mechanism</th><th>Separate coef</th><th>Separate p</th><th>Joint coef</th><th>Joint p</th></tr>
    {% for r in acsi_mechanisms %}
    {% set j = acsi_mechanisms_joint_map.get(r.term) if acsi_mechanisms_joint_map else none %}
    <tr>
      <td>{{ r.label }}</td>
      <td>{% if r.coef is not none %}{{ "%.4f"|format(r.coef) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pvalue is not none %}{{ "%.4f"|format(r.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{% if j and j.coef is not none %}{{ "%.4f"|format(j.coef) }}{% else %}N/A{% endif %}</td>
      <td>{% if j and j.pvalue is not none %}{{ "%.4f"|format(j.pvalue) }}{% else %}N/A{% endif %}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  {% if binary_did %}
  <h3>Binary DiD Consistency Check</h3>
  <div class="reg">
    Treatment x Post coef = {% if binary_did.coef is not none %}{{ "%.4f"|format(binary_did.coef) }}{% else %}N/A{% endif %}
    SE {% if binary_did.se is not none %}{{ "%.4f"|format(binary_did.se) }}{% else %}N/A{% endif %}
    p={% if binary_did.pvalue is not none %}{{ "%.4f"|format(binary_did.pvalue) }}{% else %}N/A{% endif %}
    <br>Effect: {% if binary_did.percent_effect is not none %}{{ "%.1f"|format(binary_did.percent_effect) }}%{% else %}N/A{% endif %}
  </div>
  {% endif %}

  <h3>ACSI Permutation Inference</h3>
  <div class="reg">{% if gse_perm %}
    Observed Coef: {% if gse_perm.observed_coef is not none %}{{ "%.4f"|format(gse_perm.observed_coef) }}{% else %}N/A{% endif %}<br>
    Permutation p-value: <span class="{{ 'sig' if gse_perm.perm_pvalue_left is not none and gse_perm.perm_pvalue_left < 0.05 else 'nsig' }}">{% if gse_perm.perm_pvalue_left is not none %}{{ "%.4f"|format(gse_perm.perm_pvalue_left) }}{% else %}N/A{% endif %}</span> 
    (N = {{ gse_perm.n_perms }} permutations)
  {% else %}Not estimated.{% endif %}</div>

  <h3>Secondary Outcomes</h3>
  {% if gse_secondary %}
  <table>
    <tr><th>Outcome</th><th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th></tr>
    {% for r in gse_secondary %}
    <tr>
      <td>{{ r.label }}</td>
      <td><span class="{{ 'sig' if r.pvalue is not none and r.pvalue < 0.05 else 'nsig' }}">{% if r.coef is not none %}{{ "%.4f"|format(r.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.se is not none %}{{ "%.4f"|format(r.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pvalue is not none %}{{ "%.4f"|format(r.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.percent_effect_full_exposure) if r.percent_effect_full_exposure is not none else '' }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  <h3>Quartile Nonlinearity Check</h3>
  {% if gse_quartiles %}
  <table>
    <tr><th>ACSI Quartile</th><th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th></tr>
    {% for r in gse_quartiles %}
    <tr>
      <td>{{ r.label }}</td>
      <td><span class="{{ 'sig' if r.pvalue is not none and r.pvalue < 0.05 else 'nsig' }}">{% if r.coef is not none %}{{ "%.4f"|format(r.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.se is not none %}{{ "%.4f"|format(r.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pvalue is not none %}{{ "%.4f"|format(r.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.percent_effect) if r.percent_effect is not none else '' }}</td>
    </tr>
    {% endfor %}
  </table>
  <div class="fig"><img src="/figures/gse_quartile_check.png" alt="ACSI quartile check"></div>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  {% if gse_cv %}
  <h3>ACSI Construct Validity (Δ Keyword Rate correlation)</h3>
  <div class="reg">{% if gse_cv %}
    <strong>Broad AI Terms:</strong> r = {% if gse_cv.broad_pearson_r is not none %}{{ "%.4f"|format(gse_cv.broad_pearson_r) }}{% else %}N/A{% endif %}
    {% if gse_cv.broad_pearson_pvalue is not none %}
    <span class="{{ 'sig' if gse_cv.broad_pearson_pvalue < 0.05 and gse_cv.broad_pearson_r is not none and gse_cv.broad_pearson_r > 0 else 'nsig' }}">p={{ "%.4f"|format(gse_cv.broad_pearson_pvalue) }}</span>
    {% else %}<span class="nsig">p=N/A</span>{% endif %}<br>
    
    <strong>Narrow Tool Terms (ChatGPT/Midjourney/LLM):</strong> r = {% if gse_cv.narrow_pearson_r is not none %}{{ "%.4f"|format(gse_cv.narrow_pearson_r) }}{% else %}N/A{% endif %}
    {% if gse_cv.narrow_pearson_pvalue is not none %}
    <span class="{{ 'sig' if gse_cv.narrow_pearson_pvalue < 0.05 and gse_cv.narrow_pearson_r is not none and gse_cv.narrow_pearson_r > 0 else 'nsig' }}">p={{ "%.4f"|format(gse_cv.narrow_pearson_pvalue) }}</span>
    {% else %}<span class="nsig">p=N/A</span>{% endif %}
  {% endif %}</div>
  {% endif %}
  
  <h3>ACSI Event Study</h3>
  {% if gse_event_study %}
  <div class="reg">Joint pretrend p={% if gse_event_study.pretrend_pvalue is not none %}{{ "%.4f"|format(gse_event_study.pretrend_pvalue) }}{% else %}N/A{% endif %}</div>
  {% endif %}
  <div class="fig"><img src="/figures/gse_event_study.png" alt="ACSI event study"></div>

  <h2>ACSI Measurement Table</h2>
  <p class="desc">Highest-exposure subreddits from the current `acsi_scores.csv`; used rows exclude AI-related posts by default. Low-reliability rows stay in the current analysis but mark where we need to code more posts.</p>
  {% if score_rows %}
  <table>
    <tr><th>Subreddit</th><th>ACSI</th><th>Direct</th><th>Useful</th><th>Quality</th><th>Physical Req.</th><th>Personal Req.</th><th>Used</th><th>Reliability</th><th>Hard</th><th>AI Excl.</th></tr>
    {% for r in score_rows %}
    <tr>
      <td>{{ r.subreddit }}</td>
      <td>{{ "%.3f"|format(r.gse) if r.gse is not none else "N/A" }}</td>
      <td>{{ "%.2f"|format(r.direct_gen) if r.direct_gen is not none else "N/A" }}</td>
      <td>{{ "%.2f"|format(r.usefulness) if r.usefulness is not none else "N/A" }}</td>
      <td>{{ "%.2f"|format(r.quality_comp) if r.quality_comp is not none else "N/A" }}</td>
      <td>{{ "%.2f"|format(r.physical_req) if r.physical_req is not none else "N/A" }}</td>
      <td>{{ "%.2f"|format(r.personal_req) if r.personal_req is not none else "N/A" }}</td>
      <td>{{ r.n_used if r.n_used is not none else "" }}</td>
      <td>{{ r.score_reliability if r.score_reliability is not none else "" }}</td>
      <td>{{ r.n_hard_cases if r.n_hard_cases is not none else "" }}</td>
      <td>{{ r.n_ai_related_excluded if r.n_ai_related_excluded is not none else "" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}<div class="reg">No score table found.</div>{% endif %}

  {% if show_legacy %}
  <h2>Q1 — Total content (Legacy Binary)</h2>

  <h3>Q1a — DiD primary</h3>
  <div class="reg">{% if q1a %}DiD
    <span class="{{ q1a_c }}">{% if q1a.coef is not none %}{{ "%.4f"|format(q1a.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q1a.se is not none %}{{ "%.4f"|format(q1a.se) }}{% else %}N/A{% endif %} p={% if q1a.pvalue is not none %}{{ "%.4f"|format(q1a.pvalue) }}{% else %}N/A{% endif %} N={{ q1a.n_obs }}
    <br><span class="note">Outcome: Calibrated Output (stable creators only)</span>
  {% else %}Not estimated.{% endif %}</div>

  <h3>Attrition scenario bounds</h3>
  {% if ab %}
  <div class="reg">
    Lower bound (dropouts maintained avg output): <span class="sig">{% if ab.lower_bound is not none %}{{ "%.1f"|format(ab.lower_bound) }}%{% else %}N/A{% endif %}</span><br>
    Observed (survivors only): <span class="nsig">{% if ab.observed is not none %}{{ "%.1f"|format(ab.observed) }}%{% else %}N/A{% endif %}</span><br>
    Upper bound (dropouts went to zero): {% if ab.upper_bound is not none %}{{ "%.1f"|format(ab.upper_bound) }}%{% else %}N/A{% endif %}<br>
    Dropout rate: {% if ab.dropout_rate is not none %}{{ "%.1f"|format(ab.dropout_rate*100) }}%{% else %}N/A{% endif %} N={{ ab.dropout_n }}
  </div>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  <h2>Q2 — Creator switching (Theorem 5)</h2>

  <h3>Survival ~ log(c_i) [predicted: negative — less efficient creators exit]</h3>
  <div class="reg">{% if q2_attr %}
    <span class="{{ 'sig' if q2_attr.coef is not none and q2_attr.pvalue is not none and q2_attr.coef < 0 and q2_attr.pvalue < 0.05 else 'nsig' }}">{% if q2_attr.coef is not none %}{{ "%.4f"|format(q2_attr.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q2_attr.se is not none %}{{ "%.4f"|format(q2_attr.se) }}{% else %}N/A{% endif %} p={% if q2_attr.pvalue is not none %}{{ "%.4f"|format(q2_attr.pvalue) }}{% else %}N/A{% endif %} N={{ q2_attr.n_obs }}
  {% else %}Not estimated.{% endif %}</div>

  <h3>Δ posting rate ~ log(c_i) [predicted: positive]</h3>
  <div class="reg">{% if q2r %}
    <span class="{{ q2r_c }}">{% if q2r.coef is not none %}{{ "%.4f"|format(q2r.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q2r.se is not none %}{{ "%.4f"|format(q2r.se) }}{% else %}N/A{% endif %} p={% if q2r.pvalue is not none %}{{ "%.4f"|format(q2r.pvalue) }}{% else %}N/A{% endif %} N={{ q2r.n_obs }}
  {% else %}Not estimated.{% endif %}</div>

  <h3>Δ comment rate ~ log(c_i) [Authenticity Signaling]</h3>
  <div class="reg">{% if q2c %}
    <span class="{{ q2c_c }}">{% if q2c.coef is not none %}{{ "%.4f"|format(q2c.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q2c.se is not none %}{{ "%.4f"|format(q2c.se) }}{% else %}N/A{% endif %} p={% if q2c.pvalue is not none %}{{ "%.4f"|format(q2c.pvalue) }}{% else %}N/A{% endif %} N={{ q2c.n_obs }}<br>
  {% else %}Comment data not available.{% endif %}</div>

  <h2>Q3 — Remaining creators flourish</h2>
  <div class="reg">{% if q3 %}
    DiD <span class="{{ q3_c }}">{% if q3.coef is not none %}{{ "%.4f"|format(q3.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q3.se is not none %}{{ "%.4f"|format(q3.se) }}{% else %}N/A{% endif %} p={% if q3.pvalue is not none %}{{ "%.4f"|format(q3.pvalue) }}{% else %}N/A{% endif %} N={{ q3.n_obs }}
  {% else %}Not estimated.{% endif %}</div>

  <h2>Q4 — Topic reallocation</h2>
  <div class="reg">{% if q4 %}
    Mean delta ctrl share: <span class="{{ q4_c }}">{% if q4.coef is not none %}{{ "%+.4f"|format(q4.coef) }}{% else %}N/A{% endif %}</span>
    p_one={{ "%.4f"|format(q4.pvalue_one) if q4.pvalue_one is not none else 'N/A' }}
    N={{ q4.n_obs }} genuine={{ q4.genuine_reallocation }}
  {% else %}Not estimated.{% endif %}</div>
  {% endif %}

  <footer>Outputs in {{ output_dir }}</footer>
</div>
</body>
</html>
"""

def launch_dashboard(panel, creators, results, n_ecosystem_authors=None, port=None):
    print("\n=== Starting dashboard ===")
    app = Flask(__name__, static_folder=str(FIGURES_DIR), static_url_path="/figures")

    raw_panel_posts = int(panel["post_count"].sum()) if "post_count" in panel.columns else int(panel["posts"].sum())
    posts_label = "ACSI measurement sample"
    n_posts = raw_panel_posts
    try:
        score_table_for_count = load_acsi_scores()
        if "n_used" in score_table_for_count.columns:
            n_used = pd.to_numeric(score_table_for_count["n_used"], errors="coerce")
            if n_used.notna().any():
                n_posts = int(n_used.fillna(0).sum())
    except Exception:
        posts_label = "Posts in monthly panel"
    if n_ecosystem_authors is not None:
        n_authors = f"{int(n_ecosystem_authors):,}"
    elif "author" in creators.columns:
        n_authors = f"{int(creators['author'].nunique()):,}"
    else:
        n_authors = "N/A"
    n_subs    = int(panel["subreddit"].nunique())
    n_stable  = int(creators["is_stable"].sum()) if "is_stable" in creators.columns else 0
    show_creator_stats = n_ecosystem_authors is not None or "author" in creators.columns
    show_legacy = any(
        results.get(key)
        for key in [
            "q1a",
            "attrition_bounds",
            "q2_attrition",
            "q2_rate",
            "q2_comment",
            "q3",
            "q4",
        ]
    )

    def cls(coef, pval, pos_expected):
        if pval is None or coef is None:
            return "nsig"
        try:
            if float(pval) >= 0.05:
                return "nsig"
            return "sig" if (float(coef) > 0) == pos_expected else "neg"
        except Exception:
            return "nsig"

    def to_obj(d):
        if d is None:
            return None
        class Obj:
            pass
        o = Obj()
        for k, v in d.items():
            setattr(o, k, v)
        return o

    def res(name):
        return results.get(name) or {}

    def load_score_rows(limit=25):
        score_path = resolve_acsi_score_path()
        if not score_path.exists():
            return []
        try:
            score_preview = load_acsi_scores()
            preview_columns = [
                "subreddit", "gse", "direct_gen", "usefulness", "quality_comp",
                "physical_req", "personal_req", "n_used", "score_reliability",
                "n_hard_cases", "n_ai_related_excluded",
            ]
            preview_columns = [column_name for column_name in preview_columns if column_name in score_preview.columns]
            score_preview = score_preview[preview_columns].copy()
            text_columns = {"subreddit", "score_reliability", "hard_case_policy"}
            for column_name in score_preview.columns:
                if column_name not in text_columns:
                    score_preview[column_name] = pd.to_numeric(score_preview[column_name], errors="coerce")
            if "gse" in score_preview.columns:
                score_preview = score_preview.sort_values("gse", ascending=False)
            score_preview = score_preview.head(limit)
            return json_safe(score_preview.where(pd.notna(score_preview), None).to_dict("records"))
        except Exception:
            return []

    try:
        output_dir_label = str(OUTPUT_DIR.relative_to(ROOT))
    except ValueError:
        output_dir_label = str(OUTPUT_DIR)

    @app.route("/")
    def home():
        acsi_mechanisms_joint_map = {
            row.get("term"): to_obj(row)
            for row in (results.get("acsi_mechanisms_joint") or [])
            if row.get("term")
        }
        return render_template_string(
            DASHBOARD_HTML,
            start=START_DATE.strftime("%b %Y"), end=END_DATE.strftime("%b %Y"),
            posts_label=posts_label,
            n_posts=f"{n_posts:,}", n_authors=n_authors,
            n_subs=n_subs, n_stable=f"{n_stable:,}",
            show_creator_stats=show_creator_stats,
            show_legacy=show_legacy,
            hypothesis=to_obj(results.get("substitutability_hypothesis")),
            acsi_three_dimensional=results.get("acsi_three_dimensional"),
            gse_main=to_obj(results.get("gse_main")),
            gse_main_c=cls(res("gse_main").get("coef"), res("gse_main").get("pvalue"), False),
            gse_adj=to_obj(results.get("gse_covariate_adj")),
            gse_adj_c=cls(res("gse_covariate_adj").get("coef"), res("gse_covariate_adj").get("pvalue"), False),
            gse_dimensions=results.get("gse_dimensions"),
            acsi_component_correlations=results.get("acsi_component_correlations"),
            acsi_mechanisms=results.get("acsi_mechanisms"),
            acsi_mechanisms_joint_map=acsi_mechanisms_joint_map,
            binary_did=to_obj(results.get("binary_did_consistency")),
            gse_perm=to_obj(results.get("gse_permutation")),
            gse_secondary=results.get("gse_secondary"),
            gse_quartiles=results.get("gse_quartiles"),
            gse_cv=to_obj(results.get("gse_construct_validity")),
            gse_event_study=to_obj(results.get("gse_event_study")),
            score_rows=load_score_rows(),
            output_dir=output_dir_label,
            q1a=to_obj(results.get("q1a")), q1a_c=cls(res("q1a").get("coef"), res("q1a").get("pvalue"), False),
            q1a_r=to_obj(results.get("q1a_robust")), q1a_r_c=cls(res("q1a_robust").get("coef"), res("q1a_robust").get("pvalue"), False),
            ab=to_obj(results.get("attrition_bounds")),
            q2_attr=to_obj(results.get("q2_attrition")),
            q2r=to_obj(results.get("q2_rate")), q2r_c=cls(res("q2_rate").get("coef"), res("q2_rate").get("pvalue"), True),
            q2s=to_obj(results.get("q2_score")), q2s_c=cls(res("q2_score").get("coef"), res("q2_score").get("pvalue"), False),
            q2c=to_obj(results.get("q2_comment")), q2c_c=cls(res("q2_comment").get("coef"), res("q2_comment").get("pvalue"), False),
            q2j=to_obj(results.get("q2_joint")), q2j_c=cls(res("q2_joint").get("coef"), res("q2_joint").get("pvalue"), True),
            q3=to_obj(results.get("q3")), q3_c=cls(res("q3").get("coef"), res("q3").get("pvalue"), True),
            q4=to_obj(results.get("q4")), q4_c=cls(res("q4").get("coef"), res("q4").get("pvalue"), True),
        )

    @app.route("/api/results")
    def api_results():
        return jsonify(json_safe(results))

    def write_static_dashboard(reason):
        with app.app_context():
            html = home().replace('src="/figures/', 'src="figures/')
        path = OUTPUT_DIR / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        print(f"  Dashboard server unavailable: {reason}")
        print(f"  Wrote static dashboard snapshot -> {path}")

    try:
        port = find_open_dashboard_port() if port is None else int(port)
    except PermissionError:
        write_static_dashboard("localhost socket binding is blocked in this environment")
        return
    except RuntimeError as exc:
        write_static_dashboard(str(exc))
        return

    url = f"http://localhost:{port}"
    print(f"  Dashboard ready -> {url}")
    print("  Press Ctrl+C to stop.")
    Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except OSError as exc:
        write_static_dashboard(str(exc))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Reddit GenAI empirical validation pipeline."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer randomization/permutation draws for faster development runs.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Deprecated compatibility flag. The localhost dashboard is always started after a successful run.",
    )
    parser.add_argument(
        "--reuse-clean-posts",
        action="store_true",
        help="Load cleaned post parquet caches instead of reparsing raw JSONL post files.",
    )
    parser.add_argument(
        "--skip-comments",
        action="store_true",
        help="Skip comment parsing. Faster/lighter, but omits comment-rate and keyword-validity outputs.",
    )
    parser.add_argument(
        "--gse-only",
        action="store_true",
        help="Run only the ACSI dose-response models from streamed monthly aggregates.",
    )
    parser.add_argument(
        "--reuse-gse-panel",
        action="store_true",
        help="In --gse-only mode, load the current output/latest subreddit-month panel instead of using the data/cache aggregate.",
    )
    parser.add_argument(
        "--force-rebuild-gse-panel",
        action="store_true",
        help="In --gse-only mode, ignore cached post monthly aggregates and rescan raw post JSONL files.",
    )
    parser.add_argument(
        "--force-rebuild-comment-cache",
        action="store_true",
        help="Ignore cached comment aggregates and rescan raw comment JSONL files.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Reuse the existing subreddit-month panel, rerun lightweight ACSI analyses, and launch the dashboard.",
    )
    parser.add_argument(
        "--no-author-cap",
        action="store_true",
        help="Skip the global author post cap in --gse-only so ACSI streaming can run in one pass.",
    )
    parser.add_argument(
        "--content-validation",
        action="store_true",
        help="Run the optional text content-validation sample after analysis.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small end-to-end smoke test on representative small subreddits.",
    )
    parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=None,
        help="Read at most this many raw JSONL lines per subreddit file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write outputs to a custom directory. Smoke mode defaults to output/smoke.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global ACTIVE_SUBREDDITS, MAX_LINES_PER_FILE, QUICK_MODE, N_RANDOMIZATION_PERMS

    args = parse_args(argv)
    if args.dashboard_only:
        args.gse_only = True
        args.skip_comments = True
        print("\nDASHBOARD-ONLY MODE: reusing existing panel or cached monthly aggregates; no raw JSONL scan if cache is current.")

    if args.no_dashboard:
        print("\nNOTE: --no-dashboard is deprecated; the localhost dashboard now starts after every successful run.")

    needs_existing_output = args.dashboard_only or args.reuse_clean_posts or (args.gse_only and args.reuse_gse_panel)

    if args.output_dir is not None:
        configure_output_dir(args.output_dir)
        if not needs_existing_output:
            clean_generated_output(OUTPUT_DIR)
    elif args.smoke:
        configure_output_dir(ROOT / "output" / "smoke")
        clean_generated_output(OUTPUT_DIR)
    else:
        configure_output_dir(OUTPUT_ROOT / "latest")
        if not needs_existing_output:
            clean_generated_output(OUTPUT_DIR)

    if args.dashboard_only and SUBMONTH_PANEL_PATH.exists():
        args.reuse_gse_panel = True

    if args.smoke:
        ACTIVE_SUBREDDITS = SMOKE_SUBREDDITS
        args.quick = True
        args.gse_only = True
        args.skip_comments = True
        print("\nSMOKE MODE: using representative small subreddits, quick permutations, and no comments.")
        print("  subreddits: " + ", ".join(iter_subreddits()))

    if args.gse_only and not args.skip_comments:
        args.skip_comments = True
        print("\nACSI-ONLY MODE: skipping comments and legacy creator-level analyses.")
    if args.no_author_cap and not args.gse_only:
        print("\nNOTE: --no-author-cap only affects --gse-only ACSI streaming runs.")
    if args.force_rebuild_gse_panel and not args.gse_only:
        print("\nNOTE: --force-rebuild-gse-panel only affects --gse-only runs.")

    MAX_LINES_PER_FILE = args.max_lines_per_file
    if MAX_LINES_PER_FILE is not None:
        print(f"\nLINE LIMIT: reading at most {MAX_LINES_PER_FILE:,} raw JSONL lines per file.")

    QUICK_MODE = args.quick
    N_RANDOMIZATION_PERMS = 250 if QUICK_MODE else 1000

    if QUICK_MODE:
        print(f"\nQUICK MODE: using {N_RANDOMIZATION_PERMS} randomization/permutation draws.")

    acsi_scores = load_acsi_scores()

    if args.gse_only:
        apply_author_cap = not args.no_author_cap
        if args.reuse_gse_panel:
            if not SUBMONTH_PANEL_PATH.exists():
                raise FileNotFoundError(
                    f"--reuse-gse-panel requested, but missing {SUBMONTH_PANEL_PATH}"
                )
            print(f"\n=== Step 3b: load existing subreddit-month panel ===")
            submonth_panel = pd.read_parquet(SUBMONTH_PANEL_PATH)
            submonth_panel = attach_acsi_scores(submonth_panel, acsi_scores)
            submonth_panel.to_parquet(SUBMONTH_PANEL_PATH, index=False)
            n_ecosystem_authors = None
            print(f"  loaded: {len(submonth_panel):,} rows, {submonth_panel['subreddit'].nunique():,} subreddits (explicit --reuse-gse-panel)")
        else:
            submonth_panel, n_ecosystem_authors = build_subreddit_month_panel_streaming(
                acsi_scores,
                apply_author_cap=apply_author_cap,
                force_rebuild=args.force_rebuild_gse_panel,
            )
            submonth_panel.to_parquet(SUBMONTH_PANEL_PATH, index=False)
            write_submonth_panel_cache_metadata(apply_author_cap)

        empty = pd.DataFrame()
        results = run_analysis(
            empty, empty, empty, empty,
            df_comments_valid=empty,
            df_comments_all=empty,
            acsi_scores=acsi_scores,
            submonth_panel=submonth_panel,
            run_legacy=False,
        )
        acsi_scores.to_csv(TABLES_DIR / "acsi_scores_computed.csv", index=False)

        if args.content_validation:
            top_subs = sorted(
                [s for s in iter_subreddits() if s in TREATMENT_SUBS and (DATA_DIR / f"r_{s}_posts.jsonl").exists()],
                key=lambda s: MU_K.get(s, 0), reverse=True,
            )[:5]
            cv_results = compute_content_validation_sample(top_subs)
            results["content_validation"] = cv_results

        validate_run_outputs(
            submonth_panel=submonth_panel,
            score_table=acsi_scores,
            analysis_results=results,
            run_legacy_models=False,
            author_cap_enabled=apply_author_cap,
            valid_author_ids=None,
            require_comment_cache=False,
            include_content_validation=args.content_validation,
        )
        print("\nProcessing complete.")
        launch_dashboard(submonth_panel, empty, results, n_ecosystem_authors=n_ecosystem_authors)
        return

    if args.reuse_clean_posts:
        missing = [
            p for p in [POSTS_ECOSYSTEM_PATH, POSTS_CREATOR_PATH, POSTS_SURVIVOR_PATH]
            if not p.exists()
        ]
        if missing:
            missing_names = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"--reuse-clean-posts requested, but missing cache file(s): {missing_names}"
            )
        print("\n=== Step 1/2: load cleaned post caches ===")
        df_ecosystem = pd.read_parquet(POSTS_ECOSYSTEM_PATH)
        df_all = pd.read_parquet(POSTS_CREATOR_PATH)
        df_surv = pd.read_parquet(POSTS_SURVIVOR_PATH)
        print(f"  ecosystem cache: {len(df_ecosystem):,} posts, {df_ecosystem['author'].nunique():,} authors")
        print(f"  creator cache: {len(df_all):,} posts, {df_all['author'].nunique():,} authors")
        print(f"  survivor cache: {len(df_surv):,} posts, {df_surv['author'].nunique():,} authors")
    else:
        raw_posts = parse_posts()
        df_ecosystem, df_all = clean_post_samples(raw_posts)
        del raw_posts
        gc.collect()
        df_ecosystem.to_parquet(POSTS_ECOSYSTEM_PATH, index=False)
        df_all.to_parquet(POSTS_CREATOR_PATH, index=False)

        df_surv = restrict_to_survivors(df_all)
        df_surv.to_parquet(POSTS_SURVIVOR_PATH, index=False)

    valid_authors = set(df_all["author"].unique())

    if args.skip_comments:
        print("\n=== Step 1b: parse comments skipped (--skip-comments) ===")
        df_comments_all = pd.DataFrame()
        df_comments_valid = pd.DataFrame()
    else:
        df_comments_all, df_comments_valid = load_or_build_comment_caches(
            valid_authors=valid_authors,
            force_rebuild=args.force_rebuild_comment_cache,
        )

    panel_all = build_panel(df_all)
    panel_all.to_parquet(PANEL_PATH, index=False)
    
    submonth_panel = build_subreddit_month_panel(df_ecosystem, acsi_scores)
    submonth_panel.to_parquet(SUBMONTH_PANEL_PATH, index=False)
    write_submonth_panel_cache_metadata(apply_author_cap=True)

    creators = build_creators_v3(df_all, df_comments_valid)
    creators.to_parquet(CREATORS_PATH, index=False)

    results = run_analysis(
        df_all, df_surv, panel_all, creators, 
        df_comments_valid=df_comments_valid, 
        df_comments_all=df_comments_all, 
        acsi_scores=acsi_scores, 
        submonth_panel=submonth_panel,
        run_legacy=True,
    )
    acsi_scores.to_csv(TABLES_DIR / "acsi_scores_computed.csv", index=False)

    if args.content_validation:
        top_subs = sorted(
            [s for s in iter_subreddits() if s in TREATMENT_SUBS and (DATA_DIR / f"r_{s}_posts.jsonl").exists()],
            key=lambda s: MU_K.get(s, 0), reverse=True,
        )[:5]
        cv_results = compute_content_validation_sample(top_subs)
        results["content_validation"] = cv_results

    validate_run_outputs(
        submonth_panel=submonth_panel,
        score_table=acsi_scores,
        analysis_results=results,
        run_legacy_models=True,
        author_cap_enabled=True,
        valid_author_ids=valid_authors,
        require_comment_cache=not args.skip_comments,
        include_content_validation=args.content_validation,
    )
    print("\nProcessing complete.")
    launch_dashboard(submonth_panel, creators, results, n_ecosystem_authors=df_ecosystem["author"].nunique())

if __name__ == "__main__":
    main()
