import sys
import json
import gc
import os
import random
import re
import argparse
import shutil
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy import stats
from tqdm import tqdm
from flask import Flask, render_template_string, jsonify, send_from_directory

try:
    from scripts import robustness_checks
    from scripts import test as pipeline_tests
except ImportError:
    import robustness_checks
    import test as pipeline_tests

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw_files"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_ROOT = ROOT / "output"
OUTPUT_DIR = OUTPUT_ROOT / "latest"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
ACSI_SCORE_PATH = DATA_DIR / "acsi_scores.csv"
ACSI_MEASUREMENT_SAMPLE_RUN1_PATH = DATA_DIR / "acsi_annotated.csv"
LEGACY_GSE_SCORE_PATH = DATA_DIR / "gse_scores.csv"
POSTS_ECOSYSTEM_PATH = OUTPUT_DIR / "posts_clean_ecosystem.parquet"
POSTS_CREATOR_PATH = OUTPUT_DIR / "posts_clean_all.parquet"
SUBMONTH_PANEL_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.parquet"
CREATORS_PATH = OUTPUT_DIR / "creators.parquet"
SUBMONTH_PANEL_META_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.meta.json"
POST_MONTHLY_AGG_PATH = CACHE_DIR / "post_monthly_aggregates.parquet"
POST_MONTHLY_AGG_META_PATH = CACHE_DIR / "post_monthly_aggregates.meta.json"
DEFAULT_DASHBOARD_PORT = 8000
SUBMONTH_PANEL_CACHE_VERSION = 3
POST_MONTHLY_AGG_CACHE_VERSION = 4
WRITE_OUTPUT_CSVS = True
MAX_TERMINAL_TABLE_ROWS = 25

def bind_check_modules():
    context = sys.modules[__name__]
    robustness_checks.bind_context(context)
    pipeline_tests.bind_context(context)

ACTIVE_SUBREDDITS = None
MAX_LINES_PER_FILE = None

CI_COLOR = "#1a1a2e"
MUTED_BLUE = "#2563eb"
MUTED_GREEN = "#16a34a"
MUTED_RED = "#dc2626"
MUTED_GRAY = "#9ca3af"
MUTED_PURPLE = "#9333ea"
MUTED_ORANGE = "#ea580c"
MODERN_SERIES_COLORS = [MUTED_BLUE, MUTED_RED, MUTED_GREEN, MUTED_PURPLE, MUTED_ORANGE]
MODERN_GRID_COLOR = "#e5e7eb"
MODERN_SPINE_COLOR = "#d1d5db"
MODERN_TEXT_DARK = "#1a1a2e"
MODERN_AXIS_LABEL = "#2d2d2d"
MODERN_TICK_LABEL = "#4a4a4a"
MODERN_ANNOTATION = "#5a5a5a"
MODERN_REFERENCE = "#9ca3af"

def apply_publication_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlecolor": "#1a1a2e",
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.labelcolor": "#2d2d2d",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#4a4a4a",
        "ytick.color": "#4a4a4a",
        "legend.fontsize": 10,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

apply_publication_style()

def _modern_hires_pdf_path(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_hires.pdf")

def apply_modern_style(ax):
    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)
    if not getattr(ax, "images", []):
        ax.grid(True, axis="y", color=MODERN_GRID_COLOR, linewidth=0.5, linestyle="--", zorder=0)
        ax.grid(False, axis="x")

    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["bottom", "left"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(MODERN_SPINE_COLOR)
        ax.spines[spine_name].set_linewidth(0.8)

    ax.tick_params(axis="both", colors=MODERN_TICK_LABEL, labelsize=10, width=0.8)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontfamily("sans-serif")
        tick_label.set_fontweight("normal")
        tick_label.set_color(MODERN_TICK_LABEL)
        tick_label.set_fontsize(10)

    ax.title.set_fontfamily("sans-serif")
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("semibold")
    ax.title.set_color(MODERN_TEXT_DARK)

    for axis_label in [ax.xaxis.label, ax.yaxis.label]:
        axis_label.set_fontfamily("sans-serif")
        axis_label.set_fontsize(12)
        axis_label.set_fontweight("medium")
        axis_label.set_color(MODERN_AXIS_LABEL)

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_facecolor("#ffffff")
        for text in legend.get_texts():
            text.set_fontfamily("sans-serif")
            text.set_fontsize(10)
            text.set_fontweight("normal")
            text.set_color(MODERN_AXIS_LABEL)
    return ax

def apply_modern_figure_style(fig):
    fig.patch.set_facecolor("#ffffff")
    for ax in fig.axes:
        if hasattr(ax, "spines"):
            apply_modern_style(ax)
    return fig

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
    global POSTS_ECOSYSTEM_PATH, POSTS_CREATOR_PATH
    global SUBMONTH_PANEL_PATH, CREATORS_PATH, SUBMONTH_PANEL_META_PATH

    OUTPUT_DIR = validate_generated_output_dir(output_dir)
    TABLES_DIR = OUTPUT_DIR / "tables"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    POSTS_ECOSYSTEM_PATH = OUTPUT_DIR / "posts_clean_ecosystem.parquet"
    POSTS_CREATOR_PATH = OUTPUT_DIR / "posts_clean_all.parquet"
    SUBMONTH_PANEL_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.parquet"
    SUBMONTH_PANEL_META_PATH = OUTPUT_DIR / "subreddit_month_gse_panel.meta.json"
    CREATORS_PATH = OUTPUT_DIR / "creators.parquet"

    for _d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, DATA_DIR, RAW_DATA_DIR]:
        _d.mkdir(exist_ok=True, parents=True)

configure_output_dir(OUTPUT_DIR)

def ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True, parents=True)

def clean_generated_output(output_dir):
    output_dir = validate_generated_output_dir(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)



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

def emit_output_table(frame, path, index=False):
    """Print generated result tables by default; write CSVs only when opted in."""
    path = Path(path)
    if WRITE_OUTPUT_CSVS:
        frame.to_csv(path, index=index)
        print(f"  saved table -> {path}")
        return

    display = frame.reset_index() if index else frame
    print(f"\n--- {path.stem} (terminal only; CSV not written) ---")
    if display.empty:
        print("  [empty]")
        return
    shown = display.head(MAX_TERMINAL_TABLE_ROWS)
    print(shown.to_string(index=False))
    omitted = len(display) - len(shown)
    if omitted > 0:
        print(f"  ... {omitted:,} additional row(s) omitted from terminal preview")

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


QUICK_MODE = False
N_RANDOMIZATION_PERMS = 250 if QUICK_MODE else 1000

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)
END_DATE_EXCLUSIVE = datetime(2025, 1, 1)

EXACT_SHOCK_DATE = datetime(2022, 11, 30)
SHOCK_MONTH = pd.Timestamp("2022-12-01")
STABLE_CREATOR_PRE_END_MONTH = pd.Timestamp(EXACT_SHOCK_DATE.replace(day=1))
POST_AI_ADOPTION_PRE_MONTHS = ("2022-09", "2022-10", "2022-11")
POST_AI_ADOPTION_POST_MONTHS = ("2022-12", "2023-01", "2023-02")
POST_AI_ADOPTION_MONTHS = set(POST_AI_ADOPTION_PRE_MONTHS + POST_AI_ADOPTION_POST_MONTHS)

MIN_PRE_POSTS = 5
STABLE_CREATOR_MIN_PRE_ACTIVE_MONTHS = 2
MAX_POSTS_PER_DAY = 50
RANDOM_SEED = 42
ACSI_RELIABILITY_TARGET = 50

INDEX_LABEL = "AI Content Substitutability Index"
INDEX_SHORT = "ACSI"

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

ACSI_MEASUREMENT_SCORE_COLUMNS = {
    "direct_gen": "direct_gen_score",
    "usefulness": "usefulness_score",
    "quality_comp": "quality_comp_score",
    "physical_req": "physical_req_score",
    "personal_req": "personal_req_score",
}
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
AI_TOOL_SUBSTRINGS = (
    "chatgpt",
    "gpt-4",
    "openai",
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dall e",
    "dalle",
)
AI_BROAD_SUBSTRINGS = AI_TOOL_SUBSTRINGS + (
    "language model",
    "generative ai",
    "ai-generated",
    "ai generated",
    "ai art",
)
AI_TOOL_TOKENS = ("gpt", "llm")
AI_BROAD_TOKENS = ("ai", "gpt", "prompt", "bot", "llm")

def has_boundary_token(lowered, tokens):
    for token in tokens:
        start = 0
        token_len = len(token)
        while True:
            idx = lowered.find(token, start)
            if idx < 0:
                break
            before = lowered[idx - 1] if idx > 0 else " "
            after_idx = idx + token_len
            after = lowered[after_idx] if after_idx < len(lowered) else " "
            before_is_word = before.isalnum() or before == "_"
            after_is_word = after.isalnum() or after == "_"
            if not before_is_word and not after_is_word:
                return True
            start = idx + token_len
    return False

def has_ai_tool_mention(text):
    lowered = str(text or "").lower()
    if any(term in lowered for term in AI_TOOL_SUBSTRINGS):
        return True
    return has_boundary_token(lowered, AI_TOOL_TOKENS)

def has_ai_broad_mention(text):
    lowered = str(text or "").lower()
    if any(term in lowered for term in AI_BROAD_SUBSTRINGS):
        return True
    return has_boundary_token(lowered, AI_BROAD_TOKENS)



SUBREDDIT_ROLE_GROUPS = {
    "treatment": (
        "writing", "worldbuilding", "shortstories", "screenwriting", "poetry",
        "fanfiction", "songwriting", "art", "illustration", "conceptart",
        "comics", "digitalart", "graphic_design", "gamedev", "applyingtocollege",
        "gre", "lsat", "mcat", "sat", "fantasywriters", "scifiwriting",
        "fiction", "3Dmodeling", "personalstatement", "resume", "devops",
    ),
    "control": (
        "woodworking", "pottery", "sewing", "baking", "cooking", "knitting",
        "breadit", "carpentry", "leathercraft", "quilting", "ceramics",
        "photography", "askphotography", "learnart", "learntodraw", "chanceme",
        "college", "gradschool", "lawschool", "medicalschool", "phd", "premed",
        "books", "fermentation", "gardening", "homebrewing", "plants", "chess",
    ),
    "ambiguous": (
        "machinelearning", "learnprogramming", "learnmath", "cscareerquestions",
        "askacademia", "programminghumor", "rowing", "running", "swimming",
        "solotravel", "travel", "act", "answers",
        "AskComputerScience", "AskElectronics", "AskEngineers", "BreakUps",
        "changemyview", "explainlikeimfive", "intermittentfasting",
        "languagelearning", "marketing", "math", "nutrition", "physics",
        "raisedbynarcissists", "SEO", "statistics", "todayilearned",
        "weightlifting", "addiction", "anxiety", "apple", "APStudents",
        "bipolar", "bodyweightfitness", "castiron",
        "CFA", "consulting", "CPA", "crochet",
        "disability", "divorce", "embroidery", "freelance",
        "gainit", "google", "haiku", "history", "interviews", "investing",
        "jewelry", "jobs", "LifeProTips", "lonely", "metalworking",
        "microsoft", "nosleep", "origami", "passive_income", "philosophy",
        "printmaking", "ptsd", "smallbusiness", "startups", "stocks",
        "taxidermy", "tesla", "twosentencehorror",
        "YouShouldKnow",
    ),
}

def build_subreddit_role_map(role_groups):
    role_map = {}
    duplicate_subreddits = []
    for role, subreddit_group in role_groups.items():
        for subreddit in subreddit_group:
            if subreddit in role_map:
                duplicate_subreddits.append(subreddit)
            role_map[subreddit] = role
    if duplicate_subreddits:
        duplicates = ", ".join(sorted(set(duplicate_subreddits)))
        raise ValueError(f"Subreddits assigned to multiple role groups: {duplicates}")
    return role_map

SUBREDDITS = build_subreddit_role_map(SUBREDDIT_ROLE_GROUPS)

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

TREATMENT_SUBS = {s for s, r in SUBREDDITS.items() if r == "treatment"}

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
STABLE_CREATOR_PRE_MONTHS = ALL_MONTHS[ALL_MONTHS < STABLE_CREATOR_PRE_END_MONTH]
N_STABLE_CREATOR_PRE_MONTHS = int(len(STABLE_CREATOR_PRE_MONTHS))


def iter_subreddits():
    if ACTIVE_SUBREDDITS is None:
        return list(SUBREDDITS.keys())
    return [s for s in ACTIVE_SUBREDDITS if s in SUBREDDITS]

def post_text_for_keywords(payload):
    title = str(payload.get("title") or "")
    body = str(payload.get("selftext") or payload.get("body") or "")
    return f"{title} {body}"

def raw_post_path(sub):
    return RAW_DATA_DIR / f"r_{sub}_posts.jsonl"

def iter_post_payloads(sub, desc, start_date=None, end_date_exclusive=None):
    start_date = START_DATE if start_date is None else start_date
    end_date_exclusive = END_DATE_EXCLUSIVE if end_date_exclusive is None else end_date_exclusive

    path = raw_post_path(sub)
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
            if dt < start_date or dt >= end_date_exclusive:
                continue

            author = str(p.get("author") or "")
            if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"):
                continue

            score = p.get("score")
            try:
                score = int(score) if score is not None else 0
            except Exception:
                score = 0

            yield author, dt, score, str(p.get("id") or ""), p

def post_file_signature(sub):
    path = raw_post_path(sub)
    st = path.stat()
    return {
        "subreddit": sub,
        "path": str(path.relative_to(ROOT)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

MONTH_START_CACHE = {}

def month_label_for_datetime(dt):
    return f"{dt.year:04d}-{dt.month:02d}"

def month_start_for_datetime(dt):
    key = (dt.year, dt.month)
    month = MONTH_START_CACHE.get(key)
    if month is None:
        month = pd.Timestamp(year=dt.year, month=dt.month, day=1)
        MONTH_START_CACHE[key] = month
    return month

def available_post_subreddits():
    return [sub for sub in iter_subreddits() if raw_post_path(sub).exists()]

def use_persistent_raw_cache():
    return ACTIVE_SUBREDDITS is None and MAX_LINES_PER_FILE is None

def weighted_measurement_mean(group, column_name):
    weights = group["_weight"].astype(float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float("nan")
    return float((group[column_name].astype(float) * weights).sum() / total_weight)

def acsi_component_from_measurement_average(avg_0_to_3):
    return 1 + (avg_0_to_3 * 4 / 3)

def recompute_acsi_scores_from_run1(
    measurement_path=ACSI_MEASUREMENT_SAMPLE_RUN1_PATH,
    output_path=ACSI_SCORE_PATH,
):
    print(f"\n=== Step 3a: recompute {INDEX_SHORT} scores from run-1 measurements ===")
    measurement_path = Path(measurement_path)
    output_path = Path(output_path)
    if not measurement_path.exists():
        raise FileNotFoundError(f"Missing {INDEX_SHORT} measurement file: {measurement_path}")

    coded = pd.read_csv(measurement_path)
    required_columns = {
        "subreddit",
        "ai_related_flag",
        *ACSI_MEASUREMENT_SCORE_COLUMNS.values(),
    }
    missing_columns = required_columns - set(coded.columns)
    if missing_columns:
        raise ValueError(
            f"{measurement_path} is missing required {INDEX_SHORT} columns: "
            f"{sorted(missing_columns)}"
        )

    coded["subreddit"] = coded["subreddit"].astype(str).str.strip()
    registered_subreddits = set(SUBREDDITS.keys())
    extra_subreddits = sorted(set(coded["subreddit"]) - registered_subreddits)
    if extra_subreddits:
        print(
            f"  ignoring {len(extra_subreddits)} measurement subreddit(s) not registered in run.py: "
            + ", ".join(extra_subreddits)
        )
    coded = coded[coded["subreddit"].isin(registered_subreddits)].copy()
    if coded.empty:
        raise ValueError(f"{measurement_path} has no registered subreddit measurement rows.")

    numeric_columns = list(ACSI_MEASUREMENT_SCORE_COLUMNS.values()) + ["ai_related_flag"]
    if "hard_case_flag" in coded.columns:
        numeric_columns.append("hard_case_flag")
    for column_name in numeric_columns:
        coded[column_name] = pd.to_numeric(coded[column_name], errors="raise")

    if "hard_case_flag" not in coded.columns:
        coded["hard_case_flag"] = 0
    coded["hard_case_flag"] = coded["hard_case_flag"].fillna(0).astype(int)
    coded["ai_related_flag"] = coded["ai_related_flag"].fillna(0).astype(int)
    coded["_weight"] = 1.0
    coded.loc[coded["hard_case_flag"].eq(1), "_weight"] = 0.5

    n_coded = coded.groupby("subreddit").size().rename("n_coded")
    n_ai = coded[coded["ai_related_flag"].eq(1)].groupby("subreddit").size().rename("n_ai_related_excluded")
    n_hard = coded[coded["hard_case_flag"].eq(1)].groupby("subreddit").size().rename("n_hard_cases")
    used = coded[coded["ai_related_flag"].ne(1)].copy()

    rows = []
    missing_subreddits = []
    for subreddit in sorted(coded["subreddit"].unique()):
        group = used[used["subreddit"].eq(subreddit)]
        if group.empty:
            missing_subreddits.append(subreddit)
            continue

        avg_by_output_column = {
            output_column: weighted_measurement_mean(group, source_column)
            for output_column, source_column in ACSI_MEASUREMENT_SCORE_COLUMNS.items()
        }
        direct_gen = acsi_component_from_measurement_average(avg_by_output_column["direct_gen"])
        usefulness = acsi_component_from_measurement_average(avg_by_output_column["usefulness"])
        quality_comp = acsi_component_from_measurement_average(avg_by_output_column["quality_comp"])
        physical_req = acsi_component_from_measurement_average(avg_by_output_column["physical_req"])
        personal_req = acsi_component_from_measurement_average(avg_by_output_column["personal_req"])
        raw_gse = direct_gen + usefulness + quality_comp + (6 - physical_req) + (6 - personal_req)
        gse = max(0.0, min(1.0, (raw_gse - 5) / 20))

        n_used = int(len(group))
        rows.append({
            "subreddit": subreddit,
            "direct_gen": direct_gen,
            "usefulness": usefulness,
            "quality_comp": quality_comp,
            "physical_req": physical_req,
            "personal_req": personal_req,
            "raw_gse": raw_gse,
            "gse": gse,
            "n_coded": int(n_coded.get(subreddit, 0)),
            "n_used": n_used,
            "n_ai_related_excluded": int(n_ai.get(subreddit, 0)),
            "n_hard_cases": int(n_hard.get(subreddit, 0)),
            "hard_case_policy": "downweight",
            "score_reliability": "low" if n_used < 50 else ("medium" if n_used < 100 else "high"),
            "low_n_flag": int(n_used < 50),
            "avg_direct_gen_0_to_3": avg_by_output_column["direct_gen"],
            "avg_usefulness_0_to_3": avg_by_output_column["usefulness"],
            "avg_quality_comp_0_to_3": avg_by_output_column["quality_comp"],
            "avg_physical_req_0_to_3": avg_by_output_column["physical_req"],
            "avg_personal_req_0_to_3": avg_by_output_column["personal_req"],
        })

    if missing_subreddits:
        raise ValueError(
            "No usable run-1 measurement rows after AI exclusions for: "
            + ", ".join(missing_subreddits)
        )

    score_rows = pd.DataFrame(rows).sort_values("subreddit")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    score_rows.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(
        f"  wrote {output_path}: {len(score_rows):,} subreddits, "
        f"{int(score_rows['n_used'].sum()):,} usable rows"
    )
    return load_acsi_scores()

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
        "post_ai_adoption_months": sorted(POST_AI_ADOPTION_MONTHS),
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
        "post_ai_adoption_months": sorted(POST_AI_ADOPTION_MONTHS),
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

def regression_sample_summary(panel, model_data):
    panel_months = sorted(panel["year_month"].astype(str).dropna().unique())
    model_months = sorted(model_data["year_month"].astype(str).dropna().unique())
    excluded_months = [month for month in panel_months if month not in model_months]
    n_subreddits = safe_int(model_data["subreddit"].nunique())
    month_subreddit_counts = (
        model_data.groupby("year_month")["subreddit"].nunique()
        if not model_data.empty
        else pd.Series(dtype=int)
    )
    partial_months = [
        str(month)
        for month, count in month_subreddit_counts.items()
        if n_subreddits is not None and int(count) != n_subreddits
    ]
    transition_month = pd.Timestamp(EXACT_SHOCK_DATE).strftime("%Y-%m")
    transition_rows = model_data[model_data["year_month"].astype(str).eq(transition_month)]
    transition_values = []
    if "post_shock" in transition_rows.columns and not transition_rows.empty:
        transition_values = sorted(
            pd.to_numeric(transition_rows["post_shock"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    return {
        "n_panel_rows": safe_int(len(panel)),
        "n_model_rows": safe_int(len(model_data)),
        "n_model_subreddits": n_subreddits,
        "n_panel_months": safe_int(len(panel_months)),
        "n_model_months": safe_int(len(model_months)),
        "excluded_months": ", ".join(excluded_months) if excluded_months else "none",
        "partial_months": ", ".join(partial_months) if partial_months else "none",
        "transition_month": transition_month,
        "transition_month_included": bool(transition_month in model_months),
        "transition_month_post_shock_values": ", ".join(str(value) for value in transition_values) if transition_values else "none",
        "post_period_start": SHOCK_MONTH.strftime("%Y-%m"),
    }

def print_regression_sample_summary(label, summary):
    print(
        f"  {label} sample: N={summary['n_model_rows']:,} "
        f"of {summary['n_panel_rows']:,} panel rows; "
        f"{summary['n_model_subreddits']:,} subreddits x "
        f"{summary['n_model_months']:,} month(s)."
    )
    print(
        f"  Excluded months: {summary['excluded_months']}; "
        f"partial months: {summary['partial_months']}."
    )
    transition_status = "included" if summary["transition_month_included"] else "excluded"
    print(
        f"  Transition month {summary['transition_month']} is {transition_status}; "
        f"post starts {summary['post_period_start']}."
    )

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
        print(
            f"WARNING: {INDEX_SHORT} file missing registered subreddits; "
            f"they will be excluded until scored: {missing_subreddits}"
        )
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


def parse_posts():
    print("\n=== Step 1a: parse posts ===")
    frames = []
    for sub in iter_subreddits():
        rows = []
        for author, dt, score, post_id, payload in iter_post_payloads(sub, f"  r/{sub} posts"):
            month_label = month_label_for_datetime(dt)
            month = month_start_for_datetime(dt)
            ai_post_mention = 0
            tool_post_mention = 0
            if month_label in POST_AI_ADOPTION_MONTHS:
                post_text = post_text_for_keywords(payload)
                ai_post_mention = int(has_ai_broad_mention(post_text))
                tool_post_mention = int(has_ai_tool_mention(post_text))
            rows.append({
                "author":       author,
                "subreddit":    sub,
                "date":         dt,
                "year_month":   month_label,
                "year_month_dt": month,
                "score":        score,
                "post_id":      post_id,
                "ai_post_mention": ai_post_mention,
                "tool_post_mention": tool_post_mention,
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

def add_post_mention_rates(panel):
    for column_name in ["ai_post_mentions", "tool_post_mentions"]:
        if column_name not in panel.columns:
            panel[column_name] = 0
        panel[column_name] = pd.to_numeric(panel[column_name], errors="coerce").fillna(0)

    posts = pd.to_numeric(panel["posts"], errors="coerce").replace(0, np.nan)
    panel["ai_post_rate"] = (panel["ai_post_mentions"] / posts).fillna(0.0)
    panel["tool_post_rate"] = (panel["tool_post_mentions"] / posts).fillna(0.0)
    return panel

def panel_subreddits_with_posts(panel):
    if panel is None or panel.empty or "posts" not in panel.columns:
        return []
    post_totals = panel.groupby("subreddit")["posts"].sum()
    return sorted(post_totals[post_totals > 0].index.astype(str).tolist())

def build_subreddit_month_panel(df_all, acsi_scores):
    print(f"\n=== Step 3b: Build ecosystem subreddit-month panel for {INDEX_SHORT} dose-response ===")
    df = df_all.copy()
    df["score_pos"] = df["score"].clip(lower=0)
    df["post_value"] = 1.0 + np.log1p(df["score_pos"])

    agg_kwargs = {
        "posts": ("post_id", "count"),
        "active_creators": ("author", "nunique"),
        "avg_score": ("score", "mean"),
        "calibrated_output": ("post_value", "sum"),
    }
    if {"ai_post_mention", "tool_post_mention"}.issubset(df.columns):
        agg_kwargs.update({
            "ai_post_mentions": ("ai_post_mention", "sum"),
            "tool_post_mentions": ("tool_post_mention", "sum"),
        })

    agg = (
        df.groupby(["subreddit", "year_month_dt"], as_index=False)
        .agg(**agg_kwargs)
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
    panel = add_post_mention_rates(panel)
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

def extended_post_monthly_cache_paths():
    return (
        CACHE_DIR / "extended_post_monthly_aggregates.parquet",
        CACHE_DIR / "extended_post_monthly_aggregates.meta.json",
    )

def extended_post_monthly_cache_metadata(target_subreddits, start_date, end_date_exclusive, apply_author_cap):
    post_files = []
    for subreddit in target_subreddits:
        path = raw_post_path(subreddit)
        if path.exists():
            post_files.append(post_file_signature(subreddit))
    return {
        "cache_version": 1,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date_exclusive": end_date_exclusive.strftime("%Y-%m-%d"),
        "shock_month": SHOCK_MONTH.strftime("%Y-%m-%d"),
        "max_lines_per_file": MAX_LINES_PER_FILE,
        "target_subreddits": list(target_subreddits),
        "apply_author_cap": bool(apply_author_cap),
        "post_files": post_files,
    }

def extended_post_monthly_cache_is_current(target_subreddits, start_date, end_date_exclusive, apply_author_cap):
    panel_path, meta_path = extended_post_monthly_cache_paths()
    if not panel_path.exists() or not meta_path.exists():
        return False
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = extended_post_monthly_cache_metadata(
            target_subreddits,
            start_date,
            end_date_exclusive,
            apply_author_cap,
        )
        return old == expected
    except Exception:
        return False

def write_extended_post_monthly_cache_metadata(target_subreddits, start_date, end_date_exclusive, apply_author_cap):
    _panel_path, meta_path = extended_post_monthly_cache_paths()
    ensure_cache_dir()
    metadata = extended_post_monthly_cache_metadata(
        target_subreddits,
        start_date,
        end_date_exclusive,
        apply_author_cap,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

def build_extended_post_monthly_aggregate(target_subreddits, start_date, end_date_exclusive, apply_author_cap=True):
    print("\n=== Extended panel cache: Build post subreddit-month aggregates (2020-2024) ===")
    days = max((end_date_exclusive - start_date).days, 1)
    cap = MAX_POSTS_PER_DAY * days

    valid_authors = None
    if apply_author_cap:
        author_counts = Counter()
        for sub in target_subreddits:
            for author, _dt, _score, _post_id, _payload in iter_post_payloads(
                sub,
                f"  r/{sub} extended posts pass 1",
                start_date=start_date,
                end_date_exclusive=end_date_exclusive,
            ):
                author_counts[author] += 1
        valid_authors = {author for author, n_posts in author_counts.items() if n_posts <= cap}
        del author_counts
        gc.collect()
    else:
        print("  author cap skipped: one-pass streaming mode")

    author_ids = {}
    cells = defaultdict(lambda: {
        "posts": 0,
        "author_ids": set(),
        "score_sum": 0.0,
        "calibrated_output": 0.0,
        "ai_post_mentions": 0,
        "tool_post_mentions": 0,
    })
    for sub in target_subreddits:
        pass_label = "pass 2" if apply_author_cap else "one pass"
        for author, dt, score, _post_id, payload in iter_post_payloads(
            sub,
            f"  r/{sub} extended posts {pass_label}",
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
        ):
            if valid_authors is not None and author not in valid_authors:
                continue
            author_id = author_ids.setdefault(author, len(author_ids))
            month_label = month_label_for_datetime(dt)
            month = month_start_for_datetime(dt)
            cell = cells[(sub, month)]
            cell["posts"] += 1
            cell["author_ids"].add(author_id)
            cell["score_sum"] += score
            cell["calibrated_output"] += 1.0 + np.log1p(max(score, 0))
            if month_label in POST_AI_ADOPTION_MONTHS:
                post_text = post_text_for_keywords(payload)
                cell["ai_post_mentions"] += int(has_ai_broad_mention(post_text))
                cell["tool_post_mentions"] += int(has_ai_tool_mention(post_text))

    rows = []
    for (sub, month), cell in cells.items():
        rows.append({
            "subreddit": sub,
            "year_month_dt": month,
            "posts": cell["posts"],
            "active_creators": len(cell["author_ids"]),
            "avg_score": cell["score_sum"] / cell["posts"] if cell["posts"] else 0.0,
            "calibrated_output": cell["calibrated_output"],
            "ai_post_mentions": cell["ai_post_mentions"],
            "tool_post_mentions": cell["tool_post_mentions"],
        })

    if rows:
        agg = pd.DataFrame(rows)
    else:
        agg = pd.DataFrame(columns=[
            "subreddit", "year_month_dt", "posts", "active_creators", "avg_score",
            "calibrated_output", "ai_post_mentions", "tool_post_mentions",
        ])

    extended_months = pd.date_range(
        pd.Timestamp(start_date).replace(day=1),
        pd.Timestamp(end_date_exclusive).replace(day=1) - pd.DateOffset(months=1),
        freq="MS",
    )
    grid = pd.MultiIndex.from_product(
        [target_subreddits, extended_months],
        names=["subreddit", "year_month_dt"],
    ).to_frame(index=False)

    panel = grid.merge(agg, on=["subreddit", "year_month_dt"], how="left")
    for col in ["posts", "active_creators", "calibrated_output", "ai_post_mentions", "tool_post_mentions"]:
        panel[col] = panel[col].fillna(0)
    panel["avg_score"] = panel["avg_score"].fillna(0)

    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    panel["post_shock"] = (panel["year_month_dt"] >= SHOCK_MONTH).astype(int)
    panel["log_posts"] = np.log1p(panel["posts"])
    panel["log_active_creators"] = np.log1p(panel["active_creators"])
    panel["log_calibrated_output"] = np.log1p(panel["calibrated_output"])
    panel["log_avg_score"] = np.log1p(panel["avg_score"].clip(lower=0))
    panel = add_post_mention_rates(panel)
    print(f"  extended post aggregate panel: {len(panel):,} rows")
    return panel

def load_or_build_extended_post_monthly_aggregate(target_subreddits, start_date, end_date_exclusive, apply_author_cap=True):
    target_subreddits = sorted(str(subreddit) for subreddit in target_subreddits)
    panel_path, _meta_path = extended_post_monthly_cache_paths()
    if use_persistent_raw_cache() and extended_post_monthly_cache_is_current(
        target_subreddits,
        start_date,
        end_date_exclusive,
        apply_author_cap,
    ):
        print("\n=== Extended panel cache: load cached post subreddit-month aggregates ===")
        panel = pd.read_parquet(panel_path)
        print(f"  loaded: {len(panel):,} rows from {panel_path}")
        return panel

    panel = build_extended_post_monthly_aggregate(
        target_subreddits,
        start_date,
        end_date_exclusive,
        apply_author_cap=apply_author_cap,
    )
    if use_persistent_raw_cache():
        ensure_cache_dir()
        panel.to_parquet(panel_path, index=False)
        write_extended_post_monthly_cache_metadata(
            target_subreddits,
            start_date,
            end_date_exclusive,
            apply_author_cap,
        )
        print(f"  cached extended post aggregates -> {panel_path}")
    else:
        print("  extended post aggregate cache write skipped: non-full raw scan")
    return panel

def build_extended_subreddit_month_panel(acsi_scores, target_subreddits, apply_author_cap=True):
    start_date = datetime(2020, 1, 1)
    end_date_exclusive = datetime(2025, 1, 1)
    base_panel = load_or_build_extended_post_monthly_aggregate(
        target_subreddits,
        start_date,
        end_date_exclusive,
        apply_author_cap=apply_author_cap,
    )
    base_panel = add_post_mention_rates(base_panel)
    panel = attach_acsi_scores(base_panel, acsi_scores)
    return panel

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

def build_creators_v3(df_all, _unused_activity_frame=None):
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

    stable_pre = (
        df_all[df_all["year_month_dt"].isin(STABLE_CREATOR_PRE_MONTHS)]
        .groupby("author")
        .agg(stable_pre_active_months=("year_month", "nunique"))
        .reset_index()
    )
    creators = creators.merge(stable_pre, on="author", how="left")
    creators["stable_pre_active_months"] = (
        creators["stable_pre_active_months"].fillna(0).astype(int)
    )

    creators["is_stable"] = (
        creators["stable_pre_active_months"] >= STABLE_CREATOR_MIN_PRE_ACTIVE_MONTHS
    ).astype(int)

    print(f"  total eligible: {len(creators):,}")
    print(f"  survived: {creators['survived'].sum():,}  exits: {(creators['survived']==0).sum():,}")
    print(
        f"  stable (>={STABLE_CREATOR_MIN_PRE_ACTIVE_MONTHS} of "
        f"{N_STABLE_CREATOR_PRE_MONTHS} available pre-shock months): "
        f"{creators['is_stable'].sum():,}"
    )
    return creators


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
    emit_output_table(correlation_table, TABLES_DIR / "acsi_component_correlations.csv", index=True)
    return correlation_table.round(3).reset_index().rename(columns={"index": "component"}).to_dict("records")

def fit_three_dimensional_acsi_model(acsi_panel, outcome="log_posts", additional_terms=None):
    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    additional_terms = additional_terms or []
    model_data = acsi_panel.dropna(
        subset=terms + additional_terms + [outcome, "subreddit", "year_month"]
    ).copy()
    sample_summary = regression_sample_summary(acsi_panel, model_data)
    rhs_terms = terms + additional_terms
    model = fit_ols(
        f"{outcome} ~ " + " + ".join(rhs_terms) + " + C(subreddit) + C(year_month)",
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
        model_result.update(sample_summary)
        results.append(model_result)
    return model, results





def save_plot(fig, output_path):
    apply_publication_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_modern_figure_style(fig)
    try:
        fig.tight_layout(pad=1.5)
    except Exception:
        pass
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    if output_path.suffix.lower() == ".png":
        fig.savefig(_modern_hires_pdf_path(output_path), bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)









def evenly_spaced_subreddits(frame, n=8, ascending=True):
    ordered = frame.sort_values("non_personal_norm", ascending=ascending).reset_index(drop=True)
    if len(ordered) <= n:
        return ordered["subreddit"].astype(str).tolist()
    positions = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return ordered.iloc[positions]["subreddit"].astype(str).tolist()

def plot_subreddit_small_multiples(submonth_panel, acsi_scores, output_path):
    apply_publication_style()
    panel = submonth_panel.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    if "non_personal_norm" not in panel.columns and acsi_scores is not None:
        score_columns = ["subreddit", "non_personal_norm"]
        panel = panel.merge(acsi_scores[score_columns], on="subreddit", how="left")

    forced_exclusions = {"graphic_design", "personalstatement"}
    existing_subreddits = set(panel["subreddit"].astype(str))
    excluded_low_activity = [
        subreddit for subreddit in forced_exclusions if subreddit in existing_subreddits
    ]
    if "log_posts" in panel.columns:
        for subreddit, group in panel.sort_values("year_month_dt").groupby("subreddit"):
            if str(subreddit) in forced_exclusions:
                continue
            log_posts = group["log_posts"].astype(float)
            median_log_posts = log_posts.median()
            has_single_month_drop = bool((median_log_posts - log_posts).gt(3.0).any())
            low_months = log_posts.lt(1.0).fillna(False).to_numpy()
            run_length = 0
            for is_low in low_months:
                run_length = run_length + 1 if is_low else 0
                if run_length >= 3:
                    excluded_low_activity.append(str(subreddit))
                    break
            if has_single_month_drop:
                excluded_low_activity.append(str(subreddit))
        if excluded_low_activity:
            excluded_low_activity = sorted(set(excluded_low_activity))
            panel = panel[
                ~panel["subreddit"].astype(str).isin(set(excluded_low_activity))
            ].copy()

    sub_scores = panel[["subreddit", "non_personal_norm"]].drop_duplicates().dropna().copy()
    if sub_scores.empty:
        return None

    ranked = sub_scores["non_personal_norm"].rank(method="first")
    sub_scores["persfree_group"] = pd.qcut(
        ranked,
        q=3,
        labels=["Low PersFree", "Middle PersFree", "High PersFree"],
    )

    selected = []
    group_order = ["Low PersFree", "Middle PersFree", "High PersFree"]
    for group_label in group_order:
        group = sub_scores[sub_scores["persfree_group"].astype(str) == group_label].copy()
        ascending = group_label != "High PersFree"
        selected.extend(evenly_spaced_subreddits(group, n=8, ascending=ascending))
    selected = selected[:24]
    if len(selected) < 24:
        return None

    colors = {
        "Low PersFree": "#dc2626",
        "Middle PersFree": "#9ca3af",
        "High PersFree": "#2563eb",
    }
    alphas = {"Low PersFree": 0.8, "Middle PersFree": 0.6, "High PersFree": 0.8}
    group_lookup = sub_scores.set_index("subreddit")["persfree_group"].astype(str).to_dict()

    fig, axes = plt.subplots(3, 8, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, subreddit in zip(axes, selected):
        group_label = group_lookup.get(subreddit, "Middle")
        series = panel[panel["subreddit"].astype(str) == subreddit].sort_values("year_month_dt")
        ax.plot(
            series["year_month_dt"],
            series["log_posts"],
            color=colors.get(group_label, MUTED_GRAY),
            linewidth=1.2,
            alpha=alphas.get(group_label, 0.7),
        )
        ax.axvline(pd.Timestamp("2022-11-01"), color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.text(
            0.03,
            0.92,
            str(subreddit),
            transform=ax.transAxes,
            fontsize=7,
            color="#374151",
            ha="left",
            va="top",
        )

    for ax in axes[len(selected):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=colors[label], linewidth=2, label=label)
        for label in group_order
    ]
    for row_index, group_label in enumerate(group_order):
        axes[row_index * 8].set_ylabel(group_label)
    for col_index in range(8):
        axes[col_index].set_title(f"{col_index + 1}", fontsize=9, color="#5a5a5a")
    fig.suptitle("Post trajectories by PersFree tercile", fontsize=14, y=0.98)
    fig.supylabel("log(1 + monthly posts)", fontsize=12)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0.02, 0.06, 1.0, 0.95])
    save_plot(fig, output_path)
    return {
        "n_subreddits": safe_int(len(selected)),
        "subreddits": selected,
        "n_excluded_low_activity": safe_int(len(excluded_low_activity)),
        "excluded_low_activity_subreddits": excluded_low_activity,
    }




































def run_analysis(df_all=None, creators=None, _unused_activity_frame=None, acsi_scores=None, submonth_panel=None):
    print("\n=== Step 5: analysis ===")
    clean_analysis_artifacts()
    if acsi_scores is None:
        acsi_scores = recompute_acsi_scores_from_run1()
    else:
        acsi_scores = acsi_scores.copy()
    if submonth_panel is not None and not submonth_panel.empty:
        submonth_panel = attach_acsi_scores(submonth_panel, acsi_scores)

    results = {
        "gse_main": None,
        "gse_secondary": None,
        "gse_quartiles": None,
        "gse_event_study": None,
        "robust_time_varying_personal_context": None,
        "robust_event_study_full": None,
        "robust_matched_strict_event_study": None,
        "robust_placebo_permutation": None,
        "robust_placebo_nov2023": None,
        "subreddit_small_multiples": None,
        "gse_construct_validity": None,
        "gse_permutation": None,
        "gse_covariate_adj": None,
        "acsi_three_dimensional": None,
        "backtest_forward_simulation": None,
        "acsi_three_dimensional_covariate_adj": None,
        "acsi_three_dimensional_influence": None,
        "acsi_component_correlations": None,
        "content_validation": None,
        "post_ai_adoption": None,
        "q2_survival": None,
        "q3_engagement": None,
    }




    results["acsi_component_correlations"] = compute_acsi_component_correlation(acsi_scores)
    if results["acsi_component_correlations"]:
        print(f"\n--- {INDEX_SHORT} measurement diagnostics: component correlations saved ---")

    print("\n--- Figure: subreddit small multiples ---")
    try:
        small_multiple_result = plot_subreddit_small_multiples(
            submonth_panel.copy(),
            acsi_scores,
            FIGURES_DIR / "subreddit_small_multiples.png",
        )
        if small_multiple_result:
            results["subreddit_small_multiples"] = small_multiple_result
            print(
                "  Subreddit small multiples saved: "
                f"{small_multiple_result['n_subreddits']} subreddits"
            )
        else:
            results["subreddit_small_multiples"] = {
                "status": "skipped",
                "reason": "insufficient_subreddits_or_acsi_score_variation",
            }
            print("  Subreddit small multiples skipped: insufficient eligible subreddits or ACSI score variation.")
    except Exception as e:
        print(f"  Subreddit small multiples failed: {e}")

    print(f"\n--- MAIN: Three-dimensional {INDEX_SHORT} dose-response DiD ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Three-dimensional main dose-response")
        three_dimensional_model, three_dimensional_results = fit_three_dimensional_acsi_model(acsi_panel)
        if three_dimensional_results:
            print_regression_sample_summary(
                "Three-dimensional main dose-response",
                three_dimensional_results[0],
            )
            emit_output_table(
                pd.DataFrame(three_dimensional_results),
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

    bind_check_modules()
    results = robustness_checks.run_robustness_checks(
        submonth_panel=submonth_panel,
        acsi_scores=acsi_scores,
        df_all=df_all,
        creators=creators,
        results=results,
    )
    print(f"\n  Tables -> {TABLES_DIR}")
    print(f"  Figures -> {FIGURES_DIR}")
    return results















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
    .output-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                   gap: 1rem; margin: 1rem 0; }
    .output-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
                   padding: 1rem; overflow: hidden; }
    .output-card img { width: 100%; height: auto; display: block; border-radius: 4px;
                       background: #ffffff; }
    .output-card .name { color: #f0f6fc; font-weight: 600; overflow-wrap: anywhere; }
    .output-card .meta { color: #8b949e; font-size: .78rem; margin-top: .25rem; }
    .output-card a { color: #79c0ff; text-decoration: none; }
    .output-card a:hover { text-decoration: underline; }
    .preview { max-height: 260px; overflow: auto; margin-top: .6rem; border-top: 1px solid #30363d; }
    .preview table { margin: .4rem 0 0; }
    .preview th, .preview td { font-size: .72rem; padding: .3rem .45rem; white-space: nowrap; }
    footer { color: #484f58; font-size: .8rem; text-align: center; margin-top: 3rem; }
  </style>
</head>
<body>
<div class="container">
  <h1>Reddit GenAI Pipeline — AI Content Substitutability</h1>
  <p class="desc">Empirical validation &mdash; window: {{ start }} to {{ end }}
  &mdash; shock date: Nov 30 2022; post month starts Dec 2022</p>

  <div class="stats">
    {% for stat in top_stats %}
    <div class="stat"><div class="label">{{ stat.label }}</div><div class="value">{{ stat.value }}</div></div>
    {% endfor %}
    <div class="stat"><div class="label">Subreddits</div><div class="value">{{ n_subs }}</div></div>
    {% if show_creator_stats %}
    <div class="stat"><div class="label">Creators (Ecosystem)</div><div class="value">{{ n_authors }}</div></div>
    {% endif %}
    {% if show_stable_stats %}
    <div class="stat"><div class="label">Stable creators</div><div class="value">{{ n_stable }}</div></div>
    {% endif %}
  </div>
  {% if stats_note %}
  <p class="note">{{ stats_note }}</p>
  {% endif %}

  <h2>Backtest And Forecast Reading Guide</h2>
  <p class="desc">Plain English: PersFree is the mechanism signal. The primary holdout is March-August 2024, the peak-displacement Q6-Q7 window. June-December 2024 is a secondary late holdout covering the moderation period. Tercile gap replaces the old binary sign check, and Spearman bottom half is the theory-motivated rank metric for communities below the PersFree median.</p>
  {% if backtest_plain_summary %}
  <table>
    <tr><th>Check</th><th>Result</th><th>How to read it</th></tr>
    {% for row in backtest_plain_summary.rows %}
    <tr>
      <td>{{ row.label }}</td>
      <td>{{ row.value }}</td>
      <td>{{ row.note }}</td>
    </tr>
    {% endfor %}
  </table>
  {% if backtest_plain_summary.creator_note %}
  <p class="note">{{ backtest_plain_summary.creator_note }}</p>
  {% endif %}
  {% else %}
  <div class="reg">Backtest and forward-simulation CSVs have not been generated yet.</div>
  {% endif %}
  <p class="note">Forward simulations are bounded scenario illustrations, not point forecasts. They use each community's historical activity range to avoid runaway extrapolations.</p>

  <h2>Main Three-Dimensional Dose-Response Model</h2>
  <p class="desc">Uses each subreddit's measured profile directly: generation capability, low physical constraint, and low personal-context need interacted separately with the post-ChatGPT period, plus subreddit and month fixed effects. This avoids binary categories and avoids collapsing the three dimensions into one aggregate treatment.</p>
  {% if acsi_main_sample %}
  <p class="note">Regression sample: N={{ acsi_main_sample.n_model_rows }} of {{ acsi_main_sample.n_panel_rows }} panel rows; excluded months: {{ acsi_main_sample.excluded_months }}. Transition month {{ acsi_main_sample.transition_month }} is {{ "included" if acsi_main_sample.transition_month_included else "excluded" }}; post period starts {{ acsi_main_sample.post_period_start }}.</p>
  {% endif %}
  {% if acsi_three_dimensional_rows %}
  <table>
    <tr>
      <th rowspan="2">Dimension</th>
      <th colspan="4">Main</th>
      <th colspan="4">Covariate-Adjusted</th>
    </tr>
    <tr>
      <th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th>
      <th>Coef</th><th>SE</th><th>p-value</th><th>Effect (%)</th>
    </tr>
    {% for r in acsi_three_dimensional_rows %}
    <tr>
      <td>{{ r.label }}</td>
      <td><span class="{{ 'sig' if r.main.pvalue is not none and r.main.pvalue < 0.05 else 'nsig' }}">{% if r.main.coef is not none %}{{ "%.4f"|format(r.main.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.main.se is not none %}{{ "%.4f"|format(r.main.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.main.pvalue is not none %}{{ "%.4f"|format(r.main.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.main.percent_effect_full_exposure) if r.main.percent_effect_full_exposure is not none else '' }}</td>
      <td><span class="{{ 'sig' if r.adjusted.pvalue is not none and r.adjusted.pvalue < 0.05 else 'nsig' }}">{% if r.adjusted.coef is not none %}{{ "%.4f"|format(r.adjusted.coef) }}{% else %}N/A{% endif %}</span></td>
      <td>{% if r.adjusted.se is not none %}{{ "%.4f"|format(r.adjusted.se) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.adjusted.pvalue is not none %}{{ "%.4f"|format(r.adjusted.pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ "%.1f"|format(r.adjusted.percent_effect_full_exposure) if r.adjusted.percent_effect_full_exposure is not none else '' }}</td>
    </tr>
    {% endfor %}
  </table>
  <p class="note">Covariate-adjusted specification adds pre-shock size, pre-shock trend, and subscriber-count controls interacted with the post period.</p>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

  {% if post_ai_adoption %}
  <h3>Post AI-Mention Adoption Check</h3>
  <p class="desc">Compares the change in AI-related post-title/body mention rates from Sep-Nov 2022 to Dec 2022-Feb 2023 against measured subreddit exposure. This is mechanism evidence, not a treatment definition.</p>
  <table>
    <tr><th>Exposure</th><th>AI mention metric</th><th>Pearson r</th><th>p-value</th><th>N</th></tr>
    {% for r in post_ai_adoption.rows %}
    <tr>
      <td>{{ r.exposure_label }}</td>
      <td>{{ r.metric_label }}</td>
      <td>{% if r.pearson_r is not none %}{{ "%.4f"|format(r.pearson_r) }}{% else %}N/A{% endif %}</td>
      <td>{% if r.pearson_pvalue is not none %}{{ "%.4f"|format(r.pearson_pvalue) }}{% else %}N/A{% endif %}</td>
      <td>{{ r.n_subreddits }}</td>
    </tr>
    {% endfor %}
  </table>
  {% endif %}

  <h3>Main Model Robustness</h3>
  {% if acsi_three_dimensional_influence %}
  <div class="reg">
    Leave-one-subreddit low personal-context range:
    {% if acsi_three_dimensional_influence.max_leave_one_out_coef is not none %}{{ "%.4f"|format(acsi_three_dimensional_influence.max_leave_one_out_coef) }}{% else %}N/A{% endif %}
    to
    {% if acsi_three_dimensional_influence.min_leave_one_out_coef is not none %}{{ "%.4f"|format(acsi_three_dimensional_influence.min_leave_one_out_coef) }}{% else %}N/A{% endif %}<br>
    Negative and significant leave-one-out estimates:
    {{ acsi_three_dimensional_influence.n_significant_leave_one_out }} of {{ acsi_three_dimensional_influence.n_subreddits_tested }}.<br>
    Largest shift when omitting: r/{{ acsi_three_dimensional_influence.largest_shift_subreddit }}.
    Positive leave-one-out estimates: {{ acsi_three_dimensional_influence.n_positive_leave_one_out }} of {{ acsi_three_dimensional_influence.n_subreddits_tested }}.
  </div>
  {% endif %}

  <h2>Creator-Level Supporting Checks</h2>
  <h3>Q2 — Creator Exit</h3>
  <div class="reg">{% if q2_survival %}
    Creator exit ~ log pre-shock posting rate:
    <span class="{{ q2_survival_c }}">{% if q2_survival.coef is not none %}{{ "%.4f"|format(q2_survival.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q2_survival.se is not none %}{{ "%.4f"|format(q2_survival.se) }}{% else %}N/A{% endif %}
    p={% if q2_survival.pvalue is not none %}{{ "%.4f"|format(q2_survival.pvalue) }}{% else %}N/A{% endif %}
    N={{ q2_survival.n_obs }}
  {% else %}Not estimated.{% endif %}</div>

  {% if q2_moderation %}
  <h3>Q2 — Creator Exit Moderation</h3>
  <div class="reg">
    Log pre-shock posting rate x low personal-context need:
    <span class="{{ 'sig' if q2_moderation.pvalue is not none and q2_moderation.pvalue < 0.05 else 'nsig' }}">
      {% if q2_moderation.coef is not none %}{{ "%.4f"|format(q2_moderation.coef) }}{% else %}N/A{% endif %}
    </span>
    SE {% if q2_moderation.se is not none %}{{ "%.4f"|format(q2_moderation.se) }}{% else %}N/A{% endif %}
    p={% if q2_moderation.pvalue is not none %}{{ "%.4f"|format(q2_moderation.pvalue) }}{% else %}N/A{% endif %}
    N={{ q2_moderation.n_authors }}
  </div>
  {% endif %}

  <h3>Q3 — Per-Creator Engagement DiD</h3>
  <div class="reg">{% if q3_engagement %}
    Treatment x post DiD:
    <span class="{{ q3_engagement_c }}">{% if q3_engagement.coef is not none %}{{ "%.4f"|format(q3_engagement.coef) }}{% else %}N/A{% endif %}</span>
    SE {% if q3_engagement.se is not none %}{{ "%.4f"|format(q3_engagement.se) }}{% else %}N/A{% endif %}
    p={% if q3_engagement.pvalue is not none %}{{ "%.4f"|format(q3_engagement.pvalue) }}{% else %}N/A{% endif %}
    N={{ q3_engagement.n_obs }}
  {% else %}Not estimated.{% endif %}</div>

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

  <h3>Robustness Figures</h3>
  <div class="fig"><img src="/figures/robust_time_varying_personal_context.png" alt="Quarterly personal-context displacement effect"></div>
  <div class="fig"><img src="/figures/robust_event_study_full.png" alt="Full monthly personal-context event study"></div>
  <div class="fig"><img src="/figures/robust_matched_strict_event_study.png" alt="Strict matched-pairs monthly event study"></div>
  <div class="fig"><img src="/figures/robust_placebo_permutation.png" alt="Permutation placebo histogram"></div>
  <div class="fig"><img src="/figures/robust_placebo_nov2023.png" alt="November 2023 placebo bar chart"></div>
  <div class="fig"><img src="/figures/subreddit_small_multiples.png" alt="Subreddit post trajectories by personal-context exposure"></div>

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

	  <h3>Low Personal-Context Two-Point Summary</h3>
  {% if gse_event_study %}
  <div class="reg">
    Pretrend p={% if gse_event_study.pretrend_pvalue is not none %}{{ "%.4f"|format(gse_event_study.pretrend_pvalue) }}{% else %}N/A{% endif %}.
    The old two-point figure is intentionally omitted; use the full monthly event-study figure above.
  </div>
  {% else %}<div class="reg">Not estimated.</div>{% endif %}

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

  <h2>Generated Paper Output Bundle</h2>
  <p class="desc">Everything generated by the current run is indexed here automatically. Figures are shown as thumbnails; tables, TeX files, PDFs, and large data files are linked directly from <code>output/latest</code>.</p>
  {% if generated_outputs.figures %}
  <h3>Figures</h3>
  <div class="output-grid">
    {% for item in generated_outputs.figures %}
    <div class="output-card">
      <a href="{{ item.href }}"><img src="{{ item.href }}" alt="{{ item.name }}"></a>
      <div class="name"><a href="{{ item.href }}">{{ item.name }}</a></div>
      <div class="meta">{{ item.size_label }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if generated_outputs.tables %}
  <h3>Tables And Data</h3>
  <div class="output-grid">
    {% for item in generated_outputs.tables %}
    <div class="output-card">
      <div class="name"><a href="{{ item.href }}">{{ item.name }}</a></div>
      <div class="meta">{{ item.kind }} · {{ item.size_label }}{% if item.preview_note %} · {{ item.preview_note }}{% endif %}</div>
      {% if item.preview %}
      <div class="preview">
        <table>
          <tr>{% for col in item.preview.columns %}<th>{{ col }}</th>{% endfor %}</tr>
          {% for row in item.preview.rows %}
          <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
          {% endfor %}
        </table>
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if not generated_outputs.figures and not generated_outputs.tables %}
  <div class="reg">No generated paper outputs found yet.</div>
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
    score_table_for_dashboard = None
    score_used_count = None
    score_subreddits = None
    try:
        score_table_for_dashboard = load_acsi_scores()
        score_subreddits = set(score_table_for_dashboard["subreddit"].astype(str).str.strip())
        if "n_used" in score_table_for_dashboard.columns:
            n_used = pd.to_numeric(score_table_for_dashboard["n_used"], errors="coerce")
            if n_used.notna().any():
                score_used_count = int(n_used.fillna(0).sum())
    except Exception:
        score_table_for_dashboard = None

    def load_run1_annotation_counts():
        row_level_path = ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
        if not row_level_path.exists():
            return None
        try:
            coded = pd.read_csv(
                row_level_path,
                usecols=["subreddit", "ai_related_flag"],
                dtype={"subreddit": str},
            ).fillna("")
        except Exception:
            return None
        if score_subreddits:
            coded = coded[coded["subreddit"].astype(str).str.strip().isin(score_subreddits)].copy()
        if coded.empty:
            return None
        ai_flags = pd.to_numeric(coded["ai_related_flag"], errors="coerce").fillna(0)
        return {
            "coded": int(len(coded)),
            "usable": int(ai_flags.ne(1).sum()),
        }

    run1_counts = load_run1_annotation_counts()
    top_stats = []
    if score_used_count is not None:
        top_stats.append({"label": "ACSI aggregate used", "value": f"{score_used_count:,}"})
    if run1_counts is not None:
        top_stats.append({"label": "Run-1 usable annotations", "value": f"{run1_counts['usable']:,}"})
        top_stats.append({"label": "Run-1 coded annotations", "value": f"{run1_counts['coded']:,}"})
    top_stats.append({"label": "Monthly panel posts", "value": f"{raw_panel_posts:,}"})
    stats_note = None
    if score_used_count is not None and run1_counts is not None:
        stats_note = (
            "ACSI aggregate used is read from data/acsi_scores.csv; run-1 annotation "
            f"counts are read from {ACSI_MEASUREMENT_SAMPLE_RUN1_PATH.relative_to(ROOT)} and scoped to "
            "the scored dashboard subreddits."
        )

    if n_ecosystem_authors is not None:
        n_authors = f"{int(n_ecosystem_authors):,}"
    elif "author" in creators.columns:
        n_authors = f"{int(creators['author'].nunique()):,}"
    else:
        n_authors = "N/A"
    n_subs    = int(panel["subreddit"].nunique())
    has_creator_data = "author" in creators.columns and not creators.empty
    n_stable = int(creators["is_stable"].sum()) if "is_stable" in creators.columns else None
    show_creator_stats = n_ecosystem_authors is not None or has_creator_data
    show_stable_stats = n_stable is not None

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

    def three_dimensional_dashboard_rows():
        main_rows = results.get("acsi_three_dimensional") or []
        adjusted_by_term = {
            row.get("term"): row
            for row in (results.get("acsi_three_dimensional_covariate_adj") or [])
            if row.get("term")
        }
        empty_result = {
            "coef": None,
            "se": None,
            "pvalue": None,
            "percent_effect_full_exposure": None,
        }
        rows = []
        for main_row in main_rows:
            main_result = {**empty_result, **main_row}
            adjusted_row = {**empty_result, **adjusted_by_term.get(main_row.get("term"), {})}
            rows.append({
                "label": main_row.get("label"),
                "main": to_obj(main_result),
                "adjusted": to_obj(adjusted_row),
            })
        return rows

    def main_model_sample_summary():
        rows = results.get("acsi_three_dimensional") or []
        if not rows:
            return None
        sample_keys = [
            "n_panel_rows",
            "n_model_rows",
            "n_model_subreddits",
            "n_panel_months",
            "n_model_months",
            "excluded_months",
            "partial_months",
            "transition_month",
            "transition_month_included",
            "transition_month_post_shock_values",
            "post_period_start",
        ]
        return {key: rows[0].get(key) for key in sample_keys}

    def load_score_rows(limit=25):
        if score_table_for_dashboard is None:
            return []
        try:
            score_preview = score_table_for_dashboard.copy()
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

    def dashboard_number(value):
        numeric = safe_float(value)
        return "N/A" if numeric is None else f"{numeric:.4f}"

    def load_backtest_plain_summary():
        backtest_path = OUTPUT_DIR / "backtest_results.csv"
        forward_path = OUTPUT_DIR / "forward_simulation.csv"
        if not backtest_path.exists():
            return None
        try:
            backtest_table = pd.read_csv(backtest_path)
        except Exception:
            return None
        if "row_type" not in backtest_table.columns:
            return None

        rows = [
            {
                "label": "Primary holdout",
                "value": "Mar-Aug 2024",
                "note": "Q6-Q7 peak displacement window, trained through Feb 2024.",
            },
            {
                "label": "Secondary late holdout",
                "value": "Jun-Dec 2024",
                "note": "Moderation-period check, reported separately from the primary window.",
            },
        ]
        primary_cutoff = "train_through_2024-02"
        metrics = backtest_table[
            (backtest_table["row_type"] == "backtest_metric")
            & (backtest_table.get("group", "") == "overall")
            & (backtest_table.get("cutoff", "").astype(str).str.startswith(primary_cutoff))
        ].copy()
        validity = backtest_table[
            backtest_table["row_type"] == "mechanism_validity_metric"
        ].copy()

        def metric_row(model_name):
            if metrics.empty or "model" not in metrics.columns:
                return None
            match = metrics[metrics["model"] == model_name]
            return None if match.empty else match.iloc[0]

        coeffs = backtest_table[
            (backtest_table["row_type"] == "backtest_model_coefficient")
            & (backtest_table.get("model", "") == "mechanism_persfree_mu_lag")
            & (backtest_table.get("cutoff", "").astype(str).str.startswith(primary_cutoff))
        ].copy()
        if not coeffs.empty:
            coeff = coeffs.iloc[0]
            rows.append({
                "label": "PersFree mechanism",
                "value": f"{dashboard_number(coeff.get('coef'))} (p={dashboard_number(coeff.get('pvalue'))})",
                "note": "Negative means higher pre-shock PersFree predicts lower post-shock activity after controls.",
            })

        for label, model_name, note in [
            (
                "PersFree forecast RMSE",
                "mechanism_persfree_mu_lag",
                "Level forecast error when the mechanism model is used directly in the primary window.",
            ),
            (
                "Hybrid forecast RMSE",
                "forecast_hybrid_locf_persfree",
                "Level forecast error after blending PersFree with persistence in the primary window.",
            ),
            (
                "Persistence baseline RMSE",
                "baseline_last_observation_carried_forward",
                "Primary-window level forecast error if each community stays at its last observed level.",
            ),
        ]:
            metric = metric_row(model_name)
            if metric is not None:
                rows.append({
                    "label": label,
                    "value": dashboard_number(metric.get("rmse")),
                    "note": note,
                })

        if not metrics.empty and "rmse" in metrics.columns:
            scored_metrics = metrics.copy()
            scored_metrics["rmse_numeric"] = pd.to_numeric(scored_metrics["rmse"], errors="coerce")
            scored_metrics = scored_metrics.dropna(subset=["rmse_numeric"])
            if not scored_metrics.empty:
                best = scored_metrics.sort_values("rmse_numeric").iloc[0]
                rows.append({
                    "label": "Best Mar-Aug 2024 RMSE",
                    "value": f"{dashboard_number(best.get('rmse'))} ({best.get('model')})",
                    "note": "Lower is better; this is the honest benchmark for forecast usefulness.",
                })

        def validity_row(evaluation_window, metric_name, model_name="residual_mechanism_persfree_mu_lag", group_name=None):
            if validity.empty:
                return None
            mask = (
                (validity.get("evaluation_window", "") == evaluation_window)
                & (validity.get("metric", "") == metric_name)
                & (validity.get("model", "") == model_name)
            )
            if group_name is not None:
                mask &= validity.get("group", "") == group_name
            match = validity[mask]
            return None if match.empty else match.iloc[0]

        for label, metric_name, note, group_name in [
            (
                "Primary residual RMSE",
                "residual_rmse_mechanism",
                "Residualized mechanism-model error after subtracting subreddit trend.",
                "overall",
            ),
            (
                "Primary Spearman all",
                "spearman_all",
                "Rank correlation across all 124 communities in the primary window.",
                "overall",
            ),
            (
                "Primary Spearman bottom half",
                "spearman_bottom_half",
                "Theory-motivated rank check among communities below median PersFree.",
                "bottom_half_persfree",
            ),
            (
                "Primary Spearman low tercile",
                "spearman_low_persfree",
                "Rank check inside the bottom PersFree tercile.",
                "low_persfree",
            ),
        ]:
            metric = validity_row("primary_holdout", metric_name, group_name=group_name)
            if metric is not None:
                rows.append({
                    "label": label,
                    "value": dashboard_number(metric.get("value")),
                    "note": note,
                })

        gap = validity_row(
            "primary_holdout",
            "tercile_gap",
            group_name="bottom_minus_top_persfree",
        )
        if gap is not None:
            rows.append({
                "label": "Primary tercile gap",
                "value": f"{dashboard_number(gap.get('tercile_gap'))} (p={dashboard_number(gap.get('tercile_gap_pvalue'))})",
                "note": "Bottom PersFree mean residual minus top PersFree mean residual; one-sided t-test expects a more negative bottom tercile.",
            })

        late_spearman = validity_row("late_holdout", "spearman_all", group_name="overall")
        if late_spearman is not None:
            rows.append({
                "label": "Late holdout Spearman all",
                "value": dashboard_number(late_spearman.get("value")),
                "note": "June-Dec 2024 moderation-period rank check, kept separate from primary.",
            })

        metadata = backtest_table[
            (backtest_table["row_type"] == "backtest_model_metadata")
            & (backtest_table.get("model", "") == "forecast_hybrid_locf_persfree")
            & (backtest_table.get("cutoff", "").astype(str).str.startswith(primary_cutoff))
        ].copy()
        if not metadata.empty:
            blend = metadata.iloc[0]
            rows.append({
                "label": "Hybrid blend",
                "value": f"{dashboard_number(blend.get('mechanism_weight'))} PersFree / {dashboard_number(blend.get('locf_weight'))} persistence",
                "note": "Weight chosen on the tail of the training window before testing Mar-Aug 2024.",
            })

        creator = backtest_table[backtest_table["row_type"] == "creator_exit_validation"].copy()
        creator_note = None
        if not creator.empty:
            creator_row = creator.iloc[0]
            n_observed = safe_int(creator_row.get("n_subreddits"))
            n_expected = safe_int(creator_row.get("expected_n_subreddits")) or len(TREATMENT_SUBS)
            rows.append({
                "label": "Creator validation coverage",
                "value": f"{n_observed or 'N/A'} of {n_expected}",
                "note": "High-substitutability communities represented in the creator-exit holdout.",
            })
            missing = str(creator_row.get("missing_subreddits") or "").strip()
            if missing and missing.lower() != "nan":
                creator_note = f"Creator validation is missing these communities in the holdout: {missing}."

        if forward_path.exists():
            try:
                forward_table = pd.read_csv(forward_path)
                projected = forward_table[forward_table.get("row_type") == "projected_tercile_activity"]
                scenarios = safe_int(projected.get("scenario", pd.Series(dtype=str)).nunique())
                if scenarios:
                    rows.append({
                        "label": "Forward scenarios",
                        "value": f"{scenarios} bounded scenarios",
                        "note": "Flat, Step, Accelerating, Partial Regression, and Full Regression.",
                    })
            except Exception:
                pass

        if not rows:
            return None
        return {
            "rows": json_safe(rows),
            "creator_note": creator_note,
        }

    def output_size_label(path):
        try:
            size = path.stat().st_size
        except OSError:
            return "unknown size"
        units = ["B", "KB", "MB", "GB"]
        value = float(size)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{value:.1f} GB"

    def csv_preview(path, max_bytes=1_500_000, max_rows=5, max_cols=8):
        try:
            if path.stat().st_size > max_bytes:
                return None, "preview skipped for large file"
            frame = pd.read_csv(path, nrows=max_rows)
            if frame.empty:
                return None, "empty CSV"
            frame = frame.iloc[:, :max_cols].where(pd.notna(frame), "")
            preview = {
                "columns": [str(column) for column in frame.columns],
                "rows": [
                    [str(value)[:90] for value in row]
                    for row in frame.astype(str).to_numpy().tolist()
                ],
            }
            return preview, f"showing first {len(frame)} row(s)"
        except Exception as exc:
            return None, f"preview unavailable: {exc.__class__.__name__}"

    def generated_output_item(path):
        rel = path.relative_to(OUTPUT_DIR).as_posix()
        item = {
            "name": rel,
            "href": rel,
            "kind": path.suffix.lower().lstrip(".") or "file",
            "size_label": output_size_label(path),
            "preview": None,
            "preview_note": None,
        }
        if path.suffix.lower() == ".csv":
            item["preview"], item["preview_note"] = csv_preview(path)
        return item

    def build_generated_output_index():
        figure_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
        table_suffixes = {".csv", ".tex", ".pdf", ".parquet", ".json", ".html"}
        figure_files = []
        table_files = []
        if FIGURES_DIR.exists():
            figure_files = [
                path for path in sorted(FIGURES_DIR.rglob("*"))
                if path.is_file() and path.suffix.lower() in figure_suffixes
            ]
        if TABLES_DIR.exists():
            table_files.extend(
                path for path in sorted(TABLES_DIR.rglob("*"))
                if path.is_file() and path.suffix.lower() in table_suffixes
            )
        if FIGURES_DIR.exists():
            table_files.extend(
                path for path in sorted(FIGURES_DIR.rglob("*"))
                if path.is_file() and path.suffix.lower() in {".pdf"}
            )
        for root_file in sorted(OUTPUT_DIR.glob("*")):
            if root_file.is_file() and root_file.suffix.lower() in table_suffixes:
                table_files.append(root_file)
        return {
            "figures": json_safe([generated_output_item(path) for path in figure_files]),
            "tables": json_safe([generated_output_item(path) for path in table_files]),
        }

    try:
        output_dir_label = str(OUTPUT_DIR.relative_to(ROOT))
    except ValueError:
        output_dir_label = str(OUTPUT_DIR)

    def render_dashboard_html():
        return render_template_string(
            DASHBOARD_HTML,
            start=START_DATE.strftime("%b %Y"), end=END_DATE.strftime("%b %Y"),
            top_stats=top_stats,
            stats_note=stats_note,
            n_authors=n_authors,
            n_subs=n_subs, n_stable=f"{n_stable:,}" if n_stable is not None else None,
            show_creator_stats=show_creator_stats,
            show_stable_stats=show_stable_stats,
            acsi_main_sample=to_obj(main_model_sample_summary()),
            acsi_three_dimensional_rows=three_dimensional_dashboard_rows(),
            acsi_three_dimensional_influence=to_obj(results.get("acsi_three_dimensional_influence")),
            q2_survival=to_obj(results.get("q2_survival")),
            q2_survival_c=cls(res("q2_survival").get("coef"), res("q2_survival").get("pvalue"), False),
            q2_moderation=to_obj(results.get("q2_survival_moderation")),
            q3_engagement=to_obj(results.get("q3_engagement")),
            q3_engagement_c=cls(res("q3_engagement").get("coef"), res("q3_engagement").get("pvalue"), True),
            gse_main=to_obj(results.get("gse_main")),
            gse_main_c=cls(res("gse_main").get("coef"), res("gse_main").get("pvalue"), False),
            gse_adj=to_obj(results.get("gse_covariate_adj")),
            gse_adj_c=cls(res("gse_covariate_adj").get("coef"), res("gse_covariate_adj").get("pvalue"), False),
            acsi_component_correlations=results.get("acsi_component_correlations"),
            gse_perm=to_obj(results.get("gse_permutation")),
            gse_secondary=results.get("gse_secondary"),
            gse_quartiles=results.get("gse_quartiles"),
            gse_cv=to_obj(results.get("gse_construct_validity")),
            post_ai_adoption=to_obj(results.get("post_ai_adoption")),
            gse_event_study=to_obj(results.get("gse_event_study")),
            backtest_plain_summary=to_obj(load_backtest_plain_summary()),
            score_rows=load_score_rows(),
            generated_outputs=to_obj(build_generated_output_index()),
            output_dir=output_dir_label,
        )

    @app.route("/")
    def home():
        return render_dashboard_html()

    @app.route("/api/results")
    def api_results():
        return jsonify(json_safe(results))

    @app.route("/tables/<path:filename>")
    def table_file(filename):
        return send_from_directory(TABLES_DIR, filename)

    def write_dashboard_snapshot():
        with app.app_context():
            html = render_dashboard_html().replace('src="/figures/', 'src="figures/')
        path = OUTPUT_DIR / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        print(f"  Wrote static dashboard snapshot -> {path}")

    write_dashboard_snapshot()

    try:
        port = find_open_dashboard_port() if port is None else int(port)
    except PermissionError:
        print("  Dashboard server unavailable: localhost socket binding is blocked in this environment")
        return
    except RuntimeError as exc:
        print(f"  Dashboard server unavailable: {exc}")
        return

    url = f"http://localhost:{port}"
    print(f"  Dashboard ready -> {url}")
    print("  Press Ctrl+C to stop.")
    try:
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except OSError as exc:
        print(f"  Dashboard server unavailable: {exc}")


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
        help="Deprecated no-op kept for old commands; a static dashboard snapshot is always written.",
    )
    parser.add_argument(
        "--content-validation",
        action="store_true",
        help="Run the optional text content-validation sample after analysis.",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run only the saved-panel PersFree backtest and forward simulation outputs.",
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
    parser.add_argument(
        "--write-output-csvs",
        action="store_true",
        help="Compatibility flag; generated result CSVs are now written by default for the dashboard.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global ACTIVE_SUBREDDITS, MAX_LINES_PER_FILE, QUICK_MODE, N_RANDOMIZATION_PERMS, WRITE_OUTPUT_CSVS

    args = parse_args(argv)
    WRITE_OUTPUT_CSVS = True
    if args.no_dashboard:
        print("\nNOTE: --no-dashboard is deprecated; writing the static dashboard snapshot is part of validation.")

    if args.output_dir is not None:
        configure_output_dir(args.output_dir)
    elif args.smoke:
        configure_output_dir(ROOT / "output" / "smoke")
    else:
        configure_output_dir(OUTPUT_ROOT / "latest")

    if args.backtest_only:
        if args.smoke:
            print("\nNOTE: --smoke is ignored in --backtest-only mode.")
        bind_check_modules()
        robustness_checks.run_backtest_only_from_outputs()
        return

    clean_generated_output(OUTPUT_DIR)

    if args.smoke:
        ACTIVE_SUBREDDITS = SMOKE_SUBREDDITS
        args.quick = True
        print("\nSMOKE MODE: using representative small subreddits and quick permutations.")
        print("  subreddits: " + ", ".join(iter_subreddits()))

    MAX_LINES_PER_FILE = args.max_lines_per_file
    if MAX_LINES_PER_FILE is not None:
        print(f"\nLINE LIMIT: reading at most {MAX_LINES_PER_FILE:,} raw JSONL lines per file.")

    QUICK_MODE = args.quick
    N_RANDOMIZATION_PERMS = 250 if QUICK_MODE else 1000

    if QUICK_MODE:
        print(f"\nQUICK MODE: using {N_RANDOMIZATION_PERMS} randomization/permutation draws.")

    raw_posts = parse_posts()
    df_ecosystem, df_all = clean_post_samples(raw_posts)
    del raw_posts
    gc.collect()
    df_ecosystem.to_parquet(POSTS_ECOSYSTEM_PATH, index=False)
    df_all.to_parquet(POSTS_CREATOR_PATH, index=False)

    score_path = resolve_acsi_score_path()
    if score_path.exists():
        print(f"\n=== Step 3a: load existing {INDEX_SHORT} scores ===")
        print(f"  using {score_path}")
        acsi_scores = load_acsi_scores()
    else:
        acsi_scores = recompute_acsi_scores_from_run1()
    submonth_panel = build_subreddit_month_panel(df_ecosystem, acsi_scores)
    submonth_panel.to_parquet(SUBMONTH_PANEL_PATH, index=False)
    write_submonth_panel_cache_metadata(apply_author_cap=True)

    creators = build_creators_v3(df_all)
    creators.to_parquet(CREATORS_PATH, index=False)

    bind_check_modules()
    pipeline_tests.print_subreddit_coverage_diagnostics(df_ecosystem, acsi_scores, creator_posts=df_all)

    results = run_analysis(
        df_all=df_all,
        creators=creators,
        acsi_scores=acsi_scores,
        submonth_panel=submonth_panel,
    )
    emit_output_table(acsi_scores, TABLES_DIR / "acsi_scores_computed.csv", index=False)

    if args.content_validation:
        top_subs = sorted(
            [s for s in iter_subreddits() if s in TREATMENT_SUBS and raw_post_path(s).exists()],
            key=lambda s: MU_K.get(s, 0), reverse=True,
        )[:5]
        cv_results = pipeline_tests.compute_content_validation_sample(top_subs)
        results["content_validation"] = cv_results

    pipeline_tests.validate_run_outputs(
        submonth_panel=submonth_panel,
        score_table=acsi_scores,
        analysis_results=results,
        author_cap_enabled=True,
        include_content_validation=args.content_validation,
        expected_panel_subreddits_override=panel_subreddits_with_posts(submonth_panel),
    )
    pipeline_tests.run_pytest_suite()
    print("\nProcessing complete.")
    launch_dashboard(submonth_panel, creators, results, n_ecosystem_authors=df_ecosystem["author"].nunique())

if __name__ == "__main__":
    main()
