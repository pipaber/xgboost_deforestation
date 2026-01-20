"""
Plot/export one or more XGBoost trees from a trained model.

Outputs:
- reports/trees/tree_<index>.png  (if graphviz is available)
- reports/trees/tree_<index>.txt  (always)
- reports/trees/tree_<index>_PNG_FAILED.txt (if PNG render fails; includes actionable diagnostics)

Usage examples:
  # Single tree (compatible with older usage)
  uv run python src/deforestation/plot_tree.py --bundle models/xgb_timecv_v1_gpu/bundle.joblib --tree 0 --out reports/trees

  # Explicit list of trees
  uv run python src/deforestation/plot_tree.py --bundle models/xgb_timecv_v1_gpu/bundle.joblib --trees 0,50,100 --out reports/trees

  # A handful: first, middle, last
  uv run python src/deforestation/plot_tree.py --bundle models/xgb_timecv_v1_gpu/bundle.joblib --handful --out reports/trees

  # Deterministic random sample
  uv run python src/deforestation/plot_tree.py --bundle models/xgb_timecv_v1_gpu/bundle.joblib --n-sample 10 --seed 42 --out reports/trees

Notes:
- XGBoost models are ensembles with many trees. Plotting all trees is not useful.
- This script exports selected trees (by explicit list, a standard handful, or a deterministic random sample).
- PNG rendering requires Graphviz installed and the `dot` executable available on PATH.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Set

import joblib


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _graphviz_dot_available() -> tuple[bool, str]:
    """
    Returns (ok, details). `ok` is True if `dot` is available on PATH.
    """
    dot = shutil.which("dot")
    if dot:
        return True, f"dot found at: {dot}"
    return False, "dot not found on PATH"


def _print_debug(title: str, msg: str, enabled: bool) -> None:
    if not enabled:
        return
    print(f"[DEBUG] {title}: {msg}")


def _parse_tree_list(arg: str) -> List[int]:
    """
    Parse a comma-separated list of integers like: "0,5,10".
    """
    raw = (arg or "").strip()
    if raw == "":
        return []
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p == "":
            continue
        out.append(int(p))
    return out


def _resolve_tree_indices(
    num_trees: int,
    tree_arg: Optional[int],
    trees_arg: str,
    handful: bool,
    n_sample: int,
    seed: int,
) -> List[int]:
    """
    Determine which tree indices to export.
    Priority:
      1) --trees (explicit list)
      2) --handful (0, mid, last)
      3) --n-sample (deterministic random sample)
      4) fallback: --tree (single index)
    """
    selected: List[int] = []
    if trees_arg.strip():
        selected = _parse_tree_list(trees_arg)
    elif handful:
        if num_trees <= 0:
            selected = []
        elif num_trees == 1:
            selected = [0]
        else:
            mid = num_trees // 2
            selected = sorted(set([0, mid, num_trees - 1]))
    elif n_sample > 0:
        rng = random.Random(int(seed))
        k = min(int(n_sample), num_trees)
        selected = rng.sample(list(range(num_trees)), k=k)
        selected = sorted(set(selected))
    else:
        selected = [int(tree_arg or 0)]

    # Validate and normalize
    bad = [i for i in selected if i < 0 or i >= num_trees]
    if bad:
        raise ValueError(
            f"Tree indices out of range for model with {num_trees} trees: {bad}"
        )
    return selected


def _write_tree_txt(dump: List[str], out_dir: Path, tree_idx: int) -> Path:
    txt_path = out_dir / f"tree_{tree_idx}.txt"
    txt_path.write_text(dump[tree_idx], encoding="utf-8")
    return txt_path


def _try_write_tree_png(
    booster: Any,
    out_dir: Path,
    tree_idx: int,
    fmt: str,
    verbose: bool,
) -> None:
    ok_dot, dot_details = _graphviz_dot_available()
    _print_debug("dot_check", dot_details, verbose)
    if not ok_dot:
        err_path = out_dir / f"tree_{tree_idx}_PNG_FAILED.txt"
        err_path.write_text(
            "Graphviz PNG render skipped because `dot` was not found on PATH.\n"
            f"{dot_details}\n\n"
            "Install Graphviz and ensure `dot` is on PATH.\n"
            "Ubuntu/Debian: sudo apt-get install graphviz\n"
            "Conda: conda install -c conda-forge graphviz python-graphviz\n",
            encoding="utf-8",
        )
        print(f"[WARN] Could not render PNG for tree {tree_idx}. Wrote {err_path}")
        if verbose:
            print(f"[DEBUG] PNG failure details saved in: {err_path}")
        return
    try:
        import xgboost as xgb

        _print_debug("xgboost_version", getattr(xgb, "__version__", "unknown"), verbose)
        _print_debug("to_graphviz", f"tree_idx={tree_idx}", verbose)
        g = xgb.to_graphviz(booster, tree_idx=tree_idx)

        png_path = out_dir / f"tree_{tree_idx}.png"
        render_base = str(png_path.with_suffix(""))
        _print_debug(
            "graphviz_render",
            f"filename={render_base} format={fmt} cleanup=True",
            verbose,
        )
        _ = g.render(filename=render_base, format=fmt, cleanup=True)
        print(f"[OK] wrote {png_path}")
    except Exception:
        err_path = out_dir / f"tree_{tree_idx}_PNG_FAILED.txt"
        tb = traceback.format_exc()
        ok_dot2, dot_details2 = _graphviz_dot_available()
        err_path.write_text(
            "Failed to render tree to PNG via graphviz.\n\n"
            f"Python: {sys.version}\n"
            f"Executable: {sys.executable}\n"
            f"dot: {dot_details2}\n"
            f"PATH: {os.environ.get('PATH', '')}\n\n"
            "Traceback:\n"
            f"{tb}",
            encoding="utf-8",
        )
        print(f"[WARN] Could not render PNG for tree {tree_idx}. Wrote {err_path}")
        if verbose:
            print("[DEBUG] Render exception traceback follows:")
            print(tb)
            print(f"[DEBUG] PNG failure details saved in: {err_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to bundle.joblib")
    # Backwards-compatible single-tree flag
    ap.add_argument(
        "--tree", type=int, default=0, help="Single tree index to export (fallback)."
    )
    # New: multiple tree indices
    ap.add_argument(
        "--trees", default="", help="Comma-separated tree indices, e.g. 0,50,100"
    )
    ap.add_argument(
        "--handful",
        action="store_true",
        help="Export a standard handful: tree 0, mid tree, and last tree.",
    )
    ap.add_argument(
        "--n-sample",
        type=int,
        default=0,
        help="Export a deterministic random sample of N trees (requires --seed).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic sampling when using --n-sample.",
    )
    ap.add_argument(
        "--out", required=True, help="Output directory (e.g., reports/trees)"
    )
    ap.add_argument(
        "--format",
        default="png",
        choices=["png"],
        help="Output image format (currently only png).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debug info to help diagnose graphviz/PNG failures.",
    )
    args = ap.parse_args()

    verbose = bool(args.verbose) or (
        os.environ.get("PLOT_TREE_VERBOSE", "").strip()
        not in ("", "0", "false", "False")
    )

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    _print_debug("bundle", args.bundle, verbose)
    _print_debug("tree_idx", str(args.tree), verbose)
    _print_debug("trees", str(args.trees), verbose)
    _print_debug("handful", str(bool(args.handful)), verbose)
    _print_debug("n_sample", str(int(args.n_sample)), verbose)
    _print_debug("seed", str(int(args.seed)), verbose)
    _print_debug("out_dir", str(out_dir), verbose)
    _print_debug("python", sys.version.replace("\n", " "), verbose)
    _print_debug("executable", sys.executable, verbose)

    bundle = joblib.load(args.bundle)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise KeyError(
            "Bundle does not look like the expected dict with key 'model'. "
            f"Got type={type(bundle)!r}, keys={list(bundle.keys()) if isinstance(bundle, dict) else 'N/A'}"
        )

    model = bundle["model"]

    if not hasattr(model, "get_booster"):
        raise TypeError(
            "Loaded bundle['model'] does not have get_booster(). "
            "Expected an XGBoost sklearn estimator (e.g., xgboost.XGBClassifier/XGBRegressor). "
            f"Got type={type(model)!r}"
        )

    _print_debug("model_type", repr(type(model)), verbose)

    booster = model.get_booster()

    # Always: dump as text
    dump = booster.get_dump(with_stats=True)
    _print_debug("num_trees_in_model", str(len(dump)), verbose)

    tree_indices = _resolve_tree_indices(
        num_trees=len(dump),
        tree_arg=args.tree,
        trees_arg=str(args.trees),
        handful=bool(args.handful),
        n_sample=int(args.n_sample),
        seed=int(args.seed),
    )
    print(f"[INFO] exporting trees: {tree_indices}")

    for tree_idx in tree_indices:
        # 1) Text dump (always)
        txt_path = _write_tree_txt(dump, out_dir, tree_idx)
        print(f"[OK] wrote {txt_path}")

        # 2) PNG (best effort; per-tree failure files)
        _try_write_tree_png(
            booster=booster,
            out_dir=out_dir,
            tree_idx=tree_idx,
            fmt=str(args.format),
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
