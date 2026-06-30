from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


TASK_DIR_RE = re.compile(r"^(\d+)_(.+)$")
GAINLORA_TASK_DIR_RE = re.compile(r"^(\d+)-(.+)$")
SUPERNI_TASK_ID_RE = re.compile(r"^task(\d+)_")
CLASSIFICATION_TASK_IDS = {363, 875, 1687}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute continual-learning transfer metrics from GainLoRA logs_and_outputs or "
            "runs/residual_proj-style outputs. "
            "FWT is reported only when the required pre-learning future-task scores exist."
        )
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment directories to inspect. If omitted, uses all subdirectories under --root.",
    )
    parser.add_argument(
        "--root",
        default="logs_and_outputs",
        help=(
            "Root directory that contains experiment subdirectories. "
            "For this repo's generated training scripts, this is usually logs_and_outputs."
        ),
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "gainlora", "residual_proj"],
        help="Output layout to parse. 'auto' detects logs_and_outputs and residual_proj layouts.",
    )
    parser.add_argument(
        "--metric",
        default="auto",
        choices=["auto", "eval_rougeL", "rouge1", "exact_match"],
        help=(
            "Scalar metric to use. 'auto' uses eval_rougeL for SuperNI-style task names "
            "and exact_match for Long_Sequence-style task names."
        ),
    )
    parser.add_argument(
        "--metric-mode",
        default="single",
        choices=["mixed", "single"],
        help=(
            "Metric selection mode. "
            "'mixed' uses exact_match for classification tasks and --metric for the rest. "
            "'single' uses --metric for every task."
        ),
    )
    parser.add_argument(
        "--write-json",
        help="Optional path to save the computed summary as JSON.",
    )
    parser.add_argument(
        "--baseline-root",
        default=None,
        help=(
            "Directory that stores individually trained task baselines. For GainLoRA logs, "
            "pass logs_and_outputs/<single_run> or logs_and_outputs/<single_run>/outputs. "
            "Used for SAPT-style FWT: mean_t(a_{t,t} - a_{0,t})."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON summary instead of the compact text report.",
    )
    return parser.parse_args()


def _list_experiments(root: Path, experiments: list[str]) -> list[Path]:
    if experiments:
        paths = []
        for exp in experiments:
            path = Path(exp).expanduser()
            if not path.is_absolute() and not path.exists() and (root / path).exists():
                path = root / path
            paths.append(path.resolve())
        return paths

    if not root.exists():
        return []
    if _looks_like_experiment(root):
        return [root.resolve()]
    return sorted(path.resolve() for path in root.iterdir() if path.is_dir())


def _looks_like_experiment(path: Path) -> bool:
    return _gainlora_outputs_dir(path) is not None or (
        (path / "mixing_gate").is_dir() and bool(_task_dirs(path))
    )


def _task_dirs(exp_dir: Path) -> list[Path]:
    dirs = []
    for path in exp_dir.iterdir():
        if not path.is_dir():
            continue
        if TASK_DIR_RE.match(path.name):
            dirs.append(path)
    return sorted(dirs, key=lambda path: int(path.name.split("_", 1)[0]))


def _gainlora_outputs_dir(exp_dir: Path) -> Path | None:
    for candidate in (exp_dir / "outputs", exp_dir):
        if not candidate.is_dir():
            continue
        if (candidate / "task_order.txt").is_file() or _gainlora_task_dirs(candidate):
            return candidate
    return None


def _gainlora_task_dirs(outputs_dir: Path) -> list[Path]:
    dirs = []
    for path in outputs_dir.iterdir():
        if not path.is_dir():
            continue
        if GAINLORA_TASK_DIR_RE.match(path.name):
            dirs.append(path)
    return sorted(dirs, key=lambda path: int(path.name.split("-", 1)[0]))


def _task_id_from_name(task_name: str) -> int | None:
    match = SUPERNI_TASK_ID_RE.match(task_name)
    return int(match.group(1)) if match else None


def _is_superni_like(exp_dir: Path, task_names: list[str]) -> bool:
    return "superni" in str(exp_dir).lower() or any(_task_id_from_name(task_name) is not None for task_name in task_names)


def _default_metric_for_gainlora(exp_dir: Path, task_names: list[str]) -> str:
    return "eval_rougeL" if _is_superni_like(exp_dir, task_names) else "exact_match"


def _select_metric(
    task_id: int | None,
    task_name: str,
    metric: str,
    metric_mode: str,
    default_metric: str = "eval_rougeL",
) -> str:
    del task_name
    selected = default_metric if metric == "auto" else metric
    if metric_mode == "mixed" and task_id in CLASSIFICATION_TASK_IDS:
        return "exact_match"
    return selected


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_task_metric(
    data: dict[str, Any],
    task_name: str,
    metric_name: str,
    *,
    allow_overall: bool = False,
    prefix: str = "predict",
) -> tuple[float, str]:
    candidates = [
        f"{prefix}_{metric_name}_for_{task_name}",
        f"{metric_name}_for_{task_name}",
    ]
    if allow_overall:
        candidates.extend([f"{prefix}_{metric_name}", metric_name])

    for key in candidates:
        if key in data:
            return float(data[key]), key

    available = ", ".join(sorted(data.keys()))
    raise KeyError(
        f"Could not find metric for task '{task_name}' using candidates {candidates}. "
        f"Available keys: {available}"
    )


def _read_gainlora_task_order(outputs_dir: Path) -> list[str]:
    task_order_path = outputs_dir / "task_order.txt"
    if task_order_path.exists():
        text = task_order_path.read_text(encoding="utf-8").strip()
        if text:
            return [task.strip() for task in text.split(",") if task.strip()]
    return [path.name.split("-", 1)[1] for path in _gainlora_task_dirs(outputs_dir)]


def _gainlora_step_entries(outputs_dir: Path, task_order: list[str]) -> list[tuple[int, str, Path]]:
    by_step = {
        int(match.group(1)): (match.group(2), path)
        for path in _gainlora_task_dirs(outputs_dir)
        if (match := GAINLORA_TASK_DIR_RE.match(path.name))
    }

    entries = []
    if task_order:
        for step, task_name in enumerate(task_order, start=1):
            if step not in by_step:
                break
            dir_task_name, path = by_step[step]
            if dir_task_name != task_name:
                raise ValueError(
                    f"Task order mismatch at step {step}: task_order.txt has '{task_name}', "
                    f"but directory is '{dir_task_name}'"
                )
            entries.append((step, task_name, path))
    else:
        entries = [(step, task_name, path) for step, (task_name, path) in sorted(by_step.items())]

    return entries


def _load_gainlora_scores(
    result_path: Path,
    task_names: list[str],
    metric: str,
    metric_mode: str,
    default_metric: str,
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    data = _read_json(result_path)
    scores = {}
    metric_keys = {}
    metric_names = {}
    allow_overall = len(task_names) == 1

    for task_name in task_names:
        task_id = _task_id_from_name(task_name)
        metric_name = _select_metric(task_id, task_name, metric, metric_mode, default_metric)
        score, key = _extract_task_metric(data, task_name, metric_name, allow_overall=allow_overall)
        scores[task_name] = score
        metric_keys[task_name] = key
        metric_names[task_name] = metric_name

    return scores, metric_keys, metric_names


def _load_gainlora_individual_baseline(
    baseline_root: Path,
    task_name: str,
    metric: str,
    metric_mode: str,
    default_metric: str,
) -> tuple[float | None, str | None]:
    outputs_dir = _gainlora_outputs_dir(baseline_root)
    if outputs_dir is None:
        return None, None

    task_order = _read_gainlora_task_order(outputs_dir)
    entries = _gainlora_step_entries(outputs_dir, task_order)
    for _, baseline_task_name, task_dir in entries:
        if baseline_task_name != task_name:
            continue
        result_path = task_dir / "all_results.json"
        if not result_path.exists():
            return None, str(result_path)
        scores, _, _ = _load_gainlora_scores(
            result_path, [task_name], metric, metric_mode, default_metric
        )
        return scores[task_name], str(result_path)

    return None, None


def _metric_key(task_id: int, metric: str, metric_mode: str) -> str:
    if metric_mode == "mixed" and task_id in CLASSIFICATION_TASK_IDS:
        return "exact_match"
    return "eval_rougeL" if metric == "auto" else metric


def _mixing_gate_dirs(exp_dir: Path) -> list[Path]:
    mixing_root = exp_dir / "mixing_gate"
    if not mixing_root.is_dir():
        return []
    return sorted(
        [path for path in mixing_root.iterdir() if path.is_dir()],
        key=lambda path: (len(path.name.split("_")), path.name),
    )


def _load_score_from_task_dir(task_dir: Path, task_id: int, metric: str, metric_mode: str) -> float:
    for filename in ("rouge.json", "metric.json"):
        path = task_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            return float(data[_metric_key(task_id, metric, metric_mode)])
    raise FileNotFoundError(f"No rouge/metric json found under {task_dir}")


def _load_scores(path: Path, metric: str, metric_mode: str) -> dict[int, float]:
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    return {
        int(task_id): float(scores[_metric_key(int(task_id), metric, metric_mode)])
        for task_id, scores in raw.items()
    }


def _load_individual_baseline(
    baseline_root: Path, task_id: int, task_name: str, metric: str, metric_mode: str
) -> tuple[float | None, str | None]:
    task_dir = baseline_root / task_name
    if not task_dir.is_dir():
        return None, None

    for filename in ("metric.json", "rouge.json"):
        path = task_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            key = _metric_key(task_id, metric, metric_mode)
            if key not in data:
                return None, str(path)
            return float(data[key]), str(path)

    return None, None


def _validate_sequence(task_ids: list[int], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        step = row["step"]
        expected_prefix = task_ids[:step]
        if row["task_ids"] != expected_prefix:
            raise ValueError(
                f"Sequence mismatch at step {step}: expected {expected_prefix}, got {row['task_ids']}"
            )


def _compute_fwt(rows: list[dict[str, Any]], task_ids: list[int], task_names: list[str]) -> tuple[Any, str, list[dict[str, Any]]]:
    missing = []
    for row, next_task_id, next_task_name in zip(rows[:-1], task_ids[1:], task_names[1:]):
        score = row["scores"].get(str(next_task_id))
        if score is None:
            missing.append(
                {
                    "after_step": row["step"],
                    "trained_prefix": row["task_ids"],
                    "missing_future_task_id": next_task_id,
                    "missing_future_task_name": next_task_name,
                }
            )

    if missing:
        return (
            None,
            "Stored outputs only contain seen-task evaluations at each step, so the required pre-learning future-task scores are absent.",
            missing,
        )

    return (
        None,
        "Future-task scores exist, but a baseline term is still required for the standard FWT definition.",
        [],
    )


def _compute_forgetting_rate(
    rows: list[dict[str, Any]], task_ids: list[int], task_names: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    terms = []
    for idx, (task_id, task_name) in enumerate(zip(task_ids[:-1], task_names[:-1])):
        history_before_final = [float(row["scores"][str(task_id)]) for row in rows[idx:-1]]
        final_score = float(rows[-1]["scores"][str(task_id)])
        best_before_final = max(history_before_final)
        terms.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "best_score_before_final": best_before_final,
                "final_score": final_score,
                "delta": best_before_final - final_score,
            }
        )

    forgetting_rate = mean(term["delta"] for term in terms) if terms else 0.0
    return forgetting_rate, terms


def _compute_gainlora_fwt(
    rows: list[dict[str, Any]], task_names: list[str]
) -> tuple[Any, str, list[dict[str, Any]]]:
    missing = []
    for row, next_task_name in zip(rows[:-1], task_names[1:]):
        score = row["scores"].get(next_task_name)
        if score is None:
            missing.append(
                {
                    "after_step": row["step"],
                    "trained_prefix": row["task_names"],
                    "missing_future_task_name": next_task_name,
                }
            )

    if missing:
        return (
            None,
            "Stored outputs only contain seen-task evaluations at each step, so the required pre-learning future-task scores are absent.",
            missing,
        )

    return (
        None,
        "Future-task scores exist, but a baseline term is still required for the standard FWT definition.",
        [],
    )


def _compute_gainlora_forgetting_rate(
    rows: list[dict[str, Any]], task_names: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    terms = []
    for idx, task_name in enumerate(task_names[:-1]):
        history_before_final = [float(row["scores"][task_name]) for row in rows[idx:-1]]
        final_score = float(rows[-1]["scores"][task_name])
        best_before_final = max(history_before_final)
        terms.append(
            {
                "task_id": _task_id_from_name(task_name),
                "task_name": task_name,
                "best_score_before_final": best_before_final,
                "final_score": final_score,
                "delta": best_before_final - final_score,
            }
        )

    forgetting_rate = mean(term["delta"] for term in terms) if terms else 0.0
    return forgetting_rate, terms


def _summarize_gainlora_experiment(
    exp_dir: Path,
    metric: str,
    metric_mode: str,
    baseline_root: Path | None,
) -> dict[str, Any]:
    outputs_dir = _gainlora_outputs_dir(exp_dir)
    if outputs_dir is None:
        raise ValueError(f"No GainLoRA outputs directory found under {exp_dir}")

    declared_task_order = _read_gainlora_task_order(outputs_dir)
    entries = _gainlora_step_entries(outputs_dir, declared_task_order)
    if not entries:
        raise ValueError(f"No '<step>-<task>' directories found under {outputs_dir}")

    task_names = [task_name for _, task_name, _ in entries]
    task_ids = [_task_id_from_name(task_name) for task_name in task_names]
    default_metric = _default_metric_for_gainlora(exp_dir, task_names)
    rows: list[dict[str, Any]] = []
    metric_by_task: dict[str, str] = {}

    for ordinal, (step, trained_task_name, task_dir) in enumerate(entries, start=1):
        result_path = task_dir / "all_results.json"
        if not result_path.exists():
            raise ValueError(f"No all_results.json found under {task_dir}")

        seen_task_names = task_names[:ordinal]
        scores, metric_keys, metric_names = _load_gainlora_scores(
            result_path, seen_task_names, metric, metric_mode, default_metric
        )
        metric_by_task.update(metric_names)
        rows.append(
            {
                "step": step,
                "trained_task_id": _task_id_from_name(trained_task_name),
                "trained_task_name": trained_task_name,
                "task_ids": [_task_id_from_name(task_name) for task_name in seen_task_names],
                "task_names": seen_task_names,
                "scores": scores,
                "metric_keys": metric_keys,
                "source": _relative_to(result_path, exp_dir),
            }
        )

    diagonal = [float(row["scores"][row["trained_task_name"]]) for row in rows]
    final_scores = rows[-1]["scores"]
    acc = mean(float(final_scores[task_name]) for task_name in task_names)

    bwt_terms = []
    for task_name, learned_score in zip(task_names[:-1], diagonal[:-1]):
        final_score = float(final_scores[task_name])
        bwt_terms.append(
            {
                "task_id": _task_id_from_name(task_name),
                "task_name": task_name,
                "final_score": final_score,
                "score_when_learned": learned_score,
                "delta": final_score - learned_score,
            }
        )

    bwt = mean(term["delta"] for term in bwt_terms) if bwt_terms else 0.0
    fwt, fwt_message, fwt_missing = _compute_gainlora_fwt(rows, task_names)
    forgetting_rate, forgetting_rate_terms = _compute_gainlora_forgetting_rate(rows, task_names)

    sapt_fwt_terms = []
    sapt_missing = []
    if baseline_root is None:
        sapt_missing = [{"task_id": task_id, "task_name": task_name} for task_id, task_name in zip(task_ids, task_names)]
    else:
        for task_id, task_name in zip(task_ids, task_names):
            baseline_score, baseline_source = _load_gainlora_individual_baseline(
                baseline_root, task_name, metric, metric_mode, default_metric
            )
            if baseline_score is None:
                sapt_missing.append({"task_id": task_id, "task_name": task_name})
                continue
            learned_score = float(rows[task_names.index(task_name)]["scores"][task_name])
            sapt_fwt_terms.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "a_tt": learned_score,
                    "a_0t": baseline_score,
                    "delta": learned_score - baseline_score,
                    "baseline_source": baseline_source,
                }
            )
    sapt_fwt = mean(term["delta"] for term in sapt_fwt_terms) if len(sapt_fwt_terms) == len(task_names) else None

    return {
        "experiment": exp_dir.name if exp_dir.name != "outputs" else exp_dir.parent.name,
        "path": str(exp_dir),
        "format": "gainlora",
        "metric": metric,
        "metric_mode": metric_mode,
        "resolved_default_metric": default_metric,
        "metric_by_task": metric_by_task,
        "task_order": [
            {"task_id": task_id, "task_name": task_name}
            for task_id, task_name in zip(task_ids, task_names)
        ],
        "acc": acc,
        "bwt": bwt,
        "forgetting_rate": forgetting_rate,
        "fwt": fwt,
        "fwt_message": fwt_message,
        "fwt_missing": fwt_missing,
        "sapt_fwt": sapt_fwt,
        "sapt_fwt_message": (
            "Computed as mean_t(a_{t,t} - a_{0,t}) using individually trained task baselines."
            if sapt_fwt is not None
            else (
                "Could not compute SAPT-style FWT because --baseline-root was not supplied."
                if baseline_root is None
                else "Could not compute SAPT-style FWT because one or more individual baselines are missing."
            )
        ),
        "sapt_fwt_missing": sapt_missing,
        "diagonal": diagonal,
        "final_scores": [float(final_scores[task_name]) for task_name in task_names],
        "rows": rows,
        "bwt_terms": bwt_terms,
        "forgetting_rate_terms": forgetting_rate_terms,
        "sapt_fwt_terms": sapt_fwt_terms,
    }


def _summarize_residual_proj_experiment(exp_dir: Path, metric: str, metric_mode: str, baseline_root: Path | None) -> dict[str, Any]:
    task_dirs = _task_dirs(exp_dir)
    mixing_dirs = _mixing_gate_dirs(exp_dir)
    if not task_dirs:
        raise ValueError(f"No numbered task directories found under {exp_dir}")
    if not mixing_dirs:
        raise ValueError(f"No mixing_gate directories found under {exp_dir}")

    final_task_ids = [int(token) for token in mixing_dirs[-1].name.split("_")]
    task_names = [task_dir.name.split("_", 1)[1] for task_dir in task_dirs]
    if len(final_task_ids) != len(task_names):
        raise ValueError(
            f"Task count mismatch under {exp_dir}: "
            f"{len(final_task_ids)} ids in final mixing dir vs {len(task_names)} task dirs"
        )

    rows: list[dict[str, Any]] = []

    first_scores = {
        str(final_task_ids[0]): _load_score_from_task_dir(task_dirs[0], final_task_ids[0], metric, metric_mode)
    }
    rows.append(
        {
            "step": 1,
            "trained_task_id": final_task_ids[0],
            "trained_task_name": task_names[0],
            "task_ids": final_task_ids[:1],
            "scores": first_scores,
            "source": str((task_dirs[0] / ("rouge.json" if (task_dirs[0] / "rouge.json").exists() else "metric.json")).relative_to(exp_dir)),
        }
    )

    for step, mixing_dir in enumerate(mixing_dirs, start=2):
        task_ids = [int(token) for token in mixing_dir.name.split("_")]
        scores = _load_scores(mixing_dir / "metrics.json", metric, metric_mode)
        rows.append(
            {
                "step": step,
                "trained_task_id": task_ids[-1],
                "trained_task_name": task_names[step - 1],
                "task_ids": task_ids,
                "scores": {str(task_id): score for task_id, score in scores.items()},
                "source": str((mixing_dir / "metrics.json").relative_to(exp_dir)),
            }
        )

    _validate_sequence(final_task_ids, rows)

    diagonal = []
    for row in rows:
        trained_task_id = row["trained_task_id"]
        diagonal.append(float(row["scores"][str(trained_task_id)]))

    final_scores = rows[-1]["scores"]
    acc = mean(float(final_scores[str(task_id)]) for task_id in final_task_ids)

    bwt_terms = []
    for task_id, task_name, learned_score in zip(final_task_ids[:-1], task_names[:-1], diagonal[:-1]):
        final_score = float(final_scores[str(task_id)])
        bwt_terms.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "final_score": final_score,
                "score_when_learned": learned_score,
                "delta": final_score - learned_score,
            }
        )

    bwt = mean(term["delta"] for term in bwt_terms) if bwt_terms else 0.0
    fwt, fwt_message, fwt_missing = _compute_fwt(rows, final_task_ids, task_names)
    forgetting_rate, forgetting_rate_terms = _compute_forgetting_rate(rows, final_task_ids, task_names)

    sapt_fwt_terms = []
    sapt_missing = []
    for row, task_id, task_name in zip(rows, final_task_ids, task_names):
        baseline_score, baseline_source = (
            _load_individual_baseline(baseline_root, task_id, task_name, metric, metric_mode)
            if baseline_root is not None
            else (None, None)
        )
        if baseline_score is None:
            sapt_missing.append({"task_id": task_id, "task_name": task_name})
            continue
        learned_score = float(row["scores"][str(task_id)])
        sapt_fwt_terms.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "a_tt": learned_score,
                "a_0t": baseline_score,
                "delta": learned_score - baseline_score,
                "baseline_source": baseline_source,
            }
        )
    sapt_fwt = mean(term["delta"] for term in sapt_fwt_terms) if len(sapt_fwt_terms) == len(final_task_ids) else None

    return {
        "experiment": exp_dir.name,
        "path": str(exp_dir),
        "format": "residual_proj",
        "metric": metric,
        "metric_mode": metric_mode,
        "task_order": [
            {"task_id": task_id, "task_name": task_name}
            for task_id, task_name in zip(final_task_ids, task_names)
        ],
        "acc": acc,
        "bwt": bwt,
        "forgetting_rate": forgetting_rate,
        "fwt": fwt,
        "fwt_message": fwt_message,
        "fwt_missing": fwt_missing,
        "sapt_fwt": sapt_fwt,
        "sapt_fwt_message": (
            "Computed as mean_t(a_{t,t} - a_{0,t}) using individually trained task baselines."
            if sapt_fwt is not None
            else (
                "Could not compute SAPT-style FWT because --baseline-root was not supplied."
                if baseline_root is None
                else "Could not compute SAPT-style FWT because one or more individual baselines are missing."
            )
        ),
        "sapt_fwt_missing": sapt_missing,
        "diagonal": diagonal,
        "final_scores": [float(final_scores[str(task_id)]) for task_id in final_task_ids],
        "rows": rows,
        "bwt_terms": bwt_terms,
        "forgetting_rate_terms": forgetting_rate_terms,
        "sapt_fwt_terms": sapt_fwt_terms,
    }


def _summarize_experiment(
    exp_dir: Path,
    metric: str,
    metric_mode: str,
    baseline_root: Path | None,
    output_format: str,
) -> dict[str, Any]:
    if output_format in {"auto", "gainlora"} and _gainlora_outputs_dir(exp_dir) is not None:
        return _summarize_gainlora_experiment(exp_dir, metric, metric_mode, baseline_root)
    if output_format in {"auto", "residual_proj"}:
        return _summarize_residual_proj_experiment(exp_dir, metric, metric_mode, baseline_root)
    raise ValueError(f"Could not detect a supported experiment layout under {exp_dir}")


def _compact_report(summary: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in summary:
        lines.append(item["experiment"])
        metric = item["metric"]
        if metric == "auto":
            metric = f"auto -> {item.get('resolved_default_metric', 'eval_rougeL')}"
        lines.append(f"  format: {item.get('format', 'unknown')}")
        lines.append(f"  metric: {metric} ({item['metric_mode']})")
        lines.append(f"  ACC: {item['acc']:.4f}")
        lines.append(f"  BWT: {item['bwt']:.4f}")
        lines.append(f"  Forgetting_Rate: {item['forgetting_rate']:.4f}")
        if item["sapt_fwt"] is None:
            lines.append("  SAPT_FWT: N/A")
            lines.append(f"  reason: {item['sapt_fwt_message']}")
        else:
            lines.append(f"  SAPT_FWT: {item['sapt_fwt']:.4f}")
        if item["fwt"] is None:
            lines.append("  Standard_FWT: N/A")
            lines.append(f"  reason: {item['fwt_message']}")
        else:
            lines.append(f"  Standard_FWT: {item['fwt']:.4f}")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    args = _parse_args()
    root = Path(args.root).resolve()
    baseline_root = Path(args.baseline_root).resolve() if args.baseline_root else None
    experiments = _list_experiments(root, args.experiments)

    summary = []
    for exp_dir in experiments:
        try:
            summary.append(
                _summarize_experiment(
                    exp_dir, args.metric, args.metric_mode, baseline_root, args.format
                )
            )
        except ValueError:
            if args.experiments:
                raise
            continue

    if args.write_json:
        out_path = Path(args.write_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(_compact_report(summary))


if __name__ == "__main__":
    main()
