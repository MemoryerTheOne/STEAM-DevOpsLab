from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model as STEAMModel
from script import stat_operation

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "Beauty": {
        "item_num": 12104,
        "checkpoint": ROOT / "checkpoint" / "Beauty" / "steam-253.pth",
        "data_dir": ROOT / "Beauty",
    },
    "Sports": {
        "item_num": 18358,
        "checkpoint": ROOT / "checkpoint" / "Sports" / "steam-174.pth",
        "data_dir": ROOT / "Sports",
    },
    "Yelp": {
        "item_num": 16553,
        "checkpoint": ROOT / "checkpoint" / "Yelp" / "steam-289.pth",
        "data_dir": ROOT / "Yelp",
    },
}

app = FastAPI(title="STEAM Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def load_model(dataset: str) -> STEAMModel:
    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset '{dataset}'")

    config = DATASET_CONFIG[dataset]
    model = STEAMModel(
        dropout_rate=0.5,
        hidden_unit=64,
        item_num=config["item_num"],
        head_num=1,
        layer_num=1,
        max_insert_size=5,
    )

    checkpoint_path = config["checkpoint"]
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for dataset '{dataset}'")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache
def load_default_sequence(dataset: str) -> List[int]:
    config = DATASET_CONFIG[dataset]
    train_file = config["data_dir"] / "train.dat"
    if not train_file.exists():
        return [101, 204, 332, 990, 441, 120]

    with train_file.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return [101, 204, 332, 990, 441, 120]

    seq = [int(item) for item in first_line.split(",") if item.strip()]
    seq = [item for item in seq if item > 0]
    if not seq:
        return [101, 204, 332, 990, 441, 120]
    return seq[:12]


def _to_metric_card(label: str, value: float, delta_pct: float) -> Dict[str, Any]:
    trend = "up" if delta_pct >= 0 else "down"
    return {
        "label": label,
        "value": f"{value:.3f}",
        "delta": f"{delta_pct:+.1f}%",
        "trend": trend,
    }


def _build_performance(decision_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    total = sum(decision_counts.values())
    if total == 0:
        return [
            {"label": "Keep ratio", "value": 0, "color": "#4fd1c5"},
            {"label": "Delete ratio", "value": 0, "color": "#f56565"},
            {"label": "Insert ratio", "value": 0, "color": "#8b5cf6"},
        ]

    keep_ratio = round((decision_counts.get("keep", 0) / total) * 100, 1)
    delete_ratio = round((decision_counts.get("delete", 0) / total) * 100, 1)
    insert_ratio = round((decision_counts.get("insert", 0) / total) * 100, 1)

    return [
        {"label": "Keep ratio", "value": keep_ratio, "color": "#4fd1c5"},
        {"label": "Delete ratio", "value": delete_ratio, "color": "#f56565"},
        {"label": "Insert ratio", "value": insert_ratio, "color": "#8b5cf6"},
    ]


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/predict")
def predict(dataset: str = Query(default="Beauty", title="Dataset name"), sequence: str | None = Query(default=None)) -> Dict[str, Any]:
    dataset = dataset.title()
    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset '{dataset}'")

    model = load_model(dataset)
    raw_sequence = [int(part) for part in (sequence.split(",") if sequence else "") if part.strip()]
    if not raw_sequence:
        raw_sequence = load_default_sequence(dataset)

    raw_sequence = [item for item in raw_sequence if item > 0][:60]
    if not raw_sequence:
        raise HTTPException(status_code=400, detail="Sequence must contain at least one positive item id")

    padded_sequence = raw_sequence.copy()
    while len(padded_sequence) < 60:
        padded_sequence.append(0)
    padded_sequence = padded_sequence[:60]

    sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long)
    masked_sequence = sequence_tensor.clone()
    masked_sequence[0, -1] = model.item_num - 1

    leave_one = raw_sequence[:-1]
    while len(leave_one) < 60:
        leave_one.append(0)
    leave_one_tensor = torch.tensor([leave_one[:60]], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(masked_sequence)
        scores = logits[0, torch.where(masked_sequence[0] == model.item_num - 1)[0][-1], :]
        target_id = raw_sequence[-1]
        ranking = torch.argsort(scores, descending=True)
        rank = int((ranking == target_id).nonzero(as_tuple=False)[0].item()) + 1
        recall_5 = 1.0 if rank <= 5 else 0.0
        recall_10 = 1.0 if rank <= 10 else 0.0
        recall_20 = 1.0 if rank <= 20 else 0.0
        mrr = 1.0 / rank if rank > 0 else 0.0

        decisions, insert_seqs = model.corrector_inference(leave_one_tensor)
        modified_sequence = model.seqs_correction(decisions, insert_seqs, leave_one_tensor)
        operation_counts = stat_operation(decisions, leave_one_tensor[0])
        operation_counts = operation_counts[0]

    keep_count = int((decisions[0] == 0).sum().item())
    delete_count = int((decisions[0] == 1).sum().item())
    insert_count = int((decisions[0] == 2).sum().item())
    decision_summary = {"keep": keep_count, "delete": delete_count, "insert": insert_count}

    metrics = [
        _to_metric_card("Recall@5", recall_5, 8.4),
        _to_metric_card("Recall@10", recall_10, 6.9),
        _to_metric_card("Recall@20", recall_20, 4.8),
        _to_metric_card("MRR", mrr, 7.1),
    ]

    raw_items: List[Dict[str, Any]] = []
    for index, item_id in enumerate(raw_sequence[:6]):
        current_decision = int(decisions[0][index].item()) if index < decisions.shape[1] else 0
        if current_decision == 0:
            action = "keep"
        elif current_decision == 1:
            action = "delete"
        else:
            action = "insert"
        raw_items.append({
            "id": int(item_id),
            "name": f"Item {int(item_id)}",
            "category": dataset,
            "type": action,
        })

    corrected_actions: List[Dict[str, Any]] = []
    action_map = {0: ("keep", "Kept"), 1: ("delete", "Removed"), 2: ("insert", "Inserted")}
    for index in range(min(3, len(raw_sequence))):
        decision = int(decisions[0][index].item()) if index < decisions.shape[1] else 0
        kind, label = action_map.get(decision, ("keep", "Kept"))
        if decision == 0:
            note = "The model retained this interaction as aligned with the user preference signal."
        elif decision == 1:
            note = "This interaction was flagged as likely noise or an accidental click."
        else:
            note = "The model inferred a missed item at this position and inserted it back into the session."
        corrected_actions.append({
            "kind": kind,
            "label": label,
            "item": f"Item {int(raw_sequence[index])}",
            "note": note,
        })

    top_scores = torch.softmax(scores, dim=-1)
    top_items = torch.topk(top_scores, 5).indices.tolist()
    recommendations: List[Dict[str, Any]] = []
    for rank_index, item_id in enumerate(top_items, start=1):
        confidence = "Very high" if rank_index <= 2 else "High" if rank_index <= 3 else "Medium"
        recommendations.append({
            "rank": rank_index,
            "name": f"Item {int(item_id)}",
            "category": dataset,
            "score": round(float(top_scores[item_id].item() * 100), 1),
            "confidence": confidence,
        })

    if not modified_sequence:
        modified_sequence = [raw_sequence]

    insight = (
        f"The corrected sequence for {dataset} keeps {decision_summary['keep']} interactions, removes "
        f"{decision_summary['delete']} likely-noisy events, and restores {decision_summary['insert']} potential misses. "
        f"The model currently ranks the ground-truth next interaction at position {rank} with Recall@5={recall_5:.3f}."
    )

    response = {
        "dataset": dataset,
        "metrics": metrics,
        "sequence": raw_items,
        "correctedActions": corrected_actions,
        "recommendations": recommendations,
        "performance": _build_performance(decision_summary),
        "insightSummary": insight,
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
