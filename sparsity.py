from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data import LIKE_THRESHOLD, MovieLens, split_by_time
from experiment import SHRINKAGE_GRID, EvaluationWindow, evaluate, evaluation_window, write_csv
from models import MODEL_LABELS, Recommender


VISIBLE_LIKES = (0, 1, 3, 5, 10)
TOP_K = 10
COLORS = ("#666f80", "#2c6aa6", "#168877", "#a55a9b")


@dataclass(frozen=True)
class SparsityData:
    model: Recommender
    window: EvaluationWindow
    earliest_likes: np.ndarray
    shrinkage: float
    validation_scores: list[tuple[int, float, float]]


def prepare_sparsity_data(dataset: MovieLens) -> SparsityData:
    train, validation, test = split_by_time(dataset.ratings)
    past = np.concatenate((train, validation))
    full_history = dataset.rating_matrix(past)
    future = dataset.rating_matrix(test)
    held_out = ((full_history >= LIKE_THRESHOLD).sum(axis=1) >= max(VISIBLE_LIKES)) & (
        future >= LIKE_THRESHOLD
    ).any(axis=1)

    # Keep the whole evaluation cohort out of fitting, catalog construction, and tuning.
    tuning_history = dataset.rating_matrix(train[~held_out[train[:, 0] - 1]])
    tuning_future = dataset.rating_matrix(validation[~held_out[validation[:, 0] - 1]])
    tuning_model = Recommender(tuning_history)
    tuning_window = evaluation_window(tuning_history, tuning_future, tuning_model.catalog)
    baseline = evaluate(tuning_model, tuning_window, "itemcf", TOP_K)
    validation_scores = [(0, float(baseline.recall.mean()), float(baseline.ndcg.mean()))]
    for shrinkage in SHRINKAGE_GRID:
        result = evaluate(tuning_model, tuning_window, "shrinkage", TOP_K, shrinkage)
        validation_scores.append((shrinkage, float(result.recall.mean()), float(result.ndcg.mean())))
    # Keep a genuinely weighted model in the comparison, and report λ=0 as its reference.
    shrinkage = max(validation_scores[1:], key=lambda row: row[2])[0]

    model = Recommender(dataset.rating_matrix(past[~held_out[past[:, 0] - 1]]))
    enough_known_likes = ((full_history >= LIKE_THRESHOLD) & model.catalog).sum(axis=1) >= max(VISIBLE_LIKES)
    cohort_users = np.flatnonzero(held_out & enough_known_likes)
    selected = evaluation_window(full_history[cohort_users], future[cohort_users], model.catalog)
    users = cohort_users[selected.users]
    window = EvaluationWindow(users, selected.profiles, selected.targets, selected.reachable_positives, selected.total_positives)

    # past is sorted by timestamp. Keep stable source order for ties.
    candidates = past[(past[:, 2] >= LIKE_THRESHOLD) & model.catalog[past[:, 1] - 1]]
    earliest_likes = np.stack([
        candidates[candidates[:, 0] == user + 1, 1][:max(VISIBLE_LIKES)] - 1 for user in users
    ])
    return SparsityData(model, window, earliest_likes, shrinkage, validation_scores)


def visible_profiles(data: SparsityData, count: int) -> np.ndarray:
    profiles = np.zeros_like(data.window.profiles)
    profiles[np.arange(len(profiles))[:, None], data.earliest_likes[:, :count]] = LIKE_THRESHOLD
    return profiles


def plot_sparsity(path: Path, rows: list[tuple]) -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 4.2), layout="constrained")
    axes = fig.subplots(1, 2)
    for column, axis, label in zip((3, 4), axes, ("Recall@10", "NDCG@10")):
        for (method, name), color in zip(MODEL_LABELS.items(), COLORS):
            axis.plot(VISIBLE_LIKES, [row[column] for row in rows if row[1] == method],
                      marker="o", linewidth=2, color=color, label=name)
        axis.set(xlabel="Visible positive ratings", ylabel=label, xticks=VISIBLE_LIKES)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(path, dpi=160)


def run_sparsity_experiment(dataset: MovieLens, output_dir: Path) -> None:
    data = prepare_sparsity_data(dataset)
    users = len(data.window.users)
    rows = []
    for count in VISIBLE_LIKES:
        profiles = visible_profiles(data, count)
        for method in MODEL_LABELS:
            result = evaluate(data.model, data.window, method, TOP_K, data.shrinkage, visible_profiles=profiles)
            rows.append((count, method, users, float(result.recall.mean()), float(result.ndcg.mean())))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", ("visible_likes", "model", "users", "recall@10", "ndcg@10"), rows)
    write_csv(output_dir / "validation.csv", ("shrinkage", "recall@10", "ndcg@10"), data.validation_scores)
    plot_sparsity(output_dir / "comparison.png", rows)
    summary = [
        "# 固定用户的稀疏历史实验", "",
        f"同一批 {users} 位用户参与全部实验，测试期共有 {data.window.reachable_positives:,} 条可达高分记录。",
        "从测试前至少有 10 条可用高分记录、测试期至少有 1 条可达高分记录的用户中构建评估队列。",
        "推荐模型、热门度和电影目录由其他用户在时间截点前的数据构建。模型按时间只看到每位用户最早的 0、1、3、5 或 10 条目录内高分记录。",
        "所有条件使用相同的测试目标和候选集，统一用完整的测试前历史排除已评分电影。",
        "这里的 0 条表示不给模型评分信号，候选集仍排除已评电影。实验用于衡量已积累历史的用户在信息减少时的表现。",
        f"加权 ItemCF 在排除评估用户后的验证集选出 λ = {data.shrinkage:g}。验证表的 λ = 0 对应原版 ItemCF。",
        "", "![不同可见高分记录数下的推荐指标](comparison.png)", "",
        "| 可见高分记录数 | 方法 | Recall@10 | NDCG@10 |",
        "| ---: | --- | ---: | ---: |",
    ]
    for count, method, _, recall, ndcg in rows:
        summary.append(f"| {count} | {MODEL_LABELS[method]} | {recall:.4f} | {ndcg:.4f} |")
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"稀疏历史实验评估 {users} 位用户，结果已保存至 {output_dir.resolve()}")
