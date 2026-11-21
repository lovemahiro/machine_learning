import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from als import ALPHA, FACTORS, ITERATIONS, REGULARIZATION, SEED
from data import LIKE_THRESHOLD, MovieLens, split_by_time
from models import MODEL_LABELS, Recommender


SHRINKAGE_GRID = (1, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class EvaluationWindow:
    users: np.ndarray
    profiles: np.ndarray
    targets: np.ndarray
    reachable_positives: int
    total_positives: int


@dataclass(frozen=True)
class Scores:
    ranked: np.ndarray
    recall: np.ndarray
    ndcg: np.ndarray


def evaluation_window(history: np.ndarray, future: np.ndarray, catalog: np.ndarray) -> EvaluationWindow:
    positives = future >= LIKE_THRESHOLD
    targets = positives & catalog & (history == 0)
    users = np.flatnonzero(targets.any(axis=1))
    if not len(users):
        raise ValueError("这个时间窗口内没有可评估的高分电影。")
    return EvaluationWindow(users, history[users], targets[users], int(targets.sum()), int(positives.sum()))


def evaluate(
    model: Recommender, window: EvaluationWindow, method: str, top_k: int,
    shrinkage: float = 10, *, visible_profiles: np.ndarray | None = None,
) -> Scores:
    profiles = window.profiles if visible_profiles is None else visible_profiles
    predictions = model.score_profiles(profiles, method, shrinkage)
    ranked = model.rank(window.profiles, predictions, top_k)
    hits = np.take_along_axis(window.targets, ranked.clip(min=0), axis=1) & (ranked >= 0)
    target_counts = window.targets.sum(axis=1)
    discounts = 1 / np.log2(np.arange(2, ranked.shape[1] + 2))
    ideal = np.cumsum(discounts)[np.minimum(target_counts, ranked.shape[1]) - 1]
    return Scores(ranked, hits.sum(axis=1) / target_counts, (hits @ discounts) / ideal)


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot_comparison(path: Path, overall: list[tuple], groups: list[tuple], top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = ("#727b90", "#2c6aa6", "#168877", "#a55a9b")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    axes[0].set_title("Overall ranking metrics")
    axes[1].set_title(f"NDCG@{top_k} by prior ratings")
    axes[0].set_xticks([0, 1], [f"Recall@{top_k}", f"NDCG@{top_k}"])
    group_labels = list(dict.fromkeys(row[0] for row in groups))
    axes[1].set_xticks(range(len(group_labels)), group_labels)
    axes[1].set_xlabel("Number of ratings before the test window")

    for index, ((_, method, _, recall, ndcg), color) in enumerate(zip(overall, colors)):
        offset = (index - (len(overall) - 1) / 2) * 0.20
        axes[0].bar(np.array([0, 1]) + offset, [recall, ndcg], width=0.19, color=color)
        values = [row[4] for row in groups if row[1] == method]
        axes[1].bar(np.arange(len(group_labels)) + offset, values, width=0.19, color=color, label=MODEL_LABELS[method])

    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=axes[1].containers, loc="outside lower center", ncols=len(overall), frameon=False, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def case_lines(dataset: MovieLens, window: EvaluationWindow, scores: dict[str, Scores], top_k: int) -> list[str]:
    delta = scores["shrinkage"].ndcg - scores["itemcf"].ndcg
    cases = (("相似度加权有收益的案例", int(delta.argmax())), ("相似度加权表现下降的案例", int(delta.argmin())))
    lines = []
    for index, (label, row) in enumerate(cases):
        if (index == 0 and delta[row] <= 0) or (index == 1 and delta[row] >= 0):
            continue

        user_id = int(window.users[row]) + 1
        historical_ratings = int(np.count_nonzero(window.profiles[row]))
        targets = int(window.targets[row].sum())
        lines += [f"### {label}", "", f"用户 {user_id} 有 {historical_ratings} 条历史评分，测试期有 {targets} 部可推荐的高分电影。", ""]
        for method, result in scores.items():
            hit_titles = [
                f"第 {rank} 名 {dataset.titles[int(movie) + 1]}"
                for rank, movie in enumerate(result.ranked[row], start=1)
                if movie >= 0 and window.targets[row, movie]
            ]
            detail = f"命中电影包括{'、'.join(hit_titles)}" if hit_titles else "推荐列表未命中高分电影"
            lines.append(f"- **{MODEL_LABELS[method]}** 的 NDCG@{top_k} 为 {result.ndcg[row]:.4f}。{detail}。")
        lines.append("")
    return lines


def run_experiment(dataset: MovieLens, output_dir: Path, top_k: int) -> None:
    train, validation, test = split_by_time(dataset.ratings)
    train_history = dataset.rating_matrix(train)
    train_model = Recommender(train_history)
    valid_window = evaluation_window(train_history, dataset.rating_matrix(validation), train_model.catalog)

    validation_scores = []
    for shrinkage in SHRINKAGE_GRID:
        result = evaluate(train_model, valid_window, "shrinkage", top_k, shrinkage)
        validation_scores.append((shrinkage, float(result.recall.mean()), float(result.ndcg.mean())))
    best_shrinkage, _, best_validation_ndcg = max(validation_scores, key=lambda row: row[2])

    # The validation period becomes historical data when we move to the test cutoff.
    test_history = dataset.rating_matrix(np.concatenate((train, validation)))
    test_model = Recommender(test_history)
    test_window = evaluation_window(test_history, dataset.rating_matrix(test), test_model.catalog)
    scores = {method: evaluate(test_model, test_window, method, top_k, best_shrinkage) for method in MODEL_LABELS}

    counts = np.count_nonzero(test_window.profiles, axis=1)
    masks = {
        "All": np.ones(len(counts), dtype=bool),
        "0": counts == 0,
        "1–19": (counts >= 1) & (counts < 20),
        "20–49": (counts >= 20) & (counts < 50),
        "50+": counts >= 50,
    }
    metrics = [
        (group, method, int(mask.sum()), float(result.recall[mask].mean()), float(result.ndcg[mask].mean()))
        for group, mask in masks.items() if mask.any()
        for method, result in scores.items()
    ]
    overall = metrics[:len(MODEL_LABELS)]
    groups = metrics[len(MODEL_LABELS):]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", ("history_group", "model", "users", f"recall@{top_k}", f"ndcg@{top_k}"), metrics)
    write_csv(output_dir / "validation.csv", ("shrinkage", f"recall@{top_k}", f"ndcg@{top_k}"), validation_scores)
    plot_comparison(output_dir / "model_comparison.png", overall, groups, top_k)

    val_start = datetime.fromtimestamp(int(validation[0, 3]), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    test_start = datetime.fromtimestamp(int(test[0, 3]), timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary = [
        "# MovieLens 100K 实验结果", "",
        f"训练、验证和测试集分别有 {len(train):,}、{len(validation):,} 和 {len(test):,} 条评分。验证集从 {val_start} 开始，测试集从 {test_start} 开始。",
        f"验证集以 NDCG@{top_k} 选择加权系数 λ = {best_shrinkage}，对应分数为 {best_validation_ndcg:.4f}。",
        f"ALS 使用固定配置训练，潜因子维度为 {FACTORS}，迭代 {ITERATIONS} 次，正反馈置信度为 {1 + ALPHA:g}，L2 正则化系数为 {REGULARIZATION:g}，随机种子为 {SEED}。",
        f"测试集有 {test_window.total_positives:,} 条高分记录，其中 {test_window.reachable_positives:,} 条符合候选条件，覆盖 {len(test_window.users)} 位参与评估的用户。",
        "", "## 测试集指标", "",
        f"| 历史评分数 | 模型 | 用户数 | Recall@{top_k} | NDCG@{top_k} |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group, method, users, recall, ndcg in metrics:
        summary.append(f"| {group} | {MODEL_LABELS[method]} | {users} | {recall:.4f} | {ndcg:.4f} |")
    summary += [
        "", "![测试集推荐指标对比](model_comparison.png)", "",
        f"指标先按用户计算，再取平均。{int(masks['0'].sum())} 位用户在测试前没有评分记录，四种方法都按热门度给这些用户推荐。",
        "", "## 用户案例", "",
    ]
    summary += case_lines(dataset, test_window, scores, top_k)
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"验证集选择 λ = {best_shrinkage}，测试集评估用户数 {len(test_window.users)}。")
    print(f"{'Model':<22}{f'Recall@{top_k}':>12}{f'NDCG@{top_k}':>12}")
    for _, method, _, recall, ndcg in overall:
        print(f"{MODEL_LABELS[method]:<22}{recall:>12.4f}{ndcg:>12.4f}")
    print(f"结果已保存至 {output_dir.resolve()}")
