import argparse
import math
from pathlib import Path
from zipfile import BadZipFile

from data import MovieLens, load_dataset
from experiment import run_experiment
from models import MODEL_LABELS
from recommendations import RecommendationSession


ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("请输入大于 0 的整数。")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("请输入大于 0 的有限数值。")
    return number


def show_movies(dataset: MovieLens, query: str, limit: int) -> None:
    matches = [(movie, title) for movie, title in dataset.titles.items() if query.casefold() in title.casefold()]
    if not matches:
        print("没有找到匹配的电影。试试英文片名的一部分。")
        return
    for movie, title in matches[:limit]:
        print(f"{movie:>4}  {title}")


def recommend(dataset: MovieLens, args: argparse.Namespace) -> None:
    rows = RecommendationSession(dataset).recommend(
        method=args.model, top_k=args.top_k, shrinkage=args.shrinkage,
        user=args.user, likes=tuple(args.likes or ()),
    )
    heading = f"MovieLens 用户 {args.user}" if args.user is not None else "自定义观影偏好"
    model_label = MODEL_LABELS[args.model]
    if args.model == "shrinkage":
        model_label += f" (λ = {args.shrinkage:g})"
    print(f"{heading} · {model_label}")
    if not rows:
        print("当前没有可推荐的电影。")
        return

    for rank, row in enumerate(rows, start=1):
        print(f"{rank:>2}. [{row.movie_id:>4}] {row.title}")
        print(f"    {row.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens 100K 电影推荐与离线评估")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ml-100k", help="本地数据目录")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="比较四种推荐方法并生成实验结果")
    evaluate_parser.add_argument("--top-k", type=positive_int, default=10, help="评估列表长度，默认 10")
    evaluate_parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="实验结果目录")

    search_parser = commands.add_parser("search", help="按英文片名查电影编号")
    search_parser.add_argument("query", help="英文片名或其一部分")
    search_parser.add_argument("--limit", type=positive_int, default=10, help="最多显示多少条")

    recommend_parser = commands.add_parser("recommend", help="按用户评分历史或自选电影生成推荐")
    profile = recommend_parser.add_mutually_exclusive_group(required=True)
    profile.add_argument("--user", type=positive_int, help="MovieLens 用户编号")
    profile.add_argument("--likes", type=positive_int, nargs="+", metavar="MOVIE_ID", help="喜欢的电影编号列表")
    recommend_parser.add_argument("--model", choices=MODEL_LABELS, default="itemcf", help="推荐算法，默认 itemcf")
    recommend_parser.add_argument("--shrinkage", type=positive_float, default=10.0, help="shrinkage 模型的 λ，默认 10")
    recommend_parser.add_argument("--top-k", type=positive_int, default=10, help="推荐列表长度，默认 10")

    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data_dir)
        if args.command == "evaluate":
            run_experiment(dataset, args.output_dir, args.top_k)
        elif args.command == "search":
            show_movies(dataset, args.query, args.limit)
        else:
            recommend(dataset, args)
    except (OSError, ValueError, BadZipFile, KeyError) as error:
        parser.exit(1, f"执行失败：{error}\n")


if __name__ == "__main__":
    main()
