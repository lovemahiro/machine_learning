import csv
import math
import os
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk

from data import load_dataset
from experiment import run_experiment
from recommendations import RecommendationSession


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXAMPLE = (50, 172, 181)
MOVIES_PER_PAGE = 50
MODELS = {
    "ALS 矩阵分解": "als",
    "ItemCF 协同过滤": "itemcf",
    "加权 ItemCF": "shrinkage",
    "热门推荐": "popular",
}
MODEL_NAMES = {value: name for name, value in MODELS.items()}
MODEL_HELP = {
    "als": "根据评分训练用户和电影的潜因子。",
    "itemcf": "按电影与已选片单的相似度排序。",
    "shrinkage": "降低共同高分用户较少的电影对的相似度。",
    "popular": "按电影收到的高分数量排序。",
}
GROUPS = {"全部用户": "All", "无历史评分": "0", "1–19 条评分": "1–19", "20–49 条评分": "20–49", "50 条及以上": "50+"}
CHART_COLORS = {"als": "#963e30", "itemcf": "#536a77", "shrinkage": "#517967", "popular": "#757581"}
PAPER = "#f4f1ea"
WHITE = "#fffdf8"
INK = "#222d32"
MUTED = "#536167"
RULE = "#dcd7cd"
NAV = "#18272e"
NAV_TEXT = "#f9f5eb"
ACCENT = "#963e30"
ACCENT_HOVER = "#783125"
SELECTION = "#eeded4"
STRIPE = "#faf7f0"
FONT = "Microsoft YaHei UI"


def table(parent, columns, *, height, style="Catalog.Treeview"):
    frame = ttk.Frame(parent, style="Paper.TFrame")
    keys = [column[0] for column in columns]
    view = ttk.Treeview(frame, columns=keys, show="headings", height=height, selectmode="browse", style=style)
    for key, heading, width, minimum, grow, anchor in columns:
        view.heading(key, text=heading, anchor=anchor)
        view.column(key, width=width, minwidth=minimum, stretch=grow, anchor=anchor)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=view.yview)
    view.configure(yscrollcommand=scrollbar.set)
    view.pack(side="left", expand=True, fill="both")
    scrollbar.pack(side="right", fill="y")
    view.tag_configure("stripe", background=STRIPE)
    return frame, view


def replace_rows(view, rows):
    view.delete(*view.get_children())
    for index, (key, values) in enumerate(rows):
        view.insert("", "end", iid=key, values=values, tags=("stripe",) if index % 2 else ())


class MovieApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MovieLens 电影推荐")
        self.configure(bg=PAPER)
        width = min(1180, self.winfo_screenwidth() - 72)
        height = min(795, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}")
        self.minsize(880, 642)
        self.option_add("*Font", (FONT, 10))
        self._configure_styles()

        self.session = None
        self.picks = []
        self.catalog_page = 0
        self.details = {}
        self.metrics = []
        self.metric_keys = ()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.job = None
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.source = tk.StringVar(value="movies")
        self.query = tk.StringVar()
        self.user_id = tk.StringVar(value="42")
        self.model = tk.StringVar(value=next(iter(MODELS)))
        self.top_k = tk.StringVar(value="10")
        self.group = tk.StringVar(value="全部用户")
        self.build_header()
        self.pages = ttk.Frame(self, style="Paper.TFrame")
        self.pages.pack(side="top", expand=True, fill="both")
        self.recommendation_page = ttk.Frame(self.pages, style="Paper.TFrame", padding=(26, 20))
        self.experiment_page = ttk.Frame(self.pages, style="Paper.TFrame", padding=(26, 20))
        self.build_recommendations()
        self.build_experiments()
        footer = ttk.Frame(self, style="Footer.TFrame", padding=(26, 7))
        footer.pack(side="bottom", fill="x")
        self.data_status = ttk.Label(footer, text="正在读取 MovieLens 100K…", style="Footer.TLabel")
        self.data_status.pack(side="left")
        ttk.Label(footer, text="机器学习课程项目", style="Footer.TLabel").pack(side="right")
        self.query.trace_add("write", self.search_catalog)
        self.user_id.trace_add("write", self.form_changed)
        self.model.trace_add("write", self.form_changed)
        self.top_k.trace_add("write", self.form_changed)
        self.show_page("recommendations")
        self.load_results()
        self.after(0, self.initialize_dataset)

    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=WHITE)
        style.configure("Paper.TFrame", background=PAPER)
        style.configure("Footer.TFrame", background="#e9e3d8")
        style.configure("TLabel", background=WHITE, foreground=INK, font=(FONT, 10))
        style.configure("Paper.TLabel", background=PAPER, foreground=INK)
        style.configure("Muted.TLabel", background=WHITE, foreground=MUTED, font=(FONT, 9))
        style.configure("PaperMuted.TLabel", background=PAPER, foreground=MUTED, font=(FONT, 10))
        style.configure("Footer.TLabel", background="#e9e3d8", foreground=MUTED, font=(FONT, 9))
        style.configure("PaperTitle.TLabel", background=PAPER, foreground=INK, font=(FONT, 19, "bold"))
        style.configure("Section.TLabel", background=WHITE, foreground=INK, font=(FONT, 11, "bold"))
        style.configure("Film.TLabel", background=WHITE, foreground=INK, font=(FONT, 12, "bold"))
        style.configure("TButton", font=(FONT, 10), padding=(11, 6), background=WHITE, foreground=INK, bordercolor=RULE, lightcolor=WHITE, darkcolor=WHITE)
        style.map("TButton", background=[("disabled", PAPER), ("active", SELECTION)], foreground=[("disabled", "#716d66")], bordercolor=[("focus", ACCENT)])
        style.configure("Pager.TButton", font=(FONT, 9), padding=(6, 3))
        style.configure("Primary.TButton", font=(FONT, 10, "bold"), padding=(13, 7), background=ACCENT, foreground=WHITE, bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)
        style.map("Primary.TButton", background=[("disabled", "#e2d6ce"), ("active", ACCENT_HOVER)], foreground=[("disabled", "#67534b")], bordercolor=[("focus", INK)])
        style.configure("Tab.TButton", background=NAV, foreground="#cdd5d5", borderwidth=0, lightcolor=NAV, darkcolor=NAV, padding=(16, 9))
        style.map("Tab.TButton", background=[("active", "#31434a")], foreground=[("active", NAV_TEXT)], bordercolor=[("focus", NAV_TEXT)])
        style.configure("ActiveTab.TButton", background="#35494f", foreground=NAV_TEXT, borderwidth=0, lightcolor="#35494f", darkcolor="#35494f", padding=(16, 9), font=(FONT, 10, "bold"))
        style.map("ActiveTab.TButton", background=[("active", "#3f555b")], bordercolor=[("focus", NAV_TEXT)])
        style.configure("TRadiobutton", background=WHITE, foreground=INK, padding=(0, 2, 10, 2))
        style.map("TRadiobutton", background=[("active", WHITE)], indicatorcolor=[("selected", ACCENT), ("!selected", WHITE)])
        for control in ("TEntry", "TCombobox"):
            style.configure(control, padding=5, bordercolor=RULE, fieldbackground=WHITE, foreground=INK)
            style.map(control, bordercolor=[("focus", ACCENT)], fieldbackground=[("readonly", WHITE)], selectbackground=[("readonly", WHITE)], selectforeground=[("readonly", INK)])
        for name, row_height in (("Catalog", 25), ("Result", 35), ("Metric", 36)):
            style.configure(f"{name}.Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK, bordercolor=RULE, rowheight=row_height, font=(FONT, 10))
            style.map(f"{name}.Treeview", background=[("selected", SELECTION)], foreground=[("selected", INK)], bordercolor=[("focus", ACCENT)])
            style.configure(f"{name}.Treeview.Heading", background="#eee9df", foreground=INK, font=(FONT, 9, "bold"), padding=(4, 5), relief="flat")

    def build_header(self):
        header = tk.Frame(self, bg=NAV, padx=26, pady=17)
        header.pack(fill="x")
        tk.Label(header, text="MovieLens 电影推荐", bg=NAV, fg=NAV_TEXT, font=(FONT, 17, "bold")).pack(side="left")
        self.experiment_tab = ttk.Button(header, text="实验对比", style="Tab.TButton", command=lambda: self.show_page("experiments"))
        self.experiment_tab.pack(side="right")
        self.recommendation_tab = ttk.Button(header, text="电影推荐", style="Tab.TButton", command=lambda: self.show_page("recommendations"))
        self.recommendation_tab.pack(side="right", padx=(0, 6))

    def build_recommendations(self):
        controls = ttk.Frame(self.recommendation_page, padding=(15, 13, 15, 12))
        controls.pack(side="left", fill="y", padx=(0, 23))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(2, weight=1)
        ttk.Label(controls, text="推荐设置", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        options = ttk.Frame(controls)
        options.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        for label, value in (("自选电影", "movies"), ("已有用户", "user")):
            ttk.Radiobutton(options, text=label, value=value, variable=self.source, command=self.switch_source).pack(side="left")

        self.source_body = ttk.Frame(controls)
        self.source_body.grid(row=2, column=0, sticky="nsew")
        self.movie_controls = ttk.Frame(self.source_body)
        self.movie_controls.columnconfigure(0, weight=1)
        self.movie_controls.rowconfigure(1, weight=1)
        search_line = ttk.Frame(self.movie_controls)
        search_line.grid(row=0, column=0, sticky="ew")
        ttk.Label(search_line, text="英文片名或电影编号", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        self.search_entry = ttk.Entry(search_line, textvariable=self.query)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 7))
        self.add_button = ttk.Button(search_line, text="添加", width=5, command=self.add_selected, state="disabled")
        self.add_button.pack(side="right")
        movie_columns = (("id", "编号", 52, 42, False, "center"), ("title", "电影", 238, 185, True, "w"))
        catalog, self.catalog_view = table(self.movie_controls, movie_columns, height=4)
        catalog.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.catalog_view.bind("<Double-1>", self.add_on_double_click)
        self.catalog_view.bind("<Return>", self.add_selected)
        self.catalog_view.bind("<<TreeviewSelect>>", self.update_catalog_action)
        pagination = ttk.Frame(self.movie_controls)
        pagination.grid(row=2, column=0, sticky="ew", pady=(4, 7))
        self.search_status = ttk.Label(pagination, text="正在读取电影目录…", style="Muted.TLabel")
        self.search_status.pack(side="left")
        self.next_page_button = ttk.Button(pagination, text="下一页", style="Pager.TButton", command=lambda: self.change_catalog_page(1), state="disabled")
        self.next_page_button.pack(side="right")
        self.previous_page_button = ttk.Button(pagination, text="上一页", style="Pager.TButton", command=lambda: self.change_catalog_page(-1), state="disabled")
        self.previous_page_button.pack(side="right", padx=(5, 4))

        picked_heading = ttk.Frame(self.movie_controls)
        picked_heading.grid(row=3, column=0, sticky="ew")
        self.picks_label = ttk.Label(picked_heading, text="喜欢的电影", style="Section.TLabel")
        self.picks_label.pack(side="left")
        ttk.Button(picked_heading, text="移除", command=self.remove_selected).pack(side="right")
        ttk.Button(picked_heading, text="载入示例", command=self.load_example).pack(side="right", padx=(0, 5))
        picked, self.picked_view = table(self.movie_controls, movie_columns, height=3)
        picked.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.picked_view.bind("<Delete>", self.remove_selected)

        self.user_controls = ttk.Frame(self.source_body)
        ttk.Label(self.user_controls, text="MovieLens 用户编号", style="Section.TLabel").pack(anchor="w")
        user_entry = ttk.Entry(self.user_controls, textvariable=self.user_id, width=12)
        user_entry.pack(anchor="w", pady=(6, 6))
        user_entry.bind("<Return>", self.generate)
        self.user_help = ttk.Label(self.user_controls, text="", style="Muted.TLabel", wraplength=285, justify="left")
        self.user_help.pack(anchor="w")
        self.movie_controls.pack(fill="both", expand=True)

        ttk.Separator(controls).grid(row=3, column=0, sticky="ew", pady=9)
        settings = ttk.Frame(controls)
        settings.grid(row=4, column=0, sticky="ew")
        ttk.Label(settings, text="推荐算法").grid(row=0, column=0, sticky="w")
        ttk.Label(settings, text="数量").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Combobox(settings, textvariable=self.model, values=list(MODELS), width=16, state="readonly").grid(row=1, column=0, sticky="ew", pady=(3, 0))
        ttk.Combobox(settings, textvariable=self.top_k, values=("5", "10", "15", "20"), width=5, state="readonly").grid(row=1, column=1, padx=(8, 0), pady=(3, 0))
        settings.columnconfigure(0, weight=1)
        self.model_help = ttk.Label(controls, text=MODEL_HELP["als"], style="Muted.TLabel", wraplength=275)
        self.model_help.grid(row=5, column=0, sticky="w", pady=(5, 7))
        self.generate_button = ttk.Button(controls, text="生成推荐", style="Primary.TButton", command=self.generate, state="disabled")
        self.generate_button.grid(row=6, column=0, sticky="ew")

        results = ttk.Frame(self.recommendation_page, style="Paper.TFrame")
        results.pack(side="left", fill="both", expand=True)
        ttk.Label(results, text="推荐片单", style="PaperTitle.TLabel").pack(anchor="w")
        self.result_status = ttk.Label(results, text="正在准备推荐模型…", style="PaperMuted.TLabel", wraplength=480, justify="left")
        self.result_status.pack(anchor="w", pady=(3, 13))
        result_columns = (("rank", "排名", 65, 50, False, "center"), ("title", "电影", 283, 210, True, "w"), ("id", "编号", 72, 52, False, "center"))
        result_frame, self.result_view = table(results, result_columns, height=8, style="Result.Treeview")
        result_frame.pack(fill="both", expand=True)
        self.result_view.bind("<<TreeviewSelect>>", self.show_reason)
        reason = ttk.Frame(results, padding=(13, 10))
        reason.pack(fill="x", pady=(10, 0))
        ttk.Label(reason, text="推荐依据", style="Muted.TLabel").pack(anchor="w")
        self.reason_title = ttk.Label(reason, text="", style="Film.TLabel", wraplength=490, justify="left")
        self.reason_title.pack(anchor="w")
        self.reason_text = ttk.Label(reason, text="选中一部推荐电影可查看依据。", style="Muted.TLabel", wraplength=490, justify="left")
        self.reason_text.pack(anchor="w", pady=(2, 0))
        reason.bind("<Configure>", lambda event: self.set_detail_width(event.width))

    def build_experiments(self):
        bar = ttk.Frame(self.experiment_page, style="Paper.TFrame")
        bar.pack(fill="x")
        ttk.Label(bar, text="实验对比", style="PaperTitle.TLabel").pack(side="left")
        self.evaluate_button = ttk.Button(bar, text="重新评估", command=self.evaluate, state="disabled")
        self.evaluate_button.pack(side="right")
        ttk.Button(bar, text="打开结果目录", command=self.open_results).pack(side="right", padx=(0, 8))
        self.experiment_status = ttk.Label(self.experiment_page, text="", style="PaperMuted.TLabel", wraplength=760, justify="left")
        self.experiment_status.pack(anchor="w", pady=(4, 12))

        options = ttk.Frame(self.experiment_page, style="Paper.TFrame")
        options.pack(fill="x", pady=(0, 7))
        ttk.Label(options, text="用户分组", style="Paper.TLabel").pack(side="left", padx=(0, 9))
        group_choice = ttk.Combobox(options, textvariable=self.group, values=list(GROUPS), width=16, state="readonly")
        group_choice.pack(side="left")
        group_choice.bind("<<ComboboxSelected>>", self.render_metrics)

        columns = (("method", "模型", 218, 160, True, "w"), ("users", "用户数", 120, 80, True, "center"), ("recall", "Recall", 140, 100, True, "center"), ("ndcg", "NDCG", 140, 100, True, "center"))
        metrics, self.metrics_view = table(self.experiment_page, columns, height=4, style="Metric.Treeview")
        metrics.pack(fill="x")
        self.plot = tk.Canvas(self.experiment_page, bg=WHITE, highlightthickness=1, highlightbackground=RULE, height=185)
        self.plot.pack(fill="both", expand=True, pady=(12, 0))
        self.plot.bind("<Configure>", self.draw_metrics_chart)

    def show_page(self, name):
        self.recommendation_page.pack_forget()
        self.experiment_page.pack_forget()
        is_recommendation = name == "recommendations"
        page = self.recommendation_page if is_recommendation else self.experiment_page
        page.pack(fill="both", expand=True)
        self.recommendation_tab.configure(style="ActiveTab.TButton" if is_recommendation else "Tab.TButton")
        self.experiment_tab.configure(style="Tab.TButton" if is_recommendation else "ActiveTab.TButton")

    def set_detail_width(self, width):
        self.reason_title.configure(wraplength=max(200, width - 26))
        self.reason_text.configure(wraplength=max(200, width - 26))

    def switch_source(self):
        self.movie_controls.pack_forget()
        self.user_controls.pack_forget()
        panel = self.movie_controls if self.source.get() == "movies" else self.user_controls
        panel.pack(fill="both", expand=True)
        self.form_changed()

    def start_job(self, work, done, status):
        self.job = self.worker.submit(work)
        self.generate_button.state(["disabled"])
        self.evaluate_button.state(["disabled"])
        self.after(45, self.poll_job, self.job, done, status)

    def poll_job(self, future: Future, done, status):
        if not future.done():
            self.after(45, self.poll_job, future, done, status)
            return
        self.job = None
        try:
            result = future.result()
        except Exception as error:
            status.configure(text=f"操作失败。{error}")
            if self.session is None:
                self.data_status.configure(text="数据加载失败")
                self.generate_button.configure(text="重试加载数据", command=self.initialize_dataset)
                self.generate_button.state(["!disabled"])
        else:
            done(result)
        if self.session is not None and self.job is None:
            self.generate_button.state(["!disabled"])
            self.evaluate_button.state(["!disabled"])

    def initialize_dataset(self):
        self.result_status.configure(text="正在读取电影评分。首次运行需要下载 MovieLens 100K…")
        self.start_job(lambda: RecommendationSession(load_dataset(ROOT / "data" / "ml-100k")), self.dataset_ready, self.result_status)

    def dataset_ready(self, session):
        self.session = session
        self.generate_button.configure(text="生成推荐", command=self.generate)
        self.data_status.configure(text=f"MovieLens 100K   ·   {len(session.dataset.titles):,} 部电影   ·   {len(session.dataset.ratings):,} 条评分")
        self.user_help.configure(text=f"输入 1–{len(session.history)} 之间的编号。系统会根据该用户的历史高分电影推荐。")
        self.load_example()
        self.update_catalog()
        self.generate()

    def search_catalog(self, *_):
        self.catalog_page = 0
        self.update_catalog()

    def change_catalog_page(self, step):
        self.catalog_page += step
        self.update_catalog()

    def update_catalog(self):
        if self.session is None:
            return
        matches = self.session.matching_movies(self.query.get())
        page_count = (len(matches) + MOVIES_PER_PAGE - 1) // MOVIES_PER_PAGE
        self.catalog_page = max(0, min(self.catalog_page, page_count - 1))
        start = self.catalog_page * MOVIES_PER_PAGE
        rows = matches[start:start + MOVIES_PER_PAGE]
        replace_rows(self.catalog_view, ((str(movie_id), (movie_id, title)) for movie_id, title in rows))
        self.catalog_view.yview_moveto(0)
        if rows:
            first = str(rows[0][0])
            self.catalog_view.focus(first)
            self.catalog_view.selection_set(first)
        label = f"{self.catalog_page + 1} / {page_count} 页 · 共 {len(matches):,} 部" if rows else "没有匹配的电影"
        self.search_status.configure(text=label)
        self.previous_page_button.state(["!disabled" if self.catalog_page > 0 else "disabled"])
        self.next_page_button.state(["!disabled" if self.catalog_page + 1 < page_count else "disabled"])
        self.update_catalog_action()

    def update_catalog_action(self, *_):
        chosen = self.catalog_view.selection()
        already_picked = bool(chosen) and int(chosen[0]) in self.picks
        self.add_button.configure(text="已添加" if already_picked else "添加")
        self.add_button.state(["!disabled" if chosen and not already_picked else "disabled"])

    def update_picks(self):
        titles = self.session.dataset.titles
        replace_rows(self.picked_view, ((str(movie_id), (movie_id, titles[movie_id])) for movie_id in self.picks))
        self.picks_label.configure(text=f"喜欢的电影（{len(self.picks)}）")
        self.update_catalog_action()
        self.form_changed()

    def load_example(self):
        if self.session is not None:
            self.picks = [movie_id for movie_id in EXAMPLE if movie_id in self.session.dataset.titles]
            self.update_picks()

    def add_on_double_click(self, event):
        row = self.catalog_view.identify_row(event.y)
        if row:
            self.catalog_view.selection_set(row)
            self.add_selected()

    def add_selected(self, *_):
        chosen = self.catalog_view.selection()
        if not chosen or int(chosen[0]) in self.picks:
            return
        self.picks.append(int(chosen[0]))
        self.update_picks()
        self.picked_view.see(chosen[0])

    def remove_selected(self, *_):
        chosen = self.picked_view.selection()
        if chosen:
            self.picks.remove(int(chosen[0]))
            self.update_picks()

    def form_changed(self, *_):
        method = MODELS[self.model.get()]
        self.model_help.configure(text=MODEL_HELP[method])
        if self.session is not None:
            self.result_status.configure(text="设置已更新。点击「生成推荐」刷新片单。")

    def generate(self, *_):
        if self.session is None or self.job is not None:
            return
        method = MODELS[self.model.get()]
        top_k = int(self.top_k.get())
        user = None
        likes = ()
        if self.source.get() == "user":
            entry = self.user_id.get().strip()
            if not entry.isdecimal() or len(entry) > 7 or not 1 <= int(entry) <= len(self.session.history):
                self.result_status.configure(text=f"请输入 1–{len(self.session.history)} 之间的用户编号。")
                return
            user = int(entry)
        else:
            likes = tuple(self.picks)
            if not likes and method != "popular":
                self.result_status.configure(text="请先搜索并添加至少一部喜欢的电影。")
                return

        source_label = f"用户 {user}" if user is not None else f"自选偏好 · {len(likes)} 部电影"
        status = f"正在使用 {MODEL_NAMES[method]} 生成推荐…"
        self.result_status.configure(text=status)
        session = self.session
        self.start_job(
            lambda: session.recommend(method=method, top_k=top_k, shrinkage=5, user=user, likes=likes),
            lambda rows: self.show_recommendations(rows, source_label, method, status),
            self.result_status,
        )

    def show_recommendations(self, rows, source, method, previous_status):
        self.details = {str(row.movie_id): row for row in rows}
        replace_rows(self.result_view, ((str(row.movie_id), (rank, row.title, row.movie_id)) for rank, row in enumerate(rows, start=1)))
        if self.result_status.cget("text") == previous_status:
            summary = f"{source}   ·   {MODEL_NAMES[method]}   ·   {len(rows)} 部推荐"
            self.result_status.configure(text=summary if rows else "当前没有可推荐的电影。")
        if rows:
            self.result_view.selection_set(str(rows[0].movie_id))
        self.show_reason()

    def show_reason(self, *_):
        chosen = self.result_view.selection()
        row = self.details.get(chosen[0]) if chosen else None
        self.reason_title.configure(text=row.title if row else "")
        self.reason_text.configure(text=row.detail if row else "选中一部推荐电影可查看依据。")

    def load_results(self):
        try:
            with (RESULTS / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                recall = next(key for key in reader.fieldnames or () if key.startswith("recall@"))
                ndcg = next(key for key in reader.fieldnames or () if key.startswith("ndcg@"))
                self.metric_keys = (recall, ndcg)
                self.metrics = [(row["history_group"], row["model"], int(row["users"]), float(row[recall]), float(row[ndcg])) for row in reader]
        except FileNotFoundError:
            self.metrics = []
            self.experiment_status.configure(text="暂无实验结果。点击「重新评估」运行四种推荐方法。")
        except (OSError, ValueError, KeyError, StopIteration):
            self.metrics = []
            self.experiment_status.configure(text="无法读取指标文件。点击「重新评估」生成新的结果。")
        else:
            self.experiment_status.configure(text="按时间划分的离线测试。点击用户分组查看不同历史评分数量下的表现。")
        self.render_metrics()

    def active_metrics(self):
        return [row[1:] for row in self.metrics if row[0] == GROUPS[self.group.get()]]

    def render_metrics(self, *_):
        rows = self.active_metrics()
        if self.metric_keys:
            recall, ndcg = self.metric_keys
            self.metrics_view.heading("recall", text=recall.replace("recall", "Recall"))
            self.metrics_view.heading("ndcg", text=ndcg.upper())
        replace_rows(self.metrics_view, ((method, (MODEL_NAMES.get(method, method), users, f"{recall:.4f}", f"{ndcg:.4f}")) for method, users, recall, ndcg in rows))
        self.draw_metrics_chart()

    def draw_metrics_chart(self, *_):
        self.plot.delete("all")
        rows = self.active_metrics()
        if not rows:
            self.plot.create_text(17, 20, text="当前分组暂无可显示的指标。", anchor="nw", fill=MUTED)
            return
        width = self.plot.winfo_width()
        if width < 350:
            return
        label = self.metric_keys[1].upper()
        self.plot.create_text(17, 15, text=label, anchor="nw", fill=INK, font=(FONT, 11, "bold"))
        chart_left = 144
        bar_width = width - chart_left - 72
        steps = max(1, math.ceil(max(row[4] for row in self.metrics) * 10))
        maximum = steps / 10
        for step in range(steps + 1):
            x = chart_left + bar_width * step / steps
            self.plot.create_line(x, 41, x, 151, fill=RULE)
            self.plot.create_text(x, 163, text=f"{step / 10:.1f}", fill=MUTED, font=(FONT, 8))
        for index, (method, _, _, ndcg) in enumerate(rows):
            y = 54 + index * 28
            end = chart_left + bar_width * ndcg / maximum
            self.plot.create_text(17, y, text=MODEL_NAMES.get(method, method), anchor="w", fill=INK, font=(FONT, 9))
            self.plot.create_rectangle(chart_left, y - 5, end, y + 5, fill=CHART_COLORS.get(method, MUTED), width=0)
            self.plot.create_text(end + 9, y, text=f"{ndcg:.4f}", anchor="w", fill=INK, font=(FONT, 9))

    def evaluate(self):
        if self.session is None or self.job is not None:
            return
        self.experiment_status.configure(text="正在重新训练并评估四种方法…")
        dataset = self.session.dataset
        self.start_job(lambda: run_experiment(dataset, RESULTS, 10), lambda _: self.load_results(), self.experiment_status)

    def open_results(self):
        try:
            os.startfile(RESULTS if RESULTS.is_dir() else ROOT)
        except OSError as error:
            self.experiment_status.configure(text=f"无法打开结果目录。{error}")

    def _close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    MovieApp().mainloop()
