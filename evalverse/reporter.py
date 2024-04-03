"""
Copyright (c) 2024-present Upstage Co., Ltd.
Apache-2.0 license
"""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from typing import Union, List

from evalverse.utils import (
    EVALVERSE_DB_PATH,
    EVALVERSE_MODULE_PATH,
    EVALVERSE_OUTPUT_PATH,
    get_eqbench_score,
    get_figure,
    get_h6_en_scores,
    get_ifeval_scores,
    get_logger,
    get_mt_bench_scores,
)

KST = timezone(timedelta(hours=9))
AVAILABLE_BENCHMARKS = ["h6_en", "mt_bench", "ifeval", "eq_bench"]

H6EN_NAMES = ["H6-ARC", "H6-Hellaswag", "H6-MMLU", "H6-TruthfulQA", "H6-Winogrande", "H6-GSM8k"]
MTBENCH_NAMES = [
    "MT-Bench-Coding",
    "MT-Bench-Extraction",
    "MT-Bench-Humanities",
    "MT-Bench-Math",
    "MT-Bench-Reasoning",
    "MT-Bench-Roleplay",
    "MT-Bench-Stem",
    "MT-Bench-Writing",
]
IFEVAL_NAMES = [
    "IFEval-strict-prompt",
    "IFEval-strict-instruction",
    "IFEval-loose-prompt",
    "IFEval-loose-instruction",
]
EQBENCH_NAME = ["EQ-Bench"]


class Reporter:
    def __init__(self, db_path=EVALVERSE_DB_PATH, output_path=EVALVERSE_OUTPUT_PATH, log_path=None):
        self.db_path = db_path
        self.output_path = output_path
        self.logger = get_logger(log_path)

        self.score_path = os.path.join(self.db_path, "score_df.csv")
        self.table_dir = os.path.join(self.db_path, "scores")
        self.figure_dir = os.path.join(self.db_path, "figures")

        self.model_list = self._get_dirname_list(self.output_path)

        for path in [self.db_path, self.table_dir, self.figure_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        if os.path.exists(self.score_path):
            self.score_df = pd.read_csv(self.score_path)
        else:
            self.update_db(git_fetch=False)

    def _get_dirname_list(self, path):
        return sorted(os.listdir(path), key=str.lower)

    def update_db(self, save=False, git_fetch=False):
        if git_fetch:
            import git

            repo = git.Repo("../")
            repo.remotes.origin.fetch()

        self.model_list = self._get_dirname_list(self.output_path)
        if len(self.model_list) > 0:
            values_list = []
            for model_name in self.model_list:
                bench_list = self._get_dirname_list(os.path.join(self.output_path, model_name))
                if len(bench_list) > 0:
                    values = [model_name]
                    if "h6_en" in bench_list:
                        h6_en_path = os.path.join(self.output_path, model_name, "h6_en")
                        h6_en_scores = get_h6_en_scores(h6_en_path)
                        values += h6_en_scores
                        self.logger.info(f"DB updated: h6_en for {model_name}")
                    else:
                        values += [0] * len(H6EN_NAMES)
                    if "mt_bench" in bench_list:
                        mtbench_path = os.path.join(self.output_path, model_name, "mt_bench")
                        question_file = os.path.join(
                            EVALVERSE_MODULE_PATH,
                            "submodules/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl",
                        )
                        judgement_file = os.path.join(
                            mtbench_path, "model_judgment", "gpt-4_single.jsonl"
                        )
                        mt_scores = get_mt_bench_scores(model_name, question_file, judgement_file)
                        values += mt_scores
                        self.logger.info(f"DB updated: mt_bench for {model_name}")
                    else:
                        values += [0] * len(MTBENCH_NAMES)
                    if "ifeval" in bench_list:
                        score_file = os.path.join(
                            self.output_path, model_name, "ifeval", "scores.txt"
                        )
                        ifeval_scores = get_ifeval_scores(score_file)
                        values += ifeval_scores
                        self.logger.info(f"DB updated: ifeval for {model_name}")
                    else:
                        values += [0] * len(IFEVAL_NAMES)
                    if "eq_bench" in bench_list:
                        eqbench_result_file = os.path.join(
                            self.output_path, model_name, "eq_bench", "raw_results.json"
                        )
                        eqbench_score = get_eqbench_score(eqbench_result_file)
                        values += eqbench_score
                        self.logger.info(f"DB updated: eq_bench for {model_name}")
                    else:
                        values += [0] * len(EQBENCH_NAME)
                    values_list.append(values)
                else:
                    pass
            column_list = ["Model"] + H6EN_NAMES + MTBENCH_NAMES + IFEVAL_NAMES + EQBENCH_NAME
            self.score_df = pd.DataFrame(data=values_list, columns=column_list)
            if save:
                self.score_df.to_csv(self.score_path, index=False)
                self.logger.info(f"DB saved to {self.score_path}")
        else:
            pass

    def run(self, model_list: Union[List, str] = "all", benchmark_list: Union[List, str] = "all", save: bool = False):

        if type(model_list) == list:
            for m in model_list:
                if m in self.model_list:
                    pass
                else:
                    raise ValueError(f'"{m}" is not in Available_Models: {self.model_list}')
        elif type(model_list) == str:
            if model_list in self.model_list:
                model_list = [model_list]
            elif model_list == "all":
                model_list = self.model_list
            else:
                raise ValueError(f'"{model_list}" is not in Available_Models: {self.model_list}')
        else:
            raise TypeError

        if type(benchmark_list) == list:
            for b in benchmark_list:
                if b in AVAILABLE_BENCHMARKS:
                    pass
                else:
                    raise ValueError(
                        f'"{b}" is not in Available_Benchmarks: {AVAILABLE_BENCHMARKS}'
                    )
        elif type(benchmark_list) == str:
            if benchmark_list in AVAILABLE_BENCHMARKS:
                benchmark_list = [benchmark_list]
            elif benchmark_list == "all":
                benchmark_list = AVAILABLE_BENCHMARKS
            else:
                raise ValueError(
                    f'"{benchmark_list}" is not in Available_Benchmarks: {AVAILABLE_BENCHMARKS}'
                )
        selected_benchmarks = []
        for b in benchmark_list:
            if b == "h6_en":
                selected_benchmarks += H6EN_NAMES
            if b == "mt_bench":
                selected_benchmarks += MTBENCH_NAMES
            if b == "ifeval":
                selected_benchmarks += IFEVAL_NAMES
            if b == "eq_bench":
                selected_benchmarks += EQBENCH_NAME

        score_df = self.score_df.copy()
        score_df = score_df[(score_df["Model"].isin(model_list))]
        score_df["total_avg"] = score_df[selected_benchmarks].mean(axis=1).round(2)
        score_df = score_df.sort_values("total_avg", ascending=False).reset_index(drop=True)
        score_df["Ranking"] = score_df["total_avg"].rank(ascending=False).astype(int)
        score_df = score_df[["Model", "Ranking", "total_avg"] + selected_benchmarks]

        if save:
            request_time = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
            table_name = f"table_{request_time}.csv"
            figure_name = f"figure_{request_time}.jpeg"
            table_path = os.path.join(self.table_dir, table_name)
            figure_path = os.path.join(self.figure_dir, figure_name)

            score_df.to_csv(table_path, index=False)
            get_figure(score_df, selected_benchmarks, figure_path, save=True)
            self.logger.info(f"Table saved to {table_path}")
            self.logger.info(f"Figure saved to {figure_path}")
            return table_path, figure_path
        else:
            get_figure(score_df, selected_benchmarks, save=False)
            return score_df
