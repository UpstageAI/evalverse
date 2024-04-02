"""
Copyright (c) 2024-present Upstage Co., Ltd.
Apache-2.0 license
"""
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px

EVALVERSE_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
EVALVERSE_DB_PATH = os.path.join(os.path.dirname(EVALVERSE_MODULE_PATH), "db")
EVALVERSE_OUTPUT_PATH = os.path.join(os.path.dirname(EVALVERSE_MODULE_PATH), "results")
EVALVERSE_LOG_FORMAT = (
    "[%(asctime)s][%(levelname)s][evalverse - %(filename)s:%(lineno)d] >> %(message)s"
)


def print_command(command, only_cmd=False):
    cmd = re.sub(r"\s+", " ", command).strip()
    if only_cmd:
        return cmd
    else:
        print(cmd)


def get_logger(log_path=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        fmt=EVALVERSE_LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if log_path:
        fileHandler = logging.FileHandler(filename=log_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger


def get_figure(score_df, benchmarks_list, figure_path=None, save=False):
    scores = []
    for b in benchmarks_list:
        for m, n in score_df[["Model", b]].values:
            scores.append([m, b, n])
    figure_df = pd.DataFrame(scores, columns=["model", "benchmark", "score"])

    fig = px.line_polar(
        figure_df,
        r="score",
        theta="benchmark",
        line_close=True,
        category_orders={"benchmark": benchmarks_list},
        color="model",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="LLM Evaluation Report (by Evalverse)",
        width=800,
    )
    if save:
        fig.write_image(figure_path, scale=2)
    else:
        fig.show()


def get_h6_en_scores(exp_path, stderr=False):
    acc_metric = "acc,none"
    acc_norm_metric = "acc_norm,none"
    gsm8k_metrics = ["exact_match,get-answer", "exact_match,strict-match"]
    if stderr:
        acc_metric = "acc_stderr,none"
        acc_norm_metric = "acc_norm_stderr,none"
        gsm8k_metrics = ["exact_match_stderr,get-answer", "exact_match_stderr,strict-match"]

    with open(os.path.join(exp_path, "arc_challenge_25.json"), "r") as json_file:
        arc_challenge_25 = json.load(json_file)
        arc_score = arc_challenge_25["results"]["arc_challenge"][acc_norm_metric]

    with open(os.path.join(exp_path, "hellaswag_10.json"), "r") as json_file:
        hellaswag_10 = json.load(json_file)
        hellaswag_score = hellaswag_10["results"]["hellaswag"][acc_norm_metric]

    with open(os.path.join(exp_path, "mmlu_5.json"), "r") as json_file:
        mmlu_5 = json.load(json_file)
        mmlu_score = mmlu_5["results"]["mmlu"][acc_metric]

    with open(os.path.join(exp_path, "truthfulqa_mc2_0.json"), "r") as json_file:
        truthfulqa_mc2_0 = json.load(json_file)
        truthfulqa_score = truthfulqa_mc2_0["results"]["truthfulqa_mc2"][acc_metric]

    with open(os.path.join(exp_path, "winogrande_5.json"), "r") as json_file:
        winogrande_5 = json.load(json_file)
        winogrande_score = winogrande_5["results"]["winogrande"][acc_metric]

    with open(os.path.join(exp_path, "gsm8k_5.json"), "r") as json_file:
        gsm8k_5 = json.load(json_file)
        match_key = next((key for key in gsm8k_metrics if key in gsm8k_5["results"]["gsm8k"]), None)
        gsm8k_score = gsm8k_5["results"]["gsm8k"][match_key]

    score_list = [
        arc_score,
        hellaswag_score,
        mmlu_score,
        truthfulqa_score,
        winogrande_score,
        gsm8k_score,
    ]
    score_list = list(np.round((np.array(score_list) * 100), 2))

    return score_list


def get_mt_bench_scores(model_id, question_path, judgement_path):
    question_df = pd.read_json(question_path, lines=True)
    judgement_df = pd.read_json(judgement_path, lines=True)

    df = judgement_df[["question_id", "model", "score", "turn"]]
    df = df[(df["model"] == model_id) & (df["score"] != -1)]
    df = df.merge(question_df[["question_id", "category"]], how="left")
    df = df[["category", "score"]].groupby(["category"]).mean()
    df = df.sort_values("category")

    score_list = df.score.values.tolist()
    score_list = list(np.round((np.array(score_list) * 10), 2))

    return score_list


def get_ifeval_scores(score_txt_file):
    score_list = []
    with open(score_txt_file, "r") as file:
        content = file.read()

    pattern = r"(prompt-level|instruction-level):\s([\d.]+)"
    matches = re.findall(pattern, content)

    for _, score in matches:
        score_list.append(float(score))
    score_list = list(np.round((np.array(score_list) * 100), 2))

    return score_list


def get_eqbench_score(eqbench_results_json):
    with open(eqbench_results_json, "r") as f:
        data = json.load(f)

    final_score = data[list(data.keys())[0]]["iterations"]["1"]["benchmark_results_fullscale"][
        "final_score"
    ]
    score_list = [final_score]

    return score_list


if __name__ == "__main__":
    print(f"EVALVERSE_MODULE_PATH: {EVALVERSE_MODULE_PATH}")
    print(f"EVALVERSE_DB_PATH: {EVALVERSE_DB_PATH}")
    print(f"EVALVERSE_OUTPUT_PATH: {EVALVERSE_OUTPUT_PATH}")
