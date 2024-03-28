import json
import logging
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px

EVALVERSE_LOG_FORMAT = (
    "[%(asctime)s][%(levelname)s][evalverse - %(filename)s:%(lineno)d] >> %(message)s"
)

H6_BENCHMARKS = ["ARC", "Hellaswag", "MMLU", "TruthfulQA", "Winogrande", "GSM8k"]


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


def save_figure(score_df, benchmarks_list, figure_path):
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
    fig.write_image(figure_path, scale=2)


def update_db(output_path="../results", db_path="./db", git_pull=False):
    if git_pull:
        import git

        repo = git.Repo("../")
        repo.remotes.origin.pull()

    model_list = sorted(os.listdir(output_path), key=str.lower)
    bench_list = H6_BENCHMARKS

    values = []
    for model_name in model_list:
        h6_en_scores = get_h6_en_scores(f"../results/{model_name}/h6_en")
        values.append([model_name] + h6_en_scores)

    score_df = pd.DataFrame(values, columns=["Model"] + bench_list)
    score_df.to_csv(os.path.join(db_path, "score_df.csv"), index=False)


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
