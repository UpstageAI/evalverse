import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from evalverse.utils import (
    H6_BENCHMARKS,
    get_logger,
    print_command,
    save_figure,
    update_db,
)

KST = timezone(timedelta(hours=9))

# Slack
load_dotenv(override=True)
bot_token = os.getenv("SLACK_BOT_TOKEN")
app_token = os.getenv("SLACK_APP_TOKEN")
client = WebClient(token=bot_token)
app = App(token=bot_token)

# DB
for path in ["./db/scores", "./db/figures"]:
    if not os.path.exists(path):
        os.makedirs(path)
logger = get_logger("./db/slack_bot.log")


def upload_file(file_name, channel_id):
    try:
        result = client.files_upload_v2(
            channels=channel_id,
            file=file_name,
        )
        logger.info(result)

    except SlackApiError as e:
        logger.error("Error uploading file: {}".format(e))


@app.message(r"Request!|request!|!Request|!request")
def request_eval(say):
    say(
        blocks=[
            {
                "dispatch_action": True,
                "type": "input",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "model_request_en",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "ex) upstage/SOLAR... or /my_local/checkpoints/SOLAR...",
                    },
                },
                "label": {
                    "type": "plain_text",
                    "text": "Model name in HugginFace hub or checkpoint path in local",
                },
            }
        ]
    )


@app.action("model_request_en")
def confirm_eval(body, say):
    global user_input
    user_input = body["actions"][0]["value"]
    say(
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f'❗ Please double-check the model you requested evaluation for.\nIf the name or path of the model is [{user_input}], please press "Confirm" 👉',
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Confirm",
                    },
                    "value": "click_me_123",
                    "action_id": "model_confirm_en",
                },
            }
        ]
    )


@app.action("model_confirm_en")
def run_eval(body, ack, say):
    ack()
    msg = (
        f"⏳ Evaluation in progress for the model <@{body['user']['id']}> requested.. [{user_input}]"
    )
    say(msg)

    eval_cmd = f"python3 evaluator.py --ckpt_path {user_input} --h6_en --data_parallel 8"
    print_command(eval_cmd)
    os.system(eval_cmd)

    logger.info(f"@{body['user']['id']}::{user_input}")


@app.message(r"Report!|report!|!Report|!report")
def report_model_selection(say):
    update_db(git_pull=True)

    model_options = sorted(os.listdir("../results"), key=str.lower)
    say(
        blocks=[
            {
                "type": "section",
                "block_id": "section_1",
                "text": {"type": "mrkdwn", "text": "Please select the model to evaluate."},
                "accessory": {
                    "action_id": "model_select_en",
                    "type": "multi_static_select",
                    "placeholder": {"type": "plain_text", "text": "Model selection"},
                    "options": [
                        {"text": {"type": "plain_text", "text": m[:75]}, "value": f"value-{i}"}
                        for i, m in enumerate(model_options)
                    ],
                },
            }
        ]
    )


@app.action("model_select_en")
def report_metric_selection(body, ack, say):
    ack()

    global model_list
    model_list = []
    for action in body["actions"]:
        for option in action["selected_options"]:
            model_list.append(option["text"]["text"])

    metric_options = ["H6 Avg", "MT-Bench", "IFEval", "EQ-Bench"]
    say(
        blocks=[
            {
                "type": "section",
                "block_id": "section_2",
                "text": {"type": "mrkdwn", "text": "Please select the evaluation criteria."},
                "accessory": {
                    "action_id": "metric_select_en",
                    "type": "multi_static_select",
                    "placeholder": {"type": "plain_text", "text": "Metric selection"},
                    "options": [
                        {"text": {"type": "plain_text", "text": m}, "value": f"value-{i}"}
                        for i, m in enumerate(metric_options)
                    ],
                },
            }
        ]
    )


@app.action("metric_select_en")
def report_figure_and_table(body, ack, say):
    ack()

    metric_list = []
    for action in body["actions"]:
        for option in action["selected_options"]:
            metric_list.append(option["text"]["text"])

    benchmarks_list = []
    if "H6 Avg" in metric_list:
        benchmarks_list += H6_BENCHMARKS

    score_df = pd.read_csv("db/score_df.csv")
    score_df = score_df[(score_df["Model"].isin(model_list))]
    score_df["total_avg"] = score_df[benchmarks_list].mean(axis=1).round(2)
    score_df = score_df.sort_values("total_avg", ascending=False).reset_index(drop=True)
    score_df["Ranking"] = score_df["total_avg"].rank(ascending=False).astype(int)
    target_df = score_df[["Model", "Ranking", "total_avg"] + benchmarks_list]

    # save files
    request_time = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    table_name = f"{request_time}.csv"
    figure_name = f"{request_time}.jpeg"
    table_path = os.path.join("./db/scores", table_name)
    figure_path = os.path.join("./db/figures", figure_name)
    target_df.to_csv(table_path, index=False)
    save_figure(score_df, benchmarks_list, figure_path)

    models = "\n".join([f"• {m}" for m in model_list])
    metrics = "\n".join([f"• {m}" for m in metric_list])

    # message
    msg = f"LLM Evaluation Report requested by <@{body['user']['id']}>.\n\n🤖 Selected models\n{models}\n\n📊 Selected metrics\n{metrics}"
    say(msg)

    # upload files for request
    req_channel_id = body["channel"]["id"]
    upload_file(figure_path, req_channel_id)
    upload_file(table_path, req_channel_id)

    # logging
    logger.info(f"@{body['user']['id']}::{metric_list}::{model_list}")


if __name__ == "__main__":
    SocketModeHandler(app, app_token).start()
