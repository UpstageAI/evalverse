"""
Copyright (c) 2024-present Upstage Co., Ltd.
Apache-2.0 license
"""
import os

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from evalverse.reporter import AVAILABLE_BENCHMARKS, Reporter
from evalverse.utils import EVALVERSE_DB_PATH, EVALVERSE_OUTPUT_PATH, get_logger

# Slack
load_dotenv(override=True)
bot_token = os.getenv("SLACK_BOT_TOKEN")
app_token = os.getenv("SLACK_APP_TOKEN")
client = WebClient(token=bot_token)
app = App(token=bot_token)

# Reporter
reporter = Reporter(db_path=EVALVERSE_DB_PATH, output_path=EVALVERSE_OUTPUT_PATH)

# Logger
logger = get_logger(os.path.join(EVALVERSE_DB_PATH, "slack_bot.log"))


def send_msg(msg, channel_id):
    try:
        result = client.chat_postMessage(channel=channel_id, text=msg)
        logger.info(result)

    except SlackApiError as e:
        logger.error(f"Error posting message: {e}")


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
def request_eval(ack, body, say, logger):
    ack()
    logger.info(body)
    say(
        text="",
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
        ],
    )


@app.action("model_request_en")
def confirm_eval(ack, body, say, logger):
    ack()
    logger.info(body)

    global user_input
    user_input = body["actions"][0]["value"]
    say(
        text="",
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
        ],
    )


@app.action("model_confirm_en")
def run_eval(ack, body, say, logger):
    ack()
    logger.info(body)

    # Start
    start_msg = (
        f"⏳ Evaluation in progress for the model <@{body['user']['id']}> requested.. [{user_input}]"
    )
    say(start_msg)

    # Run an evaluation
    from evalverse import Evaluator

    evaluator = Evaluator()
    evaluator.run(model=user_input, benchmark="all")

    # End
    req_channel_id = body["channel"]["id"]
    complete_msg = f"Done! <@{body['user']['id']}>\n[{user_input}] is added."
    send_msg(complete_msg, req_channel_id)

    logger.info(f"@{body['user']['id']}::{user_input}")


@app.message(r"Report!|report!|!Report|!report")
def report_model_selection(ack, body, say, logger):
    ack()
    logger.info(body)

    reporter.update_db(save=True, git_fetch=False)

    model_options = sorted(os.listdir(EVALVERSE_OUTPUT_PATH), key=str.lower)
    say(
        text="",
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
        ],
    )


@app.action("model_select_en")
def report_bench_selection(ack, body, say, logger):
    ack()
    logger.info(body)

    global model_list
    model_list = []
    for action in body["actions"]:
        for option in action["selected_options"]:
            model_list.append(option["text"]["text"])

    say(
        text="",
        blocks=[
            {
                "type": "section",
                "block_id": "section_2",
                "text": {"type": "mrkdwn", "text": "Please select the evaluation criteria."},
                "accessory": {
                    "action_id": "bench_select_en",
                    "type": "multi_static_select",
                    "placeholder": {"type": "plain_text", "text": "Metric selection"},
                    "options": [
                        {"text": {"type": "plain_text", "text": m}, "value": f"value-{i}"}
                        for i, m in enumerate(AVAILABLE_BENCHMARKS)
                    ],
                },
            }
        ],
    )


@app.action("bench_select_en")
def report_figure_and_table(ack, body, say, logger):
    ack()
    logger.info(body)

    bench_list = []
    for action in body["actions"]:
        for option in action["selected_options"]:
            bench_list.append(option["text"]["text"])

    table_path, figure_path = reporter.run(
        model_list=model_list, benchmark_list=bench_list, save=True
    )

    models = "\n".join([f"• {m}" for m in model_list])
    benchs = "\n".join([f"• {m}" for m in bench_list])

    # message
    msg = f"LLM Evaluation Report requested by <@{body['user']['id']}>.\n\n🤖 Selected models\n{models}\n\n📊 Selected benchmarks\n{benchs}"
    say(msg)

    # upload files for request
    req_channel_id = body["channel"]["id"]
    upload_file(figure_path, req_channel_id)
    upload_file(table_path, req_channel_id)

    # logging
    logger.info(f"@{body['user']['id']}::{bench_list}::{model_list}")


if __name__ == "__main__":
    SocketModeHandler(app, app_token).start()
