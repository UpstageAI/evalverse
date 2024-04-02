"""
Copyright (c) 2024-present Upstage Co., Ltd.
Apache-2.0 license
"""
import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path

from evalverse.connector import (
    eq_bench,
    fastchat_llm_judge,
    instruction_following_eval,
    lm_evaluation_harness,
)
from evalverse.reporter import AVAILABLE_BENCHMARKS
from evalverse.utils import EVALVERSE_LOG_FORMAT, EVALVERSE_OUTPUT_PATH, get_logger

logging.basicConfig(format=EVALVERSE_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


class Evaluator:
    def __init__(self, mode="lib", log_path=None):
        self.mode = mode  # lib or cli
        self.logger = get_logger(log_path)

    def get_args(self):
        parser = ArgumentParser()

        # Common Args
        parser.add_argument("--ckpt_path", type=str, default="upstage/SOLAR-10.7B-Instruct-v1.0")
        parser.add_argument("--output_path", type=str, default=EVALVERSE_OUTPUT_PATH)
        parser.add_argument("--model_name", type=str, help="using in save_path")
        parser.add_argument("--use_fast_tokenizer", action="store_true", default=False)
        parser.add_argument("--devices", type=str, default="0", help="The size of data parallel.")
        parser.add_argument("--use_flash_attention_2", action="store_true", default=False)

        # lm-evaluation-harness
        parser.add_argument("--h6_en", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--use_vllm", action="store_true", default=False)
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
        parser.add_argument(
            "--model_parallel", type=int, default=1, help="The size of model parallel"
        )
        parser.add_argument(
            "--data_parallel", type=int, default=1, help="The size of data parallel"
        )
        parser.add_argument("--load_in_8bit", action="store_true", default=False)
        parser.add_argument("--load_in_4bit", action="store_true", default=False)

        # FastChat
        parser.add_argument("--mt_bench", action="store_true", default=False)
        parser.add_argument("--baselines", type=str, default=None)
        parser.add_argument("--judge_model", type=str, default="gpt-4")
        parser.add_argument(
            "--num_gpus_total", type=int, default=1, help="The total number of GPUs."
        )
        parser.add_argument(
            "--num_gpus_per_model", type=int, default=1, help="The number of GPUs per model."
        )
        parser.add_argument(
            "--parallel_api", type=int, default=1, help="The number of concurrent API calls."
        )

        # Instruction Following Eval
        parser.add_argument("--ifeval", action="store_true", default=False)
        parser.add_argument(
            "--gpu_per_inst_eval", type=int, default=1, help="The number of GPUs per model."
        )

        # EQ-Bench
        parser.add_argument("--eq_bench", action="store_true", default=False)
        parser.add_argument("--eq_bench_prompt_type", type=str, default="ChatML")
        parser.add_argument("--eq_bench_lora_path", type=str, default=None)
        parser.add_argument(
            "--eq_bench_quantization", type=str, default=None, choices=["8bit", "4bit", None]
        )

        if self.mode == "lib":
            args = parser.parse_args(args=[])
        elif self.mode == "cli":
            args = parser.parse_args()

        # update path to work regardless of /
        args.ckpt_path = str(Path(args.ckpt_path))
        args.output_path = str(Path(args.output_path))

        # handle model name
        if args.model_name is None:
            args.model_name = args.ckpt_path.split("/")[-1]

        # change relative path to absolute path
        if not os.path.isabs(args.output_path):
            args.output_path = os.path.abspath(args.output_path)

        return args

    def update_args(self, args, model, benchmark, kwargs):
        for k, v in kwargs.items():
            if k in args:
                setattr(args, k, v)
                self.logger.info(f'The value of argument "{k}" has been changed to "{v}".')
            else:
                self.logger.warning(f'The argument "{k}" does not exist.')
        if model:
            args.ckpt_path = model
        if benchmark:
            if benchmark == "all":
                benchmark = AVAILABLE_BENCHMARKS
                self.logger.info(f"All available benchmarks are selected: {AVAILABLE_BENCHMARKS}")
            if benchmark in AVAILABLE_BENCHMARKS:
                setattr(args, benchmark, True)
                self.logger.info(f'The value of argument "{benchmark}" has been changed to "True".')
            elif type(benchmark) == list:
                for b in benchmark:
                    if b in AVAILABLE_BENCHMARKS:
                        setattr(args, b, True)
                        self.logger.info(f'The value of argument "{b}" has been changed to "True".')
                    else:
                        raise ValueError(
                            f'"{b}" is not in Available_Benchmarks: {AVAILABLE_BENCHMARKS}'
                        )
            else:
                raise ValueError(
                    f'"{benchmark}" is not in Available_Benchmarks: {AVAILABLE_BENCHMARKS}'
                )
        else:
            self.logger.info(
                f"No selected benchmarks. Available_Benchmarks: {AVAILABLE_BENCHMARKS}"
            )
        self.logger.info(f"Args {vars(args)}")

        return args

    def run(self, model: str = None, benchmark: str | list = None, **kwargs):
        # update args
        args = self.get_args()
        args = self.update_args(args, model, benchmark, kwargs)

        # h6_en (with lm-evaluation-harness)
        if args.h6_en:
            task_and_shot = [
                ("arc_challenge", 25),
                ("hellaswag", 10),
                ("mmlu", 5),
                ("truthfulqa_mc2", 0),
                ("winogrande", 5),
                ("gsm8k", 5),
            ]
            model_name = args.ckpt_path.split("/")[-1]
            h6_en_output_path = os.path.join(args.output_path, model_name, "h6_en")
            for _task_name, _num_fewshot in task_and_shot:
                start_time = time.time()
                #############################################
                lm_evaluation_harness(
                    model_path=args.ckpt_path,
                    tasks=_task_name,
                    batch_size=args.batch_size,
                    use_vllm=args.use_vllm,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    tensor_parallel_size=args.model_parallel,
                    data_parallel_size=args.data_parallel,
                    num_fewshot=_num_fewshot,
                    use_fast_tokenizer=args.use_fast_tokenizer,
                    use_flash_attention_2=args.use_flash_attention_2,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                    output_path=h6_en_output_path,
                )
                #############################################
                end_time = time.time()
                total_min = round((end_time - start_time) / 60)
                bench_name = _task_name + "_" + str(_num_fewshot) + "shot"
                self.logger.info(
                    f"{bench_name} done! exec_time: {total_min} min for {args.ckpt_path}"
                )
        # mt_bench (with evalverse-FastChat)
        if args.mt_bench:
            if "OPENAI_API_KEY" not in os.environ:
                self.logger.warning("No OPENAI_API_KEY provided. Please add it.")
            start_time = time.time()
            #############################################
            fastchat_llm_judge(
                model_path=args.ckpt_path,
                model_id=args.model_name,
                mt_bench_name="mt_bench",
                baselines=args.baselines,
                judge_model=args.judge_model,
                num_gpus_per_model=args.num_gpus_per_model,
                num_gpus_total=args.num_gpus_total,
                parallel_api=args.parallel_api,
                output_path=args.output_path,
            )
            #############################################
            end_time = time.time()
            total_min = round((end_time - start_time) / 60)
            bench_name = "mt_bench"
            self.logger.info(f"{bench_name} done! exec_time: {total_min} min for {args.ckpt_path}")

        # ifeval (with evalverse-IFEval)
        if args.ifeval:
            start_time = time.time()
            #############################################
            instruction_following_eval(
                model_path=args.ckpt_path,
                model_name=args.model_name,
                gpu_per_inst_eval=args.gpu_per_inst_eval,
                devices=args.devices,
                output_path=args.output_path,
            )
            #############################################
            end_time = time.time()
            total_min = round((end_time - start_time) / 60)
            bench_name = "ifeval"
            self.logger.info(f"{bench_name} done! exec_time: {total_min} min for {args.ckpt_path}")

        # eq_bench (with evalverse-EQBench)
        if args.eq_bench:
            start_time = time.time()
            #############################################
            eq_bench(
                model_name=args.model_name,
                prompt_type=args.eq_bench_prompt_type,
                model_path=args.ckpt_path,
                lora_path=args.eq_bench_lora_path,
                quantization=args.eq_bench_quantization,
                devices=args.devices,
                use_fast_tokenizer=args.use_fast_tokenizer,
                use_flash_attention_2=args.use_flash_attention_2,
                output_path=args.output_path,
            )
            #############################################
            end_time = time.time()
            total_min = round((end_time - start_time) / 60)
            bench_name = "eq_bench"
            self.logger.info(f"{bench_name} done! exec_time: {total_min} min for {args.ckpt_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)

    evaluator_cli = Evaluator(mode="cli")
    evaluator_cli.run()
