"""
Copyright (c) 2024-present Upstage Co., Ltd.
Apache-2.0 license
"""
import json
import os

from evalverse.utils import EVALVERSE_MODULE_PATH, print_command, print_txt_file


def lm_evaluation_harness(
    model_path="upstage/SOLAR-10.7B-Instruct-v1.0",
    tasks="arc_challenge",
    batch_size=16,
    use_vllm=False,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=1,
    data_parallel_size=1,
    num_fewshot=0,
    use_fast_tokenizer=False,
    use_flash_attention_2=False,
    load_in_8bit=False,
    load_in_4bit=False,
    output_path="../results",
):
    output_json_path = os.path.join(output_path, f"{tasks}_{num_fewshot}.json")

    if not os.path.exists(output_json_path):
        if use_vllm:
            tokenizer_mode = "auto" if use_fast_tokenizer else "slow"
            eval_cmd = f"""
            lm_eval --model vllm \
                --model_args pretrained={model_path},trust_remote_code=True,tensor_parallel_size={tensor_parallel_size},dtype=float16,gpu_memory_utilization={gpu_memory_utilization},data_parallel_size={data_parallel_size},tokenizer_mode={tokenizer_mode} \
                --tasks {tasks} \
                --batch_size {batch_size} \
                --num_fewshot {num_fewshot} \
                --output_path {output_json_path} \
            """
        else:
            hf_cmd = "lm_eval --model hf"
            model_args = f"pretrained={model_path},trust_remote_code=True,dtype=float16,use_fast_tokenizer={use_fast_tokenizer},use_flash_attention_2={use_flash_attention_2}"

            if data_parallel_size > 1:
                hf_cmd = "accelerate launch -m " + hf_cmd
            if tensor_parallel_size > 1:
                model_args = model_args + ",parallelize=True"
            if load_in_8bit:
                model_args = model_args + ",load_in_8bit=True"
            if load_in_4bit:
                model_args = model_args + ",load_in_4bit=True"

            eval_cmd = f"""
            NCCL_P2P_DISABLE=1 {hf_cmd} \
                --model_args {model_args}  \
                --tasks {tasks} \
                --batch_size {batch_size} \
                --num_fewshot {num_fewshot} \
                --output_path {output_json_path} \
            """
        print_command(eval_cmd)
        os.system(eval_cmd)

    else:
        print(f"The result already exists: {os.path.abspath(output_json_path)}")


def fastchat_llm_judge(
    model_path="upstage/SOLAR-10.7B-Instruct-v1.0",
    model_id="SOLAR-10.7B-Instruct-v1.0",
    mt_bench_name="mt_bench",
    baselines=None,
    judge_model="gpt-4",
    num_gpus_per_model=1,
    num_gpus_total=1,
    parallel_api=1,
    output_path="../results",
):
    scores_file = os.path.join(output_path, model_id, "mt_bench", "scores.txt")

    if not os.path.exists(scores_file):
        if baselines:
            model_list = " ".join([model_id] + baselines.split(","))
        else:
            model_list = model_id

        eval_code_path = os.path.join(
            EVALVERSE_MODULE_PATH, "submodules/FastChat/fastchat/llm_judge"
        )
        answer_path = os.path.join(output_path, model_id, "mt_bench", "model_answer")
        answer_file = os.path.join(answer_path, f"{model_id}.jsonl")
        judgement_path = os.path.join(output_path, model_id, "mt_bench", "model_judgment")
        judgement_file = os.path.join(judgement_path, "gpt-4_single.jsonl")

        gen_answer_cmd = f"python3 gen_model_answer.py --model-path {model_path} --model-id {model_id} --bench-name {mt_bench_name} --answer-file {answer_file} --num-gpus-per-model {num_gpus_per_model} --num-gpus-total {num_gpus_total}"
        gen_judgment_cmd = f"echo -e '\n' | python3 gen_judgment.py --model-list {model_list} --bench-name {mt_bench_name} --model-answer-dir {answer_path} --model-judgement-dir {judgement_path} --judge-model {judge_model} --parallel {parallel_api}"
        save_result_cmd = f"python3 show_result.py --model-list {model_list} --bench-name {mt_bench_name} --judge-model {judge_model} --input-file {judgement_file} > {os.path.join(output_path, model_id, 'mt_bench', 'scores.txt')}"

        eval_cmd = f"cd {eval_code_path}"
        if not os.path.exists(answer_file):
            eval_cmd += f" && {gen_answer_cmd}"
        if not os.path.exists(judgement_file):
            eval_cmd += f" && {gen_judgment_cmd}"
        eval_cmd += f" && {save_result_cmd}"
        print_command(eval_cmd)
        os.system(eval_cmd)
    else:
        print(f"The result already exists: {os.path.abspath(scores_file)}")
    # print results
    print_txt_file(scores_file)


def instruction_following_eval(
    model_path="upstage/SOLAR-10.7B-Instruct-v1.0",
    model_name="SOLAR-10.7B-Instruct-v1.0",
    gpu_per_inst_eval=1,
    devices="0",
    output_path="../results",
):
    scores_file = os.path.join(output_path, model_name, "ifeval", "scores.txt")

    if not os.path.exists(scores_file):
        eval_code_path = os.path.join(os.path.join(EVALVERSE_MODULE_PATH, "submodules/IFEval"))

        eval_cmd = f"""
        cd {eval_code_path} && python3 inst_eval.py \
            --model {model_path} \
            --model_name {model_name} \
            --gpu_per_inst_eval {gpu_per_inst_eval} \
            --output_path {output_path} \
            --devices {devices}
        """
        print_command(eval_cmd)
        os.system(eval_cmd)
    else:
        print(f"The result already exists: {os.path.abspath(scores_file)}")
    # print results
    print_txt_file(scores_file)


def eq_bench(
    model_name="SOLAR-1-10.7B-dev1.0-chat1.1.3.f8-enko",  # model name for saving results
    prompt_type="ChatML",  # Chat template
    model_path="/data/project/public/checkpoints/SOLAR-1-10.7B-dev1.0-chat1.1.3.f8-enko/",  # model path
    lora_path=None,  # lora adapter path
    quantization=None,  # quantization, [None, "8bit", "4bit"] for load_in_8bit etc.
    n_iterations=1,  # number of iterations to repeat the inference
    devices="0",  # cuda devices
    use_fast_tokenizer=False,  # use fast tokenizer
    gpu_per_proc=1,  # gpu per process, currently only supports 1
    use_flash_attention_2=True,  # use flash attention 2
    torch_dtype="b16",  # torch dtype, [b16, f16, f32]
    output_path="../results",  # output path
):
    result_file = os.path.join(output_path, model_name, "eq_bench", "raw_results.json")
    if not os.path.exists(result_file):
        assert gpu_per_proc == 1, "Currently only supports 1 gpu per process"

        eval_code_path = os.path.join(os.path.join(EVALVERSE_MODULE_PATH, "submodules/EQBench"))
        single_eval_code = f"""
        CUDA_VISIBLE_DEVICES={devices} python3 eq-bench.py --model_name {model_name} --prompt_type {prompt_type} \
            --model_path {model_path} --quantization {quantization} --n_iterations {n_iterations} \
            --gpu_per_proc {gpu_per_proc} --torch_dtype {torch_dtype} --output_path {output_path} \
            --devices {devices}"""
        if use_fast_tokenizer:
            single_eval_code += " --use_fast_tokenizer"
        if use_flash_attention_2:
            single_eval_code += " --use_flash_attention_2"
        if lora_path is not None:
            single_eval_code += f" --lora_path {lora_path}"

        eval_cmd = f"""
        cd {eval_code_path} && {single_eval_code}
        """
        print_command(eval_cmd)
        os.system(eval_cmd)
    else:
        print(f"The result already exists: {os.path.abspath(result_file)}")
    # print results
    with open(result_file, "r") as f:
        data = json.load(f)
    result = data[list(data.keys())[0]]["iterations"]["1"]["benchmark_results_fullscale"]
    print(json.dumps(result, indent=4))
