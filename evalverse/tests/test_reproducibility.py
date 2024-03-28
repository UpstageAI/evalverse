import os
import unittest

from evalverse.evaluator import Evaluator
from evalverse.utils import get_h6_en_scores

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(mode="lib")

    def test_run_all_benchmarks(self):
        model = "upstage/SOLAR-10.7B-Instruct-v1.0"
        benchmark = "all"
        original_output_path = os.path.join(TEST_PATH, "test_results")
        reproduced_output_path = os.path.join(TEST_PATH, "test_results_reproduced")
        self.evaluator.run(
            model=model,
            benchmark=benchmark,
            data_parallel=8,
            num_gpus_total=8,
            parallel_api=4,
            devices="0,1,2,3,4,5,6,7",
            output_path=reproduced_output_path,
        )

        # h6_score reproducilbility check
        model_name = model.split("/")[-1]
        original_scores = get_h6_en_scores(os.path.join(original_output_path, model_name, "h6_en"))
        original_stderr = get_h6_en_scores(
            os.path.join(original_output_path, model_name, "h6_en"), stderr=True
        )
        reproduced_scores = get_h6_en_scores(
            os.path.join(reproduced_output_path, model_name, "h6_en")
        )

        h6_list = ["arc_c_25", "hellaswag_10", "mmlu_5", "truthfulqa_0", "winogrande_5", "gsm8k_5"]
        for benchmark, original, stderr, reproduced in zip(
            h6_list, original_scores, original_stderr, reproduced_scores
        ):
            difference = abs(original - reproduced)
            print(
                f"[{benchmark}] \t original: {original} \t reproduced: {reproduced} \t difference: {round(difference, 2)} \t stderr: {stderr}"
            )
            self.assertLessEqual(difference, stderr)


if __name__ == "__main__":
    unittest.main()
