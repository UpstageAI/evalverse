import os
import unittest

from evalverse.evaluator import Evaluator

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(mode="lib")

    def test_get_args_default(self):
        args = self.evaluator.get_args()
        self.assertEqual(args.ckpt_path, "upstage/SOLAR-10.7B-Instruct-v1.0")

    def test_run_args_overriding(self):
        your_model = "your/Model"
        your_output_path = "/your/output_path"
        self.evaluator.run(model=your_model, output_path=your_output_path)
        self.assertEqual(self.evaluator.args.ckpt_path, your_model)
        self.assertEqual(self.evaluator.args.output_path, your_output_path)

    def test_run_h6_en_existing(self):
        benchmark = "h6_en"
        output_path = os.path.join(TEST_PATH, "test_results")
        self.evaluator.run(benchmark=benchmark, output_path=output_path)


if __name__ == "__main__":
    unittest.main()
