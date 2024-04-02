import os
import unittest

from evalverse.reporter import Reporter

TEST_PATH = os.path.dirname(os.path.abspath(__file__))


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        output_path = os.path.join(TEST_PATH, "test_results")
        self.reporter = Reporter(output_path=output_path)

    def test_update_db(self):
        self.reporter.update_db()

    def test_run(self):
        model_list = ["SOLAR-10.7B-Instruct-v1.0"]
        benchmark_list = ["h6_en"]
        self.reporter.run(model_list=model_list, benchmark_list=benchmark_list)


if __name__ == "__main__":
    unittest.main()
