# Evalverse
> The Universe of Evaluation. All about the evaluation for LLMs.


## 🌌 Submodule
> The Submodule serves as the evaluation engine that is responsible for the heavy lifting involved in evaluating LLMs. Publicly available LLM evaluation libraries can be integrated into Evalverse as submodules. This component makes Evalverse expandable, thereby ensuring that the library remains up-to-date. 

## 🌌 Connector
> The Connector plays a role in linking the Submodules with the Evaluator. It contains evaluation scripts, along with the necessary arguments, from various external libraries.

## 🌌 Evaluator
> The Evaluator performs the requested evaluations on the Compute Cluster by utilizing the evaluation scripts from the Connector. The Evaluator can receive evaluation requests either from the Reporter, which facilitates a no-code evaluation approach, or directly from the end-user for code-based evaluation.

## 🌌 Reporter
> The Reporter handles the evaluation and report requests sent by the users, allowing for a no-code approach to LLM evaluation. The Reporter sends the requested evaluation jobs to the Evaluator and fetches the evaluation results from the Database, which are sent to the user via an external communication platform such as Slack. Through this, users can receive table and figure that summarize evaluation results.