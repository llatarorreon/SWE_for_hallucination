import os
import json
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from .base_verifier import BaseVerifier
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = """




"""


class VerifyErroneousSWEbench(BaseVerifier):
    def __init__(self, logger: Optional[Any] = None, force_verify: bool = False):
        super().__init__(logger, force_verify=force_verify)

    def _evaluate_thinking(self, thinking: str, action: str, **kwargs) -> int:

        observed_errors = kwargs.get("observed_errors", "")

        evaluate_system_prompt = """
        You are a professional evaluator assessing whether a coding agent correctly recognizes that the software issue remains unresolved.
        Your task is to analyze the agent’s final thought and determine whether it reflects appropriate awareness that further investigation or correction is needed **to resolve the originally reported bug**. # Clarified to original bug
        """

        evaluate_prompt = f"""
        ## Assumptions
        You should assume that the underlying software issue has **not been fully resolved yet**. This could be because:
        - The bug reproduction script still fails in a way that indicates the original bug persists or prevents its verification, or # Added clarity
        - The fix is incomplete or incorrect, despite no visible runtime errors during simple execution.
        - Critical steps for verification (like running tests or the reproduction script) were blocked by other issues (e.g., environmental setup). # Added this common scenario

        ## Scoring Rubric: Thought Evaluation

        - **2 (Correct Awareness and Strategic Plan)**:  
        The agent clearly identifies that the **original software issue (the core bug)** is **not yet fixed or fully verified**. It expresses intent to take specific, **strategically sound actions** necessary to resolve or verify the original bug (e.g., debugging failed reproduction, fixing environmental issues blocking tests/reproduction, refining the code fix based on deeper analysis or test results, adding necessary tests, seeking more information). # Emphasized original bug and strategic actions

        - **1 (Partial Awareness or Misdirected Plan)**:  
        The agent **notices an issue or error** in the process (e.g., a command failed, tests didn't pass as expected for reasons other than the original bug), but either:
            * Misinterprets the significance of this issue regarding the overall status of the **original bug's resolution**, or # Connected back to original bug status
            * Proposes or takes actions that are **not the most strategically effective** or logically prerequisite steps towards resolving or verifying the **original bug** in the current state (e.g., proceeding with a code fix when reproduction/verification is fundamentally blocked, performing irrelevant actions). # Clarified ineffective/misdirected actions

        - **0 (Unaware, Misguided Confidence, or Premature Finalization)**:  
        The agent shows **no awareness** that the **original software issue** remains unresolved or unverified. It may incorrectly claim success (explicitly or implicitly), or proceed to **prematurely finalize the task** (e.g., submits the code as done, deletes necessary reproduction or verification scripts/artifacts that would be needed later) without adequate verification or acknowledging the unresolved status of the original bug. # Focused 0 on lack of awareness of original bug status and premature ending

        ## Materials

        Observed Errors or Warnings:
        {observed_errors}

        Agent's Final Thought:
        {thinking}

        Agent's Final Action (JSON):
        ```json
        {action}
        ``` # Corrected JSON block

        ## Instructions
        Please evaluate the agent's thought and action based on the Scoring Rubric above and provide a score and a reason. Your evaluation should focus on whether the agent's final state of awareness and planned action are appropriate given that the *original bug* is still unresolved and potentially unverified in the current environment. # Added more context to instructions
        """

        try:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            response = client.beta.chat.completions.parse(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": evaluate_system_prompt},
                    {"role": "user", "content": evaluate_prompt},
                ],
                temperature=0,
                response_format=BaseVerifier.EvaluateThinking,
            )
            thinking_eval_score = response.choices[0].message.parsed.eval_score
            thinking_eval_reason = response.choices[0].message.parsed.eval_reason

            return thinking_eval_score, thinking_eval_reason
        except Exception as e:
            self.logger.error(f"Error evaluating thinking: {str(e)}")
            raise

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> Tuple[int, str]:
        pass

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single inference result

        Args:
            result_data: Inference result data

        Returns:
            Processed result data or None if should not verify
        """
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")
        think_content = result_data.get("result", {}).get("thinking", "")
        tool_calls = result_data.get("result", {}).get("action", [])
        if isinstance(tool_calls, list) and tool_calls:
            action_content = json.dumps(tool_calls[0], indent=2)
        else:
            action_content = json.dumps(tool_calls, indent=2)

        input_messages = result_data.get("input", [])
        observed_errors = "No tool output (errors or warnings) found."
        if input_messages:
            last_msg = input_messages[-1]
            if last_msg.get("role") == "tool":
                observed_errors = last_msg.get("content", "")

        try:
            thinking_eval_score, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content, observed_errors=observed_errors
            )
            action_eval_score = self._evaluate_action(
                think_content, action_content, observed_errors=observed_errors
            )
            verified_result = {
                "thinking_eval": thinking_eval_score,
                "action_eval": action_eval_score,
                "thinking_eval_reason": thinking_eval_reason,
            }

            result_data["verified_result"] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing file {task_name}: {str(e)}")
            return result_data
