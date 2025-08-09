def match_judge_metric(example, pred, trace=None):
    example_str = str(example.satisfied).lower().strip()
    # this is going to be True or False
    pred_str = str(pred.satisfied).lower().strip()

    if example_str == pred_str:
        return 1
    else:
        return 0


class LLMJudgeMetric:
    def __init__(self, optimized_judge_program):
        self.optimized_judge_program = optimized_judge_program

    def __call__(self, example, pred, trace=None):
        # Ensure you pass the transcript the judge needs.
        # If your generate_reasoning produces pred.transcript, use that; otherwise map appropriately.
        # Common patterns: example.transcript, pred.transcript, pred.answer — adapt to your pipeline.
        transcript_text = getattr(pred, "transcript", None) or getattr(
            example, "transcript", None
        )
        if not transcript_text:
            # Fallback or raise; metric must be deterministic
            return False

        judged = self.optimized_judge_program(transcript=transcript_text)
        # judged.satisfied is expected to be a bool with JSONAdapter or a string convertible to bool
        return bool(judged.satisfied)
