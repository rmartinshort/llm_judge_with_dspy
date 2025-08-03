def match_judge_metric(example, pred, trace=None):
    example_str = str(example.satisfied).lower().strip()
    # this is going to be True or False
    pred_str = str(pred.satisfied).lower().strip()

    if example_str == pred_str:
        return 1
    else:
        return 0