import sys
sys.path.append('/opt/huawei/explorer-env/dataset/qjh_train/code/verl-tool/verl/verl/utils/reward_score')
def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == 'videocaption_cgbench':
        import get_caption_more_smooth_multi_score
        res = get_caption_more_smooth_multi_score.compute_score_linear_addwanswer(solution_str, ground_truth, extra_info)
    elif data_source == 'videocaption_videomme':
        import get_caption_more_smooth_multi_score
        res = get_caption_more_smooth_multi_score.compute_score_only_answer(solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])