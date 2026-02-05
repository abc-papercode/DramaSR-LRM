# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import ast
import string

def interval_length(iv):
    return max(0.0, iv[1] - iv[0])

def intersection(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def total_intersection_with_gt(visited_intervals, gt_intervals):
    # 计算 visited 与 GT 的总交集长度
    total = 0.0
    for v in visited_intervals:
        for g in gt_intervals:
            total += intersection(v, g)
    return total

def union_visited_length(visited_intervals):
    # 合并 visited_intervals 的长度（防重复计数）
    if not visited_intervals:
        return 0.0
    segs = sorted(visited_intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = segs[0]
    for s,e in segs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(interval_length(x) for x in merged)

def gt_total_length(gt_intervals):
    return sum(interval_length(g) for g in gt_intervals)


def merge_intervals(intervals):
    if not intervals: return []
    if len(intervals) == 1: return intervals
    intervals = sorted(intervals, key=lambda x: x[0])
    id_list = [0 for _ in range(len(intervals))]
    id_init = 0
    for i in range(len(intervals) - 1):
        if intervals[i+1][0] - intervals[i][1] <= 5:
            id_list[i+1] = id_init
        else:
            id_init += 1
            id_list[i+1] = id_init
    
    merged = []
    for i in range(id_init + 1):
        cur_intervals = [intervals[j] for j in range(len(intervals)) if id_list[j] == i]
        merged.append([min([x[0] for x in cur_intervals]), max([x[1] for x in cur_intervals])])
    return merged

# get id
def get_id(std, duration, width):
    L_high = duration / (width)
    L_med  = duration / (width**2)
    L_low  = duration / (width**3)
    high_id = int(std // L_high) + 1
    med_id = int(std % L_high // L_med) + 1
    low_id = int(std % L_high % L_med // L_low) + 1
    return (high_id, med_id, low_id)

# get covered ids
def get_covered_ids(std_id, end_id, width):
    """
    返回从 std_id 到 end_id（包含）之间所有 (high, med, low) id 的列表
    """
    covered = []
    for h in range(std_id[0], end_id[0]+1):
        med_start = std_id[1] if h == std_id[0] else 1
        med_end = end_id[1] if h == end_id[0] else width
        for m in range(med_start, med_end+1):
            low_start = std_id[2] if (h == std_id[0] and m == std_id[1]) else 1
            low_end = end_id[2] if (h == end_id[0] and m == end_id[1]) else width
            for l in range(low_start, low_end+1):
                covered.append((h, m, l))
    return covered

# get options segment ids
def get_options_segment_ids(duration, clues, width):
    options_segment_ids = []
    for clue in clues:
        std, end = clue
        std_id = get_id(std, duration, width)
        end_id = get_id(end, duration, width)
        covered_ids = get_covered_ids(std_id, end_id, width)
        options_segment_ids = options_segment_ids + covered_ids
        if end - std >= 10:
            mid = (std + end) / 2
            mid_id = get_id(mid, duration, width)
            options_segment_ids.append((mid_id[0], mid_id[1]))
    # for i in range(len(options_segment_ids)):
    #     if (options_segment_ids[i][0], options_segment_ids[i][1]) not in options_segment_ids:
    #         options_segment_ids.append((options_segment_ids[i][0], options_segment_ids[i][1]))
    
    return options_segment_ids

# 计算区间距离
def interval_distance(pred, gt):
    """计算 pred 区间和 gt 区间的距离 = 左端点差 + 右端点差 的绝对值和"""
    return abs(pred[0] - gt[0]) + abs(pred[1] - gt[1])

# 片段转换为时间区间
def segment_to_range(seg, duration, width):
    """根据 seg=(h,), (h,m,), (h,m,l) 计算对应的时间区间"""
    L_high = duration / width
    L_med = duration / (width ** 2)
    L_low = duration / (width ** 3)
    if len(seg) == 1:
        start = (seg[0] - 1) * L_high
        return [start, min(duration, start + L_high)]
    elif len(seg) == 2:
        start = (seg[0] - 1) * L_high + (seg[1] - 1) * L_med
        return [start, min(duration, start + L_med)]
    elif len(seg) == 3:
        start = (seg[0] - 1) * L_high + (seg[1] - 1) * L_med + (seg[2] - 1) * L_low
        return [start, min(duration, start + L_low)]
    else:
        # raise ValueError("invalid seg length")
        return [0,0]






def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_tool_call(solution_str):
    """Extract the tool call from the solution string."""
    tool_pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(tool_pattern, solution_str, re.DOTALL)
    if matches:
        results = [m.strip() for m in matches]
    elif len(matches) < 1:
        results = None
    return results

def parse_function_call(call_str):
    """
    解析形如 function_name((9,1,2)) 的字符串
    返回函数名和参数（转为 Python 对象）
    """
    func_pattern = r"(\w+)\((.*)\)"
    match = re.match(func_pattern, call_str.strip())
    arg_legal = False
    if match:
        func_name = match.group(1)
        arg_str = match.group(2).strip()
        try:
            # 直接解析
            args = ast.literal_eval(arg_str) if arg_str else ()
            arg_legal = True
        except Exception as e:
            print(f"参数解析失败: {arg_str}，错误: {e}")
            args = arg_str  # 保留原字符串
            arg_legal = False
        return func_name, args, arg_legal
    else:
        return None, None, False


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


# def compute_score(solution_str, ground_truth, extra_info):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
    
#     answer = extract_solution(solution_str=solution_str)

#     if answer is None:
#         return 0
#     else:
#         if answer == ground_truth:
#             return 1
#         else:
#             return 0


# def compute_score(solution_str, ground_truth, extra_info):
#     """The scoring function for exact match (EM).

#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#         method: the method to extract the solution, choices are 'strict' and 'flexible'
#         format_score: the score for the format
#         score: the score for the correct answer
#     """
    
#     reward = 0
    
#     answer = extract_solution(solution_str=solution_str)
    
#     clue_intervals = extra_info.get('clue_intervals')
#     width = extra_info.get('width')
#     duration = extra_info.get('duration')

#     if answer is None:
#         return 0
#     else:
#         if answer == ground_truth:
#             reward += 1
#         else:
#             return 0
    
#     try:
    
#         tool_calls = extract_tool_call(solution_str=solution_str)
#         tool_call_segments = []
#         if tool_calls is None:
#             return reward
        
#         for tool_call in tool_calls:
#             func_name, args, arg_legal = parse_function_call(tool_call)
#             if func_name == 'get_caption':
#                 if arg_legal and len(args) < 4 and args not in tool_call_segments:
#                     tool_call_segments.append(args)
        
#         if len(tool_call_segments) == 0:
#             return reward  

#         merged_clue_intervals = merge_intervals(clue_intervals)
#         options_segment_ids = get_options_segment_ids(duration, merged_clue_intervals, width)
        
#         hit_score = round(1/len(options_segment_ids), 2)
        
#         for tool_call_segment in tool_call_segments:
#             if tool_call_segment in options_segment_ids:
#                 reward += hit_score
        
#         return reward
    
#     except Exception as e:
#         return reward 
    


def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    w_repeat = -0.2
    w_f1 = 1.0
    w_answer = 1.0

    answer_reward = 0
    f1_reward = 0
    repeat_reward = 0
    
    answer = extract_solution(solution_str=solution_str)
    
    clue_intervals = extra_info.get('clue_intervals')
    width = extra_info.get('width')
    duration = extra_info.get('duration')

    if answer is None:
        answer_reward += 0
    else:
        if answer == ground_truth:
            answer_reward += 1
        else:
            answer_reward += 0
    
    try:
    
        tool_calls = extract_tool_call(solution_str=solution_str)
        tool_call_segments = []
        if tool_calls is None:
            rewards = {
                "score": w_answer * answer_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
            return rewards
        
        for tool_call in tool_calls:
            func_name, args, arg_legal = parse_function_call(tool_call)
            if func_name == 'get_caption':
                if arg_legal and len(args) < 4:
                    tool_call_segments.append(args)
        
        if len(tool_call_segments) == 0:
            rewards = {
                "score": w_answer * answer_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
            return rewards
        
        repeat_reward = (len(tool_call_segments) - len(set(tool_call_segments))) / len(set(tool_call_segments))

        # merged_clue_intervals = merge_intervals(clue_intervals)
        # visited_intervals = [segment_to_range(seg, duration, width) for seg in tool_call_segments]
        # # 基础量
        # gt_len = gt_total_length(merged_clue_intervals)  # 可能为0（保护）
        # inter_len = total_intersection_with_gt(visited_intervals, merged_clue_intervals)
        # visited_len = union_visited_length(visited_intervals)

        # # Coverage
        # cov = (inter_len / gt_len) if gt_len > 0 else 0.0
        # # Precision
        # prec = (inter_len / visited_len) if visited_len > 0 else 0.0
        # # F1-style
        # if cov + prec > 0:
        #     f1_reward = 2 * cov * prec / (cov + prec)
        # else:
        #     f1_reward = 0.0
        # rewards = {
        #         "score": w_answer * answer_reward + w_f1 * f1_reward + w_repeat * repeat_reward,
        #         "answer_reward": answer_reward,
        #         "f1_reward": f1_reward,
        #         "repeat_reward": repeat_reward,
        #     }
        rewards = {
                "score": w_answer * answer_reward + w_repeat * repeat_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
        return rewards
    
    except Exception as e:
        # rewards = {
        #         "score": w_answer * answer_reward + w_f1 * f1_reward + w_repeat * repeat_reward,
        #         "answer_reward": answer_reward,
        #         "f1_reward": f1_reward,
        #         "repeat_reward": repeat_reward,
        #     }
        rewards = {
                "score": w_answer * answer_reward + w_repeat * repeat_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
        return rewards
    



def compute_score_only_answer(solution_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return 1
        else:
            return 0
    
            
    
    
    

