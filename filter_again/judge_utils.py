import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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


def f1_score(prediction, ground_truth, **_):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction, ground_truth, **_):
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def exact_match_score(prediction, ground_truth):
    assert isinstance(prediction, str) and isinstance(ground_truth, str), (
        f"Expected strings, got {type(prediction)} and {type(ground_truth)}"
    )
    # print(normalize_answer(prediction), normalize_answer(ground_truth))
    res = int(
        (
            normalize_answer(prediction) in normalize_answer(ground_truth)
            or normalize_answer(ground_truth) in normalize_answer(prediction)
        )
    )
    return res


# ---------- boxed helper utils (unchanged from original) ---------- #
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left) :]
    left = "\\boxed{"
    return s[len(left) : -1]


def remove_text(s):
    if "\\text " in s:
        left = "\\text "
        return s[len(left) :]
    left = "\\text{"
    if s.startswith(left):
        s = s[len(left) :]
        right = s.rfind("}")
        if right == -1:
            s = s[:-1]
        return s
    return s


def boxed_exact_match_judge(output_pair):
    question = output_pair["question"]
    standard_answer = output_pair["outputs"]
    pred_list = output_pair["pred_list"]
    judge_results = []
    for pred in pred_list:
        jr = {}
        lower_pred = pred.lower().replace("\\boxed{}", "")
        response = lower_pred
        response = response.replace("*", "")
        ans = last_boxed_only_string(lower_pred)
        ans = remove_boxed(ans) if ans else None
        if ans is not None:
            jr["reason"] = ans
            score = max(exact_match_score(ans, sa) for sa in standard_answer)
        else:
            if "the answer is" in response:
                ans = (
                    response.rsplit("the answer is", 1)[-1]
                    .strip()
                    .strip()
                    .strip(".")
                    .strip()
                )
                jr["reason"] = ans
                score = max(exact_match_score(ans, sa) for sa in standard_answer)
            else:
                jr["reason"] = "boxed extraction failed"
                score = 0
        jr["is_correct"] = score
        judge_results.append(jr)
    return judge_results
