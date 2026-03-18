"""Reward function for countdown task"""

import random
import re


_EQUATION_PATTERN = r"^[\d+\-*/().\s]+$"


def extract_solution(solution_str: str):
    """Extract the equation from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    solution_str = solution_str.split("\n")[-1]
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None


def validate_equation(equation_str: str, available_numbers) -> bool:
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def evaluate_equation(equation_str: str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        if not re.match(_EQUATION_PATTERN, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    """Scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    
    Returns:
        float: reward score (0.0, 0.1, or 1.0)
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score

        if do_print:
            print(f"Wrong result: equation = {result}, target = {target}")
        return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score
