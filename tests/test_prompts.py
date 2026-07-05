from mmcot_fusion.data.prompts import build_train_pair

PROBLEMS = {
    "1": {
        "question": "Which force acts on the cabinet door?",
        "hint": "A baby opens it.",
        "caption": "",
        "choices": ["pull", "push"],
        "answer": 0,
        "lecture": "A force is a push or a pull.",
        "solution": "The baby pulls the door open.",
    }
}


def test_rationale_stage_prompt():
    prompt, target = build_train_pair(PROBLEMS, "1", "QCM-LE")
    assert prompt.startswith("Question: Which force acts on the cabinet door?")
    assert "Context: A baby opens it." in prompt
    assert "Options: (A) pull (B) push" in prompt
    assert prompt.rstrip().endswith("Solution:")
    assert target == "Solution: A force is a push or a pull. The baby pulls the door open.."


def test_answer_stage_prompt_with_generated_rationale():
    generated = "Solution: The baby pulls the door open."
    prompt, target = build_train_pair(PROBLEMS, "1", "QCMG-A", curr_le_data=generated)
    assert generated in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert target == "The answer is (A)."


def test_answer_stage_prompt_gold_rationale():
    prompt, target = build_train_pair(PROBLEMS, "1", "QCMG-A")
    assert "Solution: A force is a push or a pull." in prompt
    assert target == "The answer is (A)."
