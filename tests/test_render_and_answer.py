from __future__ import annotations

from conformal_routing.eval.answer_check import check_answer, extract_answer, extract_boxed_answer
from conformal_routing.models.render import render_for_continuation


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if continue_final_message:
            rendered += "<continue>"
        return rendered


def test_render_for_continuation_does_not_force_think_prefill():
    tokenizer = FakeTokenizer()

    rendered = render_for_continuation("What is 1+1?", "", tokenizer)

    assert rendered == "<user>What is 1+1?<assistant>"
    assert "<think>" not in rendered


def test_render_for_continuation_continues_assistant_history():
    tokenizer = FakeTokenizer()

    rendered = render_for_continuation("What is 1+1?", "</think>\n\nFinal.", tokenizer)

    assert rendered == "<user>What is 1+1?<assistant></think>\n\nFinal.<continue>"


def test_extract_nested_boxed_answer():
    assert extract_boxed_answer(r"final \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_answer(r"<think>work</think> final \boxed{\dfrac{1}{2}}") == r"\frac{1}{2}"


def test_check_answer_with_nested_latex_box():
    assert check_answer(r"The result is \boxed{\frac{1}{2}}.", r"\frac{1}{2}")
