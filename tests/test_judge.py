from localsight.rl.judge import build_judge_prompt, parse_judge_score


def test_parse_score_field():
    assert parse_judge_score('{"score": 8}') == 8.0
    assert parse_judge_score("评分：7.5") == 7.5


def test_parse_score_trailing_number():
    assert parse_judge_score("我认为这个回答可以得 9 分。") == 9.0
    assert parse_judge_score("没有数字") is None


def test_parse_score_clamped():
    assert parse_judge_score("score: 15") == 10.0
    assert parse_judge_score("score: -3") == 0.0


def test_judge_prompt_contains_question_and_answer():
    prompt = build_judge_prompt("1+1=?", "答案是 2")
    assert "1+1=?" in prompt
    assert "答案是 2" in prompt
