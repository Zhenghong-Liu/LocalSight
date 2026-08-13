from localsight.rl.rewards import composite_reward, format_reward, gt_match, normalize_answer


def test_normalize_answer():
    assert normalize_answer(" 1,430, 2730 ") == "14302730"
    assert normalize_answer("B") == "b"


def test_gt_match_numeric_and_text():
    assert gt_match("答案是 14302730", ["14302730"]) == 1.0
    assert gt_match("答案是 42", ["142"]) == 0.0
    assert gt_match("北京", ["答案是北京。"]) == 1.0


def test_format_reward_requires_think_and_tool():
    assert format_reward("<think>ok</think>", False) == 0.5
    assert format_reward("no tags", False) == 0.0
    text = '<think>t</think>\n<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>'
    assert format_reward(text, True) == 1.0


def test_composite_reward_bounds():
    ok = '<think>t</think>\n<tool_call>{"name":"calculate_math","arguments":{"expression":"2*3"}}</tool_call>\n结果是 6'
    assert 0.9 < composite_reward(ok, True, ["6"]) <= 1.0
    bad = "随便说点"
    assert composite_reward(bad, True, ["6"]) < 0.3
