from commerce_ai.query_rules import classify_query_product, parse_query


def test_parse_query_extracts_bilingual_constraints():
    intent = parse_query("red wireless headphones under $100 without microphone")
    assert intent.colors == ["red"]
    assert intent.attributes == ["wireless"]
    assert intent.max_price == 100
    assert intent.exclude_terms == ["microphone"]
    assert parse_query("knot headband for women").exclude_terms == []

    chinese = parse_query("预算 500 到 900，蓝色无线键盘，不要机械")
    assert chinese.min_price == 500
    assert chinese.max_price == 900
    assert "无线" in chinese.attributes


def test_rule_proxy_handles_exact_and_irrelevant_examples():
    exact, _ = classify_query_product(
        "fusion ringke iphone 11 pro",
        "Ringke Fusion-X Matte Compatible with iPhone 11 Pro Case",
    )
    irrelevant, _ = classify_query_product(
        "motor trend digital tire gauge safety tool",
        "Vehicle Safety Hammer and Seatbelt Cutter",
    )
    assert exact == "A"
    assert irrelevant == "D"
