from kon.ui.latex import preprocess_latex


def test_inline_dollar_math():
    text = "Energy is $E = mc^2$ and that's it."
    result = preprocess_latex(text)
    assert "E = mc²" in result
    assert "$" not in result


def test_display_double_dollar_math():
    text = "Here is an equation:\n$$\\lambda = \\frac{b}{T}$$\nDone."
    result = preprocess_latex(text)
    assert "λ = b/T" in result
    assert "$$" not in result
    assert ">" not in result


def test_latex_in_code_fence_preserved():
    text = "```\n$E = mc^2$\n```"
    result = preprocess_latex(text)
    assert "$E = mc^2$" in result


def test_greek_letters():
    text = r"$\alpha + \beta = \gamma$"
    result = preprocess_latex(text)
    assert "α + β = γ" in result


def test_fractions():
    text = r"$\frac{a+b}{c}$"
    result = preprocess_latex(text)
    assert "(a+b)/(c)" in result


def test_sqrt():
    text = r"$\sqrt{x}$"
    result = preprocess_latex(text)
    assert "√(x)" in result


def test_no_math_unchanged():
    text = "Just plain text with $5 dollars."
    result = preprocess_latex(text)
    assert result == text


def test_backslash_bracket_display():
    text = r"\[\sum_{i=0}^{n} x_i\]"
    result = preprocess_latex(text)
    assert "∑" in result
    assert r"\[" not in result


def test_backslash_paren_inline():
    text = r"\(\\alpha\\)"
    result = preprocess_latex(text)
    assert "α" in result
    assert r"\(" not in result
