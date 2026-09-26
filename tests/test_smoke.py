import llmlab


def test_version() -> None:
    assert isinstance(llmlab.__version__, str)
    assert llmlab.__version__.startswith("0.1.0")
