"""规划阶段的最小包级测试：本地无需 GPU 即可运行。"""


def test_package_imports_and_version() -> None:
    import localsight

    assert localsight.__version__ == "0.1.0"
