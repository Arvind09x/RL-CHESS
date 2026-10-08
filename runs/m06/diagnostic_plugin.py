"""Historical M05 tests use the documented constructor diagnostic path.

Only tests.network_checks' factory is adapted. Production ChessNetwork and
M06 tests retain relative-enabled defaults. M01–M04 retain standalone defaults.
"""
def pytest_collection_modifyitems():
    from functools import partial
    from chess_rl.model import ChessNetwork
    import tests.network_checks as checks
    checks.ChessNetwork = partial(ChessNetwork, relative_enabled=False)
