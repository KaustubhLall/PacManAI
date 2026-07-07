def test_deepql_module_imports_without_starting_training():
    import ai.deepQL as deepql

    assert callable(deepql.train_dqn)
    assert callable(deepql.main)
