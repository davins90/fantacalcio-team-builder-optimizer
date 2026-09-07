from engine.data_loader import BUNDLED_LIST_PATH, load_bundled_dataset


def test_bundled_official_list_is_primary_and_parses_mantra_columns():
    assert BUNDLED_LIST_PATH.exists()
    out, health = load_bundled_dataset(enrich_online=False)
    assert len(out) >= 500
    assert health.roles_coverage > 0.99

    def one(name_norm):
        row = out.loc[out["name_norm"].eq(name_norm)]
        assert len(row) == 1
        return row.iloc[0]

    dimarco = one("dimarco")
    assert dimarco["roles"] == "E/W"
    assert int(dimarco["fvm"]) == 240
    assert int(dimarco["quote"]) == 29

    calha = one("calhanoglu")
    assert calha["roles"] == "M/C"
    assert int(calha["fvm"]) == 273  # FVM M, not Classic FVM 243
    assert int(calha["quote"]) == 29

    hojlund = one("hojlund")
    assert hojlund["roles"] == "Pc"
    assert int(hojlund["fvm"]) == 260

    paz = one("paz n")
    assert paz["roles"] == "T/A"
    assert int(paz["fvm"]) == 245
