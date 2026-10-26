import newsindex as ni


def test_countries_covers_the_full_iso_3166_1_list():
    table = ni.countries()
    assert len(table) == 249
    assert list(table.columns) == [
        "name",
        "iso2",
        "iso3",
        "continent",
        "sub_region",
        "intermediate_region",
    ]


def test_countries_codes_are_unique_and_well_formed():
    table = ni.countries()
    assert table["iso2"].str.len().eq(2).all()
    assert table["iso3"].str.len().eq(3).all()
    assert table["iso2"].is_unique
    assert table["iso3"].is_unique


def test_countries_spot_check_known_entries():
    table = ni.countries().set_index("iso3")
    nigeria = table.loc["NGA"]
    assert nigeria["name"] == "Nigeria"
    assert nigeria["iso2"] == "NG"
    assert nigeria["continent"] == "Africa"
    assert nigeria["sub_region"] == "Sub-Saharan Africa"
    assert nigeria["intermediate_region"] == "Western Africa"

    france = table.loc["FRA"]
    assert france["continent"] == "Europe"
    assert france["sub_region"] == "Western Europe"
    # Europe has no third regional level.
    assert france["intermediate_region"] == ""


def test_countries_blank_classification_for_entries_outside_un_m49():
    table = ni.countries().set_index("iso3")
    antarctica = table.loc["ATA"]
    assert antarctica["continent"] == ""
    assert antarctica["sub_region"] == ""


def test_countries_only_africa_and_americas_have_an_intermediate_region():
    table = ni.countries()
    with_intermediate = table[table["intermediate_region"] != ""]
    assert set(with_intermediate["continent"]) == {"Africa", "Americas"}


def test_countries_accepts_a_path_override(tmp_path):
    path = tmp_path / "custom_countries.csv"
    path.write_text(
        "name,iso2,iso3,continent,sub_region,intermediate_region\n"
        "Freedonia,FD,FRD,Europe,Western Europe,\n",
        encoding="utf-8",
    )
    table = ni.countries(path)
    assert len(table) == 1
    assert table.iloc[0]["name"] == "Freedonia"
