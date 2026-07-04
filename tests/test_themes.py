"""Network-free tests for thematic auto-tagging (live-view markets)."""

from stockscan import themes as T


def test_tag_themes_hits_named_themes():
    assert "AI" in T.tag_themes("a platform powered by artificial intelligence and machine learning")
    assert "SaaS" in T.tag_themes("a subscription-based software-as-a-service product")
    assert "Electric Vehicles" in T.tag_themes("manufactures battery electric vehicles and EV charging")
    assert "Cybersecurity" in T.tag_themes("endpoint security and threat detection for enterprises")
    assert "Clean Energy" in T.tag_themes("develops utility-scale solar and wind power projects")


def test_tag_themes_avoids_obvious_false_positives():
    # bare ambiguous tokens must not trip a theme
    assert T.tag_themes("the aircraft maintains altitude over the airfield") == []   # no 'AI'
    assert T.tag_themes("a chain of family restaurants") == []                        # no 'AI'
    assert "Electric Vehicles" not in T.tag_themes("every product we ship is durable")  # no bare 'ev'
    assert T.tag_themes("") == [] and T.tag_themes(None) == []


def test_tag_themes_multi_membership():
    hits = T.tag_themes("a cloud computing platform using artificial intelligence")
    assert set(hits) == {"AI", "Cloud"}


def test_store_round_trip_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "THEMES_DB_PATH", tmp_path / "themes.sqlite")
    with T.ThemeStore() as store:
        store.put(320193, ["AI", "Cloud"])
        store.put(2, ["SaaS"])
        store.commit()
    assert T.load_theme_tags() == {320193: ["AI", "Cloud"], 2: ["SaaS"]}
    with T.ThemeStore() as store:
        store.clear(2)
        store.commit()
    assert T.load_theme_tags() == {320193: ["AI", "Cloud"]}


def test_refresh_tags_builds_from_descriptions(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "THEMES_DB_PATH", tmp_path / "themes.sqlite")
    descs = {
        1: "artificial intelligence analytics",           # AI
        2: "software-as-a-service for accounting",        # SaaS
        3: "a regional bank holding company",             # nothing
    }
    stats = T.refresh_theme_tags([1, 2, 3], get_desc=descs.get, max_workers=2)
    assert stats["scanned"] == 3 and stats["tagged"] == 2
    assert stats["by_theme"] == {"AI": 1, "SaaS": 1}
    tags = T.load_theme_tags()
    assert tags == {1: ["AI"], 2: ["SaaS"]}               # untagged cik 3 not stored
