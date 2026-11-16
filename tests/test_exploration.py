from banking_risk.exploration import explore
from banking_risk.data import synthetic_customers, FEATURES


def test_exploration_excludes_target_and_returns_projection():
    data = synthetic_customers(100)
    result = explore(data[FEATURES], algorithm="kmeans", projection="pca", clusters=3)
    assert len(result["projection"]) == 100
    assert len(result["sizes"]) == 3
    assert (
        result["scope"]
        == "Exploratory training partition only; never changes the scoring model"
    )
