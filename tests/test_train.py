from sklearn.linear_model import LogisticRegression

def test_model_creation():
    model = LogisticRegression()
    assert model is not None
