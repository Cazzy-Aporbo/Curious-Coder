import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA


@pytest.mark.parametrize("name", ["LinearRegressionTutorial", "LogisticRegressionTutorial"])
def test_refit_resets_loss_history(core, name):
    model = getattr(core, name)(n_iterations=10)
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model.fit(X, y, verbose=False)
    model.fit(X, y, verbose=False)
    assert len(model.losses) == 10


def test_linear_regression_recovers_known_relationship(core):
    X = np.linspace(-2, 2, 80).reshape(-1, 1)
    y = 3 * X[:, 0] + 2
    model = core.LinearRegressionTutorial(learning_rate=0.1, n_iterations=400)
    model.fit(X, y, verbose=False)
    assert_allclose(model.predict(X), y, atol=1e-6)
    assert model.losses[-1] < model.losses[0]


def test_logistic_probabilities_and_classification(core):
    X = np.array([[-3.0], [-2.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = core.LogisticRegressionTutorial(learning_rate=0.1, n_iterations=200)
    model.fit(X, y, verbose=False)
    assert_allclose(model.predict(X), y)
    probabilities = model.sigmoid(np.array([-1e6, 0, 1e6]))
    assert np.isfinite(probabilities).all()
    assert_allclose(probabilities, [0, 0.5, 1], atol=1e-12)


def test_pca_matches_reference_subspace(core):
    X = np.random.default_rng(42).normal(size=(40, 5))
    model = core.PCATutorial(n_components=3)
    model.fit(X)
    reference = PCA(n_components=3).fit(X)
    assert_allclose(model.explained_variance_, reference.explained_variance_, atol=1e-12)
    assert_allclose(model.components_.T @ model.components_,
                    reference.components_.T @ reference.components_, atol=1e-12)
    reconstructed = model.inverse_transform(model.transform(X))
    assert np.linalg.norm(X - reconstructed) < np.linalg.norm(X - X.mean(axis=0))


@pytest.mark.parametrize("X", [np.ones((1, 3)), np.ones((4, 2)) * np.nan])
def test_pca_rejects_undefined_covariance(core, X):
    with pytest.raises(ValueError):
        core.PCATutorial().fit(X)


def test_kmeans_duplicate_points_remain_finite(core):
    X = np.ones((6, 2))
    model = core.KMeansTutorial(n_clusters=3)
    model.fit(X, verbose=False)
    assert model.centroids.shape == (3, 2)
    assert np.isfinite(model.centroids).all()
    assert model.inertia_ == 0


def test_kmeans_labels_match_final_centroids(core):
    X = np.random.default_rng(8).normal(size=(30, 2))
    model = core.KMeansTutorial(n_clusters=3, max_iters=1)
    model.fit(X, verbose=False)
    assert_allclose(model.labels_, model.predict(X))
    assert_allclose(model.inertia_, ((X - model.centroids[model.labels_]) ** 2).sum())


def test_kmeans_does_not_reset_global_rng(core):
    np.random.seed(19)
    expected = np.random.random(3)
    np.random.seed(19)
    core.KMeansTutorial(n_clusters=2).fit(np.arange(12).reshape(6, 2), verbose=False)
    assert_allclose(np.random.random(3), expected)


@pytest.mark.parametrize("kwargs", [{"n_clusters": 0}, {"n_clusters": 8}, {"max_iters": 0}])
def test_kmeans_rejects_invalid_parameters(core, kwargs):
    with pytest.raises(ValueError):
        core.KMeansTutorial(**kwargs).fit(np.ones((6, 2)), verbose=False)
