"""Smoke tests: verifica que os módulos importam corretamente."""



def test_import_config():
    from churn_prediction.config import LEAKAGE_COLS, V_RETIDO

    assert isinstance(LEAKAGE_COLS, list)
    assert len(LEAKAGE_COLS) > 0
    assert V_RETIDO > 0


def test_import_data_cleaning():
    from churn_prediction.data_cleaning import clean_dataset, remove_duplicates

    assert callable(clean_dataset)
    assert callable(remove_duplicates)


def test_import_preprocessing():
    from churn_prediction.preprocessing import build_preprocessor, build_sklearn_pipeline

    assert callable(build_preprocessor)
    assert callable(build_sklearn_pipeline)


def test_import_evaluation():
    from churn_prediction.evaluation import (
        compute_business_metrics,
        compute_classification_metrics,
        optimize_threshold,
    )

    assert callable(compute_business_metrics)
    assert callable(compute_classification_metrics)
    assert callable(optimize_threshold)


def test_import_model():
    from churn_prediction.model import MLP

    assert callable(MLP)


def test_import_pipeline():
    from churn_prediction.pipelines import prepare_data, train_and_evaluate

    assert callable(prepare_data)
    assert callable(train_and_evaluate)


def test_import_api():
    from churn_prediction.api.main import app

    assert app is not None
    assert app.title == "Churn Prediction API"
