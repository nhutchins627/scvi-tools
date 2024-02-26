from ray import tune

import scvi
from scvi.autotune import ModelTuner


def test_model_tuner_init():
    model_cls = scvi.model.SCVI
    ModelTuner(model_cls)


def test_model_tuner_fit(save_path: str):
    model_cls = scvi.model.SCVI
    tuner = ModelTuner(model_cls)

    adata = scvi.data.synthetic_iid()
    model_cls.setup_anndata(adata)
    results = tuner.fit(
        adata,
        num_samples=1,
        max_epochs=1,
        logging_dir=save_path,
        search_space={
            "n_layers": tune.choice([1, 2, 3]),
        },
    )
    assert results is not None


def test_model_tuner_info():
    model_cls = scvi.model.SCVI
    tuner = ModelTuner(model_cls)

    tuner.info()
