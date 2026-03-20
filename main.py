import warnings
import logging
import numpy as np
import mlflow
import hydra
from omegaconf import DictConfig

from steps.ingest import ingest
from steps.clean import clean
from steps.feature_engineer import feature_engineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="pipeline", version_base=None)
def main(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(cfg.model.random_seed)

    exp = mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id):

        # ── Step 1: Ingest ────────────────────────────────────────────────────
        logger.info("Step 1: Ingest")
        raw_df = ingest(cfg.data.raw_path)

        # ── Step 2: Clean ─────────────────────────────────────────────────────
        logger.info("Step 2: Clean")
        df = clean(raw_df, cleaned_filename=cfg.data.cleaned_filename)

        # ── Step 3: Feature Engineering ───────────────────────────────────────
        logger.info("Step 3: Feature Engineering")
        df_features = feature_engineer(df, engineered_filename=cfg.data.engineered_filename)

        logger.info(
            "Pipeline steps 1-3 complete. Shape after feature engineering: %s",
            df_features.shape,
        )
        # Step 4 (model training) coming next.


if __name__ == "__main__":
    main()
