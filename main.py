import warnings
import logging
import numpy as np
import mlflow
import hydra
from omegaconf import DictConfig

from steps.ingest import ingest
from steps.clean import clean

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

        logger.info("Pipeline steps 1-2 complete. Shape after cleaning: %s", df.shape)
        # Step 3 (feature engineering) and Step 4 (training) coming next.


if __name__ == "__main__":
    main()
