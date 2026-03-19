import warnings
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig

from steps.ingest import ingest
from steps.clean import clean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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

        # ── Step 3: Train ─────────────────────────────────────────────────────
        logger.info("Step 3: Train")
        train, test = train_test_split(df, test_size=cfg.model.test_size, random_state=cfg.model.random_state)

        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        alpha = cfg.model.alpha
        l1_ratio = cfg.model.l1_ratio

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=cfg.model.random_state)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        logger.info("ElasticNet (alpha=%.4f, l1_ratio=%.4f) — RMSE: %.4f  MAE: %.4f  R2: %.4f",
                    alpha, l1_ratio, rmse, mae, r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "linear_model")


if __name__ == "__main__":
    main()
