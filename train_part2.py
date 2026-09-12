"""Validate, train, and export all Project Part II deliverables."""

from pathlib import Path

from proj import load_csv, train_and_export, validate_pipeline


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "cook_county_contest_data" / "cook_county_contest_train.csv"
TEST_PATH = BASE_DIR / "cook_county_contest_test.csv"
MODEL_PATH = BASE_DIR / "pipeline.joblib.gz"
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"


def main():
    training_data = load_csv(TRAIN_PATH)
    contest_data = load_csv(TEST_PATH)

    _, metrics = validate_pipeline(training_data)
    print(
        "Validation: "
        f"log RMSE={metrics['log_rmse']:.4f}, "
        f"dollar RMSE=${metrics['dollar_rmse']:,.0f} "
        f"on {metrics['validation_rows']:,} rows"
    )

    _, predictions = train_and_export(
        training_data,
        contest_data,
        model_path=MODEL_PATH,
        predictions_path=PREDICTIONS_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved {len(predictions):,} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
