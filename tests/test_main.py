import unittest
from pathlib import Path
import shutil
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.main import (
    assign_maintenance_status,
    authenticate_user,
    create_user,
    get_required_columns,
    get_saved_result_csv,
    get_user_runs,
    init_db,
    preprocess_features,
    save_prediction_run,
)


class TestMainHelpers(unittest.TestCase):
    def setUp(self):
        fit_df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, np.nan],
                "b": [10.0, np.nan, 30.0, 40.0],
            }
        )

        imp = SimpleImputer(strategy="median")
        fit_imp = pd.DataFrame(
            imp.fit_transform(fit_df),
            columns=fit_df.columns,
        )

        scl = StandardScaler()
        scl.fit(fit_imp)

        self.pre = {
            "feature_names": ["a", "b"],
            "columns_to_drop": [],
            "imputer": imp,
            "scaler": scl,
        }

    def test_status_labels(self):
        p = np.array([0.05, 0.25, 0.80])
        got = assign_maintenance_status(p).tolist()
        self.assertEqual(got, ["Healthy", "Warning", "Critical"])

    def test_get_required_columns(self):
        got = get_required_columns(model=object(), preprocessor=self.pre)
        self.assertEqual(got, ["a", "b"])

    def test_preprocess_ok(self):
        raw = pd.DataFrame({"a": [1.5, np.nan], "b": [20.0, 50.0], "class": [0, 1]})
        out = preprocess_features(raw, ["a", "b"], self.pre)

        self.assertEqual(list(out.columns), ["a", "b"])
        self.assertEqual(out.shape, (2, 2))
        self.assertFalse(out.isna().any().any())

    def test_preprocess_missing_col(self):
        raw = pd.DataFrame({"a": [1.5, 2.0]})
        with self.assertRaises(ValueError):
            preprocess_features(raw, ["a", "b"], self.pre)

    def _local_db_path(self) -> tuple[Path, Path]:
        root = Path("tests") / ".tmp_db"
        root.mkdir(exist_ok=True)
        folder = root / f"run_{uuid4().hex}"
        folder.mkdir()
        return folder, folder / "app.db"

    def test_user_register_and_login(self):
        folder, db_path = self._local_db_path()
        try:
            init_db(db_path)

            ok, _ = create_user("alex", "secret12", db_path)
            self.assertTrue(ok)

            bad_user_id, bad_name = authenticate_user("alex", "wrong", db_path)
            self.assertIsNone(bad_user_id)
            self.assertIsNone(bad_name)

            user_id, user_name = authenticate_user("alex", "secret12", db_path)
            self.assertIsNotNone(user_id)
            self.assertEqual(user_name, "alex")
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_saved_run_persistence(self):
        folder, db_path = self._local_db_path()
        try:
            init_db(db_path)
            ok, _ = create_user("alex", "secret12", db_path)
            self.assertTrue(ok)
            user_id, _ = authenticate_user("alex", "secret12", db_path)
            self.assertIsNotNone(user_id)

            raw_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            result_df = raw_df.copy()
            result_df["Failure Probability (%)"] = [12.0, 74.0]
            result_df["Maintenance Status"] = ["Healthy", "Critical"]

            save_prediction_run(
                user_id=int(user_id),
                uploaded_filename="input.csv",
                model_name="best_model_xgb.joblib",
                original_df=raw_df,
                result_df=result_df,
                db_path=db_path,
            )

            history = get_user_runs(int(user_id), db_path=db_path)
            self.assertEqual(len(history), 1)
            self.assertEqual(int(history.iloc[0]["total_rows"]), 2)
            self.assertEqual(int(history.iloc[0]["machines_at_risk"]), 1)

            run_id = int(history.iloc[0]["run_id"])
            saved = get_saved_result_csv(int(user_id), run_id, db_path=db_path)
            self.assertIsNotNone(saved)
            file_name, saved_csv = saved
            self.assertEqual(file_name, "input.csv")
            self.assertIn("Failure Probability (%)", saved_csv)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
