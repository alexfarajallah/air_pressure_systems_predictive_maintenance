from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path
import secrets
import sqlite3
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

HEALTHY_THRESHOLD = 0.20
CRITICAL_THRESHOLD = 0.50
SHAP_SAMPLE_SIZE = 500

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DB_PATH = ROOT_DIR / "data" / "app.db"


class TfModel:
    def __init__(self, net):
        self.net = net

    def predict_proba(self, x):
        p = self.net.predict(np.asarray(x), verbose=0).reshape(-1)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def open_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with open_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                uploaded_filename TEXT,
                model_name TEXT NOT NULL,
                total_rows INTEGER NOT NULL,
                machines_at_risk INTEGER NOT NULL,
                avg_failure_probability REAL NOT NULL,
                input_csv TEXT NOT NULL,
                result_csv TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def hash_password(password: str, salt: str | None = None) -> str:
    salt_text = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_text.encode("utf-8"),
        100_000,
    ).hex()
    return f"{salt_text}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_text, digest = stored_hash.split("$", 1)
    except ValueError:
        return False

    candidate = hash_password(password, salt_text)
    return secrets.compare_digest(candidate, f"{salt_text}${digest}")


def create_user(username: str, password: str, db_path: Path = DB_PATH) -> tuple[bool, str]:
    name = username.strip()
    if len(name) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    password_hash = hash_password(password)
    try:
        with open_db(db_path) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (name, password_hash),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def authenticate_user(
    username: str,
    password: str,
    db_path: Path = DB_PATH,
) -> tuple[int | None, str | None]:
    name = username.strip()
    if not name or not password:
        return None, None

    with open_db(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (name,),
        ).fetchone()

    if row is None:
        return None, None

    if not verify_password(password, row["password_hash"]):
        return None, None

    return int(row["id"]), str(row["username"])


def save_prediction_run(
    user_id: int,
    uploaded_filename: str,
    model_name: str,
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    db_path: Path = DB_PATH,
) -> None:
    total_rows = int(len(result_df))
    machines_at_risk = int((result_df["Maintenance Status"] != "Healthy").sum())
    avg_failure_probability = (
        float(result_df["Failure Probability (%)"].mean()) if total_rows else 0.0
    )

    input_csv = original_df.to_csv(index=False)
    output_csv = result_df.to_csv(index=False)

    with open_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prediction_runs
            (user_id, uploaded_filename, model_name, total_rows, machines_at_risk, avg_failure_probability, input_csv, result_csv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                uploaded_filename,
                model_name,
                total_rows,
                machines_at_risk,
                avg_failure_probability,
                input_csv,
                output_csv,
            ),
        )


def get_user_runs(user_id: int, limit: int = 20, db_path: Path = DB_PATH) -> pd.DataFrame:
    with open_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                id AS run_id,
                created_at,
                uploaded_filename,
                model_name,
                total_rows,
                machines_at_risk,
                ROUND(avg_failure_probability, 2) AS avg_failure_probability
            FROM prediction_runs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(row) for row in rows])


def get_saved_result_csv(
    user_id: int,
    run_id: int,
    db_path: Path = DB_PATH,
) -> tuple[str, str] | None:
    with open_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT uploaded_filename, result_csv
            FROM prediction_runs
            WHERE user_id = ? AND id = ?
            """,
            (user_id, run_id),
        ).fetchone()

    if row is None:
        return None

    file_name = row["uploaded_filename"] or "saved_result.csv"
    return str(file_name), str(row["result_csv"])


@st.cache_resource(show_spinner=False)
def load_model(model_dir: Path):
    model_candidates = list(model_dir.glob("best_model_*.joblib")) + list(
        model_dir.glob("best_model_*.keras")
    )
    model_candidates = sorted(
        model_candidates,
        key=lambda path_obj: path_obj.stat().st_mtime,
        reverse=True,
    )
    if not model_candidates:
        raise FileNotFoundError(
            "No trained model was found in models/. Expected best_model_*.joblib or best_model_*.keras"
        )

    model_path = model_candidates[0]
    if model_path.suffix == ".keras":
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow model found but tensorflow is not installed."
            ) from exc
        net = tf.keras.models.load_model(model_path)
        return TfModel(net), model_path.name

    return joblib.load(model_path), model_path.name


@st.cache_resource(show_spinner=False)
def load_preprocessor(model_dir: Path):
    candidate_paths = [
        model_dir / "aps_preprocessing.joblib",
        model_dir / "preprocessing.joblib",
        model_dir / "scaler.pkl",
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return joblib.load(candidate_path), candidate_path.name

    return None, None


def decode_csv_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode CSV. Use UTF-8 or Latin-1 encoded files.")


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    file_text = decode_csv_bytes(uploaded_file.getvalue())

    try:
        return pd.read_csv(StringIO(file_text), na_values=["na"])
    except Exception:
        lines = file_text.splitlines()
        header_row = next(
            (
                idx
                for idx, line in enumerate(lines)
                if line.startswith("class,") or line.startswith("aa_000,")
            ),
            None,
        )

        if header_row is None:
            raise ValueError(
                "Invalid CSV format. Ensure the file has a valid header row with sensor columns."
            )

        cleaned_text = "\n".join(lines[header_row:])
        return pd.read_csv(StringIO(cleaned_text), na_values=["na"])


def get_required_columns(model: Any, preprocessor: Any) -> list[str]:
    if isinstance(preprocessor, dict) and "feature_names" in preprocessor:
        return list(preprocessor["feature_names"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise ValueError(
        "Could not infer required sensor columns. Ensure preprocessing artifact includes feature names."
    )


def preprocess_features(
    raw_df: pd.DataFrame,
    required_columns: list[str],
    preprocessor: Any,
) -> pd.DataFrame:
    sensor_df = raw_df.copy()

    if "class" in sensor_df.columns:
        sensor_df = sensor_df.drop(columns=["class"])

    missing_columns = [column for column in required_columns if column not in sensor_df.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:15])
        raise ValueError(
            f"Uploaded CSV is missing {len(missing_columns)} required sensor columns. "
            f"First missing columns: {missing_preview}"
        )

    if isinstance(preprocessor, dict):
        columns_to_drop = preprocessor.get("columns_to_drop", [])
        if columns_to_drop:
            sensor_df = sensor_df.drop(columns=columns_to_drop, errors="ignore")

    sensor_df = sensor_df.reindex(columns=required_columns)
    sensor_df = sensor_df.apply(pd.to_numeric, errors="coerce")

    imputer = preprocessor.get("imputer") if isinstance(preprocessor, dict) else None
    scaler = preprocessor.get("scaler") if isinstance(preprocessor, dict) else None

    if imputer is not None:
        sensor_df = pd.DataFrame(
            imputer.transform(sensor_df),
            columns=required_columns,
            index=sensor_df.index,
        )
    elif sensor_df.isna().any().any():
        raise ValueError(
            "Missing numeric values detected but no imputer artifact is available."
        )

    if scaler is not None:
        sensor_df = pd.DataFrame(
            scaler.transform(sensor_df),
            columns=required_columns,
            index=sensor_df.index,
        )

    return sensor_df


def predict_failure_probabilities(model: Any, transformed_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(transformed_df)[:, 1]

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(transformed_df)
        return 1.0 / (1.0 + np.exp(-decision_scores))

    return model.predict(transformed_df).astype(float)


def assign_maintenance_status(probabilities: np.ndarray) -> np.ndarray:
    return np.where(
        probabilities >= CRITICAL_THRESHOLD,
        "Critical",
        np.where(probabilities >= HEALTHY_THRESHOLD, "Warning", "Healthy"),
    )


def resolve_shap_array(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        return shap_array[:, :, -1]

    return shap_array


def calculate_shap_values(model: Any, feature_df: pd.DataFrame) -> np.ndarray:
    if shap is None:
        raise RuntimeError("SHAP is not installed. Install dependencies from requirements.txt.")

    try:
        explainer = shap.TreeExplainer(model)
        raw_shap_values = explainer.shap_values(feature_df)
    except Exception:
        explainer = shap.Explainer(model, feature_df)
        raw_shap_values = explainer(feature_df).values

    shap_array = resolve_shap_array(raw_shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    return shap_array


def choose_identifier_column(result_df: pd.DataFrame) -> str | None:
    candidates = ["machine_id", "MachineID", "machineId", "unit_id", "UnitID", "id", "ID"]
    return next((column for column in candidates if column in result_df.columns), None)


def init_session_state() -> None:
    defaults = {
        "is_logged_in": False,
        "user_id": None,
        "username": None,
        "last_result_df": None,
        "last_transformed_df": None,
        "last_uploaded_file_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_result_state() -> None:
    st.session_state["last_result_df"] = None
    st.session_state["last_transformed_df"] = None
    st.session_state["last_uploaded_file_name"] = None


def show_login_page() -> None:
    st.title("Industrial Predictive Maintenance")
    st.caption("Sign in to upload sensor data and save each prediction run.")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            login_user = st.text_input("Username")
            login_pass = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login", width="stretch")

        if login_submit:
            user_id, user_name = authenticate_user(login_user, login_pass)
            if user_id is None:
                st.error("Invalid username or password.")
            else:
                st.session_state["is_logged_in"] = True
                st.session_state["user_id"] = user_id
                st.session_state["username"] = user_name
                clear_result_state()
                st.rerun()

    with tab_register:
        with st.form("register_form", clear_on_submit=False):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            new_pass_confirm = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Create Account", width="stretch")

        if register_submit:
            if new_pass != new_pass_confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_user, new_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def main() -> None:
    st.set_page_config(
        page_title="Industrial Predictive Maintenance",
        page_icon="PM",
        layout="wide",
    )

    init_db()
    init_session_state()
    if not st.session_state["is_logged_in"]:
        show_login_page()
        return

    with st.sidebar:
        st.write(f"Signed in as `{st.session_state['username']}`")
        if st.button("Logout", width="stretch"):
            st.session_state["is_logged_in"] = False
            st.session_state["user_id"] = None
            st.session_state["username"] = None
            clear_result_state()
            st.rerun()

    st.title("Industrial Predictive Maintenance")
    st.caption("Upload machine sensor data, evaluate failure risk in batch, and store runs.")

    try:
        model, model_name = load_model(MODELS_DIR)
        preprocessor, preprocessor_name = load_preprocessor(MODELS_DIR)
        required_columns = get_required_columns(model, preprocessor)
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()

    with st.expander("Loaded Artifacts", expanded=False):
        st.write(f"Model: `{model_name}`")
        st.write(
            f"Preprocessing: `{preprocessor_name}`"
            if preprocessor_name
            else "Preprocessing: Not found (expects already transformed features)."
        )
        st.write(f"Required sensor columns: `{len(required_columns)}`")

    with st.expander("View Required Sensor Columns", expanded=False):
        st.dataframe(
            pd.DataFrame({"required_sensor_column": required_columns}),
            width="stretch",
            height=320,
        )

    with st.form("batch_score_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Upload Sensor CSV",
            type=["csv"],
            help="The CSV must contain the required sensor columns.",
        )
        run_batch = st.form_submit_button("Run Batch Prediction", width="stretch")

    if run_batch:
        if uploaded_file is None:
            st.error("Upload a CSV file before running predictions.")
        else:
            try:
                original_df = read_uploaded_csv(uploaded_file)
                transformed_df = preprocess_features(original_df, required_columns, preprocessor)
                failure_probabilities = predict_failure_probabilities(model, transformed_df)

                result_df = original_df.copy()
                result_df["Failure Probability (%)"] = (failure_probabilities * 100).round(2)
                result_df["Maintenance Status"] = assign_maintenance_status(failure_probabilities)

                save_prediction_run(
                    user_id=int(st.session_state["user_id"]),
                    uploaded_filename=str(uploaded_file.name),
                    model_name=model_name,
                    original_df=original_df,
                    result_df=result_df,
                )

                st.session_state["last_result_df"] = result_df
                st.session_state["last_transformed_df"] = transformed_df
                st.session_state["last_uploaded_file_name"] = str(uploaded_file.name)
                st.success("Prediction completed and saved to database.")
            except ValueError as exc:
                st.error(f"Input validation error: {exc}")
            except Exception as exc:
                st.error(
                    "Prediction failed. Ensure the CSV has the expected structure and required sensor columns."
                )
                with st.expander("Technical details", expanded=False):
                    st.code(str(exc))

    result_df = st.session_state.get("last_result_df")
    transformed_df = st.session_state.get("last_transformed_df")
    uploaded_name = st.session_state.get("last_uploaded_file_name")

    if result_df is None or transformed_df is None:
        st.info("Upload a CSV and click Run Batch Prediction.")
    else:
        st.subheader("Batch Prediction Table")
        st.dataframe(result_df, width="stretch", height=460)

        output_csv = result_df.to_csv(index=False).encode("utf-8")
        default_name = uploaded_name or "predictive_maintenance_predictions.csv"
        st.download_button(
            label="Download Processed Predictions CSV",
            data=output_csv,
            file_name=f"predictions_{default_name}",
            mime="text/csv",
            width="stretch",
        )

        total_machines = len(result_df)
        machines_at_risk = int((result_df["Maintenance Status"] != "Healthy").sum())
        average_failure_probability = (
            float(result_df["Failure Probability (%)"].mean()) if total_machines else 0.0
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Machines Scanned", f"{total_machines:,}")
        metric_col2.metric("Machines at Risk", f"{machines_at_risk:,}")
        metric_col3.metric("Average Failure Probability", f"{average_failure_probability:.2f}%")

        st.subheader("Global Feature Importance (SHAP Summary)")
        try:
            shap_input = (
                transformed_df
                if len(transformed_df) <= SHAP_SAMPLE_SIZE
                else transformed_df.sample(SHAP_SAMPLE_SIZE, random_state=42)
            )

            with st.spinner("Computing SHAP summary plot..."):
                shap_values = calculate_shap_values(model, shap_input)
                summary_fig = plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    shap_input,
                    max_display=20,
                    show=False,
                )
                st.pyplot(summary_fig, clear_figure=True)
        except Exception as shap_exc:
            st.warning(f"Unable to compute global SHAP summary: {shap_exc}")

        st.subheader("Individual Drill-down Risk Report")
        identifier_column = choose_identifier_column(result_df)

        if identifier_column:
            option_map = {
                f"{result_df.at[idx, identifier_column]} (row {idx})": idx
                for idx in result_df.index
            }
            selected_label = st.selectbox("Select Machine", list(option_map.keys()))
        else:
            option_map = {f"Row {idx}": idx for idx in result_df.index}
            selected_label = st.selectbox("Select Row", list(option_map.keys()))

        selected_index = option_map[selected_label]
        selected_row = result_df.loc[selected_index]

        report_col1, report_col2, report_col3 = st.columns(3)
        report_col1.metric("Selected Unit", selected_label)
        report_col2.metric("Failure Probability", f"{float(selected_row['Failure Probability (%)']):.2f}%")
        report_col3.metric("Maintenance Status", str(selected_row["Maintenance Status"]))

        status_value = str(selected_row["Maintenance Status"])
        if status_value == "Critical":
            st.error("Risk level is critical. Immediate inspection is recommended.")
        elif status_value == "Warning":
            st.warning("Risk level is warning. Schedule preventive maintenance.")
        else:
            st.success("Risk level is healthy based on current sensor values.")

        try:
            selected_features = transformed_df.loc[[selected_index]]
            local_shap_values = calculate_shap_values(model, selected_features)

            local_importance = pd.DataFrame(
                {
                    "Sensor": selected_features.columns,
                    "Sensor Value (Transformed)": selected_features.iloc[0].values,
                    "SHAP Contribution": local_shap_values[0],
                }
            )
            local_importance["Absolute Contribution"] = local_importance[
                "SHAP Contribution"
            ].abs()
            local_importance = local_importance.sort_values(
                "Absolute Contribution",
                ascending=False,
            )

            st.markdown("Top Risk Drivers For Selected Unit")
            st.dataframe(local_importance.head(15), width="stretch")

            top_contributions = local_importance.head(10).sort_values("SHAP Contribution")
            local_fig, local_ax = plt.subplots(figsize=(10, 5))
            bar_colors = [
                "#c0392b" if contribution > 0 else "#1f77b4"
                for contribution in top_contributions["SHAP Contribution"]
            ]
            local_ax.barh(
                top_contributions["Sensor"],
                top_contributions["SHAP Contribution"],
                color=bar_colors,
            )
            local_ax.axvline(0.0, color="black", linewidth=1)
            local_ax.set_xlabel("SHAP contribution to failure risk")
            local_ax.set_ylabel("Sensor")
            local_ax.set_title("Selected Unit: SHAP Contribution by Sensor")
            st.pyplot(local_fig, clear_figure=True)
        except Exception as local_shap_exc:
            st.warning(f"Unable to generate local SHAP report: {local_shap_exc}")

    st.subheader("Recent Saved Runs")
    history_df = get_user_runs(int(st.session_state["user_id"]), limit=25)
    if history_df.empty:
        st.info("No saved runs yet for this account.")
    else:
        st.dataframe(history_df, width="stretch", height=250)
        run_ids = history_df["run_id"].astype(int).tolist()
        picked_run = st.selectbox("Select saved run to download", run_ids)
        stored = get_saved_result_csv(int(st.session_state["user_id"]), int(picked_run))
        if stored is not None:
            saved_name, saved_csv = stored
            st.download_button(
                label="Download Selected Saved Result CSV",
                data=saved_csv.encode("utf-8"),
                file_name=f"saved_run_{picked_run}_{saved_name}",
                mime="text/csv",
                width="stretch",
            )


if __name__ == "__main__":
    if not st.runtime.exists():
        print("Run this app with: streamlit run src/main.py")
    else:
        main()

