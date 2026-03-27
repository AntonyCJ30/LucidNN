import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Regression Lab", page_icon="📈", layout="wide")

MAX_POINTS = 300
DEFAULT_POINTS = 60
DEFAULT_EPOCHS = 300


# -----------------------------
# Data generators
# -----------------------------
def make_linear_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = np.linspace(-8, 8, n)
    y = 1.8 * x + 3.5 + rng.normal(0, 2.2, n)
    return pd.DataFrame({"x": x, "y": y})


def make_ellipse_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    x = np.linspace(-10, 10, n)
    scale = 10.0
    arc = np.sqrt(np.clip(1 - (x / scale) ** 2, 0, None))
    y = 4.0 + 5.2 * arc + 0.3 * x + rng.normal(0, 0.45, n)
    return pd.DataFrame({"x": x, "y": y})


def make_logistic_data(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    x = np.linspace(-8, 8, n)
    probs = 1 / (1 + np.exp(-(-1.2 + 0.95 * x)))
    y = (rng.random(n) < probs).astype(int)
    return pd.DataFrame({"x": x, "y": y})


# -----------------------------
# Training loops
# -----------------------------
def train_linear(x: np.ndarray, y: np.ndarray, lr: float, epochs: int):
    params = np.zeros(2, dtype=float)  # b0, b1
    losses = []

    for _ in range(epochs):
        pred = params[0] + params[1] * x
        err = pred - y
        loss = np.mean(err ** 2)

        grad_b0 = 2.0 * np.mean(err)
        grad_b1 = 2.0 * np.mean(err * x)

        params[0] -= lr * grad_b0
        params[1] -= lr * grad_b1
        losses.append(loss)

    return params, losses


def train_ellipse_nonlinear(x: np.ndarray, y: np.ndarray, lr: float, epochs: int):
    # Ellipse-inspired basis phi(x) = sqrt(max(0, 1 - (x/s)^2))
    s = max(float(np.max(np.abs(x))), 1e-6)
    params = np.zeros(3, dtype=float)  # b0, b1, b2
    losses = []

    for _ in range(epochs):
        phi = np.sqrt(np.clip(1 - (x / s) ** 2, 0, None))
        pred = params[0] + params[1] * x + params[2] * phi
        err = pred - y
        loss = np.mean(err ** 2)

        grad_b0 = 2.0 * np.mean(err)
        grad_b1 = 2.0 * np.mean(err * x)
        grad_b2 = 2.0 * np.mean(err * phi)

        params[0] -= lr * grad_b0
        params[1] -= lr * grad_b1
        params[2] -= lr * grad_b2
        losses.append(loss)

    return params, losses, s


def train_logistic(x: np.ndarray, y: np.ndarray, lr: float, epochs: int):
    params = np.zeros(2, dtype=float)  # w0, w1
    losses = []
    eps = 1e-9

    for _ in range(epochs):
        z = params[0] + params[1] * x
        pred = 1 / (1 + np.exp(-z))

        loss = -np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps))

        grad_w0 = np.mean(pred - y)
        grad_w1 = np.mean((pred - y) * x)

        params[0] -= lr * grad_w0
        params[1] -= lr * grad_w1
        losses.append(loss)

    return params, losses


# -----------------------------
# UI helpers
# -----------------------------
def get_default_df(model_type: str, n_points: int) -> pd.DataFrame:
    if model_type == "Linear Regression":
        return make_linear_data(n_points)
    if model_type == "Non-Linear Regression (Ellipse Basis)":
        return make_ellipse_data(n_points)
    return make_logistic_data(n_points)


def sanitize_df(df: pd.DataFrame, logistic: bool):
    work = df.copy()
    work = work[["x", "y"]]
    work["x"] = pd.to_numeric(work["x"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["x", "y"]).reset_index(drop=True)

    if logistic:
        work["y"] = (work["y"] >= 0.5).astype(int)

    return work


def plot_regression_points_and_curve(points_df: pd.DataFrame, curve_df: pd.DataFrame, y_title: str):
    points = (
        alt.Chart(points_df)
        .mark_circle(size=55, color="#245e4f")
        .encode(x="x:Q", y="y:Q", tooltip=["x", "y"])
    )

    curve = (
        alt.Chart(curve_df)
        .mark_line(color="#c1392b", strokeWidth=3)
        .encode(x="x:Q", y="prediction:Q")
    )

    chart = (
        (points + curve)
        .properties(height=360)
        .configure_axis(grid=True)
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(y_title)


def plot_convergence(losses: list[float], metric_name: str):
    loss_df = pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses})
    st.line_chart(loss_df, x="epoch", y="loss", height=260)
    st.caption(metric_name)


# -----------------------------
# Page layout
# -----------------------------
st.title("Regression Lab 📈")
st.caption(
    "Train and compare three regression families with your own data or built-in datasets, "
    "and observe convergence across epochs."
)
st.markdown("---")

c1, c2, c3 = st.columns([1.25, 1.15, 1.35])

with c1:
    model_type = st.selectbox(
        "Regression model",
        [
            "Linear Regression",
            "Non-Linear Regression (Ellipse Basis)",
            "Logistic Regression",
        ],
    )

with c2:
    source_type = st.radio("Data source", ["Default data", "Custom data"], horizontal=True)

with c3:
    n_points = st.slider("Point count", min_value=10, max_value=MAX_POINTS, value=DEFAULT_POINTS)

with st.expander("Training settings", expanded=True):
    if model_type == "Logistic Regression":
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.12, step=0.01, format="%.4f")
    else:
        lr = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")

    epochs = st.slider("Epochs", min_value=20, max_value=2000, value=DEFAULT_EPOCHS, step=20)

st.markdown("---")

is_logistic = model_type == "Logistic Regression"

if source_type == "Default data":
    data_df = get_default_df(model_type, n_points)
else:
    st.info(f"Provide up to {MAX_POINTS} rows. Use columns x and y.")
    if is_logistic:
        st.caption("For logistic regression, y values are automatically converted to classes: y >= 0.5 -> 1, otherwise 0.")

    default_seed = get_default_df(model_type, min(20, n_points))
    custom_key = f"custom_{model_type.replace(' ', '_')}"
    upload_key = f"upload_{model_type.replace(' ', '_')}"

    if custom_key not in st.session_state:
        st.session_state[custom_key] = default_seed

    uploaded_csv = st.file_uploader(
        "Import custom data from CSV",
        type=["csv"],
        key=upload_key,
        help="CSV must include x and y columns. If missing, the first two columns are used.",
    )

    if uploaded_csv is not None:
        try:
            imported = pd.read_csv(uploaded_csv)
            imported.columns = [str(c).strip().lower() for c in imported.columns]

            if "x" in imported.columns and "y" in imported.columns:
                imported = imported[["x", "y"]]
            elif imported.shape[1] >= 2:
                imported = imported.iloc[:, :2].copy()
                imported.columns = ["x", "y"]
                st.warning("CSV had no explicit x/y headers. Using first two columns as x and y.")
            else:
                st.error("CSV must contain at least two columns for x and y.")
                imported = None

            if imported is not None:
                st.session_state[custom_key] = imported
                st.success(f"Imported {len(imported)} rows from CSV.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    data_df = st.data_editor(
        st.session_state[custom_key],
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{custom_key}",
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.6f"),
            "y": st.column_config.NumberColumn("y", format="%.6f"),
        },
    )

    if len(data_df) > MAX_POINTS:
        st.warning(f"Only first {MAX_POINTS} rows are used.")
        data_df = data_df.iloc[:MAX_POINTS].copy()

data_df = sanitize_df(data_df, logistic=is_logistic)

if len(data_df) < 3:
    st.error("Need at least 3 valid points to train. Please add more rows.")
    st.stop()

x = data_df["x"].to_numpy(dtype=float)
y = data_df["y"].to_numpy(dtype=float)

training_requested = st.button("Train selected model", type="primary", use_container_width=True)

if training_requested:
    with st.spinner("Training in progress..."):
        if model_type == "Linear Regression":
            params, losses = train_linear(x, y, lr, epochs)
            x_grid = np.linspace(np.min(x), np.max(x), 250)
            y_pred_grid = params[0] + params[1] * x_grid
            y_pred_points = params[0] + params[1] * x

            r2_den = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - np.sum((y - y_pred_points) ** 2) / (r2_den + 1e-9)

            left, right = st.columns(2)
            with left:
                st.subheader("Model fit")
                curve_df = pd.DataFrame({"x": x_grid, "prediction": y_pred_grid})
                plot_regression_points_and_curve(data_df, curve_df, "Observed vs fitted line")

            with right:
                st.subheader("Convergence")
                plot_convergence(losses, "Loss metric: Mean Squared Error")
                st.metric("Final MSE", f"{losses[-1]:.6f}")
                st.metric("R^2", f"{r2:.4f}")
                st.code(f"y_hat = {params[0]:.4f} + ({params[1]:.4f}) * x", language="text")

        elif model_type == "Non-Linear Regression (Ellipse Basis)":
            params, losses, s = train_ellipse_nonlinear(x, y, lr, epochs)
            x_grid = np.linspace(np.min(x), np.max(x), 250)
            phi_grid = np.sqrt(np.clip(1 - (x_grid / s) ** 2, 0, None))
            y_pred_grid = params[0] + params[1] * x_grid + params[2] * phi_grid

            phi_points = np.sqrt(np.clip(1 - (x / s) ** 2, 0, None))
            y_pred_points = params[0] + params[1] * x + params[2] * phi_points
            r2_den = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - np.sum((y - y_pred_points) ** 2) / (r2_den + 1e-9)

            left, right = st.columns(2)
            with left:
                st.subheader("Model fit")
                curve_df = pd.DataFrame({"x": x_grid, "prediction": y_pred_grid})
                plot_regression_points_and_curve(data_df, curve_df, "Observed vs ellipse-basis fitted curve")

            with right:
                st.subheader("Convergence")
                plot_convergence(losses, "Loss metric: Mean Squared Error")
                st.metric("Final MSE", f"{losses[-1]:.6f}")
                st.metric("R^2", f"{r2:.4f}")
                st.code(
                    "y_hat = b0 + b1*x + b2*sqrt(max(0, 1 - (x/s)^2))\n"
                    f"b0={params[0]:.4f}, b1={params[1]:.4f}, b2={params[2]:.4f}, s={s:.4f}",
                    language="text",
                )

        else:
            params, losses = train_logistic(x, y, lr, epochs)
            x_grid = np.linspace(np.min(x), np.max(x), 250)
            p_grid = 1 / (1 + np.exp(-(params[0] + params[1] * x_grid)))
            p_points = 1 / (1 + np.exp(-(params[0] + params[1] * x)))

            preds_cls = (p_points >= 0.5).astype(int)
            accuracy = float(np.mean(preds_cls == y))

            left, right = st.columns(2)
            with left:
                st.subheader("Model fit")
                curve_df = pd.DataFrame({"x": x_grid, "prediction": p_grid})
                plot_regression_points_and_curve(data_df, curve_df, "Observed classes (0/1) vs fitted probability")

            with right:
                st.subheader("Convergence")
                plot_convergence(losses, "Loss metric: Binary Cross Entropy")
                st.metric("Final BCE", f"{losses[-1]:.6f}")
                st.metric("Training Accuracy", f"{accuracy * 100:.2f}%")
                st.code(
                    f"p(y=1|x) = sigmoid({params[0]:.4f} + ({params[1]:.4f}) * x)",
                    language="text",
                )

    st.success("Training complete. Adjust model, data, or hyperparameters and train again.")

st.markdown("---")
st.caption(
    "Tip: Start with default data to understand behavior, then switch to custom data and paste your own points."
)
