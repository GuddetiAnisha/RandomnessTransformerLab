from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from randomness_lab.experiment import ExperimentConfig, run_experiment
from randomness_lab.sources import GENERATORS, generate, load_bits
from randomness_lab.statistics import analyze

st.set_page_config(page_title="RandomnessTransformerLab", page_icon="🎲", layout="wide")
st.title("RandomnessTransformerLab")
st.caption("Compact Transformer analysis of binary sources — research prototype, not a NIST validation tool")

with st.sidebar:
    st.header("Binary sequence")
    mode = st.radio("Input", ["Generated source", "Uploaded file"])
    source = st.selectbox("Source", sorted(GENERATORS), index=sorted(GENERATORS).index("markov"),
                          disabled=mode != "Generated source")
    uploaded = st.file_uploader("Binary or ASCII file", disabled=mode != "Uploaded file")
    fmt = st.selectbox("File format", ["raw", "ascii", "hex", "npy", "csv"], disabled=mode != "Uploaded file")
    num_bits = st.number_input("Generated bits", 5000, 1000000, 50000, 5000)
    seed = st.number_input("Seed", 0, 1000000, 42)
    analyze_button = st.button("Analyze source", type="primary", use_container_width=True)
    st.header("Transformer")
    context = st.select_slider("Context length", [16, 32, 64, 128], value=64)
    target_bits = st.select_slider("Target bits", list(range(1, 9)), value=1)
    epochs = st.slider("Epochs", 1, 10, 3)
    train_button = st.button("Train compact model", use_container_width=True)


def obtain_bits():
    if mode == "Generated source":
        return generate(source, int(num_bits), int(seed))
    if not uploaded:
        st.error("Upload a file first.")
        return None
    suffix = Path(uploaded.name).suffix
    path = Path(tempfile.gettempdir()) / f"randomness_upload{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return load_bits(path, fmt)


if analyze_button or train_button:
    bits = obtain_bits()
    if bits is not None:
        summary = analyze(bits)
        a, b, c, d = st.columns(4)
        a.metric("Bits", f"{summary['num_bits']:,}")
        b.metric("Proportion of ones", f"{summary['proportion_ones']:.4f}")
        c.metric("Markov min-entropy", f"{summary['markov_min_entropy_per_bit']:.4f} bit")
        d.metric("Failed baseline tests", len(summary["failed_tests"]))
        test_rows = [{"test": name, "p_value": result.get("p_value", 0),
                      "status": "Pass" if result.get("p_value", 0) >= summary["alpha"] else "Investigate"}
                     for name, result in summary["tests"].items()]
        st.subheader("Classical indicators")
        st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(pd.DataFrame(test_rows), x="test", y="p_value", color="status",
                               log_y=True, title="Exploratory p-values"), use_container_width=True)
        if train_button:
            with st.spinner("Training and evaluating the compact Transformer..."):
                config = ExperimentConfig(seed=int(seed), context_length=context,
                                          target_bits=target_bits, epochs=epochs,
                                          batch_size=256, stride=max(1, context // 8))
                result = run_experiment(bits, config)
            st.subheader("Transformer results")
            x, y, z = st.columns(3)
            x.metric("Test accuracy", f"{result['test']['accuracy']:.3f}")
            y.metric("Majority baseline", f"{result['baseline']['majority_accuracy']:.3f}")
            z.metric("Cross-entropy", f"{result['test']['cross_entropy_bits_per_bit']:.3f} bit/bit")
            history = pd.DataFrame(result["training"]["history"])
            st.plotly_chart(px.line(history, x="epoch", y=["train_loss", "validation_loss"],
                                    title="Training history"), use_container_width=True)
            st.json(result["test"])

st.warning("A low p-value or successful neural predictor indicates a dependency worth investigating; it does not identify its physical cause or prove that a source is non-quantum.")
