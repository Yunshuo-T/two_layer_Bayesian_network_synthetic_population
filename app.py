import streamlit as st
from omegaconf import OmegaConf

from src.train import run_training
from src.synthesize import run_synthesize


CONFIG_PATH = "config/config.yaml"


def load_config():
    return OmegaConf.load(CONFIG_PATH)


st.set_page_config(page_title="Synthetic Population Generator")
st.title("Synthetic Population Generator")

cfg = load_config()

with st.form("configuration_form"):
    st.header("Training")

    raw_h_df = st.text_input(
        "Household input file",
        value=cfg.train.raw_h_df,
    )
    raw_p_df = st.text_input(
        "Person input file",
        value=cfg.train.raw_p_df,
    )

    st.header("Synthesis")

    num_households = st.number_input(
        "Number of households",
        min_value=1,
        value=cfg.synthesis.num_households,
    )
    pool_multiplier = st.number_input(
        "Pool multiplier",
        min_value=1,
        value=cfg.synthesis.pool_multiplier,
    )
    max_retries = st.number_input(
        "Maximum retries",
        min_value=0,
        value=cfg.synthesis.max_retries,
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        value=cfg.synthesis.random_seed,
    )
    max_workers = st.number_input(
        "Maximum workers",
        min_value=1,
        value=cfg.synthesis.max_workers,
    )
    include_kids = st.checkbox(
        "Include kids",
        value=cfg.synthesis.p_data_include_kid,
    )

    st.header("Calibration")

    calibration_method = st.selectbox(
        "Calibration method",
        ["None", "ipu", "gr", "hipf", "cross_entropy"],
        index=0,
    )

    save_config = st.form_submit_button("Save configuration")

if save_config:
    OmegaConf.update(cfg, "train.raw_h_df", raw_h_df)
    OmegaConf.update(cfg, "train.raw_p_df", raw_p_df)

    OmegaConf.update(cfg, "synthesis.num_households", num_households)
    OmegaConf.update(cfg, "synthesis.pool_multiplier", pool_multiplier)
    OmegaConf.update(cfg, "synthesis.max_retries", max_retries)
    OmegaConf.update(cfg, "synthesis.random_seed", random_seed)
    OmegaConf.update(cfg, "synthesis.max_workers", max_workers)
    OmegaConf.update(cfg, "synthesis.p_data_include_kid", include_kids)

    OmegaConf.save(cfg, CONFIG_PATH)
    st.success("Configuration saved.")

st.divider()

if st.button("Train models"):
    with st.spinner("Training models..."):
        run_training(CONFIG_PATH)
    st.success("Training completed.")

if st.button("Generate population"):
    with st.spinner("Generating population..."):
        run_synthesize(
            calibration=(
                None if calibration_method == "None" else calibration_method
            ), # type: ignore
            config_path=CONFIG_PATH,
        )
    st.success("Population generation completed.")