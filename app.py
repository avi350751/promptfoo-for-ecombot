import os
import json
import io
import streamlit as st
from helper import (
    extract_text_from_html,
    extract_text_from_json,
    summarize_with_llm,
    approx_token_count,
    ResultBundle,
)

st.set_page_config(page_title="Evaluation Results Summarizer", page_icon="🧪", layout="wide")

st.header("Evaluation Results Summarizer")
st.caption("Upload an evaluation result (HTML or JSON), then generate a summary, notes, or both with an LLM.")

with st.sidebar:
    st.title("Option Panel")
    st.markdown("—" * 15)

    task = st.radio(
        "Choose what you want to do",
        ["Summarize the content", "Notes for you", "Both"],
        index=0,
    )

    desired_length = st.selectbox(
        "Target length",
        ["very short", "short", "medium", "long", "very long"],
        index=2,
        help="Guides the LLM on how much to write."
    )

    notes_bullets = st.slider(
        "Notes: number of bullets",
        min_value=3, max_value=15, value=7, step=1
    )

    temperature = st.slider(
        "LLM temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Higher = more creative; lower = more deterministic."
    )

    extra_instructions = st.text_area(
        "Extra instructions (optional)",
        placeholder="e.g., focus on failure cases, call out top regressions, keep JSON schema violations prominent"
    )

    st.markdown("—" * 10)
    submit_btn = st.button("Submit your choice", use_container_width=True)

st.markdown("---")

uploaded = st.file_uploader(
    "Upload your evaluation file (.html or .json)",
    type=["html", "json"]
)

if submit_btn:
    if not uploaded:
        st.error("Please upload an evaluation file first.")
        st.stop()

    # Read file bytes
    data = uploaded.read()

    # Parse to text
    if uploaded.name.lower().endswith(".html"):
        src_kind = "html"
        text = extract_text_from_html(data)
    else:
        src_kind = "json"
        try:
            text = extract_text_from_json(data)
        except json.JSONDecodeError as e:
            st.error(f"The JSON file couldn't be parsed: {e}")
            st.stop()

    if not text or text.strip() == "":
        st.warning("No meaningful text was extracted from the file.")
        st.stop()

    # Show some quick stats
    with st.expander("Source & size"):
        st.write(f"**File:** `{uploaded.name}`  |  **Detected as:** `{src_kind}`")
        st.write(f"**Approx tokens:** ~{approx_token_count(text):,}")
        st.text_area("First 800 characters (preview)", text[:800], height=150)

    # Summarize / Notes / Both
    with st.spinner("Calling the LLM…"):
        try:
            result: ResultBundle = summarize_with_llm(
                raw_text=text,
                mode=task,  # "Summarize the content" | "Notes for you" | "Both"
                desired_length=desired_length,
                notes_bullets=notes_bullets,
                temperature=temperature,
                extra_instructions=extra_instructions.strip() if extra_instructions else ""
            )
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    # Render results
    if result.summary_md:
        st.subheader("📄 Summary")
        st.markdown(result.summary_md)

    if result.notes_md:
        st.subheader("🗒️ Notes for You")
        st.markdown(result.notes_md)

    # Download bundle
    if result.summary_md or result.notes_md:
        bundle_md = result.to_markdown()
        st.download_button(
            "Download Results (Markdown)",
            data=bundle_md.encode("utf-8"),
            file_name="evaluation_summary_and_notes.md",
            mime="text/markdown"
        )

    st.success("Done!")
    st.markdown("---")