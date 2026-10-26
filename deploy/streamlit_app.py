
# RAG hallucination checker: from-scratch Stage 1 gate + LettuceDetect.
# CPU-only student research prototype.

import html
import json
import os
import re
import time

import numpy as np
import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
WEIGHTS_FILE = os.path.join(HERE, "stage1_weights.json")
DEEP_DEFAULT = True

st.set_page_config(page_title="RAG hallucination checker", layout="wide")


@st.cache_resource
def load_stage1():
    """Load spaCy and the from-scratch Logistic Regression weights."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        model = json.load(f)

    # Validate the model artefact before using it.
    required = {"mean", "std", "w", "b", "log_cols", "gate"}
    if not required.issubset(model):
        raise ValueError(
            f"stage1_weights.json is missing keys: {required - set(model)}"
        )

    if not (
        len(model["mean"]) == len(model["std"]) == len(model["w"]) == 9
    ):
        raise ValueError("Expected nine Stage 1 features in the JSON file.")

    if any(float(s) == 0 for s in model["std"]):
        raise ValueError("A feature standard deviation is zero.")

    return nlp, model


@st.cache_resource
def load_detector():
    """Load the more expensive LettuceDetect model."""
    from lettucedetect.models.inference import HallucinationDetector

    return HallucinationDetector(
        method="transformer",
        model_path=MODEL_ID,
    )


NUM = re.compile(r"\d[\d,]*\.?\d*")


def toks(text):
    """Lowercase and tokenise text, excluding English stopwords."""
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    ]


def extract_features(nlp, answer, ctx):
    """
    Produce the nine features in the training order:
    ov_min, ov_mean, bad_num, num_frac, bad_ent, ent_frac,
    n_sent, n_tok, len_ratio.
    """
    doc = nlp(answer)
    ctx_lower = ctx.lower()
    ctx_tokens = set(toks(ctx))
    ctx_numbers = {n.strip(".,") for n in NUM.findall(ctx)}

    rows = []

    for sentence in doc.sents:
        if not sentence.text.strip():
            continue

        sentence_tokens = toks(sentence.text)

        overlap = (
            float(np.mean([word in ctx_tokens for word in sentence_tokens]))
            if sentence_tokens
            else 1.0
        )

        numbers = [
            number.strip(".,")
            for number in NUM.findall(sentence.text)
        ]
        bad_numbers = sum(number not in ctx_numbers for number in numbers)

        entities = [entity.text.lower() for entity in sentence.ents]
        bad_entities = sum(entity not in ctx_lower for entity in entities)

        rows.append(
            (
                overlap,
                bad_numbers,
                len(numbers),
                bad_entities,
                len(entities),
            )
        )

    if not rows:
        rows = [(1.0, 0, 0, 0, 0)]

    a = np.asarray(rows, dtype=float)

    features = np.array(
        [
            a[:, 0].min(),
            a[:, 0].mean(),
            a[:, 1].sum(),
            a[:, 1].sum() / max(1, a[:, 2].sum()),
            a[:, 3].sum(),
            a[:, 3].sum() / max(1, a[:, 4].sum()),
            len(rows),
            len(toks(answer)),
            len(answer) / max(1, len(ctx)),
        ],
        dtype=float,
    )

    return features


def stage1_prob(nlp, model, answer, ctx):
    """
    Apply the same log transforms and standardisation used in training,
    then calculate sigmoid(w dot z + b).
    """
    x = extract_features(nlp, answer, ctx)

    # Apply log1p to the saved feature indices.
    log_cols = np.asarray(model["log_cols"], dtype=int)
    x[log_cols] = np.log1p(np.maximum(x[log_cols], 0.0))

    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    weights = np.asarray(model["w"], dtype=float)
    bias = float(model["b"])

    # Standardise features using the training-set statistics.
    z = (x - mean) / std

    # Logistic Regression score and sigmoid probability.
    score = float(np.dot(weights, z) + bias)
    score = np.clip(score, -500.0, 500.0)
    probability = 1.0 / (1.0 + np.exp(-score))

    return float(probability)


def highlight(answer, spans):
    """Highlight text spans returned by LettuceDetect."""
    spans = sorted(spans, key=lambda s: int(s.get("start", 0)))
    output = []
    position = 0

    for span in spans:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))

        # Ignore invalid or overlapping spans.
        if start < position or end <= start or end > len(answer):
            continue

        output.append(html.escape(answer[position:start]))
        output.append(
            '<mark style="background:#ffb3b3">'
            + html.escape(answer[start:end])
            + "</mark>"
        )
        position = end

    output.append(html.escape(answer[position:]))
    return "".join(output)


EX_CTX = (
    "Joe's Pizza is located at 12 Main Street. "
    "Hours: Monday to Sunday 11:00-22:00. "
    "It serves pizza, pasta and salads and has a 4.5 star rating."
)
EX_ANS = (
    "Joe's Pizza operates six days a week and is famous "
    "for its award-winning tiramisu."
)


st.title("RAG hallucination checker (CPU only)")
st.write(
    "Paste the **source text** a RAG system was given and the **answer** "
    "it generated. A lightweight first-stage classifier estimates risk. "
    "Suspicious answers can receive a detailed LettuceDetect check, "
    "which may highlight unsupported spans. "
    "*Student research prototype; results can be wrong.*"
)

if st.button("Load example"):
    st.session_state["ctx"] = EX_CTX
    st.session_state["ans"] = EX_ANS

ctx = st.text_area("Source text (context)", key="ctx", height=180)
ans = st.text_area("Answer to check", key="ans", height=110)

deep = st.toggle(
    "Hybrid mode: run LettuceDetect on suspicious answers "
    "(slower and requires more memory)",
    value=DEEP_DEFAULT,
)

if st.button("Check answer", type="primary"):
    if not ctx.strip() or not ans.strip():
        st.warning("Please paste both a source text and an answer.")
    else:
        try:
            nlp, model = load_stage1()
            gate = float(model["gate"])

            start_time = time.perf_counter()
            p1 = stage1_prob(nlp, model, ans, ctx)
            stage1_time = time.perf_counter() - start_time

            st.write(f"**Stage 1 risk score:** {p1:.3f}")
            st.write(f"**Gate threshold:** {gate:.2f}")

            if p1 < gate:
                st.info(
                    "Stage 1 did not flag this answer, so LettuceDetect "
                    "was skipped. This is not proof that the answer is correct."
                )
                st.caption(
                    f"Time: {stage1_time:.2f} s (Stage 1 only)"
                )

            elif not deep:
                st.warning(
                    "Stage 1 flagged this answer as suspicious, but the "
                    "detailed check has not run. Turn on hybrid mode to "
                    "request the LettuceDetect check."
                )
                st.caption(
                    f"Time: {stage1_time:.2f} s (Stage 1 only)"
                )

            else:
                with st.spinner(
                    "Running LettuceDetect. The first run may download "
                    "the model and take longer."
                ):
                    detector = load_detector()
                    spans = detector.predict(
                        context=[ctx],
                        question="",
                        answer=ans,
                        output_format="spans",
                    )

                total_time = time.perf_counter() - start_time

                if spans:
                    st.error(
                        f"Possible hallucination: {len(spans)} "
                        "unsupported span(s) returned by LettuceDetect."
                    )
                    st.markdown(
                        highlight(ans, spans),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success(
                        "LettuceDetect returned no unsupported spans. "
                        "This does not guarantee that every claim is correct."
                    )

                st.caption(
                    f"Time: {total_time:.2f} s "
                    "(Stage 1 + LettuceDetect)"
                )

        except Exception as error:
            st.error(
                "The check failed. Review the Streamlit deployment logs "
                "for details."
            )
            st.exception(error)
