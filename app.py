import streamlit as st
import torch
import json
import re
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Care Triage Assistant",
    page_icon="🏥",
    layout="centered",
)

# ============================================================
# LOAD MODEL + VOCAB (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-triage-final")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-triage-final")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource
def load_vocab():
    with open("symptom_vocab.json", "r") as f:
        return set(json.load(f))

tokenizer, model, device = load_model()
symptom_vocab = load_vocab()

# ============================================================
# SAFETY LISTS
# ============================================================
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "unresponsive", "stroke", "seizure",
    "overdose", "severe allergic reaction", "anaphylaxis",
    "blue lips", "crushing pain", "not breathing", "collapsed",
    "worst headache of my life", "sudden confusion", "can't move"
]

SELF_HARM_KEYWORDS = [
    "hurt myself", "kill myself", "suicidal", "suicide", "end my life",
    "don't want to be here", "don't want to live", "want to die",
    "self harm", "self-harm", "harm myself"
]

STOPWORDS = {"i", "the", "a", "an", "and", "or", "is", "are", "have", "having",
             "been", "my", "me", "to", "in", "on", "at", "of", "it", "with",
             "also", "for", "im", "ive", "youve", "id"}

CONFIDENCE_THRESHOLD = 0.5

# Diagnosis -> plain "possible concern" phrasing + recommended action
URGENCY_ACTION_MAP = {
    "routine": {
        "action": "Consider scheduling a routine appointment with a doctor in the next few days.",
        "color": "#2e7d32",
        "label": "🟢 Routine",
    },
    "urgent": {
        "action": "Consider seeing a doctor or visiting an urgent care clinic within 24 hours.",
        "color": "#e65100",
        "label": "🟠 Urgent",
    },
    "emergency": {
        "action": "Seek immediate emergency care or call emergency services now.",
        "color": "#c62828",
        "label": "🔴 Emergency",
    },
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def is_plausible_symptom_text(text, min_overlap=1):
    words = set(re.findall(r"[a-z]+", text.lower())) - STOPWORDS
    words = {w for w in words if len(w) > 2}
    overlap = words & symptom_vocab
    return len(overlap) >= min_overlap, overlap


def run_model(symptom_text):
    """Generate a response from the fine-tuned model and return
    (raw_text, diagnosis, urgency, confidence)."""
    prompt = f"Patient symptoms: {symptom_text}\nTriage assessment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs, max_new_tokens=60, do_sample=True, top_p=0.9,
        temperature=0.4, pad_token_id=tokenizer.eos_token_id,
        output_scores=True, return_dict_in_generate=True
    )

    scores = output.scores
    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # confidence only over the diagnosis/urgency portion, not the fixed boilerplate
    cutoff_text = result_text.split("This is a preliminary")[0]
    cutoff_token_count = len(tokenizer(cutoff_text)["input_ids"])

    probs = []
    for i in range(min(cutoff_token_count, len(scores))):
        prob_dist = torch.softmax(scores[i][0], dim=-1)
        token_id = generated_ids[i]
        probs.append(prob_dist[token_id].item())
    avg_confidence = sum(probs) / len(probs) if probs else 0

    diag_match = re.search(r"indicate \*\*(.+?)\*\*", result_text)
    urg_match = re.search(r"urgency: \*\*(.+?)\*\*", result_text)
    diagnosis = diag_match.group(1) if diag_match else "unclear"
    urgency = urg_match.group(1) if urg_match else "urgent"  # safe default

    return result_text, diagnosis, urgency, avg_confidence


def triage(symptom_text):
    """Returns a dict describing the structured triage result."""
    text_lower = symptom_text.lower()

    # 1. Self-harm — highest priority, never touches the model
    for kw in SELF_HARM_KEYWORDS:
        if kw in text_lower:
            return {
                "type": "self_harm",
                "triage_level": "🔴 Immediate Support Needed",
                "concern": "Your message suggests you may be in emotional distress.",
                "action": ("Please reach out for support right now:\n\n"
                           "- Call or text **988** (Suicide & Crisis Lifeline, US)\n"
                           "- UK: **116 123** (Samaritans)\n"
                           "- Or contact your local emergency number"),
                "reasoning": "Detected language associated with self-harm risk. "
                             "This response is rule-based and intentionally bypasses "
                             "the diagnostic model — no diagnosis is generated for this category.",
                "confidence": None,
                "color": "#6a1b9a",
            }

    # 2. Physical emergency keywords — bypass the model
    for kw in EMERGENCY_KEYWORDS:
        if kw in text_lower:
            return {
                "type": "emergency",
                "triage_level": "🔴 Emergency",
                "concern": f"Your description includes a red-flag symptom ('{kw}').",
                "action": "Seek immediate emergency care or call emergency services now.",
                "reasoning": f"Matched hard-coded emergency keyword: \"{kw}\". "
                             f"This overrides any model prediction, per the system's safety design.",
                "confidence": None,
                "color": "#c62828",
            }

    # 3. Plausibility check
    plausible, overlap = is_plausible_symptom_text(symptom_text)
    if not plausible:
        return {
            "type": "unclear",
            "triage_level": "⚪ Unable to Assess",
            "concern": "This text doesn't appear to describe medical symptoms.",
            "action": "Please describe what you're physically experiencing, "
                      "or consult a trained professional directly.",
            "reasoning": "No overlap found with symptom-related vocabulary from the training data.",
            "confidence": None,
            "color": "#616161",
        }

    # 4. Model generation
    raw_text, diagnosis, urgency, confidence = run_model(symptom_text)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "type": "low_confidence",
            "triage_level": "⚪ Low Confidence",
            "concern": "The model could not confidently interpret these symptoms.",
            "action": "Please consult a trained professional directly rather than relying on this suggestion.",
            "reasoning": f"Model confidence ({confidence:.2f}) fell below the "
                         f"{CONFIDENCE_THRESHOLD:.2f} threshold required to show a suggestion.",
            "confidence": confidence,
            "color": "#616161",
        }

    urgency_info = URGENCY_ACTION_MAP.get(urgency, URGENCY_ACTION_MAP["urgent"])
    matched_words = ", ".join(sorted(overlap)[:6]) if overlap else "general symptom pattern"

    return {
        "type": "model_result",
        "triage_level": urgency_info["label"],
        "concern": f"This could possibly indicate **{diagnosis}** (not a definitive diagnosis).",
        "action": urgency_info["action"],
        "reasoning": f"Based on symptom terms recognized by the model (e.g. {matched_words}), "
                     f"the described pattern most closely resembles cases labeled "
                     f"**{diagnosis}** in training, with an associated **{urgency}** urgency level.",
        "confidence": confidence,
        "color": urgency_info["color"],
    }


# ============================================================
# UI
# ============================================================
st.title("🏥 Care Triage Assistant")
st.caption("Every symptom deserves attention. We're here to help guide your next step.")

st.warning(
    "⚠️ **Decision-support tool only**. " \
    "This assessment is a preliminary suggestion to assist prioritization and must be confirmed by a qualified clinician before any action is taken."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("triage_form", clear_on_submit=False):
    symptoms = st.text_area(
        "Describe your symptoms:",
        height=110,
        placeholder="e.g. I've had a sore throat and mild fever since yesterday...",
    )
    submitted = st.form_submit_button("🔍 Get Triage Assessment", use_container_width=True)

if submitted:
    if not symptoms.strip():
        st.error("Please enter a symptom description first.")
    else:
        with st.spinner("Analyzing symptoms..."):
            result = triage(symptoms)

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "input": symptoms,
            "result": result,
        })

        st.markdown("---")

        st.markdown(
            f"""
            <div style="border-left: 6px solid {result['color']}; padding: 14px 18px;
                        border-radius: 6px; background-color: rgba(255,255,255,0.04); margin-bottom: 14px;">
                <h3 style="margin-top:0;">Triage Level: {result['triage_level']}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🩺 Possible Concern**")
            st.write(result["concern"])
        with col2:
            st.markdown("**✅ Recommended Action**")
            st.write(result["action"])

        st.markdown("**🧠 Reasoning**")
        st.write(result["reasoning"])

        if result["confidence"] is not None:
            st.progress(min(result["confidence"], 1.0), text=f"Model confidence: {result['confidence']:.0%}")

        st.caption(
            "This suggestion is generated for educational purposes only and has not been "
            "clinically validated. Please have a trained professional confirm any assessment."
        )

# ============================================================
# SIDEBAR — history + about
# ============================================================
with st.sidebar:
    st.header("📜 Session History")
    if not st.session_state.history:
        st.caption("No assessments yet this session.")
    else:
        for entry in st.session_state.history[:8]:
            with st.expander(f"{entry['time']} — {entry['input'][:30]}..."):
                st.write(f"**Input:** {entry['input']}")
                st.write(f"**Triage Level:** {entry['result']['triage_level']}")
                st.write(f"**Concern:** {entry['result']['concern']}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
 
    st.markdown("---")
    st.header("ℹ️ About this tool")
    st.caption(
        "A GPT-2 model fine-tuned on a small symptom-to-diagnosis dataset "
        "(gretelai/symptom_to_diagnosis) to support triage prioritization. "
        "Rule-based safety layers handle emergencies and self-harm language, "
        "a symptom-plausibility check filters unclear input, and a confidence "
        "threshold withholds low-certainty suggestions — the model never has "
        "the final say on high-risk cases."
    )
 
    st.markdown("**👥 Team**")
    st.caption("Allen Lyimo · Claverfred Mhidze · Frank Mtimbili")
 
    st.markdown("**🔗 Project Repository**")
    st.markdown("[View on GitHub](https://github.com/lyimoa/Health-Triage-Model-GPT-2.git)")