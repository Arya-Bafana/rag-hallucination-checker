# RAG hallucination checker: cheap Stage 1 gate + LettuceDetect (CPU only). Student research prototype.
import html, json, os, re, time
import numpy as np, joblib, streamlit as st, spacy
from spacy.lang.en.stop_words import STOP_WORDS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = 'KRLabsOrg/lettucedect-base-modernbert-en-v1'   # use the spelling that worked in notebook Cell 11
DEEP_DEFAULT = True      # set to False if the free host reports "over its resource limits" (then only the cheap stage runs by default)

st.set_page_config(page_title='RAG hallucination checker', layout='wide')

@st.cache_resource
def load_stage1():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    clf = joblib.load(os.path.join(HERE, 'stage1_clf.joblib'))
    gate = json.load(open(os.path.join(HERE, 'gate.json')))['gate']
    return nlp, clf, gate

@st.cache_resource
def load_detector():
    from lettucedetect.models.inference import HallucinationDetector
    return HallucinationDetector(method='transformer', model_path=MODEL_ID)

NUM = re.compile(r'\d[\d,]*\.?\d*')
def toks(t): return [w for w in re.findall(r'[a-z0-9]+', t.lower()) if w not in STOP_WORDS]

def stage1_prob(nlp, clf, answer, ctx):
    doc = nlp(answer)
    ctx_l = ctx.lower(); ctx_tok = set(toks(ctx))
    ctx_nums = {n.strip('.,') for n in NUM.findall(ctx)}
    rows = []
    for s in doc.sents:
        if not s.text.strip(): continue
        st_ = toks(s.text)
        ov = float(np.mean([w in ctx_tok for w in st_])) if st_ else 1.0
        nums = [n.strip('.,') for n in NUM.findall(s.text)]
        bad_n = sum(n not in ctx_nums for n in nums)
        ents = [e.text.lower() for e in s.ents]
        bad_e = sum(e not in ctx_l for e in ents)
        rows.append((ov, bad_n, len(nums), bad_e, len(ents)))
    if not rows: rows = [(1.0, 0, 0, 0, 0)]
    a = np.array(rows, float)
    feats = [a[:, 0].min(), a[:, 0].mean(), a[:, 1].sum(), a[:, 1].sum() / max(1, a[:, 2].sum()),
             a[:, 3].sum(), a[:, 3].sum() / max(1, a[:, 4].sum()), len(rows), len(toks(answer)), len(answer) / max(1, len(ctx))]
    return float(clf.predict_proba(np.array([feats]))[0, 1])

def highlight(answer, spans):
    spans = sorted(spans, key=lambda s: int(s.get('start', 0)))
    out, pos = [], 0
    for s in spans:
        a, b = int(s.get('start', 0)), int(s.get('end', 0))
        if a < pos or b <= a: continue
        out.append(html.escape(answer[pos:a]))
        out.append('<mark style="background:#ffb3b3">' + html.escape(answer[a:b]) + '</mark>')
        pos = b
    out.append(html.escape(answer[pos:]))
    return ''.join(out)

EX_CTX = "Joe's Pizza is located at 12 Main Street. Hours: Monday to Sunday 11:00-22:00. It serves pizza, pasta and salads and has a 4.5 star rating."
EX_ANS = "Joe's Pizza operates six days a week and is famous for its award-winning tiramisu."

st.title('RAG hallucination checker (CPU only)')
st.write('Paste the **source text** a RAG system was given and the **answer** it wrote. A cheap first stage clears obviously clean answers; suspicious ones can be checked by LettuceDetect, which highlights unsupported parts. *Student research prototype; it can be wrong.*')
if st.button('Load example'):
    st.session_state['ctx'], st.session_state['ans'] = EX_CTX, EX_ANS
ctx = st.text_area('Source text (context)', key='ctx', height=180)
ans = st.text_area('Answer to check', key='ans', height=110)
deep = st.toggle('Hybrid mode: also run the detailed LettuceDetect check on suspicious answers (slower, needs more memory)', value=DEEP_DEFAULT)

if st.button('Check answer', type='primary'):
    if not ctx.strip() or not ans.strip():
        st.warning('Please paste both a source text and an answer.')
    else:
        nlp, clf, gate = load_stage1()
        t0 = time.perf_counter()
        p1 = stage1_prob(nlp, clf, ans, ctx)
        t1 = time.perf_counter() - t0
        if p1 < gate:
            st.success(f'Not flagged: the cheap first stage found nothing suspicious (score {p1:.2f}, gate {gate:.2f}).')
            st.caption(f'Time: {t1:.2f} s (Stage 1 only)')
        elif not deep:
            st.warning(f'Suspicious: the cheap first stage flagged this answer (score {p1:.2f}, gate {gate:.2f}). Turn on hybrid mode for the detailed check.')
            st.caption(f'Time: {t1:.2f} s (Stage 1 only)')
        else:
            with st.spinner('Running the detailed check (the first run downloads the model and is slow)...'):
                det = load_detector()
                spans = det.predict(context=[ctx], question='', answer=ans, output_format='spans')
            t2 = time.perf_counter() - t0
            if spans:
                st.error(f'Possible hallucination: {len(spans)} unsupported span(s) highlighted.')
                st.markdown(highlight(ans, spans), unsafe_allow_html=True)
            else:
                st.success('No unsupported span found by the detailed check.')
            st.caption(f'Time: {t2:.2f} s (Stage 1 + LettuceDetect)')
