import os
import json
import time
from datetime import datetime
import requests
import streamlit as st

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Car Chatbox", page_icon="💬", layout="wide")

DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ----------------------------
# Visual CSS
# ----------------------------
st.markdown(
    """
<style>
/* Wider main container */
.block-container { padding-top: 1.2rem; max-width: 1100px; }

/* Slightly nicer header spacing */
h1 { margin-bottom: 0.2rem; }

/* Chat message bubble tweaks (best-effort selectors; Streamlit DOM can change) */
[data-testid="stChatMessage"] {
  border-radius: 14px;
  padding: 0.35rem 0.25rem;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > p {
  font-size: 0.95rem;
  line-height: 1.45;
}

/* Sidebar section titles */
.sidebar-title {
  font-weight: 600;
  margin-top: 0.25rem;
  margin-bottom: 0.25rem;
}

/* Small helper text */
.muted { color: rgba(49, 51, 63, 0.6); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# State
# ----------------------------
if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    # each: {"role": "user"/"assistant", "content": "...", "ts": "..."}
    st.session_state.messages = []

if "http" not in st.session_state:
    st.session_state.http = requests.Session()

if "typing_effect" not in st.session_state:
    st.session_state.typing_effect = True

if "timeout_chat" not in st.session_state:
    st.session_state.timeout_chat = 120

if "timeout_other" not in st.session_state:
    st.session_state.timeout_other = 10

# ----------------------------
# Helpers
# ----------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content, "ts": now_str()})

@st.cache_data(ttl=8)
def health_check(api_base: str, timeout: int):
    try:
        r = requests.get(f"{api_base}/health", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("ready", False)), None
    except Exception as e:
        return False, str(e)

def api_reset(api_base: str, session_id: str, timeout: int):
    try:
        st.session_state.http.post(f"{api_base}/reset/{session_id}", timeout=timeout)
    except Exception:
        pass

def api_chat(api_base: str, session_id: str | None, message: str, timeout: int):
    payload = {"session_id": session_id, "message": message, "reset": False}
    r = st.session_state.http.post(f"{api_base}/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()  # {session_id, answer}

def render_answer_with_typing(text: str, speed_chars_per_sec: int = 80):
    # Simple typewriter effect (no backend streaming needed)
    placeholder = st.empty()
    out = []
    delay = 1.0 / max(speed_chars_per_sec, 1)

    for ch in text:
        out.append(ch)
        placeholder.markdown("".join(out))
        time.sleep(delay)

    return placeholder

def export_chat_json():
    return json.dumps(
        {
            "api_base": st.session_state.api_base,
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages,
        },
        ensure_ascii=False,
        indent=2,
    )

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Cấu hình</div>', unsafe_allow_html=True)

    st.session_state.api_base = st.text_input(
        "API base URL",
        value=st.session_state.api_base,
        help="Ví dụ: http://localhost:8000 hoặc https://your-domain.com",
    )

    colA, colB = st.columns(2)
    with colA:
        st.session_state.timeout_chat = st.number_input(
            "Timeout /chat (s)", min_value=5, max_value=600, value=int(st.session_state.timeout_chat)
        )
    with colB:
        st.session_state.timeout_other = st.number_input(
            "Timeout khác (s)", min_value=2, max_value=60, value=int(st.session_state.timeout_other)
        )

    st.session_state.typing_effect = st.toggle("Hiệu ứng gõ chữ", value=st.session_state.typing_effect)

    ready, err = health_check(st.session_state.api_base, int(st.session_state.timeout_other))
    if ready:
        st.success("API: Ready")
    else:
        st.warning("API: Not ready")
        if err:
            st.caption(f"Health error: {err}")

    st.markdown("---")

    st.markdown('<div class="sidebar-title">Phiên chat</div>', unsafe_allow_html=True)
    st.caption(f"session_id: {st.session_state.session_id or '(none)'}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New chat", use_container_width=True):
            # reset backend session if exists, then clear locally
            if st.session_state.session_id:
                api_reset(st.session_state.api_base, st.session_state.session_id, int(st.session_state.timeout_other))
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    with c2:
        if st.button("Clear UI", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.download_button(
        "Export chat (JSON)",
        data=export_chat_json(),
        file_name="chat_export.json",
        mime="application/json",
        use_container_width=True,
    )

# ----------------------------
# Header / Quick prompts
# ----------------------------
st.title("💬 Car Chatbox")
st.markdown(
    f'<div class="muted">API: {st.session_state.api_base}</div>',
    unsafe_allow_html=True,
)

quick = st.columns(4)
quick_prompts = [
    "Gợi ý 3 mẫu xe phù hợp đi gia đình (ưu tiên tiết kiệm nhiên liệu).",
    "So sánh sedan vs SUV: ưu nhược điểm cho đi làm hằng ngày.",
    "Tư vấn xe tầm giá 500–700 triệu: tiêu chí chọn.",
    "Các tính năng an toàn quan trọng khi mua xe?",
]
for i, p in enumerate(quick_prompts):
    with quick[i]:
        if st.button("Gợi ý", key=f"qp_{i}", use_container_width=True):
            # push into input pipeline by storing it
            st.session_state._prefill = p

prefill = st.session_state.pop("_prefill", "")

st.markdown("---")

# ----------------------------
# Render history
# ----------------------------
for m in st.session_state.messages:
    avatar = "🙂" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        st.caption(m.get("ts", ""))

# ----------------------------
# Input
# ----------------------------
user_input = st.chat_input("Nhập câu hỏi của bạn...", key="chat_input")
if not user_input and prefill:
    user_input = prefill

if user_input:
    add_message("user", user_input)
    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)
        st.caption(now_str())

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Đang trả lời..."):
            try:
                data = api_chat(
                    st.session_state.api_base,
                    st.session_state.session_id,
                    user_input,
                    int(st.session_state.timeout_chat),
                )
                st.session_state.session_id = str(data.get("session_id") or "")

                answer = data.get("answer", "")
                if not answer:
                    answer = "(No answer returned from API)"

                if st.session_state.typing_effect:
                    # render typing effect, then finalize into markdown once done
                    render_answer_with_typing(answer, speed_chars_per_sec=90)
                else:
                    st.markdown(answer)

                st.caption(now_str())
                add_message("assistant", answer)

            except requests.HTTPError as e:
                detail = e.response.text if e.response is not None else str(e)
                st.error(f"API error: {detail}")
                add_message("assistant", f"[ERROR] {detail}")

            except Exception as e:
                st.error(f"Lỗi: {e}")
                add_message("assistant", f"[ERROR] {e}")

