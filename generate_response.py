import os
import re
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from user_profile import (
    ProfileUpdate,
    UserProfile,
    load_profile,
    log_viewed,
    merge_update,
    save_profile,
)

load_dotenv()

# DB Init
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
db_engine = create_engine(DB_CONNECTION_STRING, pool_size=10, max_overflow=20)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "car_vectorize"

# Main SQL view used for exact brand/model lookups
MAIN_VIEW = os.getenv("MAIN_POST", "[dbo].[view_post_info]")


# --- Helper Functions ---
def get_image_urls(view_name: str, vin_query: str, max_images: int = 3):
    urls = []
    with db_engine.connect() as con:
        query = text(f"SELECT image_link FROM {view_name} WHERE VIN = :vin_param")
        result = con.execute(query, {"vin_param": vin_query})
        urls = [row[0] for row in result]
    return urls[:max_images]

def get_feature(view_name: str, vin_query: str):
    # dùng defaultdict(list) để auto tạo list rỗng cho mỗi key mới
    feature = defaultdict(list)

    with db_engine.connect() as con:
        query = text(f"""
            SELECT feature_type, feature_name
            FROM {view_name}
            WHERE VIN = :vin_param
        """)
        result = con.execute(query, {"vin_param": vin_query})

        for feature_type, feature_name in result:
            if feature_name not in feature[feature_type]:
                feature[feature_type].append(feature_name)

    return dict(feature)

def format_docs(docs):
    if not docs:
        return "No relevant car listings found."

    formatted = ""
    feature_view = os.getenv("FEATURE_POST")  
    image_view = os.getenv("IMAGE_POST")      

    for i, (doc, score) in enumerate(docs, 1):
        if score > 1.3:
            continue

        meta = doc.metadata
        vin = meta.get("VIN", "N/A")

        # image URLS
        images = get_image_urls(image_view, vin, 3)
        if images:
            img_str = "\n".join([f"- {u}" for u in images])
        else:
            img_str = "- No images available"

        # Post Feature
        features = get_feature(feature_view, vin) if feature_view else {}
        if features:
            feature_lines = []
            for f_type, f_names in features.items():
                joined_names = ", ".join(f_names)
                feature_lines.append(f"{f_type}: {joined_names}")
            feature_str = "\n".join([f"- {line}" for line in feature_lines])
        else:
            feature_str = "- No feature data"

        formatted += f"""
            --- Car Option {i} (Score {score:.2f}) ---
            VIN: {vin}
            Details Metadata: {meta}

            Features:
            {feature_str}

            Images:
            {img_str}
        """
    return formatted or "No relevant car listings found."

# --- SQL Retrieval (exact brand/model lookup) ---
_NUM_RE = re.compile(r"\$?\s*([\d][\d,]*(?:\.\d+)?)\s*(k|thousand)?", re.I)
_OVER_RE = re.compile(r"(over|above|more than|at least|min|from|>)", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FUEL_MAP = {
    "gasoline": "Gasoline", "gas": "Gasoline", "petrol": "Gasoline",
    "hybrid": "Hybrid", "electric": "Electric", "ev": "Electric",
    "diesel": "Diesel", "plug-in": "Plug-In Hybrid",
}
_STATUS_MAP = {
    "brand new": "New", "new": "New", "used": "Used",
    "pre-owned": "Used", "second hand": "Used", "certified": "Certified",
}


@dataclass
class CarConstraints:
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    status: Optional[str] = None
    fuel_type: Optional[str] = None
    year: Optional[int] = None


def parse_constraints(message: str) -> CarConstraints:
    msg = message.lower()
    c = CarConstraints()

    for m in _NUM_RE.finditer(message):
        try:
            value = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if (m.group(2) or "").lower() in ("k", "thousand"):
            value *= 1_000
        if value < 1000:
            continue
        window = message[max(0, m.start() - 25):m.start()]
        if _OVER_RE.search(window):
            c.price_min = value
        else:
            c.price_max = value
        break

    for kw, canon in _STATUS_MAP.items():
        if kw in msg:
            c.status = canon
            break

    for kw, canon in _FUEL_MAP.items():
        if re.search(rf"\b{re.escape(kw)}\b", msg):
            c.fuel_type = canon
            break

    ym = _YEAR_RE.search(message)
    if ym:
        c.year = int(ym.group(0))
    return c


def sql_search_cars(brand=None, model=None, constraints=None, exclude_brands=None, limit=6):
    conditions = ["title IS NOT NULL"]
    params = {"limit": limit}

    if brand:
        conditions.append("brand = :brand")
        params["brand"] = brand
    if model:
        conditions.append("title LIKE :model")
        params["model"] = f"%{model}%"
    for i, excluded in enumerate(exclude_brands or []):
        conditions.append(f"brand <> :exb{i}")
        params[f"exb{i}"] = excluded
    if constraints:
        if constraints.status:
            conditions.append("status LIKE :status")
            params["status"] = f"%{constraints.status}%"
        if constraints.fuel_type:
            conditions.append("fuel_type LIKE :fuel_type")
            params["fuel_type"] = f"%{constraints.fuel_type}%"
        if constraints.price_min is not None:
            conditions.append("price >= :price_min")
            params["price_min"] = constraints.price_min
        if constraints.price_max is not None:
            conditions.append("price <= :price_max")
            params["price_max"] = constraints.price_max
        if constraints.year:
            conditions.append("title LIKE :year")
            params["year"] = f"%{constraints.year}%"

    query = text(f"""
        SELECT DISTINCT TOP (:limit)
            VIN, status, title, brand, exterior_color, interior_color,
            drivetrain, fuel_type, transmission, engine, price,
            monthly_payment, mileage, mpg, post_link
        FROM {MAIN_VIEW}
        WHERE {' AND '.join(conditions)}
        ORDER BY price ASC
    """)

    with db_engine.connect() as con:
        return con.execute(query, params).mappings().all()


def format_sql_cars(rows):
    if not rows:
        return "No matching car listings found in the database."

    feature_view = os.getenv("FEATURE_POST")
    image_view = os.getenv("IMAGE_POST")
    formatted = ""

    for i, row in enumerate(rows, 1):
        vin = row.get("VIN", "N/A")

        images = get_image_urls(image_view, vin, 3) if image_view else []
        img_str = "\n".join(f"- {u}" for u in images) or "- No images available"

        features = get_feature(feature_view, vin) if feature_view else {}
        if features:
            feature_str = "\n".join(
                f"- {f_type}: {', '.join(f_names)}"
                for f_type, f_names in features.items()
            )
        else:
            feature_str = "- No feature data"

        formatted += f"""
            --- Car Option {i} ---
            VIN: {vin}
            Details Metadata: {dict(row)}

            Features:
            {feature_str}

            Images:
            {img_str}
        """
    return formatted


def get_standalone_question(llm, history, user_input):
    if not history:
        return user_input

    condense_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood, contain and capture all the chat history. Do NOT answer the question, just reformulate it if needed or return it as is."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", condense_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"chat_history": history, "question": user_input})


# Core slots required before running a consultation retrieval
CORE_SLOTS = ("budget_max", "body_type", "fuel_type")


class AgenticState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    query: str
    context: str
    answer: str
    intent: str
    brand: Optional[str]
    model: Optional[str]
    session_id: str
    profile: dict


class IntentDecision(BaseModel):
    intent: str = Field(
        description="'specific' when the user names an exact car brand/model, "
        "'vague' when they describe needs without a specific car, "
        "'chitchat' for greetings or off-topic talk."
    )
    brand: Optional[str] = Field(default=None, description="Exact car brand mentioned, else null.")
    model: Optional[str] = Field(default=None, description="Exact car model/name mentioned, else null.")


def _core_complete(profile: dict) -> bool:
    core = profile.get("core_slots", {})
    return all(core.get(k) for k in CORE_SLOTS)


def _build_agentic_app(llm, vector_store):

    def route_intent(state: AgenticState):
        router_llm = llm.with_structured_output(IntentDecision)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You route a car consultation assistant.\n"
                    "intent='specific' when the user names an exact car brand or model "
                    "(e.g. 'Show me BMW i5 specs').\n"
                    "intent='vague' when the user describes needs, budget, or usage without "
                    "a specific car (e.g. 'I need a family car around $40,000').\n"
                    "intent='chitchat' for greetings or unrelated talk.\n"
                    "Extract the exact brand and model if present, else null.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        result = (prompt | router_llm).invoke(
            {"chat_history": state["messages"], "question": state["question"]}
        )
        print(f"[INTENT] intent={result.intent} brand={result.brand} model={result.model}")
        return {
            "intent": result.intent,
            "brand": result.brand,
            "model": result.model,
            "messages": [],
        }

    def update_profile(state: AgenticState):
        profile = UserProfile(**(state.get("profile") or {}))
        updater = llm.with_structured_output(ProfileUpdate)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You maintain a persistent car-shopping profile for the customer.\n"
                    "Current profile (JSON): {profile}\n"
                    "From the latest message and the history, extract NEW or CHANGED info only:\n"
                    "- core slots: budget (USD number), body_type, fuel_type, brand, condition.\n"
                    "- add_features and vibe for soft preferences.\n"
                    "- exclude_brands when the customer dislikes or wants to avoid a brand.\n"
                    "- interested_models for specific models the customer asks about.\n"
                    "Leave fields empty or null when there is nothing new.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        update = (prompt | updater).invoke(
            {
                "profile": profile.model_dump_json(),
                "chat_history": state["messages"],
                "question": state["question"],
            }
        )
        profile = merge_update(profile, update)
        print(f"[PROFILE] {profile.model_dump()}")
        return {"profile": profile.model_dump(), "messages": []}

    def ask_slot(state: AgenticState):
        core = state["profile"].get("core_slots", {})
        missing = [k for k in CORE_SLOTS if not core.get(k)]
        known = {k: v for k, v in core.items() if v}
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a friendly car sales consultant. The customer has not given "
                    "enough detail yet.\n"
                    "Known preferences: {known}\n"
                    "Missing details: {missing}\n"
                    "Ask ONE short, natural question to gather the missing details. "
                    "Offer concrete options when helpful (e.g. SUV vs Sedan, "
                    "Gasoline vs Hybrid vs Electric). Do not list cars yet. "
                    "Reply in the same language the customer is using.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "known": known,
                "missing": missing,
                "chat_history": state["messages"],
                "question": state["question"],
            }
        )
        print(f"[ASK SLOT] missing={missing} question={answer!r}")
        return {"answer": answer}

    def hybrid_retrieve(state: AgenticState):
        profile = state.get("profile") or {}
        core = profile.get("core_slots", {})
        soft = profile.get("soft_preferences", {})
        excluded = profile.get("excluded_brands", [])

        if state.get("intent") == "specific":
            brand = state.get("brand")
            model = state.get("model")
            constraints = parse_constraints(state["question"])
            soft_query = state["query"]
        else:
            brand = core.get("brand")
            model = None
            constraints = CarConstraints(
                price_min=core.get("budget_min"),
                price_max=core.get("budget_max"),
                status=core.get("condition"),
                fuel_type=core.get("fuel_type"),
            )
            soft_parts = [core.get("body_type"), soft.get("vibe"), *(soft.get("features") or [])]
            soft_query = " ".join(p for p in soft_parts if p) or state["query"]

        rows = sql_search_cars(
            brand=brand, model=model, constraints=constraints,
            exclude_brands=excluded, limit=6,
        )
        sql_ctx = format_sql_cars(rows)

        vec_results = vector_store.similarity_search_with_score(soft_query, k=5)
        if excluded:
            excluded_low = {b.lower() for b in excluded}
            vec_results = [
                (doc, score) for doc, score in vec_results
                if (doc.metadata.get("Brand", "") or "").lower() not in excluded_low
            ]
        vec_ctx = format_docs(vec_results)

        context = f"[SQL HARD-FILTER MATCHES]\n{sql_ctx}\n\n[SEMANTIC MATCHES]\n{vec_ctx}"

        profile_obj = log_viewed(UserProfile(**profile), [r.get("title") for r in rows])
        print(f"[HYBRID] sql_rows={len(rows)} vector_hits={len(vec_results)} soft_query={soft_query!r}")
        print(f"[HYBRID CONTEXT]\n{context}")
        return {"context": context, "profile": profile_obj.model_dump(), "messages": []}

    def consult(state: AgenticState):
        system_prompt = """You are an expert car sales consultant.

Use ONLY the provided 'Knowledge Context' (SQL hard-filter matches and semantic matches).
1. Recommend the 2-3 most suitable cars for the customer's stated preferences.
2. For each car show ONLY: title, brand, status (new/used), mileage, price, exterior color, key features.
3. Add a short, tailored pros/cons for each option based on the customer's preferences.
4. Always format price with comma separators (e.g., $21,950).
5. Keep it natural and persuasive, like a professional consultant.
6. Provide images or extra details ONLY when the customer explicitly requests them.
7. Include [View Details](post_link) when a link is available in the metadata.
8. If nothing fits, say so honestly and offer the closest alternatives or general advice.
Use the Customer Profile for personalization. NEVER recommend any brand in excluded_brands.
Reply in the same language the customer is using.
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("system", "Customer Profile (JSON):\n{profile}"),
                ("system", "Knowledge Context:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "profile": state.get("profile", {}),
                "context": state.get("context", ""),
                "chat_history": state["messages"],
                "question": state["question"],
            }
        )
        print(f"[CONSULT] answer={answer!r}")
        return {"answer": answer}

    def route_after_intent(state: AgenticState):
        intent = state.get("intent")
        if intent == "specific":
            return "hybrid_retrieve"
        if intent == "vague":
            return "hybrid_retrieve" if _core_complete(state["profile"]) else "ask_slot"
        return "consult"

    graph = StateGraph(AgenticState)
    graph.add_node("update_profile", update_profile)
    graph.add_node("route_intent", route_intent)
    graph.add_node("ask_slot", ask_slot)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("consult", consult)

    graph.set_entry_point("update_profile")
    graph.add_edge("update_profile", "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        route_after_intent,
        {
            "hybrid_retrieve": "hybrid_retrieve",
            "ask_slot": "ask_slot",
            "consult": "consult",
        },
    )
    graph.add_edge("ask_slot", END)
    graph.add_edge("hybrid_retrieve", "consult")
    graph.add_edge("consult", END)
    return graph.compile()


_AGENTIC_APP = None


# --- Initialize LLM + Vector Store ---
def initialize_resources():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    global _AGENTIC_APP
    _AGENTIC_APP = _build_agentic_app(llm, vector_store)
    return llm, vector_store


# --- THE MAIN API FUNCTION USED BY STREAMLIT ---
def generate_response(llm, vector_store, chat_history_buffer, user_input, session_id="default"):
    """
    Returns (response_text, updated_chat_history_buffer)
    """

    global _AGENTIC_APP
    if _AGENTIC_APP is None:
        _AGENTIC_APP = _build_agentic_app(llm, vector_store)

    # 1. Normalize to standalone query
    search_query = get_standalone_question(llm, chat_history_buffer, user_input)

    # 2. Load persistent profile and run the consultation graph
    profile = load_profile(session_id)
    graph_state = _AGENTIC_APP.invoke(
        {
            "messages": chat_history_buffer,
            "question": user_input,
            "query": search_query,
            "context": "",
            "answer": "",
            "intent": "vague",
            "brand": None,
            "model": None,
            "session_id": session_id,
            "profile": profile.model_dump(),
        }
    )
    final_answer = graph_state.get("answer", "I could not generate a response right now.")

    # 3. Persist the updated profile
    updated_profile = graph_state.get("profile")
    if updated_profile:
        save_profile(session_id, UserProfile(**updated_profile))

    # 4. Update history (preserve existing session behavior)
    chat_history_buffer.append(HumanMessage(content=user_input))
    chat_history_buffer.append(AIMessage(content=final_answer))

    # Limit memory
    chat_history_buffer = chat_history_buffer[-10:]
   
    return final_answer, chat_history_buffer
