from __future__ import annotations

import os
from typing import Literal, TypedDict

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph


# ============================================================
# CONFIG
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash"
CHROMA_PATH = "./chroma_db"


# ============================================================
# GEMINI
# ============================================================


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini client using a Replit/local environment secret."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to Replit Secrets "
            "or export GEMINI_API_KEY in your local environment."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        api_key=api_key,
        max_retries=2,
    )


llm = get_llm()


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

billing_collection = chroma_client.get_or_create_collection(name="billing_docs")
hr_collection = chroma_client.get_or_create_collection(name="hr_docs")


# ============================================================
# KNOWLEDGE BASE
# ============================================================

billing_documents = [
    "Monthly subscriptions can be cancelled anytime from Settings > Billing > Cancel. The customer will retain access until the end of the current billing period.",
    "Refunds for annual plans are available within 30 days of purchase. After 30 days, annual subscriptions are non-refundable.",
    "Accepted payment methods include Visa, Mastercard, debit cards and supported online payment methods.",
    "If a payment fails, the system retries the payment three times over seven days. After the retries fail, the account enters a 14-day grace period.",
    "Plan changes take effect immediately. Charges are prorated based on the remaining billing period.",
]

hr_documents = [
    "Employees receive 24 days of paid annual leave per year. Leave accrues at 2 days per month.",
    "More than three consecutive days of sick leave requires a medical certificate.",
    "Employees may work remotely up to two days per week after completing the probation period.",
    "During probation, either party can terminate employment with 15 days notice.",
    "Performance reviews are conducted annually. Ratings include Exceptional, Exceeds Expectations, Meets Expectations, and Needs Improvement.",
]


# ============================================================
# DATABASE INITIALIZATION
# ============================================================


def initialize_database() -> None:
    if billing_collection.count() == 0:
        billing_collection.add(
            documents=billing_documents,
            ids=[f"billing_{i:03d}" for i in range(1, len(billing_documents) + 1)],
        )

    if hr_collection.count() == 0:
        hr_collection.add(
            documents=hr_documents,
            ids=[f"hr_{i:03d}" for i in range(1, len(hr_documents) + 1)],
        )


# ============================================================
# LANGGRAPH STATE
# ============================================================


class RAGState(TypedDict, total=False):
    question: str
    category: str
    context: str
    answer: str


# ============================================================
# CLASSIFICATION
# ============================================================


def classify_question(question: str) -> str:
    """Classify with Gemini, with deterministic policy keywords as a safety net."""
    q = question.lower().strip()

    hr_terms = (
        "work from home",
        "wfh",
        "remote work",
        "remote working",
        "work remotely",
        "working remotely",
        "leave",
        "vacation",
        "annual leave",
        "sick leave",
        "probation",
        "employee",
        "employees",
        "working hours",
        "working days",
        "holiday",
        "holidays",
        "attendance",
        "performance review",
        "employment",
    )

    billing_terms = (
        "subscription",
        "payment",
        "refund",
        "invoice",
        "billing",
        "cancel subscription",
        "pricing",
        "payment failure",
        "credit card",
        "debit card",
    )

    if any(term in q for term in hr_terms):
        return "hr"

    if any(term in q for term in billing_terms):
        return "billing"

    prompt = f"""
You are a strict question classifier.

Classify the user question into exactly one category: billing, hr, or general.

billing = subscriptions, payments, refunds, invoices, billing, cancellation, pricing.
hr = employees, leave, vacation, sick leave, probation, performance, employment,
work from home, WFH, remote work, holidays, attendance, working hours.
general = anything unrelated to billing or HR.

Examples:
- How many days can I work from home? -> hr
- Can I work remotely on Fridays? -> hr
- What is the WFH policy? -> hr
- How many vacation days do I get? -> hr
- Can I get a refund? -> billing
- What is my subscription price? -> billing
- What is Kubernetes? -> general

User question: {question}

Return ONLY one word: billing, hr, or general.
"""

    response = llm.invoke(prompt)
    category = response.text.strip().lower()

    if "billing" in category:
        return "billing"
    if "hr" in category:
        return "hr"
    return "general"


def classify(state: RAGState):
    category = classify_question(state["question"])
    print(f"[CLASSIFY] Category: {category}")
    return {"category": category}


# ============================================================
# RETRIEVAL
# ============================================================


def retrieve(collection, question: str, label: str) -> str:
    results = collection.query(query_texts=[question], n_results=3)
    documents = results.get("documents", [[]])[0]

    print(f"[{label}] Retrieved {len(documents)} documents.")

    if not documents:
        return "No relevant information was found in the knowledge base."

    return "\n\n".join(documents)


def handle_billing(state: RAGState):
    print("[ROUTER] Routing to: billing")
    return {"context": retrieve(billing_collection, state["question"], "BILLING")}


def handle_hr(state: RAGState):
    print("[ROUTER] Routing to: hr")
    return {"context": retrieve(hr_collection, state["question"], "HR")}


def handle_general(state: RAGState):
    print("[ROUTER] Routing to: general")
    return {"context": ""}


def router(state: RAGState) -> Literal["billing", "hr", "general"]:
    return state["category"]  # type: ignore[return-value]


# ============================================================
# GENERATION
# ============================================================


def generate(state: RAGState):
    question = state["question"]
    category = state["category"]
    context = state.get("context", "")

    if category in {"billing", "hr"}:
        prompt = f"""
You are a company support assistant.

Answer the user's question using ONLY the knowledge-base context below.

Rules:
1. Do not invent information.
2. Do not use outside knowledge for HR or billing questions.
3. If the answer is not available in the context, say exactly:
   I don't have enough information in the knowledge base.
4. Give a concise, direct answer.

Knowledge Base:
----------------
{context}
----------------

User Question: {question}

Answer:
"""
    else:
        prompt = f"""
You are a helpful AI assistant.
Answer the user's question clearly and concisely.

User Question: {question}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.text.strip()
    print("[GENERATE] Answer generated.")
    return {"answer": answer}


# ============================================================
# GRAPH
# ============================================================


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("classify", classify)
    graph.add_node("handle_billing", handle_billing)
    graph.add_node("handle_hr", handle_hr)
    graph.add_node("handle_general", handle_general)
    graph.add_node("generate", generate)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        router,
        {
            "billing": "handle_billing",
            "hr": "handle_hr",
            "general": "handle_general",
        },
    )

    graph.add_edge("handle_billing", "generate")
    graph.add_edge("handle_hr", "generate")
    graph.add_edge("handle_general", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


initialize_database()
app = build_graph()


def ask(question: str) -> dict:
    """Run one question through the complete LangGraph RAG pipeline."""
    return app.invoke({"question": question})
