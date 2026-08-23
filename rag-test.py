from typing import TypedDict, Literal

import chromadb

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# ============================================================
# CONFIG
# ============================================================

OLLAMA_MODEL = "qwen2.5:3b"

CHROMA_PATH = "./chroma_db"


# ============================================================
# LOCAL QWEN MODEL
# ============================================================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

billing_collection = chroma_client.get_or_create_collection(
    name="billing_docs"
)

hr_collection = chroma_client.get_or_create_collection(
    name="hr_docs"
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

billing_documents = [
    """
    Monthly subscriptions can be cancelled anytime
    from Settings > Billing > Cancel.

    The customer will retain access until the end
    of the current billing period.
    """,

    """
    Refunds for annual plans are available within
    30 days of purchase.

    After 30 days, annual subscriptions are
    non-refundable.
    """,

    """
    Accepted payment methods include Visa,
    Mastercard, debit cards and supported
    online payment methods.
    """,

    """
    If a payment fails, the system retries the payment
    three times over seven days.

    After the retries fail, the account enters
    a 14-day grace period.
    """,

    """
    Plan changes take effect immediately.

    Charges are prorated based on the remaining
    billing period.
    """
]


hr_documents = [
    """
    Employees receive 24 days of paid annual leave
    per year.

    Leave accrues at 2 days per month.
    """,

    """
    More than three consecutive days of sick leave
    requires a medical certificate.
    """,

    """
    Employees may work remotely up to two days
    per week after completing the probation period.
    """,

    """
    During probation, either party can terminate
    employment with 15 days notice.
    """,

    """
    Performance reviews are conducted annually.

    Ratings include Exceptional,
    Exceeds Expectations,
    Meets Expectations,
    and Needs Improvement.
    """
]


# ============================================================
# LOAD DOCUMENTS INTO CHROMADB
# ============================================================

def initialize_database():

    print("\n[DATABASE] Initializing ChromaDB...")

    # --------------------------------------------------------
    # BILLING
    # --------------------------------------------------------

    if billing_collection.count() == 0:

        billing_collection.add(
            documents=billing_documents,
            ids=[
                "billing_001",
                "billing_002",
                "billing_003",
                "billing_004",
                "billing_005",
            ],
        )

        print(
            "[DATABASE] Billing documents added."
        )

    else:

        print(
            f"[DATABASE] Billing collection already has "
            f"{billing_collection.count()} documents."
        )

    # --------------------------------------------------------
    # HR
    # --------------------------------------------------------

    if hr_collection.count() == 0:

        hr_collection.add(
            documents=hr_documents,
            ids=[
                "hr_001",
                "hr_002",
                "hr_003",
                "hr_004",
                "hr_005",
            ],
        )

        print(
            "[DATABASE] HR documents added."
        )

    else:

        print(
            f"[DATABASE] HR collection already has "
            f"{hr_collection.count()} documents."
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
# NODE 1 - CLASSIFY
# ============================================================

def classify(state: RAGState):

    question = state["question"]

    print("\n" + "-" * 60)
    print("[CLASSIFY]")
    print("Question:", question)

    prompt = f"""
You are a question classifier.

Classify the user question into exactly ONE category.

Available categories:

billing
hr
general

Rules:

BILLING:
Questions about:
- subscription
- payment
- refund
- invoice
- billing
- cancellation
- pricing
- payment failure
- credit card
- debit card

HR:
Questions about:
- employee
- leave
- annual leave
- vacation
- sick leave
- probation
- remote work
- work from home
- performance review
- employment

GENERAL:
Anything unrelated to billing or HR.

User question:
{question}

Return ONLY one word:

billing

hr

general
"""

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    # --------------------------------------------------------
    # Clean Qwen response
    # --------------------------------------------------------

    if "billing" in category:

        category = "billing"

    elif category == "hr" or category.startswith("hr"):

        category = "hr"

    else:

        category = "general"

    print("[CLASSIFY] Category:", category)

    return {
        "category": category
    }


# ============================================================
# ROUTER
# ============================================================

def router(
    state: RAGState,
) -> Literal[
    "billing",
    "hr",
    "general",
]:

    category = state["category"]

    print("[ROUTER] Routing to:", category)

    return category


# ============================================================
# NODE 2 - BILLING RETRIEVAL
# ============================================================

def handle_billing(state: RAGState):

    question = state["question"]

    print("\n[BILLING]")
    print("Searching Billing ChromaDB...")

    results = billing_collection.query(
        query_texts=[question],
        n_results=3,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    if not documents:

        context = (
            "No relevant billing information "
            "was found in the knowledge base."
        )

    else:

        context = "\n\n".join(documents)

    print(
        f"[BILLING] Retrieved {len(documents)} documents."
    )

    return {
        "context": context
    }


# ============================================================
# NODE 3 - HR RETRIEVAL
# ============================================================

def handle_hr(state: RAGState):

    question = state["question"]

    print("\n[HR]")
    print("Searching HR ChromaDB...")

    results = hr_collection.query(
        query_texts=[question],
        n_results=3,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    if not documents:

        context = (
            "No relevant HR information "
            "was found in the knowledge base."
        )

    else:

        context = "\n\n".join(documents)

    print(
        f"[HR] Retrieved {len(documents)} documents."
    )

    return {
        "context": context
    }


# ============================================================
# NODE 4 - GENERAL
# ============================================================

def handle_general(state: RAGState):

    print("\n[GENERAL]")
    print("No ChromaDB retrieval required.")

    return {
        "context": ""
    }


# ============================================================
# NODE 5 - GENERATE ANSWER
# ============================================================

def generate(state: RAGState):

    question = state["question"]

    category = state["category"]

    context = state.get(
        "context",
        "",
    )

    print("\n[GENERATE]")
    print("Using Qwen2.5:3B...")

    # --------------------------------------------------------
    # BILLING / HR RAG
    # --------------------------------------------------------

    if category in [
        "billing",
        "hr",
    ]:

        prompt = f"""
You are a company support assistant.

Answer the user's question using ONLY the
knowledge-base context provided below.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   say:

"I don't have enough information in the knowledge base."

4. Give a concise and clear answer.

Knowledge Base:
==================================================

{context}

==================================================

User Question:
{question}

Answer:
"""

    # --------------------------------------------------------
    # GENERAL QUESTION
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly and concisely.

User Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content.strip()

    print("[GENERATE] Answer generated.")

    return {
        "answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    print("\n[GRAPH] Building LangGraph...")

    graph = StateGraph(RAGState)

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    graph.add_node(
        "classify",
        classify,
    )

    graph.add_node(
        "handle_billing",
        handle_billing,
    )

    graph.add_node(
        "handle_general",
        handle_general,
    )

    graph.add_node(
        "handle_hr",
        handle_hr,
    )

    graph.add_node(
        "generate",
        generate,
    )

    # --------------------------------------------------------
    # START -> CLASSIFY
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "classify",
    )

    # --------------------------------------------------------
    # CLASSIFY -> ROUTER
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        router,
        {
            "billing": "handle_billing",
            "general": "handle_general",
            "hr": "handle_hr",
        },
    )

    # --------------------------------------------------------
    # RETRIEVAL -> GENERATE
    # --------------------------------------------------------

    graph.add_edge(
        "handle_billing",
        "generate",
    )

    graph.add_edge(
        "handle_general",
        "generate",
    )

    graph.add_edge(
        "handle_hr",
        "generate",
    )

    # --------------------------------------------------------
    # GENERATE -> END
    # --------------------------------------------------------

    graph.add_edge(
        "generate",
        END,
    )

    app = graph.compile()

    print("[GRAPH] LangGraph compiled successfully.")

    return app


# ============================================================
# ASK QUESTION
# ============================================================

def ask(
    app,
    question: str,
):

    result = app.invoke(
        {
            "question": question,
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)

    return result


# ============================================================
# SAVE GRAPH IMAGE
# ============================================================

def save_graph(app):

    try:

        print("\n[GRAPH] Creating graph image...")

        png = (
            app
            .get_graph()
            .draw_mermaid_png()
        )

        with open(
            "rag_graph.png",
            "wb",
        ) as file:

            file.write(png)

        print(
            "[GRAPH] Saved: rag_graph.png"
        )

    except Exception as error:

        print(
            "[GRAPH] Could not create PNG."
        )

        print(
            "[GRAPH] Reason:",
            error,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("LOCAL QWEN2.5:3B - LANGGRAPH RAG")
    print("=" * 60)

    print(
        f"Model: {OLLAMA_MODEL}"
    )

    print(
        f"ChromaDB: {CHROMA_PATH}"
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    initialize_database()

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    app = build_graph()

    # --------------------------------------------------------
    # GRAPH IMAGE
    # --------------------------------------------------------

    save_graph(app)

    print("\n[READY] Application is ready.")

    # ========================================================
    # TEST 1 - BILLING
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 1 - BILLING")
    print("=" * 60)

    ask(
        app,
        "Can I get a refund for my annual subscription?",
    )

    # ========================================================
    # TEST 2 - HR
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 2 - HR")
    print("=" * 60)

    ask(
        app,
        "How many annual leave days do employees get?",
    )

    # ========================================================
    # TEST 3 - GENERAL
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 3 - GENERAL")
    print("=" * 60)

    ask(
        app,
        "What is Kubernetes?",
    )

    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)

    print("Type your question.")
    print("Type 'exit' or 'quit' to stop.")

    while True:

        try:

            question = input("\nYou: ").strip()

        except KeyboardInterrupt:

            print("\n\nExiting...")
            break

        except EOFError:

            print("\n\nExiting...")
            break

        if question.lower() in [
            "exit",
            "quit",
        ]:

            print("\nGoodbye!")
            break

        if not question:

            continue

        ask(
            app,
            question,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()