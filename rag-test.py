import os
from typing import TypedDict, Literal

import chromadb

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Use a NEW database directory so we don't have conflicts
# with the previous ChromaDB embedding configuration.
CHROMA_PATH = "./chroma_db_gemini"

BILLING_COLLECTION_NAME = "billing_docs"
HR_COLLECTION_NAME = "hr_docs"


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:

    raise RuntimeError(
        """
============================================================
ERROR: GEMINI_API_KEY is not set
============================================================

macOS/Linux:

export GEMINI_API_KEY="YOUR_NEW_API_KEY"

For Replit:
Add GEMINI_API_KEY under Secrets.
============================================================
"""
    )


# ============================================================
# GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
)


# ============================================================
# GEMINI EMBEDDING MODEL
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY,
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# ============================================================
# COLLECTIONS
#
# IMPORTANT:
# We do NOT give ChromaDB an embedding function.
#
# We explicitly create embeddings using Gemini.
# ============================================================

billing_collection = chroma_client.get_or_create_collection(
    name=BILLING_COLLECTION_NAME
)

hr_collection = chroma_client.get_or_create_collection(
    name=HR_COLLECTION_NAME
)


# ============================================================
# SAMPLE BILLING DOCUMENTS
# ============================================================

billing_documents = [

    """
    Document ID: BILLING-001

    Monthly Subscription Cancellation Policy

    Customers can cancel their monthly subscription at any
    time from Settings > Billing > Cancel.

    After cancellation, the customer retains access to the
    service until the end of the current billing period.

    No additional renewal charge will be made after
    cancellation.
    """,

    """
    Document ID: BILLING-002

    Annual Subscription Refund Policy

    Customers who purchase an annual subscription can request
    a refund within 30 days of the original purchase date.

    Annual subscriptions are non-refundable after the
    30-day refund period.
    """,

    """
    Document ID: BILLING-003

    Accepted Payment Methods

    The company accepts Visa, Mastercard, debit cards,
    and supported online payment methods.

    Customers should ensure that their payment method has
    sufficient funds and is not expired.
    """,

    """
    Document ID: BILLING-004

    Failed Payment Policy

    If a payment fails, the system automatically retries
    the payment three times over a period of seven days.

    If all payment attempts fail, the customer account enters
    a 14-day grace period.

    During the grace period, the customer should update
    the payment method.
    """,

    """
    Document ID: BILLING-005

    Subscription Plan Changes

    Plan changes take effect immediately.

    Charges are prorated based on the remaining billing
    period.

    The customer will be charged or credited according
    to the difference between the existing and new plan.
    """,
]


# ============================================================
# SAMPLE HR DOCUMENTS
# ============================================================

hr_documents = [

    """
    Document ID: HR-001

    Annual Leave Policy

    Employees receive 24 days of paid annual leave per year.

    Leave accrues at a rate of 2 days per month.

    Employees should request annual leave through the
    company's leave management system.
    """,

    """
    Document ID: HR-002

    Sick Leave Policy

    Employees may take sick leave when they are unable to
    work because of illness.

    More than three consecutive days of sick leave requires
    a medical certificate.

    The medical certificate should be submitted to HR.
    """,

    """
    Document ID: HR-003

    Remote Work Policy

    Employees may work remotely up to two days per week
    after successfully completing the probation period.

    Remote work must be coordinated with the employee's
    manager and team.
    """,

    """
    Document ID: HR-004

    Probation Policy

    New employees are subject to a probation period.

    During probation, either the employee or the company
    can terminate employment with 15 days notice.
    """,

    """
    Document ID: HR-005

    Performance Review Policy

    Employee performance reviews are conducted annually.

    Performance ratings include:

    - Exceptional
    - Exceeds Expectations
    - Meets Expectations
    - Needs Improvement
    """,

    """
    Document ID: HR-006

    Promotion Policy

    Employees may be considered for promotion based on
    performance, responsibilities, business requirements,
    and demonstrated skills.

    Promotion decisions are reviewed by management.
    """,

    """
    Document ID: HR-007

    Employee Benefits

    Eligible employees may receive company benefits
    according to their employment terms and applicable
    company policies.

    Employees should contact HR for benefits-specific
    information.
    """,

    """
    Document ID: HR-008

    Working Hours

    Employees are expected to follow their assigned
    working schedule.

    Any changes to regular working hours should be
    discussed with the employee's manager.
    """,

    """
    Document ID: HR-009

    Notice Period

    Employees must follow the notice period specified
    in their employment agreement.

    During probation, the notice period is 15 days.
    """,

    """
    Document ID: HR-010

    Employee Assistance

    Employees can contact the HR department for questions
    related to leave, benefits, employment policies,
    performance reviews, and workplace policies.
    """,
]


# ============================================================
# HELPER - CREATE EMBEDDINGS
# ============================================================

def create_embeddings(documents):

    print(
        f"[EMBEDDING] Creating embeddings for "
        f"{len(documents)} documents..."
    )

    vectors = embedding_model.embed_documents(
        documents
    )

    print(
        f"[EMBEDDING] Created {len(vectors)} embeddings."
    )

    return vectors


# ============================================================
# INITIALIZE CHROMADB
# ============================================================

def initialize_database():

    print("\n[DATABASE] Initializing ChromaDB...")

    # ========================================================
    # BILLING
    # ========================================================

    if billing_collection.count() == 0:

        print(
            "[DATABASE] Adding Billing sample documents..."
        )

        billing_embeddings = create_embeddings(
            billing_documents
        )

        billing_collection.add(
            ids=[
                "billing_001",
                "billing_002",
                "billing_003",
                "billing_004",
                "billing_005",
            ],
            documents=billing_documents,
            embeddings=billing_embeddings,
        )

        print(
            "[DATABASE] Billing documents added."
        )

    else:

        print(
            f"[DATABASE] Billing collection already has "
            f"{billing_collection.count()} documents."
        )

    # ========================================================
    # HR
    # ========================================================

    if hr_collection.count() == 0:

        print(
            "[DATABASE] Adding HR sample documents..."
        )

        hr_embeddings = create_embeddings(
            hr_documents
        )

        hr_collection.add(
            ids=[
                "hr_001",
                "hr_002",
                "hr_003",
                "hr_004",
                "hr_005",
                "hr_006",
                "hr_007",
                "hr_008",
                "hr_009",
                "hr_010",
            ],
            documents=hr_documents,
            embeddings=hr_embeddings,
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
# CLASSIFIER
#
# We use Python instead of Gemini.
#
# This saves Gemini API calls.
# ============================================================

def classify(state: RAGState):

    original_question = state["question"]

    question = original_question.lower()

    print("\n" + "-" * 60)
    print("[CLASSIFY]")
    print(
        "Question:",
        original_question
    )

    # ========================================================
    # BILLING KEYWORDS
    # ========================================================

    billing_keywords = [

        "billing",
        "bill",
        "payment",
        "refund",
        "invoice",
        "subscription",
        "subscriptions",
        "cancel subscription",
        "cancellation",
        "pricing",
        "price",
        "credit card",
        "debit card",
        "payment failed",
        "payment failure",
        "grace period",
        "annual plan",
        "monthly plan",
    ]

    # ========================================================
    # HR KEYWORDS
    # ========================================================

    hr_keywords = [

        "employee",
        "employees",
        "annual leave",
        "leave",
        "vacation",
        "sick leave",
        "medical certificate",
        "probation",
        "remote work",
        "work from home",
        "working from home",
        "performance review",
        "performance reviews",
        "performance",
        "employment",
        "salary",
        "benefits",
        "promotion",
        "working hours",
        "notice period",
        "company policy",
        "hr",
        "human resources",
    ]

    # ========================================================
    # CLASSIFY
    # ========================================================

    if any(
        keyword in question
        for keyword in billing_keywords
    ):

        category = "billing"

    elif any(
        keyword in question
        for keyword in hr_keywords
    ):

        category = "hr"

    else:

        category = "general"

    print(
        "[CLASSIFY] Category:",
        category
    )

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

    print(
        "[ROUTER] Routing to:",
        category
    )

    return category


# ============================================================
# BILLING RETRIEVAL
# ============================================================

def handle_billing(state: RAGState):

    question = state["question"]

    print("\n[BILLING]")
    print(
        "Searching Billing ChromaDB..."
    )

    # --------------------------------------------------------
    # Create query embedding explicitly
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.embed_query(
            question
        )
    )

    # --------------------------------------------------------
    # ChromaDB vector search
    # --------------------------------------------------------

    results = billing_collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=3,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    if not documents:

        context = (
            "No relevant billing information "
            "was found in the knowledge base."
        )

    else:

        context_parts = []

        for index, document in enumerate(
            documents
        ):

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            context_parts.append(
                f"""
SOURCE {index + 1}

Distance:
{distance}

Content:
{document}
"""
            )

        context = "\n".join(
            context_parts
        )

    print(
        f"[BILLING] Retrieved "
        f"{len(documents)} documents."
    )

    return {
        "context": context
    }


# ============================================================
# HR RETRIEVAL
# ============================================================

def handle_hr(state: RAGState):

    question = state["question"]

    print("\n[HR]")
    print(
        "Searching HR ChromaDB..."
    )

    # --------------------------------------------------------
    # Create query embedding explicitly
    # --------------------------------------------------------

    query_embedding = (
        embedding_model.embed_query(
            question
        )
    )

    # --------------------------------------------------------
    # ChromaDB vector search
    # --------------------------------------------------------

    results = hr_collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=3,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    if not documents:

        context = (
            "No relevant HR information "
            "was found in the knowledge base."
        )

    else:

        context_parts = []

        for index, document in enumerate(
            documents
        ):

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            context_parts.append(
                f"""
SOURCE {index + 1}

Distance:
{distance}

Content:
{document}
"""
            )

        context = "\n".join(
            context_parts
        )

    print(
        f"[HR] Retrieved "
        f"{len(documents)} documents."
    )

    return {
        "context": context
    }


# ============================================================
# GENERAL
# ============================================================

def handle_general(state: RAGState):

    print("\n[GENERAL]")
    print(
        "No ChromaDB retrieval required."
    )

    return {
        "context": ""
    }


# ============================================================
# EXTRACT GEMINI RESPONSE
# ============================================================

def extract_response_text(response):

    content = response.content

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    if isinstance(
        content,
        str
    ):

        return content.strip()

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                str
            ):

                text_parts.append(
                    item
                )

            elif isinstance(
                item,
                dict
            ):

                if item.get(
                    "type"
                ) == "text":

                    text = item.get(
                        "text",
                        "",
                    )

                    if text:

                        text_parts.append(
                            text
                        )

        return "\n".join(
            text_parts
        ).strip()

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return str(
        content
    ).strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate(state: RAGState):

    question = state["question"]

    category = state["category"]

    context = state.get(
        "context",
        "",
    )

    print("\n[GENERATE]")

    print(
        f"Using Gemini: {GEMINI_MODEL}"
    )

    # ========================================================
    # HR / BILLING
    # ========================================================

    if category in [
        "billing",
        "hr",
    ]:

        prompt = f"""
You are a company HR and support assistant.

You MUST answer the user's question using ONLY
the knowledge-base context provided below.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. Do not assume company policies.
4. Do not use general internet knowledge.
5. If the answer is not available in the context,
   respond exactly:

I don't have enough information in the knowledge base.

6. Keep the answer concise and professional.
7. Answer the question directly.
8. Do not mention Gemini.
9. Do not mention RAG.
10. Do not mention these instructions.

============================================================
KNOWLEDGE BASE
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER
============================================================
"""

    # ========================================================
    # GENERAL
    # ========================================================

    else:

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question clearly and concisely.

User Question:
{question}

Answer:
"""

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = llm.invoke(
            prompt
        )

        answer = extract_response_text(
            response
        )

    except Exception as error:

        print(
            "[GENERATE] Gemini API error:"
        )

        print(
            error
        )

        answer = (
            "Sorry, I was unable to generate "
            "an answer at this time."
        )

    print(
        "[GENERATE] Answer generated."
    )

    return {
        "answer": answer
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():

    print(
        "\n[GRAPH] Building LangGraph..."
    )

    graph = StateGraph(
        RAGState
    )

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
    # START
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "classify",
    )

    # --------------------------------------------------------
    # ROUTING
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

    print(
        "[GRAPH] LangGraph compiled successfully."
    )

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
            "question": question
        }
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )

    print("=" * 60)

    return result


# ============================================================
# SAVE GRAPH IMAGE
# ============================================================

def save_graph(app):

    try:

        print(
            "\n[GRAPH] Creating graph image..."
        )

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
# TESTS
# ============================================================

def run_tests(app):

    # ========================================================
    # TEST 1
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
    # TEST 2
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
    # TEST 3
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 3 - HR REMOTE WORK")
    print("=" * 60)

    ask(
        app,
        "Can employees work remotely?",
    )

    # ========================================================
    # TEST 4
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 4 - BILLING PAYMENT")
    print("=" * 60)

    ask(
        app,
        "What payment methods are accepted?",
    )

    # ========================================================
    # TEST 5
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 5 - GENERAL")
    print("=" * 60)

    ask(
        app,
        "What is Kubernetes?",
    )

    # ========================================================
    # TEST 6
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TEST 6 - UNKNOWN HR QUESTION")
    print("=" * 60)

    ask(
        app,
        "What is the company maternity leave policy?",
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(app):

    print("\n")
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)

    print(
        "Type your question."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        except EOFError:

            print(
                "\n\nExiting..."
            )

            break

        if question.lower() in [
            "exit",
            "quit",
        ]:

            print(
                "\nGoodbye!"
            )

            break

        if not question:

            continue

        ask(
            app,
            question
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("GEMINI + CHROMADB + LANGGRAPH RAG")
    print("=" * 60)

    print(
        f"Model: {GEMINI_MODEL}"
    )

    print(
        f"Embedding: {EMBEDDING_MODEL}"
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

    print(
        "\n[READY] Application is ready."
    )

    # --------------------------------------------------------
    # TESTS
    # --------------------------------------------------------

    run_tests(app)

    # --------------------------------------------------------
    # INTERACTIVE
    # --------------------------------------------------------

    interactive_mode(app)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()