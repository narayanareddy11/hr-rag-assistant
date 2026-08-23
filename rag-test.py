from rag_app import GEMINI_MODEL, ask


def main():
    print("=" * 60)
    print("GEMINI + LANGGRAPH RAG")
    print("=" * 60)
    print(f"Model: {GEMINI_MODEL}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        print("\n[CLASSIFY]")
        print("Question:", question)

        try:
            result = ask(question)
            print("[CLASSIFY] Category:", result.get("category", "unknown"))
            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
        except Exception as exc:
            print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
