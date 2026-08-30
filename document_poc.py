import argparse
import base64
import mimetypes
import os
import re
from pathlib import Path
from xml.etree import ElementTree


DOCUMENTS_DIR = Path("documents")

TEXT_EXTENSIONS = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".txt",
}

SVG_EXTENSIONS = {
    ".svg",
}

IMAGE_EXTENSIONS = {
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | SVG_EXTENSIONS
    | IMAGE_EXTENSIONS
)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_text_file(path):
    return path.read_text(encoding="utf-8").strip()


def extract_svg_text(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    parts = []

    for element in root.iter():
        if element.text and element.text.strip():
            parts.append(element.text.strip())

    return "\n".join(parts)


def extract_image_text_with_gemini(path):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Image text extraction skipped. "
            "Set GEMINI_API_KEY to extract text from image documents."
        )

    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
    )

    response = model.invoke([
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract the readable text from this document image. "
                        "Return only the extracted text."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{encoded}",
                },
            ]
        )
    ])

    return extract_response_text(response)


def extract_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))

        return "\n".join(parts).strip()

    return str(content).strip()


def extract_document(path):
    extension = path.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        text = extract_text_file(path)
    elif extension in SVG_EXTENSIONS:
        text = extract_svg_text(path)
    elif extension in IMAGE_EXTENSIONS:
        text = extract_image_text_with_gemini(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return {
        "path": str(path),
        "text": normalize_text(text),
    }


def load_documents(documents_dir):
    paths = [
        path
        for path in documents_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return [
        extract_document(path)
        for path in sorted(paths)
    ]


def search_documents(documents, query):
    query_terms = {
        term.lower()
        for term in re.findall(r"\w+", query)
        if len(term) > 2
    }

    matches = []

    for document in documents:
        text = document["text"].lower()
        score = sum(
            1
            for term in query_terms
            if term in text
        )

        if score:
            matches.append((score, document))

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in matches
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Simple document folder extraction POC."
    )
    parser.add_argument(
        "--documents-dir",
        default=str(DOCUMENTS_DIR),
        help="Folder containing documents to extract.",
    )
    parser.add_argument(
        "--query",
        default="annual leave refund onboarding",
        help="Simple keyword query to test retrieval.",
    )

    args = parser.parse_args()
    documents_dir = Path(args.documents_dir)

    if not documents_dir.exists():
        raise SystemExit(f"Documents folder not found: {documents_dir}")

    documents = load_documents(documents_dir)

    print(f"Loaded documents: {len(documents)}")

    for document in documents:
        preview = document["text"][:160]
        print(f"- {document['path']}: {preview}")

    print()
    print(f"Query: {args.query}")
    print("Matches:")

    for document in search_documents(documents, args.query):
        print(f"- {document['path']}")


if __name__ == "__main__":
    main()
