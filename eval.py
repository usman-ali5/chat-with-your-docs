from dotenv import load_dotenv
from rag_chain import retriever

load_dotenv()

# hand-write a few Q&A pairs where you KNOW the right answer is in the text
test_cases = [
    {"question": "What did the speech say about the economy?", "must_contain": "econom"},
    {"question": "What was said about jobs?", "must_contain": "jobs"},
    {"question": "What did the speech say about Ukraine?", "must_contain": "ukrain"},
    {"question": "What was said about climate change?", "must_contain": "climate"},
]


for case in test_cases:
    docs = retriever.invoke(case["question"])
    combined = " ".join(d.page_content.lower() for d in docs)
    passed = case["must_contain"].lower() in combined
    print(f"[{'PASS' if passed else 'FAIL'}] {case['question']}")