import re
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (RAW_CSV, POLICY_DIR, INDEX_DIR, EVAL_DIR, EMBEDDING_MODEL,
                        CHUNK_SIZE, CHUNK_OVERLAP, RESPONSES_PER_INTENT, COMPANY)

# bitext responses have slots like {{Order Number}}, fill them with ShopEase values
PLACEHOLDERS = {
    "Company Name": COMPANY,
    "Website URL": "www.shopease.com",
    "Online Company Portal Info": "the ShopEase website or app",
    "Login Page URL": "the ShopEase login page",
    "Account Recovery Page URL": "the Forgot Password page",
    "Customer Support Phone Number": "1800-123-4567",
    "Customer Support Email": "support@shopease.com",
    "Customer Support Hours": "9 AM to 9 PM, Monday to Saturday",
    "Account Type": "ShopEase Plus",
    "Account Category": "ShopEase Plus",
    "Upgrade Account": "ShopEase Plus",
    "Order Number": "your order number",
    "Invoice Number": "your invoice number",
    "Tracking Number": "your tracking number",
    "Refund Amount": "the refund amount",
    "Money Amount": "the amount",
    "Currency Symbol": "Rs.",
    "Person Name": "the customer",
    "Client Name": "the customer",
    "Client Last Name": "",
    "Salutation": "",
    "Delivery City": "your city",
    "Delivery Country": "India",
    "Date Range": "5 to 7 business days",
    "Store Location": "your nearest store",
    "Online Order Interaction": "My Orders",
    "Order Tracking": "My Orders > Track",
    "Order Status": "order status",
    "Settings": "Settings",
    "Profile": "Profile",
    "Profile Settings": "Profile",
    "Profile Type": "account type",
    "Account Change": "Change Account Type",
    "Forgot Password": "Forgot Password",
    "Forgot PIN": "Forgot PIN",
    "Forgot Access Key": "Forgot Password",
    "Forgot Key": "Forgot Password",
    "Account Recovery": "Account Recovery",
    "Security": "Security",
    "Cancel Purchase": "Cancel Order",
}


def fill_placeholders(text):
    def repl(m):
        key = m.group(1).strip()
        return PLACEHOLDERS.get(key, key.lower())   # unknown slot -> just lowercase text
    text = re.sub(r"\{\{(.*?)\}\}", repl, text)
    text = re.sub(r"[ ]{2,}", " ", text)           # double spaces from empty slots
    text = text.replace(" ,", ",").replace(" .", ".")
    return text.strip()


def load_bitext():
    df = pd.read_csv(RAW_CSV)
    df = df.drop_duplicates(subset=["instruction", "response"])
    df["instruction"] = df["instruction"].apply(fill_placeholders)
    df["response"] = df["response"].apply(fill_placeholders)
    return df


def split_bitext(df, test_size=0.1):
    # 10% held out, those questions never go into the index, used for evaluation later
    train_df, holdout_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["intent"])
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def bitext_to_documents(train_df, per_intent=RESPONSES_PER_INTENT):
    # keep only the first N unique responses per intent, the rest are near duplicates
    docs = []
    for intent, group in train_df.groupby("intent"):
        group = group.drop_duplicates(subset=["response"]).head(per_intent)
        for _, row in group.iterrows():
            docs.append(Document(
                page_content=f"Customer question: {row['instruction']}\nSupport answer: {row['response']}",
                metadata={"source": "bitext_faq", "intent": intent, "category": row["category"].lower()}
            ))
    return docs


def load_policy_documents(policy_dir=POLICY_DIR):
    docs = []
    for path in sorted(Path(policy_dir).glob("*.md")):
        docs.append(Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, "intent": "policy", "category": "policy"}
        ))
    return docs


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " "]   # markdown heading first so policy sections stay together
    )
    chunks = splitter.split_documents(docs)
    # chunk id like returns_and_refunds.md#3 so the API can show it as a source
    counters = {}
    for c in chunks:
        src = c.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        c.metadata["chunk_id"] = f"{src}#{counters[src]}"
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}   # normalised -> can convert L2 distance to cosine
    )


def build_faiss(chunks, embeddings, index_dir=INDEX_DIR):
    store = FAISS.from_documents(chunks, embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    return store


def train_intent_classifier(train_df, embeddings, index_dir=INDEX_DIR, per_intent=250):
    # logistic regression on the sentence embeddings, so finding the intent does not need an LLM call
    sample = train_df.sample(frac=1, random_state=42).groupby("intent").head(per_intent)
    X = embeddings.embed_documents(sample["instruction"].tolist())
    clf = LogisticRegression(max_iter=2000, C=5.0)
    clf.fit(X, sample["intent"])
    intent_to_category = train_df.drop_duplicates("intent").set_index("intent")["category"].str.lower().to_dict()
    joblib.dump({"model": clf, "intent_to_category": intent_to_category}, Path(index_dir) / "intent_clf.pkl")
    return clf


def make_eval_set(holdout_df, n=20, out_dir=EVAL_DIR):
    # 20 held out questions from 20 different intents, the bitext response is the reference answer
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    picked = holdout_df.groupby("intent").head(1).sample(n=n, random_state=7)
    rows = [{"question": r.instruction, "reference": r.response, "intent": r.intent, "category": r.category.lower()}
            for r in picked.itertuples()]
    with open(Path(out_dir) / "eval_questions.json", "w") as f:
        json.dump(rows, f, indent=2)
    holdout_df.to_csv(Path(out_dir) / "holdout.csv", index=False)
    return rows


def run_ingestion():
    print("loading bitext csv ...")
    df = load_bitext()
    train_df, holdout_df = split_bitext(df)
    print(f"rows: {len(df)}  train: {len(train_df)}  holdout: {len(holdout_df)}  intents: {df['intent'].nunique()}")

    docs = bitext_to_documents(train_df) + load_policy_documents()
    chunks = chunk_documents(docs)
    print(f"documents: {len(docs)}  chunks: {len(chunks)}")

    embeddings = get_embeddings()
    build_faiss(chunks, embeddings)
    print(f"faiss index saved to {INDEX_DIR}")

    train_intent_classifier(train_df, embeddings)
    print("intent classifier saved")

    make_eval_set(holdout_df)
    print(f"eval set saved to {EVAL_DIR}")


if __name__ == "__main__":
    run_ingestion()
