import os
import json

# ============================
# CONFIG
# ============================
TRAIN_ORIGINAL = r"D:\NLP\QA_NLP\qa_splits_fixed\train.jsonl"
SYNTHETIC_DIR = r"D:\NLP\QA_NLP\data\synthetic"
SYNTHETIC_MERGED = r"D:\NLP\QA_NLP\data\synthetic_train.jsonl"
FINAL_TRAIN = r"D:\NLP\QA_NLP\data\train_full.jsonl"



# ---- Load JSONL ----
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


# ---- Load JSON (array) ----
def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- MERGE synthetic parts (json or jsonl) ----
def merge_synthetic_parts():
    print("➡️ Đang merge synthetic_parts/...")

    all_items = []

    for fname in sorted(os.listdir(SYNTHETIC_DIR)):
        fpath = os.path.join(SYNTHETIC_DIR, fname)

        if fname.endswith(".jsonl"):
            data = load_jsonl(fpath)
            print(f"   + {fname}: {len(data)} dòng (JSONL)")
            all_items.extend(data)

        elif fname.endswith(".json"):
            data = load_json_array(fpath)
            print(f"   + {fname}: {len(data)} dòng (JSON)")
            all_items.extend(data)

    # Save thành synthetic_train.jsonl
    with open(SYNTHETIC_MERGED, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 Đã tạo: {SYNTHETIC_MERGED} ({len(all_items)} dòng)")
    return all_items


# ---- MERGE vào train gốc ----
def merge_final(train_original, synthetic_items):
    merged = train_original + synthetic_items

    with open(FINAL_TRAIN, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 Đã tạo train_full.jsonl ({len(merged)} dòng)")


# ---- MAIN ----
if __name__ == "__main__":
    print("📥 Đang load train gốc...")
    train_original = load_jsonl(TRAIN_ORIGINAL)
    print(f"📌 Train gốc: {len(train_original)} dòng")

    synthetic_items = merge_synthetic_parts()
    merge_final(train_original, synthetic_items)

    print("🎉 HOÀN TẤT MERGE!")