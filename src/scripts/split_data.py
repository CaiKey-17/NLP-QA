import os
import json
import random
import math

INPUT_PATH = r"qa_dataset.json"
OUTPUT_DIR = r"qa_splits_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. LOAD DỮ LIỆU
# ----------------------------
data = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

total = len(data)
print("📌 Tổng số mẫu:", total)

# ----------------------------
# 2. SHUFFLE (quan trọng)
# ----------------------------
random.shuffle(data)

# ----------------------------
# 3. TÍNH TỶ LỆ 8:1:1
# ----------------------------
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

n_train = math.floor(total * train_ratio)
n_val   = math.floor(total * val_ratio)
n_test  = total - n_train - n_val   # tránh bị lệch số cuối

print(f"📦 train: {n_train}, val: {n_val}, test: {n_test}")

# ----------------------------
# 4. CHIA DỮ LIỆU
# ----------------------------
train_data = data[:n_train]
val_data   = data[n_train : n_train + n_val]
test_data  = data[n_train + n_val :]

splits = {
    "train.jsonl": train_data,
    "val.jsonl": val_data,
    "test.jsonl": test_data,
}

# ----------------------------
# 5. GHI FILE
# ----------------------------
for filename, split_data in splits.items():
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for row in split_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"➡️ Đã tạo {filename} ({len(split_data)} mẫu)")

print("🎉 Hoàn tất chia dataset theo tỷ lệ 8-1-1!")
