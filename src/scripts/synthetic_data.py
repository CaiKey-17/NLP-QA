import os
import json
import time
import multiprocessing as mp
from tqdm import tqdm
from google import genai
from google.genai.errors import APIError


# ==============================
# CONFIG
# ==============================
API_KEYS = [
    "KEY_1",
    "KEY_2",
    "KEY_3",
    "KEY_4",
    "KEY_5",
]

INPUT_PATH = r"D:\NLP\QA_NLP\data\train.jsonl"
OUTPUT_DIR = r"D:\NLP\QA_NLP\data\synthetic_parts"
FINAL_OUTPUT = r"D:\NLP\QA_NLP\data\synthetic_train.jsonl"

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 5
BASE_WAIT = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD JSONL
# ==============================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ==============================
# PROMPT
# ==============================
def make_prompt(context, question, answer):
    return f"""
Bạn là hệ thống tạo dữ liệu QA tiếng Việt.

Dựa vào thông tin ban đầu:
- Context: {context}
- Câu hỏi gốc: {question}
- Câu trả lời gốc: {answer}

Nhiệm vụ:
- Hãy sinh thêm **3 cặp câu hỏi và câu trả lời mới**.
- Câu hỏi phải tự nhiên, không nhắc đến “đoạn văn trên”.
- Đáp án phải đúng dựa trên context.
- **KHÔNG sinh label nữa**.
- Output **PHẢI** là JSON dạng list:
[
  {{"question":"...", "answer":"..."}},
  {{"question":"...", "answer":"..."}},
  {{"question":"...", "answer":"..."}}
]
"""


def extract_json(text):
    try:
        s = text.index("[")
        e = text.rindex("]") + 1
        return json.loads(text[s:e])
    except:
        return None


# ==============================
# WORKER
# ==============================
def worker(api_key, chunk, out_file):
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client()

    results = []

    pbar = tqdm(chunk, desc=f"KEY {api_key[-6:]}", position=mp.current_process()._identity[0])

    for item in pbar:
        context = item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer", "")

        prompt = make_prompt(context, question, answer)

        for retry in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[prompt],
                    config={
                        "temperature": 0.7,
                        "response_mime_type": "application/json"
                    }
                )

                txt = resp.text.strip()
                qa_list = extract_json(txt)

                if qa_list:
                    for qa in qa_list:
                        results.append({
                            "context": context,
                            "question": qa["question"],
                            "answer": qa["answer"]
                        })
                break

            except Exception as e:
                wait = BASE_WAIT * (2 ** retry)
                print(f"⚠ KEY {api_key[-6:]} lỗi, retry sau {wait}s")
                time.sleep(wait)

    # save
    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 KEY {api_key[-6:]} hoàn tất → {out_file}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("📥 Đang load train.jsonl ...")
    data = load_jsonl(INPUT_PATH)
    total = len(data)
    print(f"📌 Tổng số mẫu train: {total}")

    # chia theo số API key
    n = len(API_KEYS)
    chunk_size = total // n

    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
    # phần dư
    if len(chunks) < n:
        chunks[-1].extend(data[n*chunk_size:])

    print(f"🔀 Đã chia thành {len(chunks)} phần")

    processes = []
    out_files = []

    for idx, api_key in enumerate(API_KEYS):
        part = chunks[idx]
        out_file = os.path.join(OUTPUT_DIR, f"synthetic_part_{idx+1}.jsonl")
        out_files.append(out_file)

        p = mp.Process(target=worker, args=(api_key, part, out_file))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # merge output
    print("📦 Ghép các phần synthetic lại...")
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as fout:
        for path in out_files:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        fout.write(line)

    print("🎉 HOÀN TẤT!")
    print(f"📁 Synthetic train lưu tại: {FINAL_OUTPUT}")
