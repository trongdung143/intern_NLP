# intern\_NLP

- [EX1 Huấn luyện Model NER bằng SPACY](#ex1-huấn-luyện-model-ner-bằng-spacy)
- [EX2 Ý tường đề xuất công nghệ chuyển đổi văn bản thành âm thanh nói (vietnamese)](#ex2-ý-tường-đề-xuất-công-nghệ-chuyển-đổi-văn-bản-thành-âm-thanh-nói-vietnamese)
<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12.3%2B-blue"/>
  <img alt="spaCy" src="https://img.shields.io/badge/spaCy-3.x-06b6d4"/>
</p>

---

## EX1 Huấn luyện Model NER bằng SPACY.
## 1) Tóm tắt

**NER (trích xuất NAME/EMAIL)** từ CV quét PDF:

   * Công cụ: `spaCy 3.x`, mô hình nền `en_core_web_lg`.
   * Quy trình: PDF → TXT → annotate(https://tecoholic.github.io/ner-annotator) → train → gợi ý nhãn → hiệu chỉnh.

Kèm theo **FastAPI** để phục vụ:

* `POST /extract-cv` – Thêm PDF và trả về danh sách entity (NAME/EMAIL).

## 2) Cấu trúc thư mục (dự kiến)

```
intern_NLP/
├─ data/                  # dữ liệu huấn luyện
├─ static/                # HTML
├─ api.py                 # FastAPI app (điểm vào uvicorn)
├─ pdf_to_txt.py          # tiện ích chuyển đổi PDF → TXT
├─ processing_data.py     # tiền xử lý dữ liệu
├─ train.py               # train NER 
├─ predict.py             # dự đoán của model
├─ test.py                # test nhanh chức năng
├─ requirements.txt       # danh sách thư viện Python
├─ render.yaml            # cấu hình deploy Render
└─ README.md
```

## 3) Chuẩn bị môi trường

### Yêu cầu

* Python **3.12.3+** 
* (Tuỳ chọn) CUDA nếu muốn train tăng tốc GPU.

### Cài đặt

```bash
# Tạo & kích hoạt venv (Windows)
python -m venv .env
.env\Scripts\activate

# (macOS/Linux)
# python3 -m venv .env
# source .env/bin/activate

# Cập nhật pip/setuptools/wheel (nếu pip cảnh báo)
python -m pip install -U pip setuptools wheel

# Cài thư viện
pip install -r requirements.txt

# Tải mô hình spaCy cần dùng (ví dụ tiếng Anh lớn)
python -m spacy download en_core_web_lg
```

## 4) Dữ liệu & Tiền xử lý
1. **Chuyển PDF → TXT**

   * Đặt file PDF vào `data/resumes/`.
   * Chạy script chuyển đổi (tham khảo `pdf_to_txt.py`). Ví dụ hàm `convert_pdf_folder_to_txt(input_dir, output_dir)` đã có sẵn – chỉnh đường dẫn phù hợp.
2. **Annotate**:

   * Format tập huấn luyện theo chuẩn spaCy `Example.from_dict` (`{"entities": [(start, end, label), ...]}`)
   * Lưu ra JSON/JSONL hoặc module python.

## 5) Huấn luyện
```bash
import spacy
import random
from processing_data import prcesssing_data
from spacy.util import minibatch, compounding
from spacy.training import Example

TRAIN_DATA = prcesssing_data()
nlp = spacy.load("en_core_web_lg")


if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")


for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    optimizer.L2 = 1e-6
    optimizer.L2_is_weight_decay = True

    best_loss = float("inf")
    best_model = None
    patience = 5
    wait = 0

    epochs = 50
    for epoch in range(epochs):
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 16.0, 1.07))
        drop = 0.2 + (epoch / epochs) * 0.1
        lr = 0.001 - (epoch / epochs) * (0.001 - 0.0003)
        optimizer.learn_rate = lr

        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)

        current_loss = losses.get("ner", 0.0)
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

        if current_loss < best_loss:
            best_loss = current_loss
            nlp.to_disk("./model")
            wait = 0
            print("Saved new best model.")
        else:
            wait += 1
            print(f"No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print("Early stopping triggered.")
            break



```
## Thuật toán huấn luyện NER

Quy trình huấn luyện NER trong repo sử dụng **spaCy** gồm các bước:

1. **Chuẩn bị dữ liệu**  
   - Dữ liệu huấn luyện dạng `(text, {"entities": [(start, end, label), ...]})`.
   - Ví dụ:  
     ```python
     ("My name is Alice", {"entities": [(11, 16, "NAME")]})
     ```
   - filter.py để phát hiện những dữ liệu được đánh nhãn lỗi (thiếu email hoặc name).
     ```python
      from pathlib import Path
      import json
      from tqdm import tqdm
      
      def check_name_no_email():
      files = sorted(Path("./data/json").glob("*.json"), key=lambda x: int(x.stem))
      
      for json_file in tqdm(files, desc="Check JSON"):
          with open(json_file, encoding="utf-8") as f:
              data = json.load(f)
        
          text = data["annotations"][0][0]
          ents = data["annotations"][0][1]["entities"]
          labels = [label for _, _, label in ents]
        
          has_name = "NAME" in labels
          has_email = "EMAIL" in labels
        
          if (
              has_email and not has_name
          ):  # or (has_name and not has_email and "@" in text):
              print(f"{json_file.name}")
              print(f"   Content: {text[:200]}...")
              print(f"   Entities: {ents}\n")
      
      
     check_name_no_email()
     ```
     
   - processing_data.py để gộp nhưng dữ liệu json đơn lẻ được đánh nhãn thành một tập dữ liệu để chuẩn bị cho quá trình training.
     ```python
     from pathlib import Path
     import json
     from tqdm import tqdm
      
      
     def prcesssing_data():
        data_all = []
    
        for json_file in tqdm(list(Path("./data/json").glob("*.json")), desc="Read JSON"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                data_all.append(data)
    
        TRAIN_DATA = []
    
        for item in data_all:
            TRAIN_DATA.append((item["annotations"][0][0], item["annotations"][0][1]))
        return TRAIN_DATA
     ```

2. **Khởi tạo mô hình**
- Load pipeline nền: `en_core_web_lg`  
- Thêm pipe `ner` nếu chưa có  
- Đăng ký các nhãn từ dữ liệu huấn luyện  

3. **Vòng lặp huấn luyện**
- **Tắt** các pipeline khác (chỉ train NER)  
- Khởi tạo `optimizer = nlp.begin_training()`  
- Thông số:
  - Epochs: 50  
  - Mini-batch: từ 4 → 16 (sử dụng `compounding`)  
  - Dropout: từ 0.2 tăng dần lên 0.3  
  - Learning rate: từ 0.001 giảm dần xuống 0.0003  
- Trong mỗi batch:
  ```python
  examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
  nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)

4. **Early Stopping**  
   - Sau mỗi epoch, so sánh `current_loss` với `best_loss`.  
   - Nếu tốt hơn:
     - Lưu mô hình (`nlp.to_disk("./model")`).  
     - Reset bộ đếm patience (`wait=0`).  
   - Nếu không:
     - Tăng bộ đếm patience (`wait += 1`).  
   - Khi `wait >= patience` → dừng sớm (**early stopping**).

### Tóm tắt
- **Thuật toán chính**: *Supervised NER training* với **ADAM + Dropout + Early Stopping**.  
- **Mục tiêu**: giảm loss, lưu mô hình tốt nhất, tránh overfitting.

```bash
#training model
python train.py
```

```
Epoch 1, Loss: 34119.7148
Saved new best model.
Epoch 2, Loss: 5391.6880
Saved new best model.
Epoch 3, Loss: 1973.6184
Saved new best model.
Epoch 4, Loss: 740.1302
Saved new best model.
Epoch 5, Loss: 998.2747
No improvement. Patience: 1/5
Epoch 6, Loss: 196.5778
Saved new best model.
Epoch 7, Loss: 158.8715
Saved new best model.
Epoch 8, Loss: 153.6830
Saved new best model.
Epoch 9, Loss: 124.1820
Saved new best model.
Epoch 10, Loss: 102.4711
Saved new best model.
Epoch 11, Loss: 220.0776
No improvement. Patience: 1/5
Epoch 12, Loss: 114.3631
No improvement. Patience: 2/5
Epoch 13, Loss: 75.3984
Saved new best model.
Epoch 14, Loss: 72.8613
Saved new best model.
Epoch 15, Loss: 83.2745
No improvement. Patience: 1/5
Epoch 16, Loss: 55.8511
Saved new best model.
Epoch 17, Loss: 63.2302
No improvement. Patience: 1/5
Epoch 18, Loss: 44.2567
Saved new best model.
Epoch 19, Loss: 52.7343
No improvement. Patience: 1/5
Epoch 20, Loss: 36.8882
Saved new best model.
Epoch 21, Loss: 57.0876
No improvement. Patience: 1/5
Epoch 22, Loss: 41.7358
No improvement. Patience: 2/5
Epoch 23, Loss: 41.3097
No improvement. Patience: 3/5
Epoch 24, Loss: 36.3888
Saved new best model.
Epoch 25, Loss: 43.9142
No improvement. Patience: 1/5
Epoch 26, Loss: 42.1220
No improvement. Patience: 2/5
Epoch 27, Loss: 27.0462
Saved new best model.
Epoch 28, Loss: 47.2889
No improvement. Patience: 1/5
Epoch 29, Loss: 43.8407
No improvement. Patience: 2/5
Epoch 30, Loss: 44.5443
No improvement. Patience: 3/5
Epoch 31, Loss: 25.4390
Saved new best model.
Epoch 32, Loss: 30.9066
No improvement. Patience: 1/5
Epoch 33, Loss: 26.6639
No improvement. Patience: 2/5
Epoch 34, Loss: 37.9954
No improvement. Patience: 3/5
Epoch 35, Loss: 31.9484
No improvement. Patience: 4/5
Epoch 36, Loss: 25.8055
No improvement. Patience: 5/5
Early stopping triggered.
```
- model tốt nhất được lưu khi loss hiện tại nhỏ hơn loss trước
- quá trình training đã dừng ở epoch 36 sau 5 lần loss không giảm kẻ từ epoch 31.

## 6) Chạy API (FastAPI)

```bash
# chạy server
uvicorn api:app --reload --host 0.0.0.0 --port 8080
```
  truy cập http://localhost:8080


## 7) Deploy lên ngrok

```bash
ngrok start --all
```
  truy cập  https://brave-eagerly-foal.ngrok-free.app

---

## EX2 Ý tường đề xuất công nghệ chuyển đổi văn bản thành âm thanh nói (vietnamese).


