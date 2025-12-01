NLP Lab Report:
Lab 1: Text Tokenization
Mục tiêu

Thực hiện bước tiền xử lý cơ bản trong NLP – tách chuỗi văn bản thành các token riêng lẻ:

Xây dựng tokenizer đơn giản SimpleTokenizer.

Cài đặt RegexTokenizer dựa trên biểu thức chính quy.

Áp dụng tokenization trên câu ví dụ và bộ dữ liệu UD_English-EWT.

Công việc thực hiện

SimpleTokenizer

Chuyển văn bản về chữ thường.

Chia từ dựa trên dấu cách.

Một số dấu câu cơ bản (. , ? !) được tách thành token riêng.

RegexTokenizer

Sử dụng regex \w+|[^\w\s] để nhận diện token.

Hoạt động chính xác hơn ở các trường hợp như từ viết tắt, dấu nháy đơn,…

Thử nghiệm trên UD_English-EWT

Lấy 500 ký tự đầu tiên trong tập dữ liệu.

So sánh token thu được từ hai tokenizer.

Kết quả chạy code
Tokenizing các câu ví dụ:
Original: Hello, world! This is a test.
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer:   ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Original: NLP is fascinating... isn't it?
SimpleTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer:   ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Original: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer:   ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

Lab 2: Count Vectorization
Mục tiêu

Chuyển văn bản sang dạng vector số bằng phương pháp Bag-of-Words:

Tận dụng tokenizer từ Lab 1.

Tạo CountVectorizer để xây dựng vocabulary và ma trận đặc trưng.

Công việc thực hiện

Vectorizer Interface

Khai báo ba phương thức cơ bản: fit, transform, fit_transform.

CountVectorizer

Nhận một tokenizer tùy chọn.

Sinh vocabulary từ corpus.

Biểu diễn mỗi document thành vector đếm token.

Thử nghiệm

Sử dụng RegexTokenizer cho corpus đơn giản.

Quan sát kết quả ma trận sau khi fit_transform.

Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

NLP Lab Report: Lab 3
I. Giải thích các bước thực hiện
Bước 1: Khai thác mô hình embedding có sẵn (Task 1 & 2)

Tải mô hình Gensim pre-trained

Sử dụng lớp WordEmbedder để nạp các mô hình như glove-wiki-gigaword-50.

Mục đích: đánh giá độ tương tự từ, tìm từ gần nghĩa và biểu diễn câu bằng vector tổng hợp.

Trích xuất thông tin từ embedding

Các bài kiểm thử được mô tả trong lab3\test\test_b3.py.

Một số thao tác:

Lấy vector của từ qua get_vector.

Tính similarity bằng get_similarity.

Tìm danh sách từ tương tự với get_most_similar.

Nhúng document bằng embed_document.

Bước 2: Huấn luyện Word2Vec trên tập dữ liệu nhỏ (Task 3)

Thực hiện trong: lab3/test/lab4_embedding_training_demo.py

Chuẩn bị dữ liệu

Đọc file en_ewt-ud-train.txt.

Dùng lớp SentenceStream để cung cấp từng câu đã tiền xử lý (simple_preprocess).

Huấn luyện Word2Vec

Tham số: vector size = 100, window = 5, min_count = 3, workers = 8, mô hình Skip-gram (sg=1).

Sau khi training xong, mô hình được lưu lại.

Thử nghiệm

Tìm từ tương tự ví dụ: gần nghĩa với “computer”.

Thực hiện analogies: king - man + woman.

Bước 3: Huấn luyện Word2Vec trên dữ liệu lớn với Spark (Task 4)

Code tại: lab3/test/lab4_spark_word2vec_demo.py

Tải tập dữ liệu lớn

Dữ liệu JSON (c4-train) được load bằng Spark DataFrame.

Các bước lọc rác, làm sạch văn bản được thực hiện trước khi tokenize.

Tokenization với Spark ML

Dùng Tokenizer để chia câu thành danh sách từ.

Huấn luyện Word2Vec

Cấu hình: vectorSize = 100, minCount = 5.

Xuất mô hình và thử tìm top từ giống “computer”.

Bước 4: Giảm chiều & trực quan hóa embedding (Task 5)

Thực hiện trong: lab3/b3.ipynb

Load embedding GloVe

Đọc file GloVe 50D.

Lấy mẫu ~40k từ để giảm thời gian xử lý.

Giảm chiều

PCA: giảm từ 50 → 2 chiều, giữ cấu trúc tổng quát.

t-SNE: giảm từ 50 → 2 chiều, làm rõ cụm từ cục bộ.

Vẽ biểu đồ

Scatter plot cho PCA & t-SNE.

Ghi nhãn hàng trăm từ để quan sát phân bố và cụm chủ đề.

II. Hướng dẫn chạy code
Task 1 & 2: Mô hình pre-trained
python lab3/test/test_b3.py


Kết quả bao gồm:

Vector của từ.

Similarity giữa các cặp từ.

Danh sách từ gần nghĩa.

Vector biểu diễn 1 câu văn.

Task 3: Train Word2Vec mini-corpus
python lab3/test/lab4_embedding_training_demo.py


Sinh ra:

Mô hình Word2Vec tự huấn luyện.

Demo tính từ gần nghĩa & analogies.

Task 4: Train Word2Vec với Spark
python lab3/test/lab4_spark_word2vec_demo.py

Task 5: PCA & t-SNE

Mở notebook .ipynb

Chạy lần lượt các cell để trực quan hóa embedding.

III. Phân tích kết quả
3.1 Mô hình embedding pre-trained

Cho kết quả ổn định với các cặp từ liên quan ("king ↔ queen").

Các từ gần nghĩa với “computer” phản ánh đúng chủ đề: software, digital, pc, internet,…

3.2 So sánh mô hình tự huấn luyện

Dataset nhỏ → chất lượng embedding kém, kết quả không thực tế.

Dataset lớn với Spark → kết quả hợp lý hơn nhưng vẫn chưa mạnh bằng mô hình pre-trained.

3.3 PCA & t-SNE

PCA: phản ánh bố cục chung, nhưng các cụm nhỏ chưa rõ ràng.

t-SNE: làm nổi bật rõ cụm từ theo chủ đề.

Các nhóm từ công nghệ, tên người, hành động… được phân chia dễ nhận thấy.

IV. Khó khăn gặp phải

Thiết lập Spark và xử lý dữ liệu lớn tốn nhiều thời gian.

Một số mô hình yêu cầu tài nguyên tính toán mạnh.

V. Tài liệu tham khảo

Ngoài các thư viện được sử dụng trong quá trình hiện thực, em còn tham khảo thêm hỗ trợ từ ChatGPT và DeepSeek để hoàn thiện bài.