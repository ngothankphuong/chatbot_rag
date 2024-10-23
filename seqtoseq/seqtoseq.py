from collections import Counter
from keras.models import load_model
import re
import numpy as np
import string
from nltk import ngrams
import nltk
from unidecode import unidecode
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam


# Define the path to the model (adjust this path based on your environment)
model_path = 'D:\\Workspace-CTU\\LUANVAN\\chatbot-rag\\rag-main-new\\seqtoseq\\model_27_9_2024.h5'
model = load_model(model_path)

"""
Mục đích: Đảm bảo rằng các hàm mã hóa và giải mã dữ liệu được định nghĩa lại để sử dụng trong phần dự đoán.
Chi tiết: Các hàm này giống như đã được định nghĩa trước đó để chuẩn hóa dữ liệu đầu vào và giải mã kết quả dự đoán.
"""
vowel = list('aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴ')
full_letters = vowel + list('bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZđĐ')

NGRAM = 5
MAXLEN = 39
alphabet = ['\x00', ' '] + list('0123456789') + full_letters
def _encoder_data(text):
  x = np.zeros((MAXLEN, len(alphabet)))
  for i, c in enumerate(text[:MAXLEN]):
    x[i, alphabet.index(c)] = 1
  if i <  MAXLEN - 1:
    for j in range(i+1, MAXLEN):
      x[j, 0] = 1
  return x
def _decoder_data(x):
  x = x.argmax(axis=-1)
  return ''.join(alphabet[i] for i in x)

"""
Mục đích: Tạo các n-gram từ câu văn sử dụng thư viện NLTK.
Chi tiết:
Sử dụng nltk.ngrams để tạo các n-gram từ danh sách từ.
Chỉ thêm các n-gram có độ dài không vượt quá maxlen.
Nếu số từ ít hơn n, thêm toàn bộ câu như một n-gram duy nhất.
"""
def _nltk_ngrams(sentence, n, maxlen):
    list_ngrams = []
    list_words = sentence.split()
    num_words = len(list_words)

    if num_words >= n:
        for ngram in nltk.ngrams(list_words, n):
            if len(' '.join(ngram)) <= maxlen:
                list_ngrams.append(ngram)
    else:
        list_ngrams.append(tuple(list_words))
    return list_ngrams

"""
Mục đích: Dự đoán sửa lỗi cho một n-gram cụ thể.
Chi tiết:
Kết hợp các từ trong n-gram thành một chuỗi văn bản.
Mã hóa chuỗi văn bản và đưa vào mô hình để dự đoán.
Giải mã kết quả dự đoán và loại bỏ ký tự padding ('\x00').
"""
def _guess(ngram):
    text = " ".join(ngram)
    preds = model.predict(np.array([_encoder_data(text)]))

    return _decoder_data(preds[0]).strip('\x00')

"""
Mục đích: Thêm lại các dấu câu vào văn bản đã được sửa lỗi.
Chi tiết:
Duyệt qua từng từ trong văn bản gốc text để xác định vị trí các dấu câu ở đầu và cuối từ.
Lưu các dấu câu này vào list_punctuation với chỉ số vị trí từ.
Sau đó, kết hợp các từ đã được sửa lỗi corrected_text với các dấu câu tương ứng từ list_punctuation.
"""
def _add_punctuation(text, corrected_text):
    list_punctuation = {}
    for (i, word) in enumerate(text.split()):
        if word[0] not in alphabet or word[-1] not in alphabet:
            # Dấu ở đầu chữ như " và '
            start_punc = ''
            for c in word:
                if c in alphabet:
                    break
                start_punc += c

            # Dấu ở sau chữ như .?!,;
            end_punc = ''
            for c in word[::-1]:
                if c in alphabet:
                    break
                end_punc += c
            end_punc = end_punc[::-1]

            # Lưu vị trí từ và dấu câu trong từ đó
            list_punctuation[i] = [start_punc, end_punc]

    # Thêm dấu câu vào vị trí các từ đã đánh dấu
    result = ''
    for (i, word) in enumerate(corrected_text.split()):
        if i in list_punctuation:
            result += (list_punctuation[i][0] + word + list_punctuation[i][1]) + ' '
        else:
            result += word + ' '

    return result.strip()

"""
Mục đích: Chỉnh sửa văn bản nhập vào bằng cách sử dụng mô hình đã huấn luyện để sửa lỗi.
Chi tiết:
Bước 1: Loại bỏ các ký tự đặc biệt không thuộc alphabet.
Bước 2: Tạo các n-gram từ văn bản đã được làm sạch.
Bước 3: Dự đoán sửa lỗi cho từng n-gram bằng hàm _guess.
Bước 4: Sử dụng Counter để xác định từ được dự đoán phổ biến nhất tại mỗi vị trí trong câu.
Bước 5: Kết hợp các từ đã được sửa lỗi thành văn bản cuối cùng và thêm lại các dấu câu.
"""
def _correct(text):
    # Xóa các ký tự đặc biệt
    new_text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
  

    ngrams = list(_nltk_ngrams(new_text, NGRAM, MAXLEN))
    guessed_ngrams = list(_guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]

    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(r'\s', ngram)):
            index = nid + wid
            # print(f'nid: {nid}, wid: {wid}, index: {index}, candidates length: {len(candidates)}')
            if index < len(candidates):
                candidates[index].update([word])
            else:
                # Safely append to candidates if the index exceeds the current list length
                candidates.append(Counter([word]))
                # print(f"Index {index} is out of range, adding new Counter!")

    corrected_text = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
    return _add_punctuation(text, corrected_text)


# -----------------------------------------------------------------------