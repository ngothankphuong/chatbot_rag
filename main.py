#docs txt
from imports import *
from seqtoseq.seqtoseq import _correct

#0
#data txt
def load_split_txt() -> Document:
    path_for_split_file = os.path.join(current_directory, f'split_docs\\{formatted_time}.txt')
    file_path = path_default_document
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = text.split('\n\n')
    documents = [Document(page_content=chunk) for chunk in chunks]
    with open(path_for_split_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc.page_content)
            f.write('\n---\n')
    print(documents)
    return documents

#0
#RERANK
def rerank_cohere(arr: list, query: str) -> dict[str, float]:
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query= query,
        documents=arr,
        top_n=5,
    )

    reranked_docs = []
    for item in response.results:
        doc = arr[item.index]
        score = item.relevance_score
        reranked_docs.append((doc, score))

    # for i, (doc, score) in enumerate(reranked_docs):
    #     print(f"Tài liệu {i+1}:")
    #     print(f"Nội dung: {doc}")
    #     print(f"Độ liên quan: {score}")
    #     print("-" * 40)

    return reranked_docs

#0
#HYDE GENERATION
def hyde_generate(user_query: str) -> str:
    # HyDE document genration
    template = """Bạn là trợ lý ảo của trường đại học Y dược Cần Thơ. Tạo ra duy nhất một đoạn văn khoa học để trả lời câu hỏi sau.
    
    Question: {question}

    Answer:"""

    prompt_hyde = ChatPromptTemplate.from_template(template)
    generate_docs_for_retrieval = (
        prompt_hyde | ChatCohere(temperature=0.3) | StrOutputParser() 
    )
    hyde_doc =  generate_docs_for_retrieval.invoke({"question":user_query})

    query_hyde = f"{user_query}. {hyde_doc}"

    print(f"TÀI LIỆU GIẢI ĐỊNH : {query_hyde}")
    return query_hyde

#0
#Router các câu hỏi bên ngoài
def response_prompt2(user_ques):
    llm = ChatCohere(model="command-r-08-2024", temperature=0.3)
    try:
        prompt_template = PromptTemplate(
            template="""
            Bạn là một trợ lý AI có tên là 'AIBOT' được phát triển bởi các sinh viên CTU. Để trả lời các câu hỏi về đại học Y dược Cần Thơ. Khi trả lời các câu hỏi, chắc chắn rằng không bao giờ trả lời các vấn đề sau đây:
            1. Chính trị: Tránh thảo luận về các vấn đề chính trị, đảng phái, hoặc quan điểm chính trị.
            2. Tôn giáo: Không đưa ra ý kiến hoặc thông tin về tôn giáo, tín ngưỡng, hay thực hành tôn giáo.
            3. Từ phản cảm: Tránh sử dụng hoặc phản hồi về các từ ngữ phản cảm, xúc phạm, hoặc không thích hợp.
            4. Gây thù dịch: Không đưa ra ý kiến hay thông tin có thể gây ra sự thù địch, phân biệt, hoặc thù địch giữa các cá nhân hoặc nhóm.
            5. Không đề cập đến các công nghệ hoặc những gì liên quan đến Cohere.
            6. Các câu hỏi về đảng và nhà nước.
            Hãy tập trung vào việc cung cấp thông tin chính xác và hữu ích liên quan đến các vấn đề học thuật và quy trình tuyển sinh của trường Đại học. Cảm ơn bạn!
            
            Trả lời ngắn ngọn, xúc tích câu hỏi sau:
            Question: {question}
            """,
            input_variables=["question"]
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True
        )
        response = chain.run({"question": user_ques})
        print(response)
        return response
    except Exception as e:
        print(f"Lỗi : {e}")
        return "Có lỗi Router câu hỏi không liên quan"

# load default document
def default_faiss_db():
    with open(path_default_document, "r", encoding="utf-8") as file:
        text = file.read()
    chunks = text.split('\n\n')
    documents = [Document(page_content=chunk) for chunk in chunks]
    index = FAISS.from_documents(
                    documents, 
                    CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")
                )
    index.save_local(path_for_FAISS)
    #save split file
    with open(path_for_split_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc.page_content)
            f.write('\n---\n')

#check is file pdf
def is_pdf(path_to_file) -> bool:
    try:
        extract_text(path_to_file)
        return True
    except:
        return False

#Add documents faiss file .pdf
def add_documents_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap = 200
    )
    documents = splitter.split_documents(docs)
    # Ghi các đoạn văn bản vào file text
    with open('my_main_copy.txt', 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc.page_content)  # Ghi nội dung của từng đoạn
            f.write('\n---\n')
    print("Thêm dữ liệu thành công")
    return documents

#Add documents faiss 
def add_documents_faiss(file_path, type:str):
    if(type=="txt"):
        #load dữ liệu & chunking
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        chunks = text.split('\n\n')
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(documents)
        #ghi chunk vao file
        with open(path_for_split_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc.page_content)
                f.write('\n---\n')

        new_docs = FAISS.from_documents(documents, CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0"))
        old_db = FAISS.load_local(
            path_for_FAISS,
            CohereEmbeddings(cohere_api_key=api_key,model="embed-multilingual-v3.0"),
            allow_dangerous_deserialization=True
        )
        old_db.merge_from(new_docs)
        old_db.save_local(path_for_FAISS)
        print("Thêm dữ liệu txt thành công")
    elif(type=="pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap = 200
        )
        documents = splitter.split_documents(docs)
        print(documents)
        #ghi chunk vao file
        with open(path_for_split_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc.page_content)
                f.write('\n---\n')

        new_docs = FAISS.from_documents(documents, CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0"))
        old_db = FAISS.load_local(
            path_for_FAISS,
            CohereEmbeddings(cohere_api_key=api_key,model="embed-multilingual-v3.0"),
            allow_dangerous_deserialization=True
        )
        old_db.merge_from(new_docs)
        old_db.save_local(path_for_FAISS)
        print("Thêm dữ liệu pdf thành công")
        
#đồng bộ dữ liệu file upload vào faiss db
def sync():
    folder_file = os.path.join(current_directory, 'file_upload','*')
    # chuyển thành mảng chứa các file con
    files = glob.glob(folder_file)
    for file in files:
        #file path
        file_data = os.path.join(current_directory, file)
        print(f"{file} : {is_pdf(file)}")
        if(is_pdf(file)==False):
            add_documents_faiss(file_data, "txt")
        elif(is_pdf(file)==True):
            add_documents_faiss(file_data, "pdf")
    # load DB mới sau khi thêm
    global docsearch
    docsearch = FAISS.load_local(path_for_FAISS, CohereEmbeddings(cohere_api_key=api_key,model="embed-multilingual-v3.0"),allow_dangerous_deserialization=True)

#retriever
def retrieval(user_ques: str, rasa_response: str, chat_history: list) -> str:
    llm = ChatCohere(model="command-r-08-2024", temperature=0.1)
    
    try:
        # Cấu hình Retrieval langchain
        retriever = docsearch.as_retriever()
        retriever.search_kwargs['fetch_k'] = 30
        retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 10

        # custom multiple retriever
        class LineListOutputParser(BaseOutputParser[List[str]]):
            """Output parser for a list of lines."""
            def parse(self, text: str) -> List[str]:
                lines = text.strip().split("\n")
                return list(filter(None, lines))
        output_parser = LineListOutputParser()
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Bạn là trợ lý mô hình ngôn ngữ AI. Nhiệm vụ của bạn là tạo ra 2 
            các phiên bản khác nhau của câu hỏi người dùng nhất định để truy xuất các tài liệu liên quan từ một vectơ cơ sở dữ liệu. Bằng cách tạo ra nhiều góc nhìn cho câu hỏi của người dùng, mục tiêu của bạn là giúp người dùng khắc phục một số hạn chế của tìm kiếm tương tự dựa trên khoảng cách.Cung cấp các câu hỏi thay thế này được phân tách bằng dòng mới.
            Câu hỏi gốc: {question}""",
        )
        llm_chain = QUERY_PROMPT | llm | output_parser
        multi_query_retriever_2 = MultiQueryRetriever(
            retriever=retriever,
            llm_chain=llm_chain,
            parser_key="lines"
        )

        #default multiple query retriever : 3 
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm,
            parser_key="lines"
        )
        
        #history
        history_prompt ="""Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        #custom history prompt
        """Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng \
        có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, hãy tạo ra một câu hỏi độc lập \
        có thể được hiểu mà không cần lịch sử trò chuyện. Đừng trả lời câu hỏi, \
        chỉ cần chỉnh sửa lại câu hỏi nếu cần, nếu không thì giữ nguyên và trả lại như vậy."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", history_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt,)

        # retriever
        custom_prompt_template =  ("Bạn tên 'AskU' chatbot về các thông tin Trường Đại học Y dược Cần Thơ. "
                                   "Dựa vào thông tin được cung cấp,trả lời câu hỏi. "
                                   "Nếu không biết câu trả lời,yêu cầu người hỏi cung cấp thêm thông tin cho câu hỏi. "
                                   "KHÔNG cố tạo ra câu trả lời. "
                                   "KHÔNG trả lời các câu hỏi KHÔNG liên quan đến trường Đại học Y dược Cần Thơ. "
                                    "Thông tin:{context}")
        #default retriever prompt
        system_prompt = (
            "You're 'AskU' chatbot.Use the given context to answer the question in Vietnamese."
            "If you don't know the answer, request additional information."
            "Keep the answer concise but informative."
            "Context: {context}."
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", custom_prompt_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        # add rasa response
        # combined_question = f"{rasa_response}\nCâu hỏi: {user_ques}"

        input_data = {
            "input": user_ques, 
            "chat_history": chat_history
        }
        result = rag_chain.invoke(
            input_data,
            # config={'callbacks': [ConsoleCallbackHandler()]}
        )
        return result['answer']
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}")
        return "Xin lỗi, đã xảy ra lỗi trong quá trình xử lý yêu cầu của bạn."

@app.route('/')
def index():
    return render_template('user.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    status = None
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            # kiểm tra tồn tại file
            if os.path.exists(file_path):
                status = "File already exists"
            else:
                file.save(file_path)
                status = "Upload success"
        elif 'sync' in request.form:
            sync()
            status = "Sync success"
        else:
            status = "No file selected or no sync request"

    return render_template('upload_data.html', status=status)

@app.route("/check_spell", methods=["POST"])
def check_spell():
    data = request.get_json()
    user_question = data.get("question") 
    if not user_question.strip():
        return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        response = _correct(user_question)
        print(f"Câu sửa lỗi: {response}")
        return jsonify({"response": response})  
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    print("data", data)

    user_message = data.get('message')
    print(f"-----------Tin nhắn từ người dùng: {user_message} -----------\n\n")
   
   #chat history của từng người dùng
    chat_history = []
    for qa in data.get('chat_history'):
        # print(qa)
        chat_history.extend([HumanMessage(content=qa.get('human')),qa.get('ai')])

    bot_reply = ''
    post_data = {"sender": "user", "message": user_message}
    rasa_response = requests.post(RASA_API_URL, json=post_data)
    rasa_response.raise_for_status()

    """Xử lý phản hồi từ Rasa"""
    if rasa_response.status_code == 200:
        rasa_data = rasa_response.json()
        if rasa_data:
            # Phản hồi từ Rasa
            print(f"PHẢN HỒI TỪ RASA : {rasa_data}")
            bot_reply = rasa_data[0].get('text', 'Xin lỗi, tôi không hiểu câu hỏi của bạn.')
        else:
            bot_reply = '' 
    #kết quả retrieval
    rag_answer = retrieval(user_message, bot_reply, [])
    return jsonify({
        "reply": rag_answer
    })

if __name__ == '__main__':
    print("Khởi tạo server.")
    # tạo thư mục spit docs
    if not os.path.exists(folder_split_doc):
        os.makedirs(folder_split_doc)
    #ttạo thư mục upload
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    default_faiss_db()
    global docsearch
    docsearch = FAISS.load_local(path_for_FAISS, CohereEmbeddings(cohere_api_key=api_key,model="embed-multilingual-v3.0"),allow_dangerous_deserialization=True)
    app.run(host='0.0.0.0', port=5000)
