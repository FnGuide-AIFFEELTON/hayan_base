# 베이스 모델
# 1. PDF 폴더 및 Chroma DB 폴더 지정
# 2. 함수화
#    - PDF 폴더에서 모든 문서를 가져오고, PDF 파일에서 텍스트 추출
#    - 문서 분할(텍스트만 있는 documents 리스트를 Document 객체로 변환)
#    - Chroma DB에 저장(이 단계에서 실제 텍스트 확인 필요)
#    - 리트리버 설정
# 3. 실행 (PDF 로드 -> 텍스트 분할 -> 벡터 DB 저장)
# ** 메모리 기능 없음**

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# .env 파일에서 OpenAI API 키 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# PDF 파일이 들어 있는 폴더
FOLDER_PATH = "./raw_pdf"
DB_PATH = "./chroma_db"

# PDF 파일에서 텍스트 추출
def load_pdfs_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)  # pypdfloader 사용
            pages = loader.load_and_split()
            text = ""
            for page in pages:
                text += page.page_content  # 페이지 내용 추출
            documents.append(text)
    return documents

# 문서 분할
def split_text(documents):
    # 텍스트만 있는 documents 리스트를 Document 객체로 변환
    documents = [Document(page_content=doc) for doc in documents]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    return split_docs

# Chroma DB에 저장
def store_in_chroma(split_docs):
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    db = Chroma.from_documents(split_docs, embedding_function, persist_directory=DB_PATH)
    return db

# 리트리버 설정
def get_retriever(db):
    return db.as_retriever(search_kwargs={"k": 5})

# 질의응답 시스템 구축
def create_qa_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == "__main__":
    # PDF 로드 -> 텍스트 분할 -> 벡터 DB 저장
    print("PDF 파일 로드 중...")
    documents = load_pdfs_from_folder(FOLDER_PATH)
    split_docs = split_text(documents)
    db = store_in_chroma(split_docs)
    
    # 리트리버 및 QA 시스템 구축
    retriever = get_retriever(db)
    qa_chain = create_qa_chain(retriever)
    
    prompt = PromptTemplate.from_template(
    '''You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in Korean.

    #Question:
    {question}
    #Context:
    {context}

    #Answer:'''
    )
    # 질의응답 시스템 실행
    print("질의응답 시스템 준비 완료! 질문을 입력하세요.")
    
    while True:
        query = input("질문: ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        
        answer = qa_chain.run(query)
        print(f"답변: {answer}")