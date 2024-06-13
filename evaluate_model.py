import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import io
from dataset import dataset  # Import the dataset

# Load environment variables
load_dotenv()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to evaluate the model
def evaluate_model(predictions, true_answers):
    y_pred = [pred['answers'][0].answer if pred['answers'] else "" for pred in predictions]
    y_true = true_answers
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1, precision, recall

# Function to run the evaluation
def run_evaluation():
    # Load a sample PDF for evaluation
    pdf_path = "D:\RAG Chatbot\policy-booklet-0923.pdf"
    with open(pdf_path, "rb") as f:
        pdf_docs = [f]
        raw_text = get_pdf_text(pdf_docs)
    
    # Get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # Create vector store
    vectorstore = get_vectorstore(text_chunks)

    # Create conversation chain
    conversation_chain = get_conversation_chain(vectorstore)

    # Evaluate the model using the dataset
    questions = [pair['question'] for pair in dataset]
    true_answers = [pair['answer'] for pair in dataset]

    predictions = [conversation_chain({'question': q}) for q in questions]
    accuracy, f1, precision, recall = evaluate_model(predictions, true_answers)
    print(f"Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

if __name__ == "__main__":
    run_evaluation()
