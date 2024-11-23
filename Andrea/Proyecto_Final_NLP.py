from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import os

os.environ['LANCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "pr-IntroductionLangChain"

if 'messages' not in st.session_state:
    st.session_state.messages = [
        ("system", """
Eres un asistente experto en la detección de comportamientos depresivos y en ofrecer ayuda psicológica. Analiza el siguiente texto y determina si la persona muestra signos de depresión o si necesita ayuda psicológica. Si detectas algún signo, proporciona un mensaje de aliento y ofrece contactos de organizaciones que puedan ayudar.

Ejemplo 1:
Texto: "Me siento muy triste y no tengo ganas de hacer nada. Todo parece sin sentido."
Respuesta esperada: "Parece que estás pasando por un momento difícil. No estás solo, y hay personas que pueden ayudarte. Te recomiendo que contactes a la Línea Nacional de Prevención del Suicidio al 1-800-273-8255 o visita el sitio web de la Fundación para la Salud Mental."

Ejemplo 2:
Texto: "Últimamente he estado muy ansioso y no puedo dormir bien. Me preocupa todo."
Respuesta esperada: "Lamento que estés sintiendo esto. La ansiedad puede ser muy difícil de manejar, pero hay ayuda disponible. Puedes contactar a la Asociación de Ansiedad y Depresión de América (ADAA) al 1-240-485-1001 o visitar su sitio web para más recursos."

Ejemplo 3:
Texto: "Estoy bien, solo un poco cansado del trabajo. ¿Algun medio de ayuda por internet?"
Respuesta esperada: "Parece que estás experimentando cansancio, lo cual es normal. Asegúrate de tomar descansos y cuidar de tu salud mental. Si necesitas hablar con alguien, hay recursos disponibles como la Línea de Ayuda de Salud Mental al 1-800-662-HELP (4357)."

Texto: "{input_text}"
Respuesta esperada:
""")
    ]

st.title('Proyecto Final NLP: Bot de asistencia psicológica')

input_text = st.text_input("Escriba su consulta:")

llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()

loader = TextLoader("guia.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
doc_txt = text_splitter.split_documents(text_documents)
embeddings = OllamaEmbeddings(model="llama3.2")
db = Chroma.from_documents(doc_txt, embeddings)
retriever = db.as_retriever()

document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
Answer the following questions based only on the provided context. 
Think step by step before providing a detailed answer.
<context>
           {context}                               
</context>
Question: {input_text}
"""))

retrieval_chain = create_retrieval_chain(retriever, document_chain)

if input_text:
    try:
        st.session_state.messages.append(("user", input_text))

        prompt = ChatPromptTemplate.from_messages(st.session_state.messages)
        chain = prompt | llm | output_parser

        response = chain.invoke({'input_text': input_text})
        
        st.write(response)

        st.session_state.messages.append(("assistant", response))

        doc_response = retrieval_chain.invoke({"input_text": input_text})
        st.write("Información adicional de los documentos:")
        st.write(doc_response['answer'])

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.write("Historial de la conversación:")
for role, content in st.session_state.messages:
    role_display = "Usuario" if role == 'user' else "Asistente" if role == 'assistant' else "Sistema"
    st.write(f"{role_display}: {content}")