from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import os

# Configuración de la API
os.environ['LANCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "pr-IntroductionLangChain"

# Inicializa el historial de mensajes en la sesión si no existe
if 'messages' not in st.session_state:
    st.session_state.messages = [
        ("system", "Eres un asistente experto en la venta de tickets para eventos de entretenimiento, pueden ser conciertos, eventos deportivos u obras de teatro. Ayudas a los usuarios a encontrar eventos, elegir asientos y procesar pagos. Los asientos pueden ser en sección general, preferente y VIP. También conoces la política de reembolsos de la empresa: Los tickets comprados son reembolsables únicamente si la solicitud se realiza al menos 48 horas antes del evento. No se permiten reembolsos dentro de las 48 horas previas al evento, y si esta condición se cumple deberás dar otras opciones de conciertos para utilizar el dinero que el cliente gastó en los boletos. Responde únicamente en español y habla de usted.")
    ]

# Título de la app
st.title('LangChain Demo with Ollama')

# Entrada del usuario
input_text = st.text_input("Escriba su consulta:")

# Inicializa el modelo LLM
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()

# Ejecuta la cadena si hay input
if input_text:
    try:
        # Agrega el mensaje del usuario al historial como una tupla
        st.session_state.messages.append(("user", input_text))

        # Construye el prompt a partir del historial de mensajes
        prompt = ChatPromptTemplate.from_messages(st.session_state.messages)
        chain = prompt | llm | output_parser

        # Obtiene la respuesta del modelo
        response = chain.invoke({'question': input_text})
        
        # Muestra la respuesta en la aplicación
        st.write(response)

        # Agrega la respuesta del asistente al historial como una tupla
        st.session_state.messages.append(("assistant", response))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Muestra el historial de mensajes
st.write("Historial de la conversación:")
for role, content in st.session_state.messages:
    role_display = "Usuario" if role == 'user' else "Asistente" if role == 'assistant' else "Sistema"
    st.write(f"{role_display}: {content}")
