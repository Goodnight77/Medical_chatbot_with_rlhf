import streamlit as st
from langchain import memory as lc_memory
#from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from langsmith import Client
from streamlit_feedback import streamlit_feedback
from utils1 import get_expression_chain, retriever, get_embeddings, create_qdrant_collection
from langchain_core.tracers.context import collect_runs
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os


load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_API_KEY =  os.getenv('HF_API_KEY')
QDRANT_API_KEY =  os.getenv('QDRANT_API_KEY')
QDRANT_URL =  os.getenv('QDRANT_URL')
COHERE_API_KEY =  os.getenv('COHERE_API_KEY')

LANGSMITH_TRACING='true'
LANGSMITH_ENDPOINT= os.getenv('LANGSMITH_ENDPOINT')
LANGSMITH_API_KEY= os.getenv('LANGSMITH_API_KEY')
LANGSMITH_PROJECT= os.getenv('LANGSMITH_PROJECT')
collection_name = os.getenv('collection_name')

#profile = st.session_state.profile
client = Client()
qdrant_client = QdrantClient(QDRANT_URL ,api_key=QDRANT_API_KEY)
st.set_page_config(page_title = "MEDICAL CHATBOT")
st.subheader(f"Hello! How can I assist you today!")

memory = lc_memory.ConversationBufferMemory(
    chat_memory=lc_memory.StreamlitChatMessageHistory(key="langchain_messages"),

    #key="langchain_messages",
    return_messages=True,
    memory_key="chat_history",
)
st.sidebar.markdown("## Feedback Scale")
feedback_option = (
    "thumbs" if st.sidebar.toggle(label="`Faces` ⇄ `Thumbs`", value=False) else "faces"
)

with st.sidebar:
    model_name = st.selectbox("**Model**", options=["llama-3.1-70b-versatile","gemma2-9b-it","gemma-7b-it","llama-3.2-3b-preview", "llama3-70b-8192", "mixtral-8x7b-32768"])
    temp = st.slider("**Temperature**", min_value=0.0, max_value=1.0, step=0.001)
    n_docs = st.number_input("**Number of retrieved documents**", min_value=0, max_value=10, value=5, step=1)
 
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    memory.clear()

retriever = retriever(n_docs=n_docs)
# Create Chain.
chain = get_expression_chain(retriever,model_name,temp)

for msg in st.session_state.langchain_messages:
    avatar = "🐒" if msg.type == "ai" else None
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)


prompt = st.chat_input(placeholder="Describe your symptoms or medical questions ?")

if prompt :
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant", avatar="🐒"):
        message_placeholder = st.empty()
        full_response = ""
        # Define the basic input structure for the chains
        input_dict = {"input": prompt.lower()}
        #used_docs = retriever.get_relevant_documents(prompt.lower())
        used_docs = retriever.invoke(prompt.lower())

        with collect_runs() as cb:
            for chunk in chain.stream(input_dict, config={"tags": ["MEDICAL CHATBOT"]}):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌") 
                memory.save_context(input_dict, {"output": full_response}) #add_user_message

            st.session_state.run_id = cb.traced_runs[0].id
        message_placeholder.markdown(full_response)
        if used_docs :
            docs_content = "\n\n".join(
                                    [
                                        f"Doc {i+1}:\n"
                                        #f"Source: {doc.metadata['source']}\n"
                                        #f"Title: {doc.metadata['title']}\n"
                                        f"Content: {doc.page_content}\n"
                                        for i, doc in enumerate(used_docs)
                                    ]
                                )
            with st.sidebar:
                st.download_button(
                label="Consulted Documents",
                data=docs_content,
                file_name="Consulted_documents.txt",
                mime="text/plain",
            )

        with st.spinner("Just a sec! Dont enter prompts while loading pelase!"):
            run_id = st.session_state.run_id
            question_embedding = get_embeddings(prompt)
            answer_embedding = get_embeddings(full_response)
            # Add question and answer to Qdrant
            qdrant_client.upload_collection(            
                collection_name = collection_name,
                payload=[
                    {"text": prompt, "type": "question", "question_ID": run_id},
                    {"text": full_response, "type": "answer", "question_ID": run_id, "used_docs":used_docs}
                ],
                vectors=[
                    question_embedding,
                    answer_embedding,
                ],
                parallel=4,
                max_retries=3,
                )

        

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            with st.spinner("Just a sec! Dont enter prompts while loading pelase!"):
                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=feedback.get("text"),
                    #source_info={"profile":profile}
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
        else:
            st.warning("Invalid feedback score.")

        with st.spinner("Just a sec! Dont enter prompts while loading pelase!"):
            if feedback.get("text"):
                comment = feedback.get("text")
                feedback_embedding = get_embeddings(comment)
            else:
                comment = "no comment"
                feedback_embedding = get_embeddings(comment)

            
            qdrant_client.upload_collection(            
                collection_name=collection_name,
                payload=[
                    {"text": comment,
                        "Score:":score, 
                        "type": "feedback", 
                        "question_ID": run_id, 
                        #"User_profile":profile
                        }
                ],
                vectors=[
                    feedback_embedding
                ],
                parallel=4,
                max_retries=3,
                )


            
