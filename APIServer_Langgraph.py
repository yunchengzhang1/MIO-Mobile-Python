from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from typing_extensions import TypedDict
from juno_prompt import get_juno_prompt, RAILGUARD_INSTRUCTIONS, UTILITY_INSTRUCTIONS
from utils.timer import Timer 
from time import perf_counter
import os
import time
import uuid
import json
from collections import deque
from typing import List

# ngrok config add-authtoken 2x1OkTFW1HQUZqasrtzyHsy7y2Q_BUfUyB6qqUWRe5xBszHQ
# uvicorn APIServer_Langgraph:app --host 0.0.0.0 --port 8000
# in second terminal run: ngrok http 8000

# local same computer testing:
# uvicorn APIServer_Langgraph:app --host 127.0.0.1 --port 8000
# http://localhost:8000

# === Load Environment ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# === Initialize Pinecone ===
pc = Pinecone(api_key=pinecone_key)
index_name = "juno-memory-test"
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# === Embeddings & VectorStore ===
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding_model,
    pinecone_api_key=pinecone_key
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === LLM ===
llm_smart = ChatOpenAI(openai_api_key=openai_key , model="gpt-4o")
llm_light = ChatOpenAI(openai_api_key=openai_key ,model="gpt-3.5-turbo-1106")

# === Define State Schema ===
class GraphState(TypedDict):
    input: str
    context: str
    response: str
    is_repeat: bool
    is_emotional: bool
    topic_shifted: bool
    chat_history: list

recent_inputs = deque(maxlen=5)

# === LangGraph Nodes ===
def analyze_input(state: GraphState) -> GraphState:
    normalized = state["input"].strip().lower()
    is_repeat = normalized in recent_inputs
    recent_inputs.append(normalized)

    print("-- Analyze Input Debug --")
    print("INPUT:", state["input"])
    print("Normalized:", normalized)
    print("Chat history sample:", state.get("chat_history", [])[-5:])

    # emotion_check = llm.invoke(f"""Is the following message emotionally significant or reflective?\nMessage: {state['input']}\nRespond 'yes' or 'no'.""")
    # is_emotional = 'yes' in emotion_check.content.lower()

    chat_history_items = state.get("chat_history", [])[-5:]
    if not chat_history_items:
        chat_snippet = "(No prior messages)"
    else:
        chat_snippet = "".join([
            f"{str(m.get('role'))}: {str(m.get('content'))}" for m in chat_history_items
        ])
    topic_check = f"""
    Given the message below and the recent chat history, would you say this message introduces a new topic that contains new information or a change in direction?

    Only respond "yes" if the topic has shifted and the previous topic contained meaningful or specific knowledge that might be worth remembering. 
    Ignore greetings, casual transitions, or filler.

    Message: "{state['input']}"
    History (most recent first):
    {chat_snippet}

    Answer with only 'yes' or 'no'.
    """
    with Timer("Topic Check LLM Flow"):
        topic_check_result = llm_light.invoke(topic_check)
    topic_shifted = 'yes' in topic_check_result.content.lower()
    print("IS_REPEAT:", is_repeat)
    # print("IS_EMOTIONAL:", is_emotional)
    print("TOPIC_SHIFTED:", topic_shifted)
    print("--------------------------")
    return {**state, "is_repeat": is_repeat, "topic_shifted": topic_shifted}

def retrieve_context(state: GraphState) -> GraphState:
    compression_prompt = f"""
    Extract the core topic or concept from the following user message so it can be used for searching memory.
    Do NOT rewrite the sentence, just identify the most relevant keywords or topics. 
    If the message is vague, include the most probable subject.
    Message: \"{state['input']}\"
    Result:"""
    with Timer("Query Refine LLM Flow"):
        refined_query = llm_light.invoke(compression_prompt).content.strip().lower()
    start = perf_counter()
    docs = retriever.invoke(refined_query)
    print(f"Embedding took {perf_counter() - start:.2f} sec")
    context_text = "\n".join([doc.page_content for doc in docs]) if docs else ""
    return {**state, "context": context_text}

def generate_response(state: GraphState) -> GraphState:
    chat_history = [
        HumanMessage(content=m['content']) if m['role'] == 'human' else AIMessage(content=m['content'])
        for m in state.get("chat_history", [])
    ]
    chat_history.append(HumanMessage(content=state["input"]))

    repeat_flag = str(state.get('is_repeat', False)).lower()
    print(repeat_flag)
    repeat_instruction = """If REPEAT_FLAG is true, the user repeated a question they already asked.\nAcknowledge it with a bit of dry humor or curiosity.\n\nIf REPEAT_FLAG is false, ignore this and continue the conversation normally."""

    prompt = get_juno_prompt(mode="casual")
    formatted_prompt = prompt.invoke({
        "chat_history": chat_history,
        "context": state.get("context", ""),
        "repeat_flag": repeat_flag,
        "repeat_instruction": repeat_instruction,
        "railguard": RAILGUARD_INSTRUCTIONS,
        "utility": UTILITY_INSTRUCTIONS
    })


    # print("\n==================== PROMPT TO LLM ====================")
    # for m in formatted_prompt.to_messages():
    #     print(f"{m.type.upper()}: {m.content}")
    # print("======================================================\n")
    with Timer("Response LLM Flow"):
        response = llm_smart.invoke(formatted_prompt)
    chat_history.append(AIMessage(content=response.content))

    return {
        **state,
        "response": response.content,
        "chat_history": [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content} for m in chat_history
        ]
    }

def write_memory(state: GraphState) -> GraphState:
    return state

# === LangGraph Setup ===
graph = StateGraph(state_schema=GraphState)
graph.add_node("analyze_input", analyze_input)
graph.add_node("retrieve", retrieve_context)
graph.add_node("respond", generate_response)
graph.add_node("write_memory", write_memory)

graph.set_entry_point("analyze_input")
graph.add_edge("analyze_input", "retrieve")
graph.add_edge("retrieve", "respond")
graph.add_edge("respond", "write_memory")
graph.add_edge("write_memory", END)

juno_graph = graph.compile()

# === FastAPI App ===
app = FastAPI()

class Message(BaseModel):
    player_input: str
    chat_history: list = []

@app.post("/preload")
async def preload():
    try:
        check = index.query(vector=[0.0]*1536, top_k=1, include_metadata=True, filter={"source": {"$eq": "lore"}})
        matches = check.get("matches", [])
        if matches:
            print(f"[Preload Check] Top match score: {matches[0].get('score', 0.0):.4f}")
            if matches[0].get("score", 0.0) > 0.95:
                return {"status": "Lore already exists, skipping preload."}

        with open("lore.json", "r", encoding="utf-8") as f:
            lore_items = json.load(f)
        for entry in lore_items:
            text = entry["text"]
            tags = entry.get("tags", [])
            embedding_input = f"[{'|'.join(tags)}] {text}"
            metadata = {
                "tags": tags,
                "source": "lore",
                "text": embedding_input,
                "timestamp": time.time()
            }
            embedding = embedding_model.embed_query(embedding_input)

            # Check for duplicates before upserting
            similar = index.query(vector=embedding, top_k=1, include_metadata=True, filter={"source": {"$eq": "lore"}})
            match = similar.get("matches", [])[0] if similar.get("matches") else None
            score = match.get("score", 0.0) if match else 0.0
            print(f"[Preload Entry] Score for potential duplicate: {score:.4f}")
            if match and score > 0.90:
                print("[Preload Entry] Skipped due to high similarity.")
                continue

            index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": embedding, "metadata": metadata}])

        return {"status": f"Preloaded {len(lore_items)} items from lore.json."}
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
async def chat(msg: Message):
    if not msg.player_input or msg.player_input.strip() == "":
        return {
            "reply": "[Error: Empty input received. Please type something to talk to Juno.]",
            "chat_history": msg.chat_history
        }
    try:
        with Timer("LangGraph LLM Flow"):
            result = juno_graph.invoke({"input": msg.player_input, "chat_history": msg.chat_history})
        return {
            "reply": f"{result['response'].strip()}",
            "chat_history": result["chat_history"]
        }
    except Exception as e:
        return {"reply": f"[Server error: {str(e)}]", "chat_history": []}

class TopicRequest(BaseModel):
    topic: str

@app.post("/summarize")
async def summarize(request: TopicRequest):
    try:
        embedding = embedding_model.embed_query(request.topic)
        results = index.query(vector=embedding, top_k=25, include_metadata=True)
        memory_texts = [m["metadata"].get("text") for m in results.get("matches", []) if "metadata" in m]
        summary_input = "\n".join(memory_texts) + f"\n\nSummarize all useful knowledge about: {request.topic}"
        response = llm_smart.invoke(summary_input)
        return {"summary": response.content.strip()}
    except Exception as e:
        return {"error": str(e)}
    


class MemoryMessage(BaseModel):
    role: str
    content: str

class ConsolidateRequest(BaseModel):
    memory: List[MemoryMessage]

@app.post("/consolidate_memory")
async def consolidate_memory(payload: ConsolidateRequest):
    try:
        memory = payload.memory
        if not memory:
            return {"error": "No memory provided."}

        dialogue = "\n".join([f"{m.role}: {m.content}" for m in memory])

        summary_prompt = f"""Summarize the following chat in a way that captures emotionally or narratively important facts.
        Only include things Ignis would want to remember, not filler.
        Also classify this memory using one or more of the following tags if applicable:
        [player_fact, player_preference, world_detail, AI_emotion, AI_preference, event, philosophy, relationship]
        ---
        {dialogue}
        ---
        Return your result in this format:
        Summary: <summary here>
        Tags: [comma-separated tags]
        """
        summary_response = llm_smart.invoke(summary_prompt)
        print(summary_response)
        response_lines = summary_response.content.strip().split("\n")
        summary_text = ""
        summary_tags = []
        for line in response_lines:
            if line.lower().startswith("summary:"):
                summary_text = line[len("summary:"):].strip()
            elif line.lower().startswith("tags:"):
                tag_string = line[len("tags:"):].strip()
                summary_tags = [t.strip() for t in tag_string.strip("[]").split(",") if t.strip()]

        embedding_input = f"[{'|'.join(summary_tags)}] {summary_text}"
        embedding = embedding_model.embed_query(embedding_input)
        similar = index.query(vector=embedding, top_k=3, include_metadata=True)
        for match in similar.get("matches", []):
            existing_text = match.get("metadata", {}).get("text", "")
            score = match.get("score", 0.0)
            if existing_text and score > 0.95:
                return {
                    "status": "Summary already exists. No update made.",
                    "summary": summary_text
                }

        metadata = {
            "source": "consolidated",
            "timestamp": time.time(),
            "text": embedding_input,
            "tags": summary_tags
        }
        index.upsert(vectors=[{
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": metadata
        }])

        return {
            "status": "Memory successfully consolidated.",
            "summary": summary_text
        }
    except Exception as e:
        return {"error": str(e)}
