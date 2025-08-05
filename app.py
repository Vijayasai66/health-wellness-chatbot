import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
from datetime import date
import random
import os

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# ✅ Check API key presence
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found in .env file.")
    st.stop()

# Setup LangChain-compatible Claude model via OpenRouter
model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    streaming=True
)

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "profile" not in st.session_state:
    st.session_state.profile = {}

st.set_page_config(page_title="🌟 Health & Wellness Chatbot", layout="wide")
st.title("Health & Wellness Chatbot")
st.header("✨ Your Personal Wellness Assistant", divider="green")

menu = st.sidebar.radio("📋 Menu", ["💬 Chat", "🌞 Daily Tips", "🢁‍♂️ Profile & Goals", "🛠️ Tools"])

# Prompt setup
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful, friendly Health and Wellness assistant. Provide accurate and empathetic responses to user queries."),
    MessagesPlaceholder(variable_name="messages")
])

trimmer = trim_messages(
    strategy="last",
    max_tokens=200,
    token_counter=lambda msgs: sum(len(msg.content) // 4 for msg in msgs if hasattr(msg, "content")),
    include_system=True
)

# LangGraph setup
class MessageState(TypedDict):
    messages: List[BaseMessage]

workflow = StateGraph(state_schema=MessageState)

def call_model(state: MessageState):
    try:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            messages = messages[:-1]
        trimmed = trimmer.invoke(messages)
        prompt_messages = trimmed
        completion = model.invoke(prompt_messages)
        return {"messages": messages + [completion]}
    except Exception as e:
        return {"messages": state["messages"] + [AIMessage(content=f"❌ Error: {e}")]}

workflow.add_node("model", call_model)
workflow.set_entry_point("model")
runnable_graph = workflow.compile(checkpointer=st.session_state.memory)
config = {"configurable": {"thread_id": "health-chat-thread-1"}}

# Chat UI
if menu == "💬 Chat":
    for message in st.session_state.chat_history:
        role = "human" if isinstance(message, HumanMessage) else "ai"
        with st.chat_message(role):
            st.write(message.content)

    prompt = st.chat_input("💬 Ask me anything about your health or wellness!")

    if prompt:
        user_message = HumanMessage(content=prompt)
        st.session_state.chat_history.append(user_message)


        with st.chat_message("human"):
            st.write(prompt)

        full_response = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk, _ in runnable_graph.stream({"messages": st.session_state.chat_history}, config, stream_mode="messages"):
                if isinstance(chunk, AIMessage):
                    full_response += chunk.content
                    placeholder.write(full_response)
                
        st.session_state.chat_history.append(AIMessage(content=full_response))

        followups = [
            "🍏 Would you like some personalized meal plans?",
            "🏋️ Shall I suggest a workout routine based on your goals?",
            "💧 Would you like to set a hydration reminder?",
            "🛌 Shall I share some sleep hygiene tips?",
            "🧘 Would you like to know more about mindfulness practices?"
        ]
        st.info(random.choice(followups))

elif menu == "🌞 Daily Tips":
    tips = {
        "🥗 Nutrition": "Add more fiber to your meals by including fruits, vegetables, and whole grains.",
        "🏃 Fitness": "Stretch for 5 minutes every hour to keep your body flexible.",
        "🧠 Mental Health": "Take 10 deep breaths whenever you feel overwhelmed or stressed.",
        "🛌 Sleep": "Try to maintain a regular bedtime and avoid screens before sleep.",
        "💧 Hydration": "Drink at least 8 glasses of water daily to stay hydrated.",
        "🧘 Mindfulness": "Spend 5 minutes each day practicing mindfulness or meditation.",
        "🪑 Posture": "Check your posture while sitting; keep your back straight and shoulders relaxed."
    }
    st.subheader("🌞 Your Daily Wellness Tips for " + str(date.today()))
    for category, tip in tips.items():
        st.markdown(f"**{category}:** {tip}")

elif menu == "🢁‍♂️ Profile & Goals":
    st.subheader("📝 Set Your Profile")
    name = st.text_input("👤 Enter your name:", value=st.session_state.profile.get("name", ""))
    age = st.number_input("🎂 Enter your age:", min_value=0, max_value=120, step=1, value=st.session_state.profile.get("age", 0))
    gender = st.selectbox("♂️ Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.profile.get("gender", "Male")))
    weight = st.number_input("⚖️ Weight (kg)", min_value=0.0, step=0.1, value=st.session_state.profile.get("weight", 0.0))
    height = st.number_input("📏 Height (cm)", min_value=0, step=1, value=st.session_state.profile.get("height", 0))
    activity = st.selectbox("🏃 Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], index=["Sedentary", "Light", "Moderate", "Active", "Very Active"].index(st.session_state.profile.get("activity", "Sedentary")))
    goal = st.selectbox("🎯 Health Goal", ["Lose weight", "Gain weight", "Maintain weight", "Build muscle", "Improve fitness", "Enhance mental health"], index=["Lose weight", "Gain weight", "Maintain weight", "Build muscle", "Improve fitness", "Enhance mental health"].index(st.session_state.profile.get("goal", "Maintain weight")))

    if st.button("📂 Save Profile"):
        st.session_state.profile = {
            "name": name,
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "activity": activity,
            "goal": goal
        }
        st.success("✅ Profile saved successfully!")

elif menu == "🛠️ Tools":
    st.subheader("🛠️ Mini Health Tools")
    tabs = st.tabs(["📏 BMI Calculator", "🔥 Calorie Tracker", "💧 Water Intake", "🫧 Breathing Timer", "😴 Sleep Hygiene"])

    with tabs[0]:
        h = st.number_input("📏 Enter your height (cm):", min_value=0, step=1)
        w = st.number_input("⚖️ Enter your weight (kg):", min_value=0.0, step=0.1)
        if h and w:
            bmi = w / ((h / 100) ** 2)
            st.metric("📊 Your BMI", round(bmi, 2))

    with tabs[1]:
        age = st.number_input("🎂 Enter your age again:", min_value=0, max_value=120, step=1)
        gender = st.selectbox("⚧️ Select your gender again:", ["Male", "Female", "Other"])
        activity = st.selectbox("🏃 Activity Level again", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
        bmr = 10 * w + 6.25 * h - 5 * age + (5 if gender == "Male" else -161)
        factor = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}[activity]
        st.metric("🔥 Daily Caloric Needs", int(bmr * factor))

    with tabs[2]:
        st.write("💧 Water goal: 35ml per kg of body weight")
        water = w * 0.035
        st.metric("💧 Water Intake Goal (liters/day)", round(water, 2))

    with tabs[3]:
        st.write("🫧 Try this guided 4-7-8 breathing exercise:")
        if st.button("🕒 Start Breathing Timer"):
            for i in range(3):
                st.markdown("### 🫁 Inhale for 4 seconds")
                time.sleep(4)
                st.markdown("### ✋ Hold for 7 seconds")
                time.sleep(7)
                st.markdown("### 🫁 Exhale for 8 seconds")
                time.sleep(8)
            st.success("✅ Done! You can repeat this exercise as needed.")

    with tabs[4]:
        st.checkbox("🛌 Go to bed at the same time every night")
        st.checkbox("☕ Avoid caffeine and heavy meals before bed")
        st.checkbox("🌙 Keep your bedroom dark, quiet, and cool")
        st.checkbox("📵 Avoid screens before bedtime")
        st.checkbox("📚 Relax before bed with a book or music")

print("✅ API key loaded from .env:", api_key)


