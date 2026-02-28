from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Project Setup
load_dotenv()
llm_openai = ChatOpenAI(model="gpt-4.1-nano")

# Context Break Demonstration (Naive Invocation)

resp1 = llm_openai.invoke("We are building an AI system for processing medical insurance claims.")
resp2 = llm_openai.invoke("What are the main risks in this system?")

print("\n----------------------------------------------\n")
print("Response 1:", resp1.content)
print("Response 2:", resp2.content)

# Why the second question may fail or behave inconsistently without conversation history ?

# The second question may fail or behave inconsistently because the model does not have access to the context of the first question.
# Without the conversation history, the model cannot understand that the second question is related to the first one,
# and it may not provide a relevant or accurate response. The model treats each invocation as an independent request,
# so it lacks the necessary information to connect the two questions and provide a coherent answer.

messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

resp3 = llm_openai.invoke(messages)
print("\n----------------------------------------------\n")
print("Response 3:", resp3.content)

# 4 Reflection Block (Mandatory)
"""
Reflection:

1. Why did string-based invocation fail?
2. Why does message-based invocation work?
3. What would break in a production AI system if we ignore message history?
"""
"""
1. String-based invocation failed because it does not provide the model with the necessary context to understand the relationship between the questions. Each invocation is treated as an independent request, so the model cannot connect the second question to the first one, leading to irrelevant or inaccurate responses.
2. Message-based invocation works because it provides the model with a structured conversation history, allowing it to understand the context and relationships between the messages. The model can use the information from previous messages to generate more relevant and coherent responses.
3. If we ignore message history in a production AI system, it would lead to a lack of context for the model, resulting in inconsistent and irrelevant responses. The system would struggle to maintain a coherent conversation, and users would likely experience frustration due to the model's inability to understand the flow of the conversation or recall previous interactions. This could significantly degrade the user experience and reduce the effectiveness of the AI system.
"""
