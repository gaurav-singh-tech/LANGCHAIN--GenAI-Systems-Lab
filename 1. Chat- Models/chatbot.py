from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#We are importing langchain core messages to create a list of messages that we will pass to the model, we will create a conversation between the user and the model, we will use the role of the message to differentiate between the user and the model, we will use the content of the message to store the actual message.
#AI message is the response from the model, human message is the prompt from the user, system message is the instruction for the model, you can use it to set the context for the model, for example you can use it to tell the model that it is a chatbot and it should respond in a conversational way.

model = ChatMistralAI(model="mistral-small-2506", temperature=0.6)

print("Choose your AI mode")
print("1. Funny AI mode")
print("2. Very Sad and Irritating AI mode")

choice = int(input("Enter your choice (1 or 2): "))

if choice == 1:
    mode="You are a funny AI agent"
elif choice == 2:
    mode="You are a very sad and irritating AI agent"

messages=[
    SystemMessage(content=mode) # We are setting the context for the model, we are telling the model that it is a funny AI agent, so it should respond in a funny way.    
    #We are doing it here because we want to set the context for the entire conversation, if we set it in the loop, it will be added to the messages list every time and it will affect the response of the model, so we are setting it outside the loop.
]
print("__________welcome type 0 to exit the application--------")
while True:
    prompt= input("You: ")
    messages.append(HumanMessage(content=prompt)) # We are adding the prompt from the user to the messages list, we are using the role of the message to differentiate between the user and the model, we are using the content of the message to store the actual message.
    if prompt == "0":
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content)) # We are adding the response from the model to the messages list, we are using the role of the message to differentiate between the user and the model, we are using the content of the message to store the actual message.
    print("Bot:", response.content)
    
print(messages) # We are printing the messages list to see the conversation between the user and the model, we can see the role of each message and the content of each message, we can also see the system message that we set at the beginning of the conversation.