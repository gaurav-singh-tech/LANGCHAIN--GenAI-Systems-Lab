from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-small-2506")

response = model.invoke("Give me a paragraph on Generative AI")

print(response.content)