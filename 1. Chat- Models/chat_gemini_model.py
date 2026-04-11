from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response = model.invoke("Give a poem on cricket")

print(response.content)