from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

from langchain_groq import ChatGroq

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.8, max_tokens=25) #FOr models supported by Groq, you have to specifically specify the model name in the format "<model_name>", you can find the list of supported models in the Langchain documentation.

response = model.invoke("Give a poem on cricket")

print(response.content)