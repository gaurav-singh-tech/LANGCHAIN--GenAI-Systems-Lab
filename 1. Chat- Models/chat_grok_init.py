from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

from langchain.chat_models import init_chat_model

model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct") #Here we are using the Groq LLM, you can change it to any other LLM supported by Langchain, 
# for groq models, you have to specifically specify the model name in the format "groq:<model_name>", you can find the list of supported models in the Langchain documentation.

response = model.invoke("Give me a paragraph on Machine Learning")

print(response.content)