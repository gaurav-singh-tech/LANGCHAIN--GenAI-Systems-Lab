import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load .env from the project root
load_dotenv()

# (Optional) sanity check
if not os.getenv("HF_TOKEN"):
    raise ValueError("HF_TOKEN not found. Make sure it exists in .env and load_dotenv() is called.")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    # keep params in model_kwargs to avoid warnings / incompatibilities
    model_kwargs={"max_tokens": 256},
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Tell me about yourself in storytelling form")
print(response.content)