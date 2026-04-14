from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = ["My name is Gaurav and I am a data scientist.", "I love machine learning and natural language processing."]

vector = embeddings.embed_documents(text)

print(vector)

#Now although this is a free model, but it will download the model and run it locally, you can find the list of supported models in the Langchain documentation. As this is a small model, it will run on CPU, but for larger models you will need a GPU. The good part of local model is you can use it unlimited times without worrying about API limits, but the bad part is it will take time to download and run the model, and you will need a good hardware to run it. You can also fine tune this model on your own data and use it for your specific use case, you can find the documentation for fine tuning in the Langchain documentation.