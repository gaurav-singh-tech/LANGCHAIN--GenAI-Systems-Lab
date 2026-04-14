from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=64)

vector = embeddings.embed_query("What is data science?")

print(vector)

#This model and code will generate a 64 dimensional vector for the query "What is data science?", you can use this vector for various applications like semantic search, clustering, etc. You can also use this model to generate embeddings for documents and then use those embeddings for various applications. You can find the documentation for this model in the Langchain documentation.
#Now this is a paid model, you will need to have an OpenAI account and set up billing to use this model, you can find the documentation for setting up billing in the OpenAI documentation. You can also use other embedding models like "text-embedding-3-small", "text-embedding-3-medium", etc. which are cheaper but have lower dimensions. You can find the list of available models in the Langchain documentation.