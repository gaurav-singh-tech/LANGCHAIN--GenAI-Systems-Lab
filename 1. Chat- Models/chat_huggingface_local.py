from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm= HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 25, "temperature": 0.8}
)

chat_model=ChatHuggingFace(llm=llm)
response=chat_model.invoke("What is data science")

print(response.content)

#this code will download the model and run it locally, you can find the list of supported models in the Langchain documentation.
#as tiny lama is a small model, it will run on CPU, but for larger models you will need a GPU.
#It will download 2gb of tiny lama model in  my local PC and run it locally, you can use it unlimited, I have not downloaded it
#Good part of local model is you can use it unlimited times without worrying about API limits, but the bad part is it will take time to download and run the model, and you will need a good hardware to run it.
#You can also fine tune this model on your own data and use it for your specific use case, you can find the documentation for fine tuning in the Langchain documentation.


