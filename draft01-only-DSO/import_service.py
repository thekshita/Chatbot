from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from llama_index import download_loader, LLMPredictor, ServiceContext

from config import CHAT_MODEL

from llama_index.vector_stores import PineconeVectorStore
from llama_index import StorageContext, VectorStoreIndex
from config import PINECONE_INDEX, PINECONE_ENVIRONMENT

load_dotenv()


def get_llm_predictor():
    return LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=512, model_name=CHAT_MODEL))


def import_web_scrape_data(urls: list):
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=urls)
    #documents = loader.load_data(urls=['https://google.com'])
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        environment=PINECONE_ENVIRONMENT
    )
    stc = StorageContext.from_defaults(vector_store=vector_store)
    llm_predictor_chatgpt = get_llm_predictor()
    svc =  ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)
    
    index = VectorStoreIndex.from_documents(documents,
                                            storage_context=stc,
                                            service_context=svc)
    return index
