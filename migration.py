from langchain import SerpAPIWrapper
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv
import sys
import aiohttp
from typing import Tuple

# from custom_wrappers import CustomSerpAPIWrapper, HiddenPrints, search_with_site

# Load environment variables from .env file
load_dotenv()

from typing import Optional, Dict, Any


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class CustomSerpAPIWrapper(SerpAPIWrapper):
    def get_params(self, query: str, site: Optional[str] = None) -> Dict[str, str]:
        """Get parameters for SerpAPI."""
        if site:
            query = f"{query} site:{site}"
        return super().get_params(query)

    async def arun(self, query: str, site: Optional[str] = None, **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result async."""
        return self._process_response(await self.aresults(query, site))

    def run(self, query: str, site: Optional[str] = None, **kwargs: Any) -> str:
        """Run query through SerpAPI and parse result."""
        return self._process_response(self.results(query, site))

    def results(self, query: str, site: Optional[str] = None) -> dict:
        """Run query through SerpAPI and return the raw result."""
        params = self.get_params(query, site)
        with HiddenPrints():
            search = self.search_engine(params)
            res = search.get_dict()
        return res

    async def aresults(self, query: str, site: Optional[str] = None) -> dict:
        """Use aiohttp to run query through SerpAPI and return the results async."""
        url, params = self.construct_url_and_params(query, site)
        if not self.aiosession:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    res = await response.json()
        else:
            async with self.aiosession.get(url, params=params) as response:
                res = await response.json()
        return res

    def construct_url_and_params(self, query: str, site: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
        params = self.get_params(query, site)
        params["source"] = "python"
        if self.serpapi_api_key:
            params["serp_api_key"] = self.serpapi_api_key
        params["output"] = "json"
        url = "https://serpapi.com/search"
        return url, params
    

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pinecone.Index(index_name)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
namespace = os.getenv("PINECONE_NAMESPACE")
vector_store = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)

# from your_library import SerpAPIWrapper  # Replace 'your_library' with the actual import


# search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
search = CustomSerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))


from langchain.tools import Tool
Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current wait times, processing times for Australian student visas"
        # coroutine= ... <- you can specify an async method if desired as well
    )
tool1 = create_retriever_tool(
    vector_store.as_retriever(),
    "search_latest_immigration",
    """Searches and returns latest data related to queries around Australian immigration in context of international students, 
    useful especially if the question is around Working Hours Cap,
    Pandemic Visa (Subclass 408)
    Cost of Student Visas,
    Minimum Wage,
    Replacement Visa for Temporary Graduate (Subclass 485),
    Intent to Migrate for students,
    New Post-Study Work Visa Category,
    English Language Proficiency,
    Extended Post-Study Work Rights,
    Benefits for Students in Regional Areas"""
)
tool2 = Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for answering queries like visa wait times, processing times "
        # coroutine= ... <- you can specify an async method if desired as well
    )

def search_with_site(query: str, **kwargs: Any) -> str:
    return search.run(query, site="https://visaenvoy.com/processing-times/student-visa/", **kwargs)

tool2 = Tool.from_function(
    func=search_with_site,
    name="Search",
    description="useful for answering queries like visa wait times, processing times "
    # coroutine= ... <- you can specify an async method if desired as well
)

tools = [tool1, tool2]
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature = 0, openai_api_key="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", model="gpt-3.5-turbo-16k")
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)
# result = agent_executor({"input": "hi, im bob"})
# result = agent_executor({"input": "Who am i?"})
result = agent_executor({"input": "What visa do I need to study in Australia"})
result = agent_executor({"input": "What are the current wait times for a student visa in Australia?"})
result = agent_executor({"input": "Are there any benefits to study in a regional area?"})
