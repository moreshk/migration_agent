from flask import Flask, request, jsonify, render_template
from langchain import SerpAPIWrapper
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from typing import Optional, Dict, Any, Tuple
import sys
import aiohttp

# Initialize Flask app
app = Flask(__name__)

# Your existing setup code
load_dotenv()

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
# Tool.from_function(
#         func=search.run,
#         name="Search",
#         description="useful for when you need to answer questions about current wait times, processing times for Australian student visas"
#         # coroutine= ... <- you can specify an async method if desired as well
#     )
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
# tool2 = Tool.from_function(
#         func=search.run,
#         name="Search",
#         description="useful for answering queries like visa wait times, processing times "
#         # coroutine= ... <- you can specify an async method if desired as well
#     )

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
llm = ChatOpenAI(temperature = 0, openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-16k")
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

# New code
# This is needed for both the memory and the prompt
memory_key = "history"

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
system_message = SystemMessage(
        content=(
            "You are a helpful migration assistant whose objective is to assist Indian students looking to pursue higher education in Australia "
            "Feel free to use any tools available to look up relevant information, only if neccessary."
            "You will only respond to queries that are directly related to immigration and politely decline to respond to queries that are not relevant to immigration."
        )
)
prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                   return_intermediate_steps=True)
# End of new code

# Initialize your chatbot agent
# agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests from the frontend."""
    user_input = request.json['user_input']
    
    print(f"Received user input: {user_input}")  # Debugging line
    
    # Use your chatbot logic to generate a response
    result = agent_executor({"input": user_input})
    
    print(f"Generated result: {result}")  # Debugging line

    response = result.get("output", "Sorry, I couldn't understand that.")
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
