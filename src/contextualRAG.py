# Small Test:- In the reason query function I have specifically specified response format
#              but not in search_query generating function

from config.config import Settings
from src.chunker import TableAwareChunker
from src.state import ReasoningQueryPlan, SearchQueryPlan, InputState, OutputState, OverallState
from src.extract import DocumentIntelligenceService
from prompts.prompts import SEARCH_QUERY_PROMPT, PLANNER_PROMPT, FEW_SHOT_EXAMPLES, GENERATOR_PROMPT

from dotenv import load_dotenv
from typing import List,Dict,Any,Union
import openai
import cohere
import logging
import chromadb
from langgraph.types import interrupt
from langchain.output_parsers import PydanticOutputParser
from langsmith.wrappers import wrap_openai
from langsmith import traceable

load_dotenv()
class ContextualizedRAG:
    def __init__(self):
        
        settings = Settings()
        self.key = settings.llm.api_key
        self.endpoint = settings.llm.base_url

        self.llm_client = wrap_openai(openai.OpenAI(
            api_key = self.key,
            base_url = self.endpoint
        ))

        self.cohere_client = cohere.Client(
            api_key=settings.cohere.api_key,
        )

        self.async_cohere_client = cohere.AsyncClient(
            api_key=settings.cohere.api_key,
        )

        self.doc_intel_client = DocumentIntelligenceService()

        client = chromadb.HttpClient(
        ssl=True,
        host='api.trychroma.com',
        tenant=settings.chromadb.tenant,
        database=settings.chromadb.database,
        headers={
            'x-chroma-token': settings.chromadb.token
        }
        )

        self.collection = client.get_or_create_collection(name="hackrx-sections")

        self.chunker_class = TableAwareChunker(child_chunk_size = 512)

        self.child_chunks = []
        self.parent_chunks = []

    async def _extract_content(self, pdf_blob_url):
        """
        Uses Microsoft Document Intelligence to retrieve information in markdown format.
        Chunks the retrieved context into parent chunk and it's child chunk
        """
        analysis_result = await self.doc_intel_client.analyze(pdf_blob_url, True)
        markdown_content = analysis_result["analyzeResult"]["content"]

        child_chunks, parent_chunks = await self.chunker_class.process_document(markdown_content) 
        self.child_chunks = child_chunks
        self.parent_chunks = parent_chunks

        return parent_chunks, child_chunks
    
    async def _create_and_store_embeddings(self, processed_docs: List[Dict[str, Any]]):
        """
        Generates embeddings for the summaries using Cohere and stores them in ChromaDB.
        """
        logging.info(f"Generating Vector embeddings for {len(processed_docs)} summaries...")
        content_to_embed = [ele['content'] for ele in processed_docs]

        try:
            resp = await self.async_cohere_client.embed(
                model="embed-english-v3.0",
                input_type="search_document",
                texts=content_to_embed,
                embedding_types=["float"]
            )
        except Exception as e:
            resp = await self.async_cohere_client.embed(
                model="embed-multilingual-v3.0",
                input_type="search_document",
                texts=content_to_embed,
                embedding_types=["float"]
            )

        embeddings = resp.embeddings.float

        self.collection.add(
            ids=[doc["child_id"] for doc in processed_docs],
            embeddings=embeddings,
            documents=[doc["content"] for doc in processed_docs], 
            metadatas=[doc['metadata'] for doc in processed_docs] 
        )

        logging.info("Storage complete.")

    @traceable
    async def data_ingestion(self, state:InputState):
        """
        This function completes the data ingestion into the vector database
        """
        logging.info("DATA ingestion starts")
        pdf_blob_url = state.pdf_blob_url

        try:
            _, processed_docs = await self._extract_content(pdf_blob_url=pdf_blob_url)
            print("\n\nPROCESSED DOCS GENERATED \n\n")
            await self._create_and_store_embeddings(processed_docs)
            return {"pdf_blob_url" : state.pdf_blob_url, "data_loaded" : True, "error" : [False, ""]}

        except Exception as e:
            print(f"[ERROR in DI]: {e}")
            return {"pdf_blob_url" : state.pdf_blob_url, "data_loaded" : False, "error" : [True, str(e)]}

    @traceable
    def get_parse_user_query(self, state:InputState):
        """
        Accepts query from user
        """
        user_query = state.user_query

        if user_query == "":
            user_query = input("Enter your query")

        plan: ReasoningQueryPlan = self._query_parser(user_query)
        fields = ReasoningQueryPlan.model_fields.keys()

        result = {}
        for field in fields:
            result[field] = getattr(plan, field)

        return result        

    @traceable
    def _query_parser(self, user_query:str):
        """
        Parses the question and generates a few reasoning and clarification
        query.
        """
        parser = PydanticOutputParser(pydantic_object=ReasoningQueryPlan)
        planner_prompt = PLANNER_PROMPT
        
        formatted_examples = ""
        for i in range(1,len(FEW_SHOT_EXAMPLES)):
            example = FEW_SHOT_EXAMPLES[i]

            formatted_examples += f"""\n\n
            ==============================
            User Query:
            {example['user_query']}

            Reasoning Queries:
            {example['reasoning_queries']}
            ==============================
            """

        prompt1 = planner_prompt + formatted_examples
        human_message = f"{user_query}\nFormat Instruction:\n{parser.get_format_instructions()}"

        try:
            response = self.llm_client.beta.chat.completions.parse(
                model="gemini-2.5-flash",
                messages=[
                    {'role':'system', 'content':prompt1},
                    {'role':'user', 'content': human_message}
                ],
                response_format=ReasoningQueryPlan
            )
            plan = response.choices[0].message.parsed
            return plan
        
        except Exception as e:
            print("Reasoning questions planning failed")

    def _human_in_loop(self, state):
        plan = SearchQueryPlan(**state)
        questions = plan.reasoning_sub_questions

        answers = []
        for q in questions:
            # Interrupt graph execution and ask for a human response to each question
            answer = interrupt(f"Human review required: {q} (Please provide an answer)")
            answers.append(answer)
        return {"reasoning_responses": answers}

    @traceable
    def _generate_search_queries(self, state):
        """
        Generates search queries which is used for retrieving info from the vectorDB
        """
        parser = PydanticOutputParser(pydantic_object=SearchQueryPlan)
        plan = SearchQueryPlan(**state)

        search_query_prompt = SEARCH_QUERY_PROMPT.format(reason_queries = plan.reasoning_sub_questions, 
                                                         user_response_to_rqs = plan.reasoning_responses
                                                        )
        
        human_message = f"""Generate relevant search queries using 
        reasoning_sub_questions and reaoning_response{plan.user_query}"""
        
        try:
            response = self.llm_client.beta.chat.completions.parse(
                model="gemini-2.5-flash",
                messages=[
                    {'role':'system', 'content':search_query_prompt},
                    {'role':'user', 'content': human_message}
                ],
                response_format=SearchQueryPlan
            )
            new_plan = response.choices[0].message.parsed
            fields = SearchQueryPlan.model_fields.keys()

            result = {}
            
            for field in fields:
                if field != "search_queries":
                    result[field] = getattr(plan, field)
                else:
                    result[field] = new_plan.search_queries

            return result
        
        except Exception as e:
            print("Planning failed")

    def _wait_for_data(self, state: OverallState):
        """
        Checks if the data ingestion has completed successfully before proceeding.
        This node acts as a gate.
        """
        if state.get("error") and state["error"][0]:
            return {"final_response": f"Could not continue due to a data ingestion error: {state['error'][1]}"}
        
        if state.get("data_loaded"):
            return {}
        
        return {"final_response": "Data is not available yet. Please try again later."}
        
    def fallback_node(self, state: OverallState) -> dict:
        """
        Handles the graph when some error has occurred during data ingestion.
        """
        error_message = state.get("error", [True, "Unknown error"])[1]
        return {"final_response": f"Could not continue the process due to the following error:\n{error_message}"}

    @traceable
    def _retrieve_and_grade(self, state):
        """
        Executes the retrieval plan, searches for child chunks,
        and returns the content of their unique parents.
        """
        data = OverallState(**state)
        search_queries = data["search_queries"]

        final_results = []

        query_embeddings = self.cohere_client.embed(
            texts=search_queries,
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )

        query_embeddings = query_embeddings.embeddings.float

        retrieved_results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=12,
            include=["documents","metadatas"],
        )

        try:
            # Flatten the lists of lists returned by ChromaDB
            all_metadatas = [item for sublist in retrieved_results["metadatas"] for item in sublist]
            all_documents = [item for sublist in retrieved_results["documents"] for item in sublist]
            
            docs_for_reranking = []
            for i, meta in enumerate(all_metadatas):
                if meta.get("content_type") == "paragraph":
                    # Assuming 'full_parent_content' is a key in your metadata
                    docs_for_reranking.append(meta.get("full_parent_content", ""))
                else:
                    docs_for_reranking.append(all_documents[i])
             
        except Exception as e:
            print(f"[ERROR in R_and_G]: {e}")
            return {"top_chunks": []}
        
        for query in search_queries:
            reranked_results = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs_for_reranking,
                top_n=4
            )

            for result in reranked_results.results:
                final_results.append(
                    [
                        result.relevance_score, 
                        query,
                        docs_for_reranking[result.index]
                    ]
                )

        # Sort in descending order
        final_results.sort(key=lambda x:-x[0])
        state["top_chunks"] = final_results

        return state

    @traceable
    def _generate_answer(self, state):
        """
        Uses the content retrieved from the vector DB and writes the output
        """

        # providing the top_5 results only
        # Changes needed for the above based on testing
        data = OverallState(**state)
        user_query = data["user_query"]
        no_chunks = 0
        final_results = ""

        if data["top_chunks"] is None:
            return {"final_response": "I could not find any relevant information in the document to answer your question."}


        # currently omitting the threshold factor
        for score, q, content in data["top_chunks"]:
            if no_chunks < 5:
                final_results += content
                final_results += "\n----\n"
                no_chunks += 1
            else:
                break

            # if score > 0.75 and no_chunks < 5:
            #     final_results.append(content)
            #     no_chunks += 1
            # else:
            #     break

        for attempt in range(2):
            generator_prompt = GENERATOR_PROMPT.format(user_query = user_query, final_results = final_results)
            
            try:
                response = self.llm_client.chat.completions.create(
                    model="gemini-2.5-pro", 
                    messages=[{'role':'user', 'content': generator_prompt}],
                    temperature=0.25, 
                )
                
                answer = response.choices[0].message.content

                if answer:
                    return {"final_response":answer}
                else:
                    print(f"[WARN] Attempt {attempt + 1}: LLM returned an empty response.")
            
            except Exception as e:
                print(f"[ERROR]: {e}")

        print("[ERROR] All attempts to generate an answer failed.")        
        return {"final_response":"An error occurred while generating the final answer."}