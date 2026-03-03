import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        
        if os.path.exists("microwave_faiss_index"):
            print("✅ Found existing FAISS index. Loading...")
            vectorstore = FAISS.load_local(
                folder_path="microwave_faiss_index",
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded successfully!")
        else:
            print("📝 No existing index found. Creating new one...")
            vectorstore = self._create_new_index()
        
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        
        try:
            # 1. Create Text loader
            loader = TextLoader(
                file_path="microwave_manual.txt",
                encoding="utf-8"
            )
            
            # 2. Load documents
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} document(s)")
            
            # 3. Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", "."]
            )
            
            # 4. Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            print(f"✅ Split into {len(chunks)} chunks")
            
            # 5. Create vectorstore from documents
            print("🔄 Creating embeddings and FAISS index...")
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # 6. Save indexed data locally
            vectorstore.save_local("microwave_faiss_index")
            print("✅ FAISS index created and saved successfully!")
            
            # 7. Return vectorstore
            return vectorstore
            
        except FileNotFoundError:
            print("❌ Error: microwave_manual.txt not found. Please ensure the file exists.")
            raise
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            raise

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        # Make similarity search with relevance scores
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )

        context_parts = []
        # Iterate through results
        for i, (doc, relevance_score) in enumerate(results, 1):
            context_parts.append(doc.page_content)
            print(f"\n📄 Chunk {i} (Score: {relevance_score:.4f}):")
            print(f"{doc.page_content}")

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        try:
            # 1. Create messages array
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=augmented_prompt)
            ]
            
            # 2. Invoke llm client
            response = self.llm_client.invoke(messages)
            
            # 3. Print response content
            print(f"\n📝 Answer:\n{response.content}")
            
            # 4. Return response content
            return response.content
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            raise


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")
    print("Type your question or 'quit'/'exit' to exit.\n")

    while True:
        user_question = input("\n> ").strip()
        
        # Check for exit commands
        if user_question.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
        
        try:
            # Step 1: Retrieval of context
            context = rag.retrieve_context(user_question)
            
            # Step 2: Augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            
            # Step 3: Generation
            answer = rag.generate_answer(augmented_prompt)
            
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")



main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY)
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version=""
        )
    )
)