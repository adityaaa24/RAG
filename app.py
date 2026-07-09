# Run this cell to install the necessary packages




# Import the required packages
from langchain_core.prompts import ChatPromptTemplate
# First, understand Prompt Templates: It is used for formatting user input into a suitable prompt for an LLM.
# ChatPromptTemplate: for chat-based prompts with multiple messages.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# ChatOpenAI class provides more chat-related methods, such as completion_with_retry,
#  get_num_tokens_from_messages to make it more user-friendly when build chatbot related applications.
# With OpenAI, the input and output are strings, while with ChatOpenAI, the input is a sequence of messages
# and the output is a message.
# They use different API endpoints and the endpoint of OpenAI has received its final
# models like Gemini offer ChatGemini, and OpenAI offers ChatOpenAI.
# These modules are specifically built for chat applications
#  and are optimized for maintaining context and handling extended interactions.

# OpenAI models use advanced algorithms and big data to achieve a much deeper
# and more nuanced representation of the data.
# The model not only analyzes individual words, but also looks at the context in which those words are used
# OpenAI embeddings are based on sophisticated machine learning models that can learn from a huge amount of data.
# This means that they can recognize subtle patterns and relationships in the data
# that go far beyond what could be achieved by simple scaling and dimensioning

from langchain_community.document_loaders import UnstructuredHTMLLoader
# Using UnstructuredHTMLLoader: Extracts plain text content from HTML files.

from langchain_core.runnables import RunnablePassthrough

# First, understand Runnable:It is a unit of work that can be invoked, batched, streamed, transformed and composed.
# In LangChain, a Runnable is a fundamental building block that represents a single task or operation,
# and it’s the core of the Lang Chain Expression Language (LCEL).
# LangChain provides a flexible framework for building LLM-powered applications.
# One of its core features is the concept of Runnables,
#  which are modular units of computation that can be composed together to build chains.
# RunnablePassthrough is the most basic Runnable.
# It simply returns the input as the output. This is useful for debugging, testing, or placeholder purposes.

# If you are using a RunnableSequence, the input flows step-by-step, where the output of one component becomes
# the input to the next. For example, if we start with a country input,
# the first prompt generates some text (e.g., a summary),
# and that output is then passed into the next prompt (e.g., to create a joke or quiz).

# However, there are scenarios where you may want to preserve the original input or
# inspect both the original input and intermediate output. This is where RunnablePassthrough becomes useful.

from langchain_text_splitters import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter
# Splitting text by recursively look at characters.

# Recursively tries to split by different characters to find one that works.

# Chunking involves dividing the document into smaller, more manageable sections that
# fit comfortably within the context window of the large language model.

# Langchain provides users with a range of chunking techniques to choose from. However,
# among these options, the RecursiveCharacterTextSplitter emerges as the favored and strongly recommended method.

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size.
#  It does this by using a set of characters.
#  The default characters provided to it are ["\n\n", "\n", " ", ""].

from langchain_chroma import Chroma

# Chroma is an open source database,
# lets developers build applications including ANN search, image retrieval, RAG,
# and ecommerce recommenders. It’s known as being a lightweight vector database
# that developers can run on a laptop for rapid prototyping, as well as in public or private cloud services.
# Chroma employs the Apache Arrow data format for fast data access.

# Load the HTML as a LangChain document loader
loader = UnstructuredHTMLLoader(file_path="/content/sample_data/How to use the various modes of the washing machine | Samsung LEVANT.html")
machine_docs = loader.load()

import os

os.environ["OPENAI_API_KEY"] = "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"



# Load the models required to complete the exercise
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Temperature in deep learning is a parameter usually used to adjust the probability distribution
# of the predicted outputs. It is also known as softmax temperature or softmax scaling.
# In simple terms,
# it controls the level of confidence that a neural network has in its predictions.
# It helps to increase the diversity of the model’s outputs.

# For temperature in OpenAI chat request, “higher values like 0.8 will make the output more random,
# while lower values like 0.2 will make it more focused and deterministic”.


# Temperature is a parameter that controls the “creativity” or randomness of the text generated by GPT.
# A higher temperature (e.g., 0.7) results in more diverse and creative output, while a lower temperature
#  (e.g., 0.2) makes the output more deterministic and focused.

# Temperature value less than 1 makes the model more deterministic
# because if you divide a number with a value less than 1, you get a greater number.


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])

# OpenAI models use advanced algorithms and big data to achieve a much deeper
# and more nuanced representation of the data.
# The model not only analyzes individual words, but also looks at the context in which those words are used
# OpenAI embeddings are based on sophisticated machine learning models that can learn from a huge amount of data.
# This means that they can recognize subtle patterns and relationships in the data
# that go far beyond what could be achieved by simple scaling and dimensioning


# Load the models required to complete the exercise
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ["OPENAI_API_KEY"])

# Initialize RecursiveCharacterTextSplitter to make chunks of HTML text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Because RAG pipelines often rely on retrieval from vector databases and large language models (LLMs)
# with limited context windows,
# smart chunking can make all the difference in delivering relevant, context-rich answers.

# Chunking is the process of breaking a large piece of text into smaller,
# manageable pieces to make it easier to process and analyze.
# The chunk size refers to the maximum number of characters or tokens allowed in a single chunk.

# What is Chunk Overlap?

# Chunk Overlap refers to the number of characters or tokens shared between consecutive chunks.
# Overlapping ensures that important context is not lost when diving the text into smaller parts.

# Common values range from 200 to 500 tokens per chunk for embedding tasks.
# Chunk Overlap: Generally, overlap is set to 10%-20% of the chunk size to ensure continuity.


# Split the machine documents with text_splitter
splits = text_splitter.split_documents(machine_docs)

# Initialize Chroma vectorstore with documents as splits and using OpenAIEmbeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Setup vectorstore as retriever
# By calling .as_retriever(), you wrap the complex vector database
# into a simple component that plugs directly into LangChain chains and LangChain Expression Language (LCEL).
retriever = vectorstore.as_retriever()

# Define RAG prompt
prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""")

# Setup the chain
# you have to send question to the vector store as well as to the prompt later in the pipeline.
# Using "question": RunnablePassthrough() ensures the original question is retained for the prompt,
# whereas without it the "question" key would not be available during prompt construction.

rag_chain = (
    {"context": retriever , "question": RunnablePassthrough()}
    | prompt
    | llm
)

# When working with LCEL we may find that we need to modify the flow of values,
# or the values themselves as they are passed between components — for this, we can use runnables.

# Initialize query
query = "What is the cycle for DRUM CLEAN?"

# Invoke the query
answer = rag_chain.invoke(query).content
print(answer)

# Initialize query
query = "WHAT IS WASHING MACHINE?"

# Invoke the query
answer = rag_chain.invoke(query).content
print(answer)

