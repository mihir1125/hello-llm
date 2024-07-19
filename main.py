#!/usr/bin/env python
from typing import List, Union
from dotenv import load_dotenv

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

load_dotenv()

# 1. Create prompt template
# system_template = "You are a helpful, professional assistant named Cob."
system_template = "You are an annoying, unprofessional meddler named Cob."
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description = "The chat messages representing the current conversation.",
    )

# 2. Create model
model = ChatCohere(model="command-r")

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
# chain = prompt_template | model | parser
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain.with_types(input_type=InputChat),
    path="/chain",
    playground_type="chat"
)

if __name__ == "__main__":
    print("Running main thread")
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)