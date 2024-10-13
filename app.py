import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.utilities import WikipediaAPIWrapper

# Title and welcome message
st.set_page_config(page_title="Text To Math Problem Solver and Data Search Assistance", page_icon="🧮")
st.title("Innovative Math and Data Search Assistant Using Google Gemma2")
st.write("""
        👋 Welcome to the \"Math & Information Retrieval Assistant\" - your go-to tool for solving math problems and accessing information from Wikipedia. 🌎
        """)

# Input for Groq API Key
groq_api_key = st.sidebar.text_input("Groq API Key:", type="password")

if groq_api_key:
    # LLM Model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    # Wikipedia Tool
    api_wrapper_wiki = WikipediaAPIWrapper()
    wiki = Tool(
        name="Wikipedia",
        func=api_wrapper_wiki.run,
        description="A tool for searching the Internet to find the various information on the mentioned topics."
    )

    # Math Tool
    math_chain = LLMMathChain.from_llm(llm=llm)
    calculator = Tool(
        name="Calculator",
        func=math_chain.run,
        description="A tool for answering math-related questions. Only input mathematical expression need to be provided."
    )

    # Create the Prompt template object
    template = """
        You are an agent tasked with solving users' mathematical questions. 
        Logically arrive at the solution and provide a detailed explanation, displayed point-wise for the question below: 
        Question: {question}
        Answer:
    """

    prompt = PromptTemplate(
        input_variables=['question'],
        template=template
    )

    # Combine Tools
    chain = LLMChain(llm=llm, prompt=prompt)
    reasoning_tool = Tool(
        name="Reasoning tool",
        func=chain.run,
        description="A tool for answering logic-based and reasoning questions."
    )
    tools = [wiki, calculator, reasoning_tool]

    # Initialize the agent
    assistant_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Initialize the session state for storing messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can answer all your math questions!"
        }]
    
    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Text area for entering a question
    question = st.text_area("Enter your question:", value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")
    
    # Button to get the answer
    if st.button("Get Answer"):
        if question:
            with st.spinner("Calculating..."):
                # Append user's question to the session state (only once)
                st.session_state["messages"].append({
                    "role": "user",
                    "content": question
                })
                st.chat_message("user").write(question)

                # Stream the agent's thoughts with callback handler
                callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                
                # Run the agent using the latest question
                response = assistant_agent.run(question, callbacks=[callback_handler])

                # Append assistant's response to the chat history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response
                })

                # Display the assistant's response
                st.write("### Response")
                st.success(response)

else:
    st.info("Please add your Groq API Key to continue.")
    st.stop()
