import os
from lyzr_agent import LyzrAgent
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LYZR_API_KEY = os.getenv("LYZR_API_KEY")

# Streamlit page configuration
st.set_page_config(
    page_title="Lyzr Newsletter Generator",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

st.title("Lyzr Newsletter Generator")
st.markdown("### Welcome to the Lyzr Newsletter Generator!")

# Initialize the LyzrAgent
Agent = LyzrAgent(api_key=LYZR_API_KEY, llm_api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=True)
def create_agent():
    """Create and return the agent for generating cold emails."""
    env_id = Agent.create_environment(
        name="Newsletter Environment",
        features=[{
            "type": "TOOL_CALLING",
            "config": {"max_tries": 3},
            "priority": 0
        },
            {
                "type": "SHORT_TERM_MEMORY",
                "config": {},
                "priority": 0
            }
        ],
        tools=["perplexity_search"]
    )

    # System prompt for guiding the agent's behavior
    prompt = """
    Act like an experienced newsletter writer skilled in crafting engaging and informative content. 
    Use a perplexity search to find the most relevant and current information on Given Topic. 
    Your goal is to summarize the key insights and trends into a concise, well-structured newsletter that keeps readers informed and interested. 
    Include engaging headlines, clear sections, and a call to action.

    Take a deep breath and work on this problem step-by-step.
    """

    agent_id = Agent.create_agent(
        env_id=env_id['env_id'],
        system_prompt=prompt,
        name="Newsletter Agent"
    )

    return agent_id


# Maintain the agent session using Streamlit's session state
if "agent_id" not in st.session_state:
    st.session_state.agent_id = create_agent()

# Input area for the product description and target audience
query = st.text_area("Enter Topic and description", height=150)

if st.button("Generate NewsLetter"):
    if query.strip():
        with st.spinner("Generating NewsLetter..."):
            response = Agent.send_message(
                agent_id=st.session_state.agent_id['agent_id'],
                user_id="default_user",
                session_id="new_session",
                message=query
            )
            # Display the generated email
            st.markdown(f"**Generated NewsLetter:**\n\n{response['response']}")
    else:
        st.warning("Please provide Topic")

# Optional footer or credits
st.markdown("---")
st.markdown("Powered by Lyzr and OpenAI")
