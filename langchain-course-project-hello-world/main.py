
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():

    # Topic 1
    information = """
    Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability
    with the use of significant indentation. Python is dynamically type-checked and garbage-collected.
    It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    Python is widely used in machine learning.
    """

    # Topic 2 (another variable, different topic)
    information2 = """
    Artificial Intelligence (AI) is the capability of machines to perform tasks that typically require human intelligence.
    AI includes areas such as machine learning, natural language processing, and computer vision.
    AI is used in healthcare, finance, autonomous vehicles, and recommendation systems.
    """

    # Extra variable (controls output level/style)
    audience = "beginners"

    summary_template = """
    Given the information about TWO topics:

    Topic 1:
    {information}

    Topic 2:
    {information2}

    Create the following for EACH topic for a {audience} audience:
    1) Basic Rules
    2) Popular frameworks related to the topic

    Format the answer clearly using headings.
    """

    prompt = PromptTemplate(
        input_variables=["information", "information2", "audience"],
        template=summary_template
    )

    llm = ChatGroq(
    temperature=0,
    model="llama-3.1-8b-instant"

    )

    # ✅ LCEL: pipe prompt into llm
    chain = prompt | llm

    # ✅ Invoke with variables as a dict
    response = chain.invoke({
        "information": information,
        "information2": information2,
        "audience": audience
    })

    print("\n--- LLM Response ---\n")
    print(response.content)


if __name__ == "__main__":
    main()
