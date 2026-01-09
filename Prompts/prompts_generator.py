from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=['length_input', 'length_instructions',
                     'paper_input', 'style_input', 'style_instructions'],
    template="""
        You are an AI research assistant specialized in explaining complex machine learning papers.

        RESEARCH PAPER: {paper_input}
        EXPLANATION STYLE: {style_input}
        EXPLANATION LENGTH: {length_input}

        Please provide an explanation of the research paper following these guidelines:

        STYLE ADJUSTMENTS:
        {style_instructions}

        LENGTH REQUIREMENTS:
        {length_instructions}

        EXPLANATION STRUCTURE:
        1. **Core Idea**: What is the main innovation/contribution?
        2. **Key Components**: Break down the methodology/architecture
        3. **Why It Matters**: Impact on the field and practical applications
        4. **Limitations**: What are the shortcomings or areas for improvement?

        IMPORTANT:
        - Use clear, structured formatting with headings
        - Include equations if style is 'Technical' or 'Mathematical'
        - Mention authors and publication year
        - Compare with previous approaches when relevant
        - Adjust technical depth based on selected style

        Begin your explanation now:
    """
)

template.save('templete.json')
