# LLM-Powered Graph Q&A and Visualization System

To know more, check out my blog - [Building an Interactive Graph-Based Question-Answering System](https://medium.com/@girishwangikar/building-an-interactive-graph-based-question-answering-system-f5508b830488)

This repository hosts an interactive Gradio-based application that leverages advanced language models and graph processing techniques to provide insightful answers to user queries based on provided text. It also visualizes relationships in the text as a graph, generates a summary, and creates an image based on the summary.

## Features

- **Graph-Based QA**: Automatically converts text into a graph of entities and relationships, and uses this graph to answer questions.
- **Graph Visualization**: Visualizes the relationships within the text using NetworkX.
- **Image Generation**: Generates images from text summaries using the Flux model.
- **DataTable of Relations**: Displays a table of relationships extracted from the text.
- **User-Friendly Interface**: Built with Gradio for an intuitive and interactive experience.

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.7+
- Gradio
- LangChain
- NetworkX
- Matplotlib
- Pandas
- NumPy
- PIL (Pillow)
- gradio_client
- langchain_experimental
- langchain_community

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/GirishWangikar/GraphRAG-Visualization-System
    cd llm-graph-qa-image-gen
    ```

2. Install the required packages:
    ```bash
    pip install gradio langchain networkx matplotlib pandas numpy pillow gradio_client langchain_experimental langchain_community
    ```

3. Set up your Groq API key as an environment variable:
    ```bash
    export API_KEY='your_api_key_here'
    ```

## Usage

1. Run the application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to the URL provided in the console output.

3. In the Gradio interface:
    - Enter the text you want to analyze in the "Input Text" field.
    - Pose a question related to the text in the "Question" field.
    - Click the "Run" button to process the input and view the results:
        - **Answer**: The AI's answer to your question.
        - **Graph Visualization**: A visual representation of the entities and relationships extracted from the text.
        - **Relations Table**: A table detailing the relationships between entities.
        - **Summary**: A one-sentence summary of the text.
        - **Generated Image**: An image generated based on the summary.

## Customization

- **LLM Parameters**: Modify the temperature or model name in the script to change the behavior of the language model.
- **Graph Parameters**: Adjust the graph layout settings in the `visualize_graph` function for different visual styles.
- **Image Generation**: Customize the image generation by changing the `width`, `height`, and `num_inference_steps` parameters in the `generate_image` function.

## Contact

Created by Girish Wangikar

Check out more on [LinkedIn](https://www.linkedin.com/in/girish-wangikar/) | [Portfolio](https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/) | [Technical Blog - Medium](https://medium.com/@girishwangikar/building-an-interactive-graph-based-question-answering-system-f5508b830488)
