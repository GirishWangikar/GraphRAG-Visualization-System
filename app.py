import os
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
from gradio_client import Client
import numpy as np
from PIL import Image as PILImage
import base64
from io import BytesIO

# Set the base directory
BASE_DIR = os.getcwd()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set up LLM and Flux client
llm = ChatGroq(temperature=0, model_name='llama-3.1-8b-instant', groq_api_key=GROQ_API_KEY)
flux_client = Client("black-forest-labs/Flux.1-schnell")

def create_graph(text):
    documents = [Document(page_content=text)]
    llm_transformer_filtered = LLMGraphTransformer(llm=llm)
    graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
    graph = NetworkxEntityGraph()

    for node in graph_documents_filtered[0].nodes:
        graph.add_node(node.id)

    for edge in graph_documents_filtered[0].relationships:
        graph._graph.add_edge(
            edge.source.id,
            edge.target.id,
            relation=edge.type
        )

    return graph, graph_documents_filtered

def visualize_graph(graph):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph._graph)
    nx.draw(graph._graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(graph._graph, 'relation')
    nx.draw_networkx_edge_labels(graph._graph, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Graph Visualization")
    plt.axis('off')

    # Save the plot as an image file
    graph_viz_path = os.path.join(BASE_DIR, 'graph_visualization.png')
    plt.savefig(graph_viz_path)
    plt.close()

    return graph_viz_path

def generate_image(prompt):
    try:
        print(f"Generating image with prompt: {prompt}")
        result = flux_client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )

        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], str):
            img_str = result[0]
            
            # Fix padding if necessary
            missing_padding = len(img_str) % 4
            if missing_padding:
                img_str += '=' * (4 - missing_padding)
                
            img_data = base64.b64decode(img_str)
            image = PILImage.open(BytesIO(img_data))
        elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], np.ndarray):
            image = PILImage.fromarray((result[0] * 255).astype(np.uint8))
        elif isinstance(result, PILImage.Image):
            image = result
        else:
            raise ValueError(f"Unexpected result format from flux_client.predict: {type(result)}")

        image_path = os.path.join(BASE_DIR, 'generated_image.png')
        image.save(image_path)

        print(f"Image saved to: {image_path}")
        return image_path
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_relations_table(graph_documents_filtered):
    df = pd.DataFrame(columns=['node1', 'node2', 'relation'])
    for edge in graph_documents_filtered[0].relationships:
        df = pd.concat([df, pd.DataFrame({'node1': [edge.source.id], 'node2': [edge.target.id], 'relation': [edge.type]})], ignore_index=True)
    return df

def process_text(text, question):
    try:
        print("Creating graph...")
        graph, graph_documents_filtered = create_graph(text)

        print("Setting up GraphQAChain...")
        graph_rag = GraphQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True
        )

        print("Running question through GraphQAChain...")
        answer = graph_rag.run(question)
        print(f"Answer: {answer}")

        print("Visualizing graph...")
        graph_viz_path = visualize_graph(graph)
        print(f"Graph visualization saved to: {graph_viz_path}")

        print("Creating relations table...")
        relations_table = create_relations_table(graph_documents_filtered)
        print("Relations table created")

        print("Generating summary...")
        summary_prompt = f"Summarize the following text in one sentence: {text}"
        summary = llm.invoke(summary_prompt).content
        print(f"Summary: {summary}")

        print("Generating image...")
        image_path = generate_image(summary)
        if image_path and os.path.exists(image_path):
            print(f"Generated image saved to: {image_path}")
        else:
            print("Failed to generate or save image")

        return answer, graph_viz_path, relations_table, summary, image_path
    except Exception as e:
        print(f"An error occurred in process_text: {str(e)}")
        import traceback
        traceback.print_exc()
        return str(e), None, None, str(e), None

def ui_function(text, question):
    answer, graph_viz_path, relations_table, summary, image_path = process_text(text, question)
    if isinstance(answer, str) and answer.startswith("An error occurred"):
        return answer, None, None, answer, None
    return answer, graph_viz_path, relations_table, summary, image_path

# Example text
example_text = """The Apollo 11 mission, launched by NASA in July 1969, was the first manned mission to land on the Moon. Commanded by Neil Armstrong and piloted by Buzz Aldrin and Michael Collins, it was the culmination of the Space Race between the United States and the Soviet Union. On July 20, 1969, Armstrong and Aldrin became the first humans to set foot on the lunar surface, while Collins orbited above in the command module."""

# Create Gradio interface
with gr.Blocks() as iface:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text")
            question = gr.Textbox(label="Question")
            example_box = gr.Markdown(f"### Example Paragraph\n\n{example_text}")
            graph_viz = gr.Image(label="Graph Visualization", type="filepath")
        with gr.Column():
            answer = gr.Textbox(label="Answer")
            relations_table = gr.Dataframe(label="Relations Table")
            summary = gr.Textbox(label="Summary")
            generated_image = gr.Image(label="Generated Image", type="filepath")

    gr.Button("Run").click(ui_function, inputs=[input_text, question], outputs=[answer, graph_viz, relations_table, summary, generated_image])
    
    footer_text = """
        <footer>
            <p>If you enjoyed the functionality of the app, please leave a like!<br>
            Check out more on <a href="https://www.linkedin.com/in/girish-wangikar/" target="_blank">LinkedIn</a> |
            <a href="https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/" target="_blank">Portfolio</a></p>
        </footer>
        """
    gr.Markdown(footer_text)
iface.launch()
