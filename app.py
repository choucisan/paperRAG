from gradio_pdf import PDF
from latex2zh.trans_tex import translate_tex_file
from latex2zh.trans_arxiv import translate_arxiv_file
from latex2zh.config import config
from typing import Optional, Tuple
import gradio as gr
from threading import Lock
from langchain.chains.conversation.base import ConversationChain
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os



class LLMEngine:
    @staticmethod
    def get_model(engine: str, api_key: str = None):
        engine = engine.lower()

        if engine == "openai":
            return ChatOpenAI(
                temperature=0.7
            )

        elif engine == "zhipu":
            return ChatZhipuAI(
                api_key=api_key,
                model="glm-4.5"
            )

        elif engine == "claude":
            raise NotImplementedError("Claude 接入未实现")

        else:
            raise ValueError(f"Unsupported engine: {engine}")




class APIManager:

    def __init__(self, engine: str = "OpenAI", api_key: str = None):
        self.engine = engine
        self.api_key = api_key
        self.llm = None
        self.chain = None


    def load_llm(self):
        self.llm = LLMEngine.get_model(self.engine, self.api_key)
        return self.llm


    def load_chain(self):

        if not self.llm:
            self.load_llm()
        memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=self.llm, memory=memory, verbose=True)
        return self.chain

    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.load_llm()
        return self.load_chain()


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
        self.vectorstore = None

    def build_vector_store_from_documents(self, docs):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        return self.vectorstore

    def build_vector_store_from_csv(self, csv_path):
        loader = CSVLoader(file_path=csv_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return self.build_vector_store_from_documents(splits)

    def __call__(
            self, inp: str, chain: Optional[ConversationChain], history: Optional[list[Tuple[str, str]]]
    ):
        with self.lock:
            history = history or []
            if chain is None:
                history.append((inp, "Please paste your AI key to use"))
                return history, history

            if self.vectorstore is None:
                history.append((inp, "Vector store not initialized. Please build vector store first."))
                return history, history

            docs = self.vectorstore.similarity_search(inp, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {inp}"
            output = chain.run(input=prompt)

            history.append((inp, output))

            return history, history



chat = ChatWrapper()
chat.build_vector_store_from_csv("cv_paper.csv")



# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
        
    }

    """


page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}


# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}



engine: dict[str, int] = {
    "Google": 0,
    "Bing": 0,
    "DeepL": 0,
    "DeepLX": 0,
    "Xinference": 0,
    "AzureOpenAI": 0,
    "OpenAI": 0,
    "Zhipu": 0,
    "ModelScope": 0,
    "Silicon": 0,
    "Gemini": 0,
    "Azure": 0,
    "Tencent": 0,
    "Dify": 0,
    "AnythingLLM": 0,
    "Argos Translate": 0,
    "Grok": 0,
    "Groq": 0,
    "DeepSeek": 0,
    "OpenAI-liked": 0,
    "Ali Qwen-Translation": 0,
}


env_config_map = {
    "Zhipu": ["API Key"],
    "Tencent": ["SecretId", "SecretKey"],
    "OpenAI": ["API Key", "Organization"],
    "AzureOpenAI": ["API Key", "Endpoint", "Deployment Name"],
    "Google": ["API Key"],
    "Bing": ["API Key"],
    "DeepL": ["API Key"],
    "DeepLX": ["Host URL"],
    "Xinference": ["Base URL"],
    "ModelScope": ["API Key"],
    "Silicon": ["API Key"],
    "Gemini": ["API Key"],
    "Azure": ["API Key", "Endpoint"],
    "Dify": ["API Key"],
    "AnythingLLM": ["API Key"],
    "Argos Translate": [],
    "Grok": ["API Key"],
    "Groq": ["API Key"],
    "DeepSeek": ["API Key"],
    "OpenAI-liked": ["API Key"],
    "Ali Qwen-Translation": ["API Key"]
}


env_field_map = {
    "Zhipu": {
        "API Key": "zhipu_key_default"
    },
    "Tencent": {
        "SecretId": "tencent_secret_id_default",
        "SecretKey": "tencent_secret_key_default"
    },
    "OpenAI": {
        "API Key": "openai_api_key",
        "Organization": "openai_org"
    },
    "AzureOpenAI": {
        "API Key": "azure_api_key",
        "Endpoint": "azure_endpoint",
        "Deployment Name": "azure_deployment"
    },
    "Google": {
        "API Key": "google_api_key"
    },
    "Bing": {
        "API Key": "bing_api_key"
    },
    "DeepL": {
        "API Key": "deepl_api_key"
    },
    "DeepLX": {
        "Host URL": "deeplx_url"
    },
    "Xinference": {
        "Base URL": "xinference_base_url"
    },
    "ModelScope": {
        "API Key": "modelscope_key"
    },
    "Silicon": {
        "API Key": "silicon_key"
    },
    "Gemini": {
        "API Key": "gemini_key"
    },
    "Azure": {
        "API Key": "azure_key",
        "Endpoint": "azure_endpoint"
    },
    "Dify": {
        "API Key": "dify_key"
    },
    "AnythingLLM": {
        "API Key": "anythingllm_key"
    },
    "Grok": {
        "API Key": "grok_key"
    },
    "Groq": {
        "API Key": "groq_key"
    },
    "DeepSeek": {
        "API Key": "deepseek_key"
    },
    "OpenAI-liked": {
        "API Key": "openai_like_key"
    },
    "Ali Qwen-Translation": {
        "API Key": "ali_qwen_key"
    }
}


env_store = {}

def save_env(service_name, env1, env2, env3, _agent_state):

    fields_map = env_field_map.get(service_name, {})
    field_names = list(fields_map.keys())       # ["API Key", ...]
    config_attrs = list(fields_map.values())    # ["zhipu_key_default", ...]

    values = [env1, env2, env3][:len(config_attrs)]

    env_store[service_name] = dict(zip(field_names, values))

    api_key = values[0] if values else ""

    for attr_name, val in zip(config_attrs, values):
        try:
            setattr(config, attr_name, val)
        except Exception as e:
            print(f"Failed to set config.{attr_name}: {e}")

    try:
        llm = LLMEngine.get_model(service_name, api_key=api_key)
        chain = ConversationChain(llm=llm, memory=ConversationBufferMemory())
    except Exception as e:
        print(f"Failed to init LLM: {e}")
        chain = None

    return (
        *[gr.update(visible=False) for _ in range(3)],
        gr.update(visible=False),
        chain
    )


def toggle_env_fields(selected_service):
    fields = env_config_map.get(selected_service, [])
    outputs = []
    for i in range(3):
        if i < len(fields):
            outputs.append(gr.update(visible=True, label=fields[i], value=""))
        else:
            outputs.append(gr.update(visible=False))

    outputs.append(gr.update(visible=True))
    return outputs




def toggle_inputs(file_type_choice):
    if file_type_choice == "File":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def toggle_page_range(choice):
    if choice == "Custom range":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)



def translate_handler(file_type_choice, tex_path, arxiv_id, eng, lang_from_code, lang_to_code):
    lang_from_code = lang_map[lang_from_code]
    lang_to_code = lang_map[lang_to_code]

    def abs_path(path):
        if not path:
            return None
        return os.path.abspath(path)

    if file_type_choice == "File":
        if not tex_path:
            return None, "Error: No .tex file uploaded."
        try:
            tex_out, pdf_out = translate_tex_file(
                input_path=tex_path,
                engine=eng,
                lang_from=lang_from_code,
                lang_to=lang_to_code,
                compile_pdf=True
            )
            return abs_path(pdf_out), "File translated successfully."
        except Exception as e:
            return None, f"Error in file translation: {e}"

    elif file_type_choice == "Link":
        if not arxiv_id:
            return None, "Error: No arXiv ID provided."
        try:
            pdf_out = translate_arxiv_file(
                number=arxiv_id,
                engine=eng,
                lang_from=lang_from_code,
                lang_to=lang_to_code
            )
            return abs_path(pdf_out), "arXiv translation successful."
        except Exception as e:
            return None, f"Error in arXiv translation: {e}"

    else:
        return None, "Invalid input type selected."

with gr.Blocks(
        title="TranslaTex",
        theme=gr.themes.Default(
            primary_hue=custom_blue, spacing_size="md", radius_size="lg"
        ),
        css=custom_css,
) as demo:
    gr.Markdown("# PaperRAG")

    state = gr.State()
    agent_state = gr.State()

    with gr.Row():
        # 左侧：上传和选项
        with gr.Column(scale=1):
            gr.Markdown("## File")

            file_type = gr.Radio(
                choices=["File", "Link"],
                label="Type",
                value="File",
            )

            tex_file_input = gr.File(label="Upload .tex file", file_types=[".tex"], type="filepath")

            arxiv_link_input = gr.Textbox(
                label="arXiv id",
                interactive=True,
            )

            file_type.change(fn=toggle_inputs, inputs=file_type, outputs=[tex_file_input, arxiv_link_input])



            gr.Markdown("## Option")
            service = gr.Dropdown(
                choices=list(engine.keys()),
                value=list(engine.keys())[0],
                label="Service"
            )


            envs = [gr.Textbox(visible=False, label=f"ENV {i + 1}") for i in range(3)]

            save_btn = gr.Button("Save Configuration", visible=False)


            service.change(
                fn=toggle_env_fields,
                inputs=service,
                outputs=envs + [save_btn]
            )


            save_btn.click(
                fn=save_env,
                inputs=[service] + envs + [agent_state],
                outputs=envs + [save_btn, agent_state],
            )


            with gr.Row():
                lang_from = gr.Dropdown(
                    label="Translate from",
                    choices=list(lang_map.keys()),
                    value="English",
                )
                lang_to = gr.Dropdown(
                    label="Translate to",
                    choices=list(lang_map.keys()),
                    value="Simplified Chinese",
                )

            page_range = gr.Radio(
                choices=list(page_map.keys()),
                label="Pages",
                value=list(page_map.keys())[0],
            )

            page_input = gr.Textbox(
                label="Page range",
                visible=False,
                interactive=True,
            )


            page_range.change(fn=toggle_page_range, inputs=page_range, outputs=page_input)

            with gr.Row():
                translate_btn = gr.Button("Translate", variant="primary", scale=1)
                cancel_btn = gr.Button("Cancel", variant="secondary", scale=1)


        with gr.Column(scale=2):
            gr.Markdown("## Preview")
            pdf_output = PDF(label="Document Preview", visible=True, height=2000)
            preview = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("## ChatRAG")
            chatbot = gr.Chatbot(label="chatBot", height=600)
            message = gr.Textbox(
                placeholder="What's the answer to life, the universe, and everything?",
                lines=2,
                scale=9,
                show_label=False,
            )

            with gr.Row():

                with gr.Column(scale=8):
                    gr.Examples(
                        examples=[
                            "Hi! How's it going?",
                            "What should I do tonight?",
                            "What's 2 + 2?",
                        ],
                        inputs=message,
                        label="💡 Example Prompts",
                    )

                with gr.Column(scale=2):
                    submit = gr.Button(
                        value="Send",
                        variant="primary",
                    )

    translate_btn.click(
        fn=translate_handler,
        inputs=[file_type, tex_file_input, arxiv_link_input, service, lang_from, lang_to],
        outputs=[pdf_output, preview]
    )

    cancel_btn.click(
        lambda: (None, "Translation canceled."),
        outputs=[pdf_output, preview]
    )

    submit.click(chat, inputs=[message, agent_state, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, agent_state, state], outputs=[chatbot, state])


demo.launch()
