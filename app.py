import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import pickle
import re

from keras.preprocessing.sequence import pad_sequences


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "bigru_model.pth"
TOKENIZER_PATH = "bigru_tokenizer.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 128
MAX_VOCAB = 50000

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

OUTPUT_DIM = 2
NUM_LAYERS = 1


# =========================================================
# MODEL ARCHITECTURE
# =========================================================

class BiGRUClassifier(nn.Module):

    def __init__(self, vocab_size):

        super(BiGRUClassifier, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM
        )

        self.bigru = nn.GRU(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            HIDDEN_DIM * 2,
            OUTPUT_DIM
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, hidden = self.bigru(embedded)

        hidden = torch.cat(
            (hidden[-2,:,:], hidden[-1,:,:]),
            dim=1
        )

        hidden = self.dropout(hidden)

        out = self.fc(hidden)

        return out


# =========================================================
# LOAD TOKENIZER
# =========================================================

with open(TOKENIZER_PATH, "rb") as f:
    _tokenizer = pickle.load(f)


# =========================================================
# LOAD MODEL
# =========================================================

_model = BiGRUClassifier(
    vocab_size=MAX_VOCAB
).to(DEVICE)

_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

_model.eval()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# CUSTOM CSS
# =========================================================

CUSTOM_CSS = """

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    background: #040d1a !important;
    font-family: 'DM Sans', sans-serif !important;
    color: white !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-top: 20px;
    color: #ffffff;
}

.sub-title {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(4,13,26,0.9));
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.result-hate {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(4,13,26,0.9));
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-confidence {
    font-size: 16px;
    color: #cbd5e1;
}

textarea {
    background: rgba(15,23,42,0.8) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #1d4ed8 !important;
}

button {
    border-radius: 12px !important;
}

"""


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_hate_speech(text):

    if not text.strip():

        return (
            """
            <div style='padding:20px;text-align:center;color:#94a3b8;'>
                Enter text to analyse.
            </div>
            """,
            None
        )

    cleaned = clean_text(text)

    sequence = _tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    ids = torch.tensor(
        padded,
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        logits = _model(ids)

        probs = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    prediction = int(np.argmax(probs))

    confidence = float(probs[prediction])

    if prediction == 1:

        result_html = f"""
        <div class="result-hate">
            <div class="result-title">
                Hate Speech Detected
            </div>

            <div class="result-confidence">
                Confidence: {confidence:.2%}
            </div>
        </div>
        """

    else:

        result_html = f"""
        <div class="result-safe">
            <div class="result-title">
                Content is Safe
            </div>

            <div class="result-confidence">
                Confidence: {confidence:.2%}
            </div>
        </div>
        """

    confidence_scores = {
        "Non-Hate Speech": float(probs[0]),
        "Hate Speech": float(probs[1])
    }

    return result_html, confidence_scores


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Hate Speech Detection using BiGRU"
) as demo:

    gr.HTML(
        """
        <div class="main-title">
            Hate Speech Detection System
        </div>

        <div class="sub-title">
            Deep Learning based NLP moderation system using BiGRU
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=3):

            input_text = gr.Textbox(
                lines=8,
                placeholder="Type or paste text here...",
                label="Input Text"
            )

            with gr.Row():

                analyse_btn = gr.Button(
                    "Analyse Text",
                    variant="primary"
                )

                clear_btn = gr.ClearButton(
                    [input_text]
                )

        with gr.Column(scale=2):

            output_html = gr.HTML()

            output_conf = gr.Label(
                num_top_classes=2,
                label="Confidence Scores"
            )

    gr.Examples(
        examples=[
            ["I love spending time with my family."],
            ["You are a horrible disgusting person."],
            ["Let's work together positively."],
            ["I hate everyone from that community."]
        ],
        inputs=input_text
    )

    analyse_btn.click(
        fn=predict_hate_speech,
        inputs=input_text,
        outputs=[output_html, output_conf]
    )

    input_text.submit(
        fn=predict_hate_speech,
        inputs=input_text,
        outputs=[output_html, output_conf]
    )


# =========================================================
# LAUNCH
# =========================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
