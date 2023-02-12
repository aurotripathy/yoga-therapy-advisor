<h1 align="center">
🧘YogaTherapyAdvisor
</h1>

Accurate answers and instant citations for your yoga therapy questions.

## 🔧 Features

- Upload documents 📁(PDF, DOCX, TXT) and answer questions about them.
- Cite sources📚 for the answers, with excerpts from the text.

## 💻 Running Locally

1. Clone the repository📂

```bash
git clone https://aurotripathy/yoga-therapy-advisor
cd yoga-therapy-advisor
```

2. Install dependencies with [Poetry](https://python-poetry.org/) and activate virtual environment🔨

```bash
poetry install
poetry shell
```

3. Run the Streamlit server🚀

```bash
cd yoga-therapy-advisor
streamlit run bot_main.py
```
In run in DEBUG mode, use: 

```
streamlit run bot_main.py --logger.level=debug
```

## 🚀 Upcoming Features

- Highlight relevant phrases in citations 🔦
- Support scanned documents with OCR 📝
- More customization options (e.g. chain type 🔗, chunk size📏, etc.)

## Note
Attribution: This repo has be built after cloning from KnowledgeGPT
