# Spider Scanner

A web application that identifies spider species from an uploaded photo.
Currently supports 15 species.

## Requirements

- Python 3.10+
- pip

## Setup for development

```bash
# 1. Clone repo
git clone https://github.com/Oskar-Collegium/MSI_Spider_Scanner

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the .env file
cp .env.example .env

# 5. Run the dev local server
uvicorn app.main:app --reload
```

App will show up at `http://localhost:8000`.

## Training

Training dataset is located at `data/spiders/`, seperated per species:

To train the model yourself, use:
```bash
python training/train.py --data_dir data/spiders --epochs 20
```

The best model will be saved to `model/spider_model.pt`.

## Testing

```bash
pytest tests/ -v
```
