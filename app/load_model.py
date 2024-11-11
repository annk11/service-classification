from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import logging


logging.info("Loading model")
tokenizer = BertTokenizer.from_pretrained("model")
model = BertForSequenceClassification.from_pretrained("model")

pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device="cpu"
)
logging.info("Model loaded")