from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

if __name__ == "__main__":
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analysis = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )

    # Define a sample movie review
    review = "I really enjoyed watching this movie."

    # Tokenize the review and convert to tensor
    inputs = tokenizer(
        review, return_tensors="pt", padding=True, truncation=True, max_length=128
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    expected_label = 0

    model.save_pretrained("./")
    tokenizer.save_pretrained("./")
    print("Model weights are saved into this folder")
