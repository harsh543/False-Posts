import transformers


MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "./bert_base_uncased/"
MODEL_PATH = "./toxic_comments_bert_model.bin"
TRAINING_FILE = "../data/train/cleaned_train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
