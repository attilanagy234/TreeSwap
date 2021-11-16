from transformers import MarianMTModel, MarianTokenizer


def download(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to('cuda:0')
    return tokenizer, model

# download model for English -> Romance
trg_lang_tokenizer, trg_lang_model = download('Helsinki-NLP/opus-mt-en-hu')
# download model for Romance -> English
src_lang_tokenizer, src_lang_model = download('Helsinki-NLP/opus-mt-hu-en')

def translate(texts, model, tokenizer, language):
    """Translate texts into a target language"""
    # Format the text as expected by the model
    formatter_fn = lambda txt: f"{txt}" if language == "en" else f">>{language}<< {txt}"
    original_texts = [formatter_fn(txt) for txt in texts]

    # Tokenize (text to tokens)
    tokens = tokenizer.prepare_seq2seq_batch(original_texts, return_tensors='pt').to('cuda:0')

    # Translate
    translated = model.generate(**tokens)

    # Decode (tokens to text)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return translated_texts

def back_translate(texts, language_src, language_dst):
    """Implements back translation"""
    # Translate from source to target language
    translated = translate(texts, trg_lang_model, trg_lang_tokenizer, language_dst)

    # Translate from target language back to source language
    back_translated = translate(translated, src_lang_model, src_lang_tokenizer, language_src)

    return translated, back_translated

src_texts = ['I might be late tonight', 'What a movie, so bad', 'That was very kind']
translated_texts, back_texts = back_translate(src_texts, "en", "hu")
print(src_texts)
print(translated_texts)
print(back_texts)