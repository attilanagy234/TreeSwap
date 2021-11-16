import datasets
import numpy as np

dataset_names = [
                "bible_para",
                "ecb",
                "emea",
                "kde4",
                "open_subtitles", 
                "php",
                "qed_amara",
                "tatoeba",
                "tilde_model",
                "opus_dgt",
                "opus_gnome",
                "opus_paracrawl",
                "opus_ubuntu",
                "opus_books"]

for name in dataset_names:
    print("---Starting "+name)
    ds = datasets.load_dataset(name, lang1="en", lang2="hu")
    value = ds['train'].map(lambda x: {"value": x['translation']["hu"]+'\t'+x['translation']["en"]})
    with open('../data/huggingface/'+name+".txt", 'w', encoding='utf8') as f:
        for l in value["value"]:
            f.write(l+'\n')
    print("---Finished "+name)
