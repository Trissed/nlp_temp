import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, set_seed,FillMaskPipeline, logging
logging.set_verbosity_error()

commandtext = 'basic text '

sentences1 = []
sentences2 = []

for x in range(50):
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    sequence = f"basic text {tokenizer.mask_token}."
    input_ids = tokenizer.encode(sequence, return_tensors="pt")
    result = model(input_ids=input_ids)
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    b = (tokenizer.decode(result.logits[:, mask_token_index].argmax()))
    b = commandtext + " " + b
    #print(b)
    sentences2.append(b)

for x in range(50):
    generator = pipeline('text-generation', model='gpt2')
    a = generator(commandtext, max_length = 2, num_return_sequences=1)
    a = str(a)
    a = a.replace ("[{'generated_text': '", "")
    a = a.replace ("'}]", "")
    #print(a)
    sentences1.append(a)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2)

print("теоретически схожие варианты: ")
for i in range(len(sentences1)):
    if cosine_scores[i][i] > 0.75:
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))