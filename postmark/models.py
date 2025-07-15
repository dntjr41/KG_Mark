import os
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import json
import re
import spacy
import nltk
import time
import random
from collections import Counter
from torch.nn.functional import cosine_similarity
from openai import OpenAI
from together import Together
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
from scipy.stats import kendalltau
import numpy as np
import pickle
import tiktoken
import pdb

torch.manual_seed(42)
random.seed(42)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class EmbeddingModel():
    def __init__(self, ratio=0.1, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ratio = ratio
        self.nlp = spacy.load("en_core_web_sm")
        self.word2idx = {}
        self.idx2word = []
        self.embedding_table = torch.zeros(1, 300).to(self.device)
        self.freq_dict = json.load(open("wikitext_freq.json", 'r'))
        self.tagger = nltk.PerceptronTagger()
    
    def get_embedding(self, word):
        if word in self.word2idx:
            return self.embedding_table[self.word2idx[word]]
        else:
            return None

    def get_embeddings(self, words):
        indices = []
        none_indices = []
        for i, word in enumerate(words):
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                none_indices.append(i)
        embeddings = self.embedding_table[indices, :]
        if none_indices:
            for i in none_indices:
                t = torch.zeros(1, self.embedding_table.shape[1]).to(self.device)
                embeddings = torch.cat((embeddings[:i], t, embeddings[i:]), dim=0)
        assert embeddings.shape[0] == len(words), f"embeddings shape mismatch with words size: {embeddings.shape[0]} != {len(words)}"
        return embeddings
    
    def get_doc_embedding(self, text):
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_stop and token.text in self.word2idx]
        embeddings = self.get_embeddings(words)
        return embeddings.mean(dim=0).unsqueeze(0).to(self.device)
    
    def get_word(self, embedding):
        idx = (self.embedding_table == embedding).all(dim=1).nonzero(as_tuple=True)[0]
        return self.idx_to_word(idx)
    
    def idx_to_word(self, idx):
        return self.idx2word[idx]
    
    def word_to_idx(self, word):
        return self.word2idx[word]

    def get_words(self, text):
        k = int(len(text.split()) * self.ratio)
        response_vec = self.get_doc_embedding(text)
        if isinstance(response_vec, list):
            response_vec = torch.tensor(response_vec).to(self.device)

        scores = cosine_similarity(self.embedding_table, response_vec, dim=1)
        assert scores.shape[0] == len(self.idx2word), f"scores shape mismatch with idx2word size: {scores.shape[0]} != {len(self.idx2word)}"
        top_k_scores, top_k_indices = torch.topk(scores, k * 3)
        
        top_k_words = [self.idx_to_word(index.item()) for index in top_k_indices]
        words = top_k_words
        try:
            word_embs = self.get_doc_embedding(words)
        except:
            return words
        word_embs = torch.tensor(word_embs).to(self.device)
        text_emb = response_vec.unsqueeze(0).to(self.device)
        scores = cosine_similarity(text_emb, word_embs, dim=1)

        top_k_scores, top_k_indices = torch.topk(scores, k)
        words = [words[idx] for idx in top_k_indices]
        words = [word.lower() for word in words]
        words = sorted(list(set(words)))
        return words


class OpenAIEmb(EmbeddingModel):
    def __init__(self, model="text-embedding-3-large", word=False, device_id=None, **kwargs):
        super().__init__(device_id=device_id, **kwargs)
        self.model = model
        from config import OPENAI_API_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        wpath = "valid_wtmk_words_in_wiki_base-only-f1000.pkl"  # only use base forms of nouns, verbs, adjectives, adverbs
        if not os.path.exists(wpath):
            with open( "wikitext_freq.json", "r") as f:
                freq_dict = json.load(f)
            freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=True))
            freq_dict = {k: v for k, v in freq_dict.items() if v >= 1000}
            all_words = list(freq_dict.keys())
            tag_filter = ['NN', 'VB', 'JJ', 'RB']
            final_list = []
            for i, word in tqdm.tqdm(enumerate(all_words), total=len(all_words)):
                if len(word) < 3 or not word.isalpha() or not word.islower():
                    continue
                tag = self.tagger.tag([word])[0][1]
                if "NNP" in tag or tag not in tag_filter:
                    continue
                final_list.append(word)
            with open(wpath, 'wb') as f:
                pickle.dump(final_list, f)
        words = pickle.load(open(wpath, 'rb'))
        redpj_embs = pickle.load(open("filtered_data_100k_unique_250w_sentbound_openai_embs.pkl", 'rb'))
        indices = random.sample(range(len(redpj_embs)), len(words))
        emb_list = [torch.tensor(redpj_embs[i]) for i in indices]
        random.shuffle(emb_list)
        emb_table = torch.stack(emb_list)
        self.embedding_table = emb_table.to(self.device)
        print(f"Embedding table shape: {self.embedding_table.shape}")
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = words

    
    def get_doc_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            dimensions=256
        )
        if type(text) == str:
            return response.data[0].embedding
        elif type(text) == list:
            return [r.embedding for r in response.data]
        else:
            raise ValueError


class NomicEmbed(EmbeddingModel):
    def __init__(self, model="nomic-embed-text-v1", word=False, device_id=None, **kwargs):
        super().__init__(device_id=device_id, **kwargs)
        self.model = model
        self.word = word
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.embedder = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True).to(self.device)
        self.embedder.eval()
        if not self.word:
            wpath = "valid_wtmk_words_in_wiki_base-only-f1000.pkl"  # only use base forms of nouns, verbs, adjectives, adverbs
            words = pickle.load(open(wpath, 'rb'))
            redpj_embs = pickle.load(open("filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl", 'rb'))
            indices = random.sample(range(len(redpj_embs)), len(words))
            emb_list = [torch.tensor(redpj_embs[i]) for i in indices]
            random.shuffle(emb_list)
            emb_table = torch.stack(emb_list)
            self.embedding_table = emb_table.to(self.device)
            print(f"Embedding table shape: {self.embedding_table.shape}")
            self.word2idx = {word: idx for idx, word in enumerate(words)}
            self.idx2word = words
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_doc_embedding(self, text):
        if type(text) == str:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)[0]
        elif type(text) == list:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.embedder(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class KEPLEREmbedding():
    def __init__(self, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # KEPLER 사전학습 모델 로드 (Wikipedia + Wikidata5M 기반)
        self.MODEL_NAME = "thunlp/KEPLER"
        print(f"Loading KEPLER model: {self.MODEL_NAME}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()  # 추론 모드
        self.model = self.model.to(self.device)
        
        print(f"KEPLER model loaded on device: {self.device}")
    
    def get_entity_embedding(self, entity_name):
        """
        Entity 이름을 통해 KEPLER embedding 생성
        Args:
            entity_name (str): Entity 이름 (예: "Neil Armstrong")
        Returns:
            numpy.ndarray: 768차원 embedding 벡터
        """
        try:
            # 입력: "Neil Armstrong" 같은 Entity 이름
            inputs = self.tokenizer(entity_name, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # [CLS] 토큰 임베딩 사용 (768차원)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{entity_name}': {e}")
            # 에러 시 zero embedding 반환
            return np.zeros(768)
    
    def get_entity_embeddings_batch(self, entity_names, batch_size=8):
        """
        여러 entity 이름에 대해 배치로 embedding 생성
        Args:
            entity_names (list): Entity 이름 리스트
            batch_size (int): 배치 크기
        Returns:
            dict: {entity_name: embedding} 형태의 딕셔너리
        """
        embeddings = {}
        
        for i in range(0, len(entity_names), batch_size):
            batch_names = entity_names[i:i+batch_size]
            try:
                # 배치로 토크나이징
                inputs = self.tokenizer(batch_names, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 각 entity의 [CLS] 토큰 embedding 추출
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                for j, entity_name in enumerate(batch_names):
                    embeddings[entity_name] = batch_embeddings[j]
                    
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # 에러 시 개별 처리
                for entity_name in batch_names:
                    embeddings[entity_name] = self.get_entity_embedding(entity_name)
        
        return embeddings
    
    def compute_similarity(self, emb1, emb2):
        """두 embedding 간의 코사인 유사도 계산"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def find_similar_entities(self, query_embedding, entity_embeddings, top_k=5):
        """Query embedding과 가장 유사한 entity들 찾기"""
        similarities = []
        for entity_name, entity_emb in entity_embeddings.items():
            sim = self.compute_similarity(query_embedding, entity_emb)
            similarities.append((entity_name, sim))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class Paragram(EmbeddingModel):
    def __init__(self, filter_vocab=0, **kwargs):
        super().__init__(**kwargs)
        words={}
        We = []
        We = pickle.load(open("paragram_xxl.pkl", 'rb'))
        words = json.load(open("paragram_xxl_words.json", 'r'))
        self.word2idx = words
        self.idx2word = list(self.word2idx.keys())
        self.embedding_table = We.to(self.device)
        indices = list(self.word2idx.values())

class Watermarker():
    def __init__(self, llm, embedder, inserter, ratio=0.1, iterate=None, device_id=None):
        self.llm = LLM(llm, device_id=device_id)
        self.inserter = LLM(inserter, device_id=device_id)
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoding = tiktoken.get_encoding('cl100k_base')
        print(f"Loading {embedder} embedder...")
        if embedder == 'openai':
            self.embedder = OpenAIEmb(ratio=ratio, device_id=device_id)
        elif embedder == 'nomic':
            self.embedder = NomicEmbed(ratio=ratio, device_id=device_id)
        else:
            raise ValueError
        print("Loaded embedding model.")
        with open(f"prompts/insert.txt", 'r') as f:
            self.watermark_template = f.read()
        self.iterate = iterate

    def get_words(self, text):
        words = self.embedder.get_words(text)
        return words
    
    def insert_watermark(self, text1, max_tokens=600):
        list1 = self.get_words(text1)
        if len(list1) == 0:
            print("No words found in text, returning...")
            return {"text1": text1, "list1": list1, "text2": text1, "list2": list1}
        
        if self.iterate:
            if self.iterate == "v2":
                sublists = [list1[i:i+10] for i in range(0, len(list1), 10)]
                input_res = text1
                for sublist in sublists:
                    init_words_str = ", ".join(sublist)
                    new_prompt = self.watermark_template.format(input_res, init_words_str)
                    sub_presence = 0
                    n_attempts = 0
                    while sub_presence < 0.5:
                        if n_attempts == 3:
                            print(f"Exceeded 3 tries, breaking...sub_presence: {sub_presence}")
                            break
                        input_res = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
                        sub_presence = sum([1 for word in sublist if word.lower() in input_res.lower()]) / len(sublist)
                        n_attempts += 1
                text2 = input_res
                presence = sum([1 for word in list1 if word.lower() in text2.lower()]) / len(list1)
            else:
                raise ValueError
        else:
            init_words_str = ", ".join(list1)
            new_prompt = self.watermark_template.format(text1, init_words_str)
            text2 = self.inserter.generate(new_prompt, max_tokens=max_tokens, temperature=0)
        
        list2 = self.get_words(text2)
        res = {
            "text1": text1,
            "list1": list1,
            "text2": text2,
            "list2": list2
        }
        return res


class LLM():
    def __init__(self, model, device_id=None):
        if "gpt" in model:
            self.model = ChatGPT(model)
        elif model == "llama-3-8b":
            self.model = Llama3_8B(device_id=device_id)
        elif model == "llama-3-8b-chat":
            self.model = Llama3_8B_Chat(device_id=device_id)
        elif model == "mistral-7b-inst":
            self.model = Mistral_7B_Inst(device_id=device_id)
        elif model == "llama-3-70b-chat":
            self.model = Llama3_70B_Chat()
    
    def generate(self, prompt, max_tokens=600, temperature=1.0):
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=float(temperature))


class ChatGPT():
    def __init__(self, llm):
        from config import OPENAI_API_KEY
        self.llm = llm
        print(f"Loading {llm}...")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def obtain_response(self, prompt, max_tokens, temperature, seed=42):
        response = None
        num_attemps = 0
        messages = []
        messages.append({"role": "user", "content": prompt})
        while response is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed)
            except Exception as e:
                if num_attemps == 5:
                    print(f"Attempt {num_attemps} failed, breaking...")
                    return None
                print(e)
                num_attemps += 1
                print(f"Attempt {num_attemps} failed, trying again after 5 seconds...")
                time.sleep(5)
        return response.choices[0].message.content.strip()
    
    def generate(self, prompt, max_tokens, temperature):
        return self.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature)


class Llama3_8B():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Meta-Llama-3-8B", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id,
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return output_text
    


class Llama3_8B_Chat():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        generation_args = {
            "eos_token_id": eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text


class Mistral_7B_Inst():
    def __init__(self, half=False, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if half:
            print("Loading half precision model...")
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto", torch_dtype=torch.float16)
        else:
            if device_id is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
        self.model.eval()
    
    def generate(self,
        prompt,
        min_new_tokens=10,
        max_tokens=512,
        do_sample=True,
        top_k=None,
        top_p=0.9,
        typical_p=None,
        temperature=1.0,
        num_return_sequences=1,
        logits_processor=None
    ):
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        generation_args = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": float(temperature),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_token_id
        }
        if logits_processor:
            generation_args["logits_processor"] = logits_processor
        with torch.inference_mode():
            outputs = self.model.generate(inputs, **generation_args)
        output_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return output_text


class Llama3_70B_Chat():
    def __init__(self, model="meta-llama/Llama-3-70b-chat-hf"):
        from config import TOGETHER_API_KEY
        self.model = model
        self.client = Together(api_key=TOGETHER_API_KEY)
    
    def generate(self, prompt, max_tokens, temperature=1.0):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=max_tokens,
        )
        try:
            return chat_completion.choices[0].message.content.strip()
        except Exception:
            return ""

from sentence_transformers import SentenceTransformer
import torch
import spacy
from functools import lru_cache
from subgraph_construction import subgraph_construction

class KGWatermarker():
    def __init__(self, llm, embedder, inserter, ratio=0.2, iterate=None, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_id}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = LLM(llm, device_id=device_id)
        self.ratio = ratio
        self.topk = 3
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        
        kg_root_path = "/home/wooseok/KG_Mark/kg/processed_wikidata5m"
        kg_entity_path = f"{kg_root_path}/entities.txt"
        kg_relation_path = f"{kg_root_path}/relations.txt"
        kg_triple_path = f"{kg_root_path}/triplets.txt"
        self.constructor = subgraph_construction(self.llm, ratio=ratio, kg_entity_path=kg_entity_path, 
                                                      kg_relation_path=kg_relation_path, kg_triple_path=kg_triple_path, 
                                                      device_id=device_id)
        
        self.entity, self.relation, self.triple = self.constructor.load_kg(kg_entity_path, kg_relation_path, kg_triple_path)
    
    def _make_triples_json_serializable(self, triples_dict):
        # tuple key를 'head|relation|tail' string으로 변환
        return {"|".join(triple): value for triple, value in triples_dict.items()}

    def _make_json_serializable(self, obj):
        import numpy as np
        if isinstance(obj, dict):
            return {self._make_json_serializable(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def build_subgraph_from_text(self, text):
        """
        주어진 텍스트를 기반으로 subgraph 구성
        """
        print(f"Building subgraph for text: {text[:100]}...")
        
        # 1. 키워드 추출 (NER 사용)
        keywords = self.constructor.extract_keywords_with_ner(text)
        print(f"Extracted keywords: {keywords}")
        
        # 2. Entity 매칭
        seed_entities = self.constructor.get_matching_entities(keywords)
        print(f"Matched entities: {len(seed_entities)}")
        
        # 3. Embedding 가져오기
        entity_embeddings = self.constructor.get_kepler_embeddings_for_matched_entities(seed_entities)
        print(f"Entity embeddings: {len(entity_embeddings)}")
        
        # 4. Subgraph 구성
        subgraph_nodes = self.constructor.construct_subgraph_semantic_bridge(
            seed_entities, entity_embeddings, 
            top_k=50, similarity_threshold=0.7, virtual_edge_ratio=0.1
        )
        
        # 5. Subgraph triples 추출
        subgraph_triples = self.constructor.get_subgraph_triples(subgraph_nodes)
        print(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_triples)} triples")
        
        return {
            'keywords': keywords,
            'seed_entities': seed_entities,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_triples': subgraph_triples,
            'entity_embeddings': entity_embeddings
        }
    
    def convert_triple_to_sentence(self, triple):
        """
        Triple을 자연스러운 문장으로 변환
        """
        head, relation, tail = triple
        head_name = self.entity[head]["entity"][0] if head in self.entity else head
        tail_name = self.entity[tail]["entity"][0] if tail in self.entity else tail
        relation_name = self.relation[relation]["name"][0] if relation in self.relation else relation
        
        # 다양한 문장 패턴으로 변환
        patterns = [
            f"{head_name} {relation_name} {tail_name}.",
            f"{head_name} is known for {relation_name} {tail_name}.",
            f"The {relation_name} of {head_name} is {tail_name}.",
            f"{head_name} has {relation_name} {tail_name}."
        ]
        
        return random.choice(patterns)
    
    def conditional_watermarking(self, text, delta=0.5):
        """
        Conditional Watermarking with strength δ
        
        Args:
            text (str): 원본 텍스트
            delta (float): 워터마킹 강도 (0.0 ~ 1.0)
        """
        print(f"Performing conditional watermarking with strength δ={delta}")
        
        # 1. Subgraph 구성
        subgraph_info = self.build_subgraph_from_text(text)
        keywords = subgraph_info['keywords']
        subgraph_triples = subgraph_info['subgraph_triples']
        # subgraph_triples를 JSON serializable하게 변환
        subgraph_info['subgraph_triples'] = self._make_triples_json_serializable(subgraph_triples)
        
        # 2. 텍스트를 문장 단위로 분리
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        modified_sentences = []
        inserted_sentences = []
        
        # 3. 각 문장에 대해 조건부 워터마킹 수행
        for i, sentence in enumerate(sentences):
            modified_sentence = sentence
            
            # 키워드가 포함된 문장인지 확인
            contains_keyword = any(k.lower() in sentence.lower() for k in keywords)
            
            if contains_keyword and random.random() < delta:
                # 키워드가 포함된 문장에 대해 워터마킹 수행
                
                # 3-1. POS Tagging 유지하면서 단어 교체
                modified_sentence = self._watermark_with_pos_preservation(
                    sentence, keywords, subgraph_triples
                )
                
                # 3-2. Subgraph에서 triple 기반 문장 삽입
                triple_sentences = self._insert_triple_based_sentences(
                    sentence, subgraph_triples, delta
                )
                inserted_sentences.extend(triple_sentences)
                
                print(f"Watermarked sentence {i}: {modified_sentence}")
                if triple_sentences:
                    print(f"  Inserted sentences: {triple_sentences}")
            
            modified_sentences.append(modified_sentence)
        
        # 4. 결과 조합
        final_sentences = modified_sentences + inserted_sentences
        watermarked_text = " ".join(final_sentences)
        
        result = {
            "original_text": text,
            "watermarked_text": watermarked_text,
            "keywords": keywords,
            "subgraph_info": subgraph_info,
            "delta": delta,
            "modified_sentences": len([s for s in modified_sentences if s != sentences[modified_sentences.index(s)]]),
            "inserted_sentences": len(inserted_sentences)
        }
        return self._make_json_serializable(result)
    
    def _watermark_with_pos_preservation(self, sentence, keywords, subgraph_triples):
        """
        POS Tagging을 유지하면서 워터마킹 수행
        """
        doc = self.nlp(sentence)
        modified_tokens = []
        
        for token in doc:
            # 키워드와 관련된 토큰인지 확인
            is_keyword_related = any(k.lower() in token.text.lower() for k in keywords)
            
            if is_keyword_related and random.random() < 0.5:  # 50% 확률로 교체
                # Subgraph에서 같은 POS tag를 가진 entity 찾기
                replacement = self._find_pos_compatible_replacement(token, subgraph_triples)
                if replacement:
                    modified_tokens.append(replacement)
                else:
                    modified_tokens.append(token.text)
            else:
                modified_tokens.append(token.text)
        
        return " ".join(modified_tokens)
    
    def _find_pos_compatible_replacement(self, token, subgraph_triples):
        """
        같은 POS tag를 가진 대체 단어 찾기
        """
        candidates = []
        
        for (head, relation, tail) in subgraph_triples.keys():
            # Head와 tail entity에서 같은 POS tag를 가진 것 찾기
            head_name = self.entity[head]["entity"][0] if head in self.entity else head
            tail_name = self.entity[tail]["entity"][0] if tail in self.entity else tail
            
            # POS tag 확인
            head_doc = self.nlp(head_name)
            tail_doc = self.nlp(tail_name)
            
            for entity_doc in [head_doc, tail_doc]:
                for entity_token in entity_doc:
                    if entity_token.pos_ == token.pos_ and entity_token.text.lower() != token.text.lower():
                        candidates.append(entity_token.text)
        
        return random.choice(candidates) if candidates else None
    
    def _insert_triple_based_sentences(self, sentence, subgraph_triples, delta):
        """
        Subgraph의 triple을 기반으로 새로운 문장 삽입
        """
        inserted_sentences = []
        
        # δ 강도에 따라 삽입할 triple 수 결정
        num_insertions = max(1, int(len(subgraph_triples) * delta * 0.1))
        
        # 랜덤하게 triple 선택
        selected_triples = random.sample(list(subgraph_triples.keys()), 
                                      min(num_insertions, len(subgraph_triples)))
        
        for triple in selected_triples:
            if random.random() < delta:  # δ 확률로 삽입
                triple_sentence = self.convert_triple_to_sentence(triple)
                inserted_sentences.append(triple_sentence)
        
        return inserted_sentences
                    
    def insert_watermark(self, prefix, target, max_tokens=600, delta=0.3):    
        combined_text = f"{prefix} {target}"
        
        # Conditional watermarking 수행
        watermark_result = self.conditional_watermarking(combined_text, delta=delta)
        
        # Ensure watermark_result is a dict before using .get()
        if not isinstance(watermark_result, dict):
            if isinstance(watermark_result, (tuple, list)) and len(watermark_result) > 0 and isinstance(watermark_result[0], dict):
                watermark_result = watermark_result[0]
            else:
                raise ValueError(f"Unexpected return type from conditional_watermarking: {type(watermark_result)}")
        res = {
            "original_text": target,
            "watermarked_text": watermark_result.get("watermarked_text"),
            "keywords": watermark_result.get("keywords"),
            "modified_sentences": watermark_result.get("modified_sentences"),
            "inserted_sentences": watermark_result.get("inserted_sentences")
        }
        return res