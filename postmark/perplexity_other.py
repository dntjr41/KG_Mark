import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk import ngrams
import statistics

# 1. JSONL 파일 읽기 함수
def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 2. Perplexity 계산 함수
def calculate_perplexity(text, model, tokenizer):
    """주어진 텍스트의 Perplexity를 계산"""
    if not text:  # 텍스트가 비어 있는 경우
        return float('inf')  # 무한대로 설정
    
    # LLaMA3 모델에 맞는 입력 처리
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    return perplexity

# 3. Log Diversity 계산 함수
def calculate_log_diversity(text, n=2):
    """주어진 텍스트의 Log Diversity를 계산 (n-gram 기반)"""
    if not text:  # 텍스트가 비어 있는 경우
        return 0
    tokens = text.split()
    n_grams = list(ngrams(tokens, n))
    unique_n_grams = set(n_grams)
    if len(unique_n_grams) == 0:
        return 0
    return math.log(len(unique_n_grams))

# 4. 메인 함수
def main():
    # 파일 경로
    # file_path = "/home/wooseok/PostMark-main/outputs/test/test.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_blackbox.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_exp.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_expedit.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_kgw.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_postmark-12.jsonl"
    file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_unigram.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    
    # LLaMA3 8B 모델과 토크나이저 로드 (모델 이름 지정)
    print("LLaMA3 8B 모델을 로드 중...")
    model_name = "meta-llama/Meta-Llama-3-8B"  # 모델 이름 지정 (캐시 경로 자동 탐색)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # GPU/CPU 자동 매핑
            torch_dtype=torch.float16  # 메모리 효율성을 위해 float16 사용
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading LLaMA3 8B model: {e}")
        print("Please check the model path and ensure all required files are present.")
        return
    
    # LLaMA3 모델의 경우 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 각 텍스트 유형별 Perplexity와 Log Diversity 저장용 리스트
    perplexities_original = []
    perplexities_method = []
    log_diversities_original = []
    log_diversities_method = []
    
    # 각 항목 처리
    for idx, item in enumerate(data):
        print(f"항목 {idx+1} 처리 중...")
        original_text = item.get("text1", "")
        method_text = item.get("text2", "")
        
        perplexity_original = calculate_perplexity(original_text, model, tokenizer)
        perplexity_method = calculate_perplexity(method_text, model, tokenizer)
        log_diversity_original = calculate_log_diversity(original_text)
        log_diversity_method = calculate_log_diversity(method_text)
        
        perplexities_original.append(perplexity_original)
        perplexities_method.append(perplexity_method)
        
        log_diversities_original.append(log_diversity_original)
        log_diversities_method.append(log_diversity_method)
        
    # 평균값 계산
    avg_perplexity_original = statistics.mean(perplexities_original)
    avg_perplexity_method = statistics.mean(perplexities_method)
    
    avg_log_diversity_original = statistics.mean(log_diversities_original)
    avg_log_diversity_method = statistics.mean(log_diversities_method)
    
    # 결과 출력
    print(f"Original Text Perplexity: {avg_perplexity_original}")
    print(f"Method Text Perplexity: {avg_perplexity_method}")
    print(f"Original Text Log Diversity: {avg_log_diversity_original}")
    print(f"Method Text Log Diversity: {avg_log_diversity_method}")
    
# 메인 함수 실행
if __name__ == "__main__":
    main()