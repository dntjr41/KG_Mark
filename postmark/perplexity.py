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
    file_path = "/home/wooseok/PostMark-main/outputs/test/test_only_change.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_blackbox.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_exp.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_expedit.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_kgw.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_postmark-12.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_unigram.jsonl"
    
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
    insert_perplexities_original = []
    insert_perplexities_llm = []
    insert_perplexities_ner = []
    insert_log_diversities_original = []
    insert_log_diversities_llm = []
    insert_log_diversities_ner = []
    
    watermark_perplexities_original = []
    watermark_perplexities_llm = []
    watermark_perplexities_ner = []
    watermark_log_diversities_original = []
    watermark_log_diversities_llm = []
    watermark_log_diversities_ner = []
    
    # 각 항목 처리
    for idx, item in enumerate(data):
        print(f"항목 {idx+1} 처리 중...")
        original_text = item.get("original_text", "")
        llm_inserted_text = item.get("llm_insertion_only", "")
        ner_inserted_text = item.get("ner_insertion_only", "")
        
        llm_watermarked_text = item.get("llm_watermarked_only", "")
        ner_watermarked_text = item.get("ner_watermarked_only", "")
        
        # Perplexity 계산
        insert_perplexity_original = calculate_perplexity(original_text, model, tokenizer)
        insert_perplexity_llm = calculate_perplexity(llm_inserted_text, model, tokenizer)
        insert_perplexity_ner = calculate_perplexity(ner_inserted_text, model, tokenizer)
        
        watermark_perplexity_original = calculate_perplexity(original_text, model, tokenizer)
        watermark_perplexity_llm = calculate_perplexity(llm_watermarked_text, model, tokenizer)
        watermark_perplexity_ner = calculate_perplexity(ner_watermarked_text, model, tokenizer)
        
        # Log Diversity 계산 (n=2)
        insert_log_diversity_original = calculate_log_diversity(original_text)
        insert_log_diversity_llm = calculate_log_diversity(llm_inserted_text)
        insert_log_diversity_ner = calculate_log_diversity(ner_inserted_text)
        
        watermark_log_diversity_original = calculate_log_diversity(original_text)
        watermark_log_diversity_llm = calculate_log_diversity(llm_watermarked_text)
        watermark_log_diversity_ner = calculate_log_diversity(ner_watermarked_text)
            
        insert_perplexities_original.append(insert_perplexity_original)
        insert_perplexities_llm.append(insert_perplexity_llm)
        insert_perplexities_ner.append(insert_perplexity_ner)
        
        watermark_perplexities_original.append(watermark_perplexity_original)
        watermark_perplexities_llm.append(watermark_perplexity_llm)
        watermark_perplexities_ner.append(watermark_perplexity_ner)
        
        insert_log_diversities_original.append(insert_log_diversity_original)
        insert_log_diversities_llm.append(insert_log_diversity_llm)
        insert_log_diversities_ner.append(insert_log_diversity_ner)
        
        watermark_log_diversities_original.append(watermark_log_diversity_original)
        watermark_log_diversities_llm.append(watermark_log_diversity_llm)
        watermark_log_diversities_ner.append(watermark_log_diversity_ner)
        
    # 평균값 계산
    avg_insert_perplexity_original = statistics.mean(insert_perplexities_original)
    avg_insert_perplexity_llm = statistics.mean(insert_perplexities_llm)
    avg_insert_perplexity_ner = statistics.mean(insert_perplexities_ner)
    
    avg_watermark_perplexity_original = statistics.mean(watermark_perplexities_original)
    avg_watermark_perplexity_llm = statistics.mean(watermark_perplexities_llm)
    avg_watermark_perplexity_ner = statistics.mean(watermark_perplexities_ner)
    
    avg_insert_log_diversity_original = statistics.mean(insert_log_diversities_original)
    avg_insert_log_diversity_llm = statistics.mean(insert_log_diversities_llm)
    avg_insert_log_diversity_ner = statistics.mean(insert_log_diversities_ner)
    
    avg_watermark_log_diversity_original = statistics.mean(watermark_log_diversities_original)
    avg_watermark_log_diversity_llm = statistics.mean(watermark_log_diversities_llm)
    avg_watermark_log_diversity_ner = statistics.mean(watermark_log_diversities_ner)
    
    # 결과 출력 (평균값)
    print("\n=== 평균 결과 ===")
    print(f"Average Original Insert Perplexity: {avg_insert_perplexity_original:.4f}")
    print(f"Average LLM Insert Perplexity: {avg_insert_perplexity_llm:.4f}")
    print(f"Average NER Insert Perplexity: {avg_insert_perplexity_ner:.4f}")
    
    print(f"Average Original Watermark Perplexity: {avg_watermark_perplexity_original:.4f}")
    print(f"Average LLM Watermark Perplexity: {avg_watermark_perplexity_llm:.4f}")
    print(f"Average NER Watermark Perplexity: {avg_watermark_perplexity_ner:.4f}")
    
    print(f"Average Original Insert Log Diversity: {avg_insert_log_diversity_original:.4f}")
    print(f"Average LLM Insert Log Diversity: {avg_insert_log_diversity_llm:.4f}")
    print(f"Average NER Insert Log Diversity: {avg_insert_log_diversity_ner:.4f}")
    
    print(f"Average Original Watermark Log Diversity: {avg_watermark_log_diversity_original:.4f}")
    print(f"Average LLM Watermark Log Diversity: {avg_watermark_log_diversity_llm:.4f}")
    print(f"Average NER Watermark Log Diversity: {avg_watermark_log_diversity_ner:.4f}")
    
if __name__ == "__main__":
    main()
    
#     # 각 텍스트 유형별 Perplexity와 Log Diversity 저장용 리스트
#     perplexities_original = []
#     perplexities_llm = []
#     perplexities_ner = []
#     log_diversities_original = []
#     log_diversities_llm = []
#     log_diversities_ner = []
    
#     # 각 항목 처리
#     for idx, item in enumerate(data):
#         print(f"항목 {idx+1} 처리 중...")
#         original_text = item.get("original_text", "")
#         llm_watermarked = item.get("llm_watermarked_text", "")
#         ner_watermarked = item.get("ner_watermarked_text", "")
        
#         # Perplexity 계산
#         perplexity_original = calculate_perplexity(original_text, model, tokenizer)
#         perplexity_llm = calculate_perplexity(llm_watermarked, model, tokenizer)
#         perplexity_ner = calculate_perplexity(ner_watermarked, model, tokenizer)
        
#         # Log Diversity 계산 (n=2)
#         log_diversity_original = calculate_log_diversity(original_text, n=2)
#         log_diversity_llm = calculate_log_diversity(llm_watermarked, n=2)
#         log_diversity_ner = calculate_log_diversity(ner_watermarked, n=2)
        
#         # 유효한 값만 리스트에 추가 (무한대 제외)
#         if perplexity_original != float('inf'):
#             perplexities_original.append(perplexity_original)
#         if perplexity_llm != float('inf'):
#             perplexities_llm.append(perplexity_llm)
#         if perplexity_ner != float('inf'):
#             perplexities_ner.append(perplexity_ner)
        
#         log_diversities_original.append(log_diversity_original)
#         log_diversities_llm.append(log_diversity_llm)
#         log_diversities_ner.append(log_diversity_ner)
    
#     # 평균값 계산
#     avg_perplexity_original = statistics.mean(perplexities_original) if perplexities_original else float('inf')
#     avg_perplexity_llm = statistics.mean(perplexities_llm) if perplexities_llm else float('inf')
#     avg_perplexity_ner = statistics.mean(perplexities_ner) if perplexities_ner else float('inf')
    
#     avg_log_diversity_original = statistics.mean(log_diversities_original) if log_diversities_original else 0
#     avg_log_diversity_llm = statistics.mean(log_diversities_llm) if log_diversities_llm else 0
#     avg_log_diversity_ner = statistics.mean(log_diversities_ner) if log_diversities_ner else 0
    
#     # 결과 출력 (평균값)
#     print("\n=== 평균 결과 ===")
#     print(f"Average Original Perplexity: {avg_perplexity_original:.4f}")
#     print(f"Average LLM Perplexity: {avg_perplexity_llm:.4f}")
#     print(f"Average NER Perplexity: {avg_perplexity_ner:.4f}")
#     print(f"Average Original Log Diversity: {avg_log_diversity_original:.4f}")
#     print(f"Average LLM Log Diversity: {avg_log_diversity_llm:.4f}")
#     print(f"Average NER Log Diversity: {avg_log_diversity_ner:.4f}")

# if __name__ == "__main__":
#     main()