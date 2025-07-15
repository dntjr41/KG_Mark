import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk import ngrams
import statistics
import openai
import re
import numpy as np
from config import OPENAI_API_KEY

# OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def load_lamma_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading LLaMA-3-8B-Instruct model: {e}")
        return
    return model, tokenizer

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

def calculate_perplexity_llama(text, model, tokenizer):
    """LLaMA를 이용한 Perplexity 계산"""
    if not text:  
        return float('inf')  # 빈 텍스트는 무한대로 설정

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)
    
    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        except Exception as e:
            print(f"Error calculating perplexity (LLaMA): {e}")
            return float('inf')
    return perplexity

def calculate_perplexity_gpt(text, model_name="gpt-3.5-turbo"):
    """GPT API를 사용한 Perplexity 추정"""
    prompt = (
        "Estimate the perplexity score of the following text:\n\n"
        f"{text}\n\n"
        "Provide the perplexity score as a single number."
    )

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )

        result_text = response.choices[0].message.content.strip()
        match = re.search(r"([\d.]+)", result_text)  # 숫자 추출
        if match:
            return float(match.group(1))  # Perplexity 값 반환

    except Exception as e:
        print(f"Error calculating perplexity (GPT API): {e}")

    return float('inf')  # 실패 시 기본값 반환
    
def evaluate_text_with_gpt(original_text, watermarked_text, metric, model_name="gpt-3.5-turbo"):
    """주어진 두 텍스트를 GPT-3.5-turbo를 사용하여 평가"""
    if metric == "perplexity":
        original_score = calculate_perplexity_gpt(original_text, model_name)
        watermarked_score = calculate_perplexity_gpt(watermarked_text, model_name)
        return original_score, watermarked_score
    
    if metric == "log_diversity":
        original_score = calculate_log_diversity(original_text)
        watermarked_score = calculate_log_diversity(watermarked_text)
        return original_score, watermarked_score
    
    if metric == "relevance":
        prompt = (
            f"Compare the following two texts and evaluate their relevance to the intended meaning on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "coherence":
        prompt = (
            f"Compare the coherence of the following two texts on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    elif metric == "interestingness":
        prompt = (
            f"Compare how interesting the following two texts are on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide your answer in the format: [Original: X, Watermarked: Y]."
        )
    else:
        raise ValueError("Invalid metric provided.")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert text evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )
        
    # 응답 텍스트 추출
    result_text = response.choices[0].message.content.strip()
    # 예시: "[Original: 7, Watermarked: 8]"
    try:
        match = re.search(r"\[Original:\s*(\d+),\s*Watermarked:\s*(\d+)\]", result_text)
        if match:
            orig_score = int(match.group(1))
            water_score = int(match.group(2))
            return orig_score, water_score  # 정규화 제거
    except Exception as e:
        print(f"Score extraction failed: {e}")
    return 0, 0

def evaulate_text_with_lamma(original_text, watermarked_text, metric):
    model, tokenizer = load_lamma_model()
    
    if metric == "perplexity":
        original_score = calculate_perplexity_llama(original_text, model, tokenizer)
        watermarked_score = calculate_perplexity_llama(watermarked_text, model, tokenizer)
        return original_score, watermarked_score
    
    if metric == "log_diversity":
        original_score = calculate_log_diversity(original_text)
        watermarked_score = calculate_log_diversity(watermarked_text)
        return original_score, watermarked_score
    
    # 평가를 위한 프롬프트 설계
    if metric == "relevance":
        prompt = (
            f"Compare the following two texts and evaluate their relevance to the intended meaning on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide scores in the format: [Original: X, Watermarked: Y]"
        )
    elif metric == "coherence":
        prompt = (
            f"Compare the coherence of the following two texts on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide scores in the format: [Original: X, Watermarked: Y]"
        )
    elif metric == "interestingness":
        prompt = (
            f"Compare how interesting the following two texts are on a scale from 0 to 10.\n"
            f"Text 1 (Original): {original_text}\n"
            f"Text 2 (Watermarked): {watermarked_text}\n"
            f"Provide scores in the format: [Original: X, Watermarked: Y]"
        )
    else:
        raise ValueError("Invalid metric")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0
        )
    
    # 생성된 텍스트에서 점수 추출
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # print(f"[{metric}] Response: {response}")  # 디버깅용 출력
    try:
        scores = re.findall(r"\[Original: (\d+), Watermarked: (\d+)\]", response)
        if scores:
            orig_score, water_score = map(int, scores[0])
            return min(max(orig_score, 0), 10) / 10, min(max(water_score, 0), 10) / 10
    except Exception as e:
        print(f"Score extraction failed: {e}")
    return 0.0, 0.0  # 기본값

def main():
    # 파일 경로
    file_path = "/home/wooseok/PostMark-main/outputs/test/test_only_change.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_kgw.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_unigram.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_expedit.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_exp.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_blackbox.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_postmark-12.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    check_test = False
    
    if file_path == "/home/wooseok/PostMark-main/outputs/test/test_03_13.jsonl":
        check_test = True
    
    # 모델 선택
    model_name = "gpt"
    # model_name = "llama-3-8b-inst"
    
    # 결과 저장용 딕셔너리
    perplexity_results = {"original": 0, "watermarked": 0}
    llm_perplexity_results = {"original": 0, "watermarked": 0}
    ner_perplexity_results = {"original": 0, "watermarked": 0}
    
    log_diversity_results = {"original": 0, "watermarked": 0}
    llm_log_diversity_results = {"original": 0, "watermarked": 0}
    ner_log_diversity_results = {"original": 0, "watermarked": 0}
    
    relevance_results = {"original_better": 0, "other_better": 0, "tie": 0}
    llm_relevance_results = {"original_better": 0, "other_better": 0, "tie": 0}
    ner_relevance_results = {"original_better": 0, "other_better": 0, "tie": 0}
    
    coherence_results = {"original_better": 0, "other_better": 0, "tie": 0}
    llm_coherence_results = {"original_better": 0, "other_better": 0, "tie": 0}
    ner_coherence_results = {"original_better": 0, "other_better": 0, "tie": 0}
    
    interestingness_results = {"original_better": 0, "other_better": 0, "tie": 0}
    llm_interestingness_results = {"original_better": 0, "other_better": 0, "tie": 0}
    ner_interestingness_results = {"original_better": 0, "other_better": 0, "tie": 0}
    
    total_items = len(data)
    
    for idx, item in enumerate(data):
        print(f"항목 {idx+1} 처리 중...")
        if check_test is True:
            original_text = item.get("original_text", "")
            llm_watermarked_text = item.get("llm_watermarked_text", "")
            ner_watermarked_text = item.get("ner_watermarked_text", "")
        else:
            original_text = item.get("text1", "")
            watermarked_text = item.get("text2", "")

        # Perplexity, Log Diversity 계산 및 저장
        if check_test is True:
            if model_name == "gpt":
                original_perplexity, llm_perplexity = evaluate_text_with_gpt(original_text, llm_watermarked_text, "perplexity")
                original_perplexity, ner_perplexity = evaluate_text_with_gpt(original_text, ner_watermarked_text, "perplexity")
            elif model_name == "llama-3-8b-inst":
                original_perplexity, llm_perplexity = evaulate_text_with_lamma(original_text, llm_watermarked_text, "perplexity")
                original_perplexity, ner_perplexity = evaulate_text_with_lamma(original_text, ner_watermarked_text, "perplexity")
            original_log_diversity = calculate_log_diversity(original_text)
            llm_log_diversity = calculate_log_diversity(llm_watermarked_text)
            ner_log_diversity = calculate_log_diversity(ner_watermarked_text)
            
            perplexity_results["original"] += original_perplexity
            llm_perplexity_results["watermarked"] += llm_perplexity
            ner_perplexity_results["watermarked"] += ner_perplexity
            
            log_diversity_results["original"] += original_log_diversity
            llm_log_diversity_results["watermarked"] += llm_log_diversity
            ner_log_diversity_results["watermarked"] += ner_log_diversity
        else:
            if model_name == "gpt":
                original_perplexity, watermarked_perplexity = evaluate_text_with_gpt(original_text, watermarked_text, "perplexity")
            elif model_name == "llama-3-8b-inst":
                original_perplexity, watermarked_perplexity = evaulate_text_with_lamma(original_text, watermarked_text, "perplexity")
            original_log_diversity = calculate_log_diversity(original_text)
            watermarked_log_diversity = calculate_log_diversity(watermarked_text)
            
            perplexity_results["original"] += original_perplexity
            perplexity_results["watermarked"] += watermarked_perplexity
            
            log_diversity_results["original"] += original_log_diversity
            log_diversity_results["watermarked"] += watermarked_log_diversity
        
        # Relevance, Coherence, Interestingness 평가 및 저장
        if check_test is True:
            if model_name == "gpt":
                original_relevance, llm_relevance = evaluate_text_with_gpt(original_text, llm_watermarked_text, "relevance")
                original_relevance, ner_relevance = evaluate_text_with_gpt(original_text, ner_watermarked_text, "relevance")
                original_coherence, llm_coherence = evaluate_text_with_gpt(original_text, llm_watermarked_text, "coherence")
                original_coherence, ner_coherence = evaluate_text_with_gpt(original_text, ner_watermarked_text, "coherence")
                original_interestingness, llm_interestingness = evaluate_text_with_gpt(original_text, llm_watermarked_text, "interestingness")
                original_interestingness, ner_interestingness = evaluate_text_with_gpt(original_text, ner_watermarked_text, "interestingness")
            elif model_name == "llama-3-8b-inst":
                original_relevance, llm_relevance = evaulate_text_with_lamma(original_text, llm_watermarked_text, "relevance")
                original_relevance, ner_relevance = evaulate_text_with_lamma(original_text, ner_watermarked_text, "relevance")
                original_coherence, llm_coherence = evaulate_text_with_lamma(original_text, llm_watermarked_text, "coherence")
                original_coherence, ner_coherence = evaulate_text_with_lamma(original_text, ner_watermarked_text, "coherence")
                original_interestingness, llm_interestingness = evaulate_text_with_lamma(original_text, llm_watermarked_text, "interestingness")
                original_interestingness, ner_interestingness = evaulate_text_with_lamma(original_text, ner_watermarked_text, "interestingness")
            
            relevance_results["original_better"] += 1 if original_relevance > llm_relevance else 0
            llm_relevance_results["other_better"] += 1 if llm_relevance > original_relevance else 0
            relevance_results["original_better"] += 1 if original_relevance > ner_relevance else 0
            ner_relevance_results["other_better"] += 1 if ner_relevance > original_relevance else 0
            
            coherence_results["original_better"] += 1 if original_coherence > llm_coherence else 0
            llm_coherence_results["other_better"] += 1 if llm_coherence > original_coherence else 0 
            coherence_results["original_better"] += 1 if original_coherence > ner_coherence else 0
            ner_coherence_results["other_better"] += 1 if ner_coherence > original_coherence else 0
            
            interestingness_results["original_better"] += 1 if original_interestingness > llm_interestingness else 0
            llm_interestingness_results["other_better"] += 1 if llm_interestingness > original_interestingness else 0
            interestingness_results["original_better"] += 1 if original_interestingness > ner_interestingness else 0
            ner_interestingness_results["other_better"] += 1 if ner_interestingness > original_interestingness else 0
        else:
            if model_name == "gpt":
                original_relevance, watermarked_relevance = evaluate_text_with_gpt(original_text, watermarked_text, "relevance")
                original_coherence, watermarked_coherence = evaluate_text_with_gpt(original_text, watermarked_text, "coherence")
                original_interestingness, watermarked_interestingness = evaluate_text_with_gpt(original_text, watermarked_text, "interestingness")
            elif model_name == "llama-3-8b-inst":
                original_relevance, watermarked_relevance = evaulate_text_with_lamma(original_text, watermarked_text, "relevance")
                original_coherence, watermarked_coherence = evaulate_text_with_lamma(original_text, watermarked_text, "coherence")
                original_interestingness, watermarked_interestingness = evaulate_text_with_lamma(original_text, watermarked_text, "interestingness")
            
            relevance_results["original_better"] += 1 if original_relevance > watermarked_relevance else 0
            relevance_results["other_better"] += 1 if watermarked_relevance > original_relevance else 0
            
            coherence_results["original_better"] += 1 if original_coherence > watermarked_coherence else 0
            coherence_results["other_better"] += 1 if watermarked_coherence > original_coherence else 0
            
            interestingness_results["original_better"] += 1 if original_interestingness > watermarked_interestingness else 0
            interestingness_results["other_better"] += 1 if watermarked_interestingness > original_interestingness else 0
            
    # 평균 계산
    if check_test is True:
        perplexity_results["original"] /= total_items
        llm_perplexity_results["watermarked"] /= total_items
        ner_perplexity_results["watermarked"] /= total_items
        
        log_diversity_results["original"] /= total_items
        llm_log_diversity_results["watermarked"] /= total_items
        ner_log_diversity_results["watermarked"] /= total_items
        
        relevance_results["original_better"] /= total_items
        llm_relevance_results["other_better"] /= total_items
        ner_relevance_results["other_better"] /= total_items
        
        coherence_results["original_better"] /= total_items
        llm_coherence_results["other_better"] /= total_items
        ner_coherence_results["other_better"] /= total_items
        
        interestingness_results["original_better"] /= total_items
        llm_interestingness_results["other_better"] /= total_items
        ner_interestingness_results["other_better"] /= total_items
        
    else:
        perplexity_results["original"] /= total_items
        perplexity_results["watermarked"] /= total_items
        
        log_diversity_results["original"] /= total_items
        log_diversity_results["watermarked"] /= total_items
        
        relevance_results["original_better"] /= total_items
        relevance_results["other_better"] /= total_items
        
        coherence_results["original_better"] /= total_items
        coherence_results["other_better"] /= total_items
        
        interestingness_results["original_better"] /= total_items
        interestingness_results["other_better"] /= total_items
    
    # 결과 출력
    print("\n=== 결과 비율 (100% 기준) ===")
    if check_test is True:
        print("Perplexity:")
        print(f"Original: {perplexity_results['original']:.2f}")
        print(f"LLM: {llm_perplexity_results['watermarked']:.2f}")
        print(f"NER: {ner_perplexity_results['watermarked']:.2f}")
        
        print("\nLog Diversity:")
        print(f"Original: {log_diversity_results['original']:.2f}")
        print(f"LLM: {llm_log_diversity_results['watermarked']:.2f}")
        print(f"NER: {ner_log_diversity_results['watermarked']:.2f}")
        
        print("\nRelevance:")
        print(f"Original Better: {relevance_results['original_better']:.2f}%")
        print(f"LLM Better: {llm_relevance_results['other_better']:.2f}%")
        print(f"NER Better: {ner_relevance_results['other_better']:.2f}%")
        
        print("\nCoherence:")
        print(f"Original Better: {coherence_results['original_better']:.2f}%")
        print(f"LLM Better: {llm_coherence_results['other_better']:.2f}%")
        print(f"NER Better: {ner_coherence_results['other_better']:.2f}%")
        
        print("\nInterestingness:")
        print(f"Original Better: {interestingness_results['original_better']:.2f}%")
        print(f"LLM Better: {llm_interestingness_results['other_better']:.2f}%")
        print(f"NER Better: {ner_interestingness_results['other_better']:.2f}%")
    else:
        print("Perplexity:")
        print(f"Original: {perplexity_results['original']:.2f}")
        print(f"Watermarked: {perplexity_results['watermarked']:.2f}")
        
        print("\nLog Diversity:")
        print(f"Original: {log_diversity_results['original']:.2f}")
        print(f"Watermarked: {log_diversity_results['watermarked']:.2f}")
        
        print("\nRelevance:")
        print(f"Original Better: {relevance_results['original_better']:.2f}%")
        print(f"Watermarked Better: {relevance_results['other_better']:.2f}%")
        
        print("\nCoherence:")
        print(f"Original Better: {coherence_results['original_better']:.2f}%")
        print(f"Watermarked Better: {coherence_results['other_better']:.2f}%")
        
        print("\nInterestingness:")
        print(f"Original Better: {interestingness_results['original_better']:.2f}%")
        print(f"Watermarked Better: {interestingness_results['other_better']:.2f}%")
    
if __name__ == "__main__":
    main()