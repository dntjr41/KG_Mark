import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import statistics
import re

# 1. JSONL 파일 읽기 함수
def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 2. LLaMA를 이용한 텍스트 평가 함수
def evaluate_text_with_llama(original_text, watermarked_text, model, tokenizer, metric):
    """LLaMA로 Original과 Watermarked 텍스트를 비교 평가"""
    if not original_text or not watermarked_text:
        return 0.0, 0.0
    
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

# 3. 메인 함수
def main():
    # 파일 경로
    file_path = "/home/wooseok/PostMark-main/outputs/test/test.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    
    # LLaMA-3-8B-Instruct 모델 로드
    print("LLaMA-3-8B-Instruct 모델을 로드 중...")
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
    
    # 결과 저장용 딕셔너리
    relevance_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    relevance_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    coherence_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    coherence_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    interestingness_llm_result = {"original_better": 0, "llm_better": 0, "tie": 0}
    interestingness_ner_result = {"original_better": 0, "ner_better": 0, "tie": 0}
    
    total_items = len(data)
    
    for idx, item in enumerate(data):
        original_text = item.get("original_text")
        llm_text = item.get("llm_watermarked_text")
        ner_text = item.get("ner_watermarked_text")
        
        relevance_original_llm, relevance_llm = evaluate_text_with_llama(original_text, llm_text, model, tokenizer, "relevance")
        relevance_original_ner, relevance_ner = evaluate_text_with_llama(original_text, ner_text, model, tokenizer, "relevance")
        
        coherence_original_llm, coherence_llm = evaluate_text_with_llama(original_text, llm_text, model, tokenizer, "coherence")
        coherence_original_ner, coherence_ner = evaluate_text_with_llama(original_text, ner_text, model, tokenizer, "coherence")
        
        interestingness_original_llm, interestingness_llm = evaluate_text_with_llama(original_text, llm_text, model, tokenizer, "interestingness")
        interestingness_original_ner, interestingness_ner = evaluate_text_with_llama(original_text, ner_text, model, tokenizer, "interestingness")
        
        # Relevance
        if relevance_original_llm > relevance_llm:
            relevance_llm_result["original_better"] += 1
        elif relevance_original_llm < relevance_llm:
            relevance_llm_result["llm_better"] += 1
        else:
            relevance_llm_result["tie"] += 1
            
        if relevance_original_ner > relevance_ner:
            relevance_ner_result["original_better"] += 1
        elif relevance_original_ner < relevance_ner:
            relevance_ner_result["ner_better"] += 1
        else:
            relevance_ner_result["tie"] += 1
            
        # Coherence
        if coherence_original_llm > coherence_llm:
            coherence_llm_result["original_better"] += 1
        elif coherence_original_llm < coherence_llm:
            coherence_llm_result["llm_better"] += 1
        else:
            coherence_llm_result["tie"] += 1
        
        if coherence_original_ner > coherence_ner:
            coherence_ner_result["original_better"] += 1
        elif coherence_original_ner < coherence_ner:
            coherence_ner_result["ner_better"] += 1
        else:
            coherence_ner_result["tie"] += 1
            
        # Interestingness
        if interestingness_original_llm > interestingness_llm:
            interestingness_llm_result["original_better"] += 1
        elif interestingness_original_llm < interestingness_llm:
            interestingness_llm_result["llm_better"] += 1
        else:
            interestingness_llm_result["tie"] += 1
        
        if interestingness_original_ner > interestingness_ner:
            interestingness_ner_result["original_better"] += 1
        elif interestingness_original_ner < interestingness_ner:
            interestingness_ner_result["ner_better"] += 1
        else:
            interestingness_ner_result["tie"] += 1
        
    # 비율 계산 및 출력
    print("\n=== 결과 비율 (100% 기준) ===")
    print("Relevance:")
    print(f"LLM Original Better: {relevance_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {relevance_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {relevance_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {relevance_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {relevance_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {relevance_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nCoherence:")
    print(f"LLM Original Better: {coherence_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {coherence_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {coherence_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {coherence_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {coherence_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {coherence_ner_result['tie'] / total_items * 100:.2f}%")
    
    print("\nInterestingness:")
    print(f"LLM Original Better: {interestingness_llm_result['original_better'] / total_items * 100:.2f}%")
    print(f"LLM Better: {interestingness_llm_result['llm_better'] / total_items * 100:.2f}%")
    print(f"LLM Tie: {interestingness_llm_result['tie'] / total_items * 100:.2f}%")
    
    print(f"NER Original Better: {interestingness_ner_result['original_better'] / total_items * 100:.2f}%")
    print(f"NER Better: {interestingness_ner_result['ner_better'] / total_items * 100:.2f}%")
    print(f"NER Tie: {interestingness_ner_result['tie'] / total_items * 100:.2f}%")
    
if __name__ == "__main__":
    main()