import openai
import re
import json
import torch
import statistics
from config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 1. JSONL 파일 읽기 함수
def load_jsonl(file_path):
    """JSONL 파일을 읽어 리스트로 반환"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def evaluate_text_with_gpt(original_text, watermarked_text, metric, model_name="gpt-3.5-turbo"):
    """
    주어진 원본 텍스트와 워터마크 텍스트의 metric(평가 기준: 'relevance', 'coherence', 'interestingness')을 비교 평가.
    모델은 model_name 인자를 통해 선택 가능 (예: "gpt-3.5-turbo" 또는 "gpt-4").
    """
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

# 3. 메인 함수
def main():
    # 파일 경로
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_kgw.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_unigram.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_expedit.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_exp.jsonl"
    # file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_blackbox.jsonl"
    file_path = "/home/wooseok/PostMark-main/outputs/opengen/llama-3-8b-inst_postmark-12.jsonl"
    
    # 데이터 로드
    print("JSONL 파일을 로드 중...")
    data = load_jsonl(file_path)
    
    # 결과 저장용 딕셔너리 (동점 추가)
    relevance_results = {"original_better": 0, "other_better": 0, "tie": 0}
    coherence_results = {"original_better": 0, "other_better": 0, "tie": 0}
    interestingness_results = {"original_better": 0, "other_better": 0, "tie": 0}
    
    total_items = len(data)
    
    # 각 항목 처리
    for idx, item in enumerate(data):
        # print(f"항목 {idx+1} 처리 중...")
        original_text = item.get("text1", "")
        other_text = item.get("text2", "")
        
        # Relevance 평가
        relevance_original, relevance_other = evaluate_text_with_gpt(original_text, other_text, "relevance")
        
        # Coherence 평가
        coherence_original, coherence_other = evaluate_text_with_gpt(original_text, other_text, "coherence")
        
        # Interestingness 평가
        interestingness_original, interestingness_other = evaluate_text_with_gpt(original_text, other_text, "interestingness")
        
        # Relevance 비교
        if relevance_other > relevance_original:
            relevance_results["other_better"] += 1
        elif relevance_original > relevance_other:
            relevance_results["original_better"] += 1
        else:
            relevance_results["tie"] += 1
            
        # Coherence 비교
        if coherence_other > coherence_original:
            coherence_results["other_better"] += 1
        elif coherence_original > coherence_other:
            coherence_results["original_better"] += 1
        else:
            coherence_results["tie"] += 1
            
        # Interestingness 비교
        if interestingness_other > interestingness_original:
            interestingness_results["other_better"] += 1
        elif interestingness_original > interestingness_other:
            interestingness_results["original_better"] += 1
        else:
            interestingness_results["tie"] += 1
    
    # 비율 계산 및 출력
    print("\n=== 결과 비율 (100% 기준) ===")
    print("Relevance:")
    print(f"Original Better: {relevance_results['original_better'] / total_items * 100:.2f}%")
    print(f"Other Better: {relevance_results['other_better'] / total_items * 100:.2f}%")
    print(f"Tie: {relevance_results['tie'] / total_items * 100:.2f}%")
    
    print("\nCoherence:")
    print(f"Original Better: {coherence_results['original_better'] / total_items * 100:.2f}%")
    print(f"Other Better: {coherence_results['other_better'] / total_items * 100:.2f}%")
    print(f"Tie: {coherence_results['tie'] / total_items * 100:.2f}%")
    
    print("\nInterestingness:")
    print(f"Original Better: {interestingness_results['original_better'] / total_items * 100:.2f}%")
    print(f"Other Better: {interestingness_results['other_better'] / total_items * 100:.2f}%")
    print(f"Tie: {interestingness_results['tie'] / total_items * 100:.2f}%")

if __name__ == "__main__":
    main()