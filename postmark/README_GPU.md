# GPU Device Specification Guide

이 코드는 특정 GPU 번호를 지정하여 사용할 수 있도록 수정되었습니다. **멀티 GPU 분산 처리를 방지**하고 지정된 GPU만 사용하도록 설정됩니다.

## 주요 개선사항

### 1. **CUDA_VISIBLE_DEVICES 설정**
- `device_id`를 지정하면 `CUDA_VISIBLE_DEVICES` 환경 변수를 설정하여 해당 GPU만 사용 가능하게 합니다.
- 이를 통해 PyTorch가 다른 GPU에 접근하지 못하도록 제한합니다.

### 2. **device_map="auto" 제거**
- `device_id`가 지정되면 `device_map="auto"` 대신 `.to(device)`를 사용합니다.
- 이는 모델이 여러 GPU에 분산되지 않고 지정된 GPU에만 로드되도록 합니다.

## 사용법

### 1. Command Line에서 GPU 지정

```bash
# GPU 7번만 사용 (다른 GPU 사용 안함)
python kg_watermark.py --device_id 7 --llm llama-3-8b-chat --embedder openai --inserter llama-3-70b-chat

# GPU 0번만 사용
python kg_watermark.py --device_id 0 --llm llama-3-8b-chat --embedder openai --inserter llama-3-70b-chat

# 기본 GPU 사용 (device_id 지정하지 않음)
python kg_watermark.py --llm llama-3-8b-chat --embedder openai --inserter llama-3-70b-chat
```

### 2. Python 코드에서 직접 사용

```python
from models import KGWatermarker

# GPU 7번만 사용
watermarker = KGWatermarker(
    llm="llama-3-8b-chat",
    embedder="openai", 
    inserter="llama-3-70b-chat",
    device_id=7
)

# GPU 0번만 사용
watermarker = KGWatermarker(
    llm="llama-3-8b-chat",
    embedder="openai", 
    inserter="llama-3-70b-chat",
    device_id=0
)
```

### 3. Subgraph Construction에서 사용

```python
from subgraph_construction import subgraph_construction

# GPU 7번만 사용
sg = subgraph_construction(
    llm="llama-3-8b-chat",
    kg_entity_path="entities.txt",
    kg_relation_path="relations.txt", 
    kg_triple_path="triples.txt",
    device_id=7
)
```

## 출력 예시

```bash
# GPU 7번 사용시
Set CUDA_VISIBLE_DEVICES to 7
Available GPUs: 1
Current GPU: 0
GPU Name: NVIDIA A100-SXM4-40GB
Using device: cuda:0
LLM device: cuda:0

# GPU 0번 사용시
Set CUDA_VISIBLE_DEVICES to 0
Available GPUs: 1
Current GPU: 0
GPU Name: NVIDIA A100-SXM4-40GB
Using device: cuda:0
LLM device: cuda:0
```

## 멀티 GPU 방지 메커니즘

### 1. **환경 변수 설정**
```python
if device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
```

### 2. **모델 로딩 방식 변경**
```python
# device_id가 지정된 경우
if device_id is not None:
    self.model = AutoModelForCausalLM.from_pretrained("model_name").to(self.device)
else:
    # 기본 동작 (device_map="auto" 사용)
    self.model = AutoModelForCausalLM.from_pretrained("model_name", device_map="auto")
```

## 지원되는 모델들

다음 모델들이 GPU device 지정을 지원합니다:

- `llama-3-8b`
- `llama-3-8b-chat` 
- `mistral-7b-inst`
- `openai` (embedder)
- `nomic` (embedder)

## 주의사항

1. **GPU 가용성 확인**: `device_id`를 지정하기 전에 해당 GPU가 사용 가능한지 확인하세요.
2. **CUDA 지원**: CUDA가 설치되어 있지 않으면 CPU로 자동 전환됩니다.
3. **메모리 관리**: 지정된 GPU의 메모리 사용량을 모니터링하세요.
4. **환경 변수**: `CUDA_VISIBLE_DEVICES`가 설정되면 다른 프로세스에도 영향을 줄 수 있습니다.

## 문제 해결

1. **GPU 메모리 부족**: 더 작은 배치 사이즈를 사용하거나 다른 GPU를 시도하세요.
2. **CUDA 오류**: CUDA 버전과 PyTorch 버전이 호환되는지 확인하세요.
3. **GPU 인덱스 오류**: `nvidia-smi`로 사용 가능한 GPU 목록을 확인하세요.
4. **여전히 멀티 GPU 사용**: `nvidia-smi`로 실제 GPU 사용량을 확인하고, 필요시 프로세스를 재시작하세요.

## 확인 방법

실행 후 다음 명령어로 GPU 사용량을 확인할 수 있습니다:

```bash
nvidia-smi
```

지정된 GPU만 사용되고 있다면, 해당 GPU의 메모리 사용량만 증가하는 것을 확인할 수 있습니다. 