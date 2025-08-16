# HyperCLOVAX 취득세 전문가 파인튜닝 프로젝트

HyperCLOVAX를 활용한 취득세 전문가 AI 모델 파인튜닝 프로젝트입니다.

## 프로젝트 구조

```
hyperclovax-tax-expert/
├── .devcontainer/      # GitHub Codespaces 설정
├── .github/            # GitHub Actions 워크플로우
├── src/
│   ├── models/         # 모델 관련 코드
│   ├── data/           # 데이터 처리 및 생성
│   └── training/       # 학습 및 파인튜닝
├── requirements.txt    # Python 의존성
└── main.py            # 메인 실행 파일
```

## 환경 설정

### 로컬 개발 환경
```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 가상 환경 활성화 (Mac/Linux)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### GitHub Codespaces
1. GitHub 저장소에서 "Code" > "Codespaces" 클릭
2. "Create codespace on main" 선택
3. 자동으로 개발 환경 구성

### Modal GPU 환경
```bash
# Modal 설치
pip install modal

# Modal 인증
modal setup

# GPU 작업 실행
modal run src/training/modal_finetuning.py
```

## 실행 방법

```bash
# 메인 프로그램 실행
python main.py

# 데이터 생성
python -m src.data.tax_dataset

# 파인튜닝 실행 (Modal GPU)
modal run src.training.modal_finetuning::run_training
```

## 기술 스택

- **LLM**: HyperCLOVAX-SEED-Text-Instruct-1.5B
- **Framework**: Transformers, PEFT (LoRA)
- **GPU Compute**: Modal
- **Development**: GitHub Codespaces
- **Language**: Python 3.11

## License

Private Repository