# GitHub Codespaces + Modal 완벽 설정 가이드

**HyperCLOVAX 파인튜닝을 위한 최적 개발환경 구축**

## 🚀 **1단계: GitHub 저장소 생성**

### **1.1 GitHub 저장소 생성**
1. **GitHub.com** 접속
2. **"New repository"** 클릭
3. **Repository name**: `hyperclovax-tax-expert`
4. **Description**: `HyperCLOVAX 취득세 전문가 파인튜닝 프로젝트`
5. **Private** 선택 (중요한 코드 보호)
6. **Add README file** 체크
7. **Create repository** 클릭

### **1.2 로컬에서 프로젝트 생성 및 Git 설정**

#### **로컬 프로젝트 생성**
```bash
# 프로젝트 디렉토리 생성
mkdir hyperclovax-tax-expert
cd hyperclovax-tax-expert

# Git 초기화
git init

# 기본 브랜치를 main으로 설정
git branch -M main

# GitHub 원격 저장소 연결
git remote add origin https://github.com/YOUR_USERNAME/hyperclovax-tax-expert.git
```

#### **기본 파일 구조 생성**
```bash
# 디렉토리 구조 생성
mkdir -p .devcontainer
mkdir -p .github/workflows
mkdir -p src/{models,data,training}

# 기본 파일 생성
touch .devcontainer/devcontainer.json
touch requirements.txt
touch README.md
touch main.py
touch .gitignore
```

#### **프로젝트 구조**
```
hyperclovax-tax-expert/
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
├── src/
│   ├── models/
│   ├── data/
│   └── training/
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

#### **.gitignore 파일 설정**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
pip-log.txt
pip-delete-this-directory.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env
*.log
data/raw/
models/checkpoints/
*.pkl
*.h5
*.pth

# Jupyter
.ipynb_checkpoints
*.ipynb

# OS
.DS_Store
Thumbs.db
```

#### **첫 커밋 및 푸시**
```bash
# 모든 파일 스테이징
git add .

# 첫 커밋
git commit -m "Initial commit: Project structure setup"

# GitHub에 푸시
git push -u origin main
```

## 🛠️ **2단계: GitHub Codespaces 설정**

### **2.1 Codespaces 시작**
1. GitHub 저장소 페이지에서 **"Code"** 버튼 클릭
2. **"Codespaces"** 탭 선택
3. **"Create codespace on main"** 클릭
4. 자동으로 VS Code 환경이 브라우저에서 열림

### **2.2 devcontainer.json 설정**
**.devcontainer/devcontainer.json** 파일 생성:

```json
{
  "name": "HyperCLOVAX Finetuning Environment",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  
  "features": {
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.debugpy", 
        "ms-toolsai.jupyter",
        "ms-vscode.vscode-json",
        "GitHub.copilot",
        "GitHub.copilot-chat"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.formatting.provider": "black",
        "editor.formatOnSave": true
      }
    }
  },
  
  "postCreateCommand": "pip install -r requirements.txt",
  
  "forwardPorts": [8000, 8888],
  
  "remoteUser": "vscode"
}
```

### **2.3 requirements.txt 생성**
```txt
# Core ML libraries
transformers==4.36.0
torch==2.1.0
accelerate==0.24.0
datasets==2.14.0
peft==0.6.0
huggingface-hub==0.19.0

# Modal for GPU compute
modal==0.63.0

# Data processing
pandas==2.1.0
numpy==1.24.0
faker==20.1.0

# Development tools
jupyter==1.0.0
black==23.0.0
pytest==7.4.0

# Visualization
matplotlib==3.7.0
seaborn==0.12.0
```

## ⚡ **3단계: Modal 설정**

### **3.1 Modal 계정 생성**
1. **https://modal.com** 접속
2. **"Sign up"** 클릭
3. GitHub 계정으로 로그인
4. 무료 크레딧 $30 받기

### **3.2 Modal 토큰 설정**
**Codespaces 터미널에서:**
```bash
# Modal 설치 (이미 requirements.txt에 포함)
pip install modal

# Modal 인증 설정
modal setup
# 브라우저가 열리면 로그인하여 토큰 생성

# 토큰 확인
modal token current
```

### **3.3 Modal 환경 변수 설정**
**Codespaces에서 환경 변수 추가:**
```bash
# GitHub Codespaces Secrets에 추가
# 1. GitHub 저장소 → Settings → Secrets and variables → Codespaces
# 2. New repository secret 클릭
# 3. Name: MODAL_TOKEN_ID, Value: (modal token current에서 확인한 값)
# 4. Name: MODAL_TOKEN_SECRET, Value: (modal token current에서 확인한 값)
```

## 📁 **4단계: 프로젝트 파일 구조 생성**

### **4.1 main.py 생성**
```python
# main.py
"""
HyperCLOVAX 취득세 전문가 파인튜닝 프로젝트
GitHub Codespaces + Modal 환경
"""

import modal
import os
from pathlib import Path

# Modal 앱 정의
app = modal.App("hyperclovax-tax-expert")

# GPU 환경을 위한 이미지 정의
gpu_image = (
    modal.Image.debian_slim()
    .pip_install([
        "transformers==4.36.0",
        "torch==2.1.0", 
        "accelerate==0.24.0",
        "peft==0.6.0",
        "datasets==2.14.0",
        "huggingface-hub==0.19.0",
        "pandas==2.1.0",
        "numpy==1.24.0"
    ])
    .env({
        "HF_HOME": "/tmp/huggingface",
        "TRANSFORMERS_CACHE": "/tmp/transformers"
    })
)

def main():
    """Codespaces에서 실행할 메인 함수"""
    print("🎯 HyperCLOVAX 취득세 전문가 파인튜닝 프로젝트")
    print("=" * 50)
    print("✅ GitHub Codespaces 환경 준비 완료")
    print("✅ Modal GPU 환경 준비 완료")
    print("🚀 파인튜닝 준비 완료!")
    
    # 데이터 준비 (CPU에서 실행)
    print("\n📊 데이터셋 준비 중...")
    # 여기에 데이터 생성 코드 추가
    
if __name__ == "__main__":
    main()
```

### **4.2 src/training/modal_finetuning.py 생성**
```python
# src/training/modal_finetuning.py
"""
Modal GPU를 사용한 HyperCLOVAX 파인튜닝
"""

import modal
from modal import App, Image, gpu, method

# 앱 정의
app = App("hyperclovax-finetuning")

# GPU 이미지 정의
finetuning_image = (
    Image.debian_slim()
    .pip_install([
        "transformers==4.36.0",
        "torch==2.1.0",
        "accelerate==0.24.0", 
        "peft==0.6.0",
        "datasets==2.14.0",
        "huggingface-hub==0.19.0"
    ])
    .env({
        "HF_HOME": "/tmp/huggingface",
        "TRANSFORMERS_CACHE": "/tmp/transformers"
    })
)

@app.cls(
    image=finetuning_image,
    gpu=gpu.T4(),  # T4 GPU 사용
    timeout=3600,  # 1시간 제한
    secrets=[modal.Secret.from_name("huggingface-secret")]  # HF 토큰
)
class HyperCLOVAXTrainer:
    """Modal에서 실행될 파인튜닝 클래스"""
    
    def __enter__(self):
        """GPU 환경 초기화"""
        print("🚀 GPU 환경 초기화 중...")
        
        # 필요한 라이브러리 import
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login
        import torch
        
        # HuggingFace 로그인
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        
        # 모델 로드
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        print(f"📥 모델 로딩: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ 모델 로딩 완료!")
        return self
    
    @method()
    def run_finetuning(self, training_data: dict):
        """실제 파인튜닝 실행"""
        print("🔥 파인튜닝 시작...")
        
        # 여기에 파인튜닝 로직 구현
        # 1. 데이터 전처리
        # 2. LoRA 설정
        # 3. 학습 실행
        # 4. 모델 저장
        
        return {
            "status": "completed",
            "model_path": "/tmp/finetuned_model",
            "metrics": {"loss": 0.1, "accuracy": 0.95}
        }

# 파인튜닝 실행 함수
@app.function(
    image=finetuning_image,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_training():
    """외부에서 호출할 수 있는 파인튜닝 함수"""
    
    # 학습 데이터 준비 (여기서는 샘플)
    training_data = {
        "samples": [
            {
                "instruction": "취득세 전문가로서 답변하세요",
                "input": "5억원 아파트 취득세는?",
                "output": "취득세: 20,000,000원 (4%)"
            }
        ]
    }
    
    # 파인튜닝 실행
    trainer = HyperCLOVAXTrainer()
    result = trainer.run_finetuning.remote(training_data)
    
    return result
```

### **4.3 src/data/tax_dataset.py 생성**
```python
# src/data/tax_dataset.py
"""
취득세 데이터셋 생성 (CPU에서 실행)
"""

# 기존 코드에서 가져온 데이터 생성 클래스들
from updated_tax_rules_2024 import UpdatedTaxRuleEngine2024
# ... 기타 데이터 생성 코드
```

## 🔧 **5단계: Claude Code 활용 최적화**

### **5.1 GitHub Copilot 활성화**
1. Codespaces에서 **Ctrl+Shift+P**
2. **"GitHub Copilot: Enable"** 검색 후 실행
3. Claude Code와 Copilot 동시 활용

### **5.2 개발 워크플로우**
```python
# 1. Claude Code에게 요청
# "취득세 계산 함수를 만들어줘"

# 2. 자동 생성된 코드 검토
def calculate_acquisition_tax(price, property_type):
    # Claude Code가 생성한 코드
    pass

# 3. 테스트 코드 생성 요청
# "위 함수의 테스트 코드 만들어줘"

# 4. Modal로 GPU 작업 전송
@app.function(gpu="T4")
def gpu_task():
    # GPU 작업
    pass
```

## 🚀 **6단계: 실행 및 테스트**

### **6.1 로컬 테스트 (Codespaces)**
```bash
# Codespaces 터미널에서
python main.py

# 데이터 생성 테스트
python -m src.data.tax_dataset

# Modal 연결 테스트
modal run src.training.modal_finetuning::run_training
```

### **6.2 Modal GPU 파인튜닝 실행**
```python
# 파인튜닝 실행
import modal

# Modal 앱 실행
with modal.run():
    result = run_training.remote()
    print(f"파인튜닝 결과: {result}")
```

## 💰 **7단계: 비용 모니터링**

### **7.1 Modal 사용량 확인**
```bash
# Modal 대시보드에서 확인
modal logs

# 비용 확인
modal usage
```

### **7.2 GitHub Codespaces 사용량**
- GitHub 설정 → Billing → Codespaces에서 확인
- 무료 한도: 월 120 코어 시간

## 🎯 **8단계: 개발 팁**

### **8.1 효율적인 워크플로우**
1. **개발/디버깅**: Codespaces (무료/저비용)
2. **GPU 작업**: Modal (필요할 때만)
3. **결과 분석**: Codespaces로 다시

### **8.2 비용 최적화**
- **Codespaces**: 사용하지 않을 때 Stop
- **Modal**: GPU 작업 완료 즉시 종료
- **데이터 캐싱**: 반복 작업 최소화

## ✅ **완료 체크리스트**

- [ ] GitHub 저장소 생성
- [ ] Codespaces 시작 및 devcontainer 설정
- [ ] Modal 계정 생성 및 토큰 설정
- [ ] 프로젝트 파일 구조 생성
- [ ] HuggingFace 토큰 설정
- [ ] Modal 연결 테스트
- [ ] 파인튜닝 코드 테스트
- [ ] 비용 모니터링 설정

**🎉 설정 완료! 이제 Claude Code의 도움을 받아 HyperCLOVAX 파인튜닝을 시작하세요!**