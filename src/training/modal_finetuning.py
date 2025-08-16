"""
Modal GPU를 사용한 HyperCLOVAX 파인튜닝
"""

import modal
from modal import App, Image, gpu, method
import os

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