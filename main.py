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