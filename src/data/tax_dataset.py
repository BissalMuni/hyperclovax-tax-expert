"""
취득세 데이터셋 생성 (CPU에서 실행)
"""

import pandas as pd
import numpy as np
from faker import Faker
import json
from datetime import datetime, timedelta
import random

fake = Faker('ko_KR')

class TaxDatasetGenerator:
    """취득세 학습 데이터셋 생성 클래스"""
    
    def __init__(self):
        self.property_types = [
            '아파트', '단독주택', '다가구주택', '오피스텔', 
            '상가', '토지', '빌라', '연립주택'
        ]
        
        self.regions = [
            '서울', '경기', '인천', '부산', '대구', 
            '광주', '대전', '울산', '세종'
        ]
        
    def generate_property_data(self, n_samples=1000):
        """부동산 거래 데이터 생성"""
        data = []
        
        for _ in range(n_samples):
            property_type = random.choice(self.property_types)
            region = random.choice(self.regions)
            
            # 가격 범위 설정 (지역별 차등)
            if region == '서울':
                price = random.randint(3, 30) * 100000000  # 3억 ~ 30억
            elif region in ['경기', '인천']:
                price = random.randint(2, 15) * 100000000  # 2억 ~ 15억
            else:
                price = random.randint(1, 10) * 100000000  # 1억 ~ 10억
            
            # 면적 설정
            if property_type == '아파트':
                area = random.randint(60, 200)
            elif property_type in ['단독주택', '다가구주택']:
                area = random.randint(100, 300)
            else:
                area = random.randint(20, 150)
            
            data.append({
                'property_type': property_type,
                'region': region,
                'price': price,
                'area': area,
                'is_first_home': random.choice([True, False]),
                'buyer_count': random.randint(1, 3),
                'transaction_date': fake.date_between(
                    start_date='-2y', 
                    end_date='today'
                ).strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)
    
    def calculate_acquisition_tax(self, row):
        """취득세 계산"""
        price = row['price']
        property_type = row['property_type']
        is_first_home = row['is_first_home']
        
        # 간단한 취득세 계산 로직 (실제는 더 복잡)
        if property_type == '아파트':
            if is_first_home and price <= 600000000:  # 6억 이하 첫 주택
                tax_rate = 0.01  # 1%
            elif price <= 900000000:  # 9억 이하
                tax_rate = 0.02  # 2%
            else:
                tax_rate = 0.03  # 3%
        else:
            tax_rate = 0.04  # 기본 4%
        
        acquisition_tax = int(price * tax_rate)
        
        # 지방교육세 (취득세의 10%)
        education_tax = int(acquisition_tax * 0.1)
        
        # 농어촌특별세 (취득세 2% 초과분의 20%)
        if tax_rate > 0.02:
            special_tax = int((price * (tax_rate - 0.02)) * 0.2)
        else:
            special_tax = 0
        
        total_tax = acquisition_tax + education_tax + special_tax
        
        return {
            'acquisition_tax': acquisition_tax,
            'education_tax': education_tax,
            'special_tax': special_tax,
            'total_tax': total_tax,
            'tax_rate': tax_rate
        }
    
    def generate_qa_pairs(self, df):
        """질문-답변 쌍 생성"""
        qa_pairs = []
        
        for _, row in df.iterrows():
            tax_info = self.calculate_acquisition_tax(row)
            
            # 다양한 질문 형식 생성
            questions = [
                f"{row['region']}에 있는 {row['price']:,}원짜리 {row['property_type']}의 취득세는 얼마인가요?",
                f"{row['property_type']} {row['price']:,}원 취득시 내야 할 세금은?",
                f"{row['area']}평 {row['property_type']} 취득세 계산 부탁드립니다. 가격은 {row['price']:,}원입니다.",
            ]
            
            question = random.choice(questions)
            
            answer = f"""취득세 계산 결과입니다:
- 부동산 유형: {row['property_type']}
- 거래금액: {row['price']:,}원
- 적용 세율: {tax_info['tax_rate']*100:.1f}%
- 취득세: {tax_info['acquisition_tax']:,}원
- 지방교육세: {tax_info['education_tax']:,}원
- 농어촌특별세: {tax_info['special_tax']:,}원
- 총 납부세액: {tax_info['total_tax']:,}원"""
            
            qa_pairs.append({
                'instruction': "취득세 전문가로서 정확한 세금 계산을 제공하세요.",
                'input': question,
                'output': answer
            })
        
        return qa_pairs
    
    def save_dataset(self, qa_pairs, output_path='training_data.jsonl'):
        """데이터셋 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in qa_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ 데이터셋 저장 완료: {output_path}")
        print(f"📊 총 {len(qa_pairs)}개의 학습 데이터 생성")

def main():
    """메인 실행 함수"""
    print("🚀 취득세 학습 데이터셋 생성 시작...")
    
    generator = TaxDatasetGenerator()
    
    # 1. 부동산 데이터 생성
    print("📊 부동산 거래 데이터 생성 중...")
    property_df = generator.generate_property_data(n_samples=500)
    
    # 2. QA 쌍 생성
    print("💬 질문-답변 쌍 생성 중...")
    qa_pairs = generator.generate_qa_pairs(property_df)
    
    # 3. 데이터셋 저장
    generator.save_dataset(qa_pairs, 'src/data/training_data.jsonl')
    
    # 샘플 출력
    print("\n📝 생성된 데이터 샘플:")
    print(json.dumps(qa_pairs[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()