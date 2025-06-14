# SCINet_Custom
Customized SCINet for forecasting financial volatility using macroeconomic features.

SCINet 모델 커스터마이징 및 실행을 위한 코드 수정

SCINet 공식 저장소(https://github.com/cure-lab/SCINet)를 기반으로 사용자 정의 변동성 데이터셋을 적용하여 모델을 학습 및 예측하기 위한 코드 수정

1. 목적

SCINet 모델을 이용하여 사용자 정의 시계열 데이터 (Volatility) 예측 수행

기존 공식 코드는 ILI (Influenza-like Illness) 데이터 전용으로 설계되어 있어, 다음과 같이 커스터마이징

2. 주요 수정 사항 개요

파일명

수정 목적

주요 변경 내용

data_provider.py

사용자 CSV/Excel 데이터 로딩 지원

pandas 기반 커스텀 데이터 로딩 함수 추가 및 시계열 인덱싱

exp/exp_main.py

실험 실행 및 파라미터 설정

인자 파싱 수정 (--data custom, --target Volatility 등)

data/data_loader.py

커스텀 features 선택 반영

target 컬럼 및 feature 구성 방식 변경

models/SCINet.py

출력 shape 조정

output dimension 자동 처리 및 예측 horizon 반영

train.py

학습/검증 로직 수정

커스텀 loss 저장 및 결과 기록 형식 조정

3. 상세 변경 사항 설명 (수정 전/후 diff 포함)

3.1 data_provider.py

변경 후:

elif args.data == 'custom':
    from data.custom_dataset import Dataset_Custom
    Data = Dataset_Custom

커스텀 데이터셋 로더 등록

3.2 data/custom_dataset.py (신규 작성)

주요 내용:

class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, target='Volatility', ...):
        self.data = pd.read_csv(os.path.join(root_path, data_path))
        self.target = self.data[target].values
        # 필요한 features만 지정

기존 데이터셋 클래스 패턴을 그대로 따르되, 사용자 정의 target 및 feature 반영

3.3 exp/exp_main.py

변경 후:

parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--target', type=str, default='Volatility')

setting = '{}_sl{}_pl{}_target{}'.format(args.model, args.seq_len, args.pred_len, args.target)

3.4 models/SCINet.py

변경 후:

self.projection = nn.Linear(channels, output_length)

출력 길이를 예측 horizon에 맞춰 자동 조정

out = self.projection(out)

3.5 train.py

변경 후:

mse = criterion(outputs, batch_y)
mae = torch.mean(torch.abs(outputs - batch_y))

MAE 병렬 계산 추가 및 로그 저장 형태 개선

4. 입력 데이터 구성 예시

입력 데이터: MK2000_with_macro_and_volatility.csv

주요 열:

Date: 날짜

Volatility: 예측 대상

open, high, low, close, volume, return, macro vars: feature로 사용

seq_len, pred_len, label_len 조정 필요 (예: 336, 96)

5. 실행 커맨드 예시

python -u run.py \
  --is_training 1 \
  --data custom \
  --root_path ./dataset/ \
  --data_path MK2000_with_macro_and_volatility.csv \
  --model SCINet \
  --target Volatility \
  --features M \
  --seq_len 336 --pred_len 96 --label_len 48 \
  --enc_in 1 --des 'custom_exp' --itr 1

6. 참고 사항

데이터 누락 또는 NaN 값 제거 필수

feature_columns는 config 파일 또는 argparse에서 동적으로 설정 가능

로그 저장: 실험 별 MSE/MAE 기록 파일 생성

데이터: DataGuide, FnGuide MK2000 지수
