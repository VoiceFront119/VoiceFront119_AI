from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torchaudio
import pandas as pd
from datasets import Dataset
import os
import evaluate
import torch
from datetime import datetime

# 현재 시각 가져오기
start_datetime = datetime.now()

print('Loading model and processor...')
# 모델과 processor 로드
# model_name = "openai/whisper-medium"
model_name = r"D:\2025캡스톤\whisper_m__250512_001"  # LoRA 모델 경로
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)
# 추론 모드로 전환
model.eval()
print('Done.')

# CSV 파일 불러오기
ts01_df = pd.read_csv(r"D:\Dysarthria Dataset\audio_text_matches_ts01.csv")
ts02_df = pd.read_csv(r"D:\Dysarthria Dataset\audio_text_matches_ts02.csv")
ts02_2_df = pd.read_csv(r"D:\Dysarthria Dataset\audio_text_matches_ts02_2.csv")
ts03_df = pd.read_csv(r"D:\Dysarthria Dataset\audio_text_matches_ts03.csv")
ts119_df = pd.read_csv(r"D:\Dysarthria Dataset\audio_text_matches_119.csv")

df = pd.concat([ts01_df, ts02_df, ts02_2_df, ts03_df, ts119_df], ignore_index=True)

dataset = Dataset.from_pandas(df)

# 오디오 데이터 로드
def load_audio(data):
    data_dir = r"D:\Dysarthria Dataset\Dataset"  # 오디오 파일이 저장된 디렉토리 경로
    data_path = os.path.join(data_dir, data["audio_file_name"])  # 전체 경로

    speech_array, original_sampling_rate = torchaudio.load(data_path)

    # 16kHz로 리샘플링
    if original_sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)

    data["speech"] = speech_array.squeeze().numpy()  # 1차원 변환한 waveform
    data["sampling_rate"] = 16000
    return data

# 오디오 불러오기 (dataset columns = 'audio_file_name', 'text', 'speech', 'sampling_rate')
print('\nLoading audio data...')
dataset = dataset.map(load_audio)
print('Data loaded.')

# 전처리
def prepare_dataset(data):
    # 입력 오디오 데이터 처리
    inputs = processor.feature_extractor(data["speech"], sampling_rate=16000, return_tensors="pt").input_features[0]
    labels = processor.tokenizer(data["text"]).input_ids

    data["input_features"] = inputs
    data["labels"] = labels
    return data

# 전처리 적용 ('input_features' 와 'labels' 만 남겨서 저장)
print('\nPreparing dataset...')
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
print('Dataset prepared.')

# 데이터 나누기
train_validtest_split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_dataset = train_validtest_split["train"]

valid_test_split = train_validtest_split["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)
valid_dataset = valid_test_split["train"]
test_dataset = valid_test_split["test"]

# metric 로드
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 예측 및 정답 리스트 초기화
predictions = []
references = []

total = len(test_dataset)
i=1

# 모델 예측 루프
for sample in test_dataset:
    print(f"------------- {i} / {total}")
    # input 준비 (batch가 아니므로 unsqueeze 필요)
    input_features = torch.tensor(sample["input_features"]).unsqueeze(0)

    # 모델 추론 (generate)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # 디코딩
    pred_str = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    label_str = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)

    predictions.append(pred_str)
    references.append(label_str)
    i += 1

print('\nCalculating WER and CER...\n')
# WER(Word Error Rate) 계산
wer_score = wer_metric.compute(predictions=predictions, references=references)
# CER(Character Error Rate) 계산
cer_score = cer_metric.compute(predictions=predictions, references=references)

print('-------------------------------------------------------------')
print(f"WER (Word Error Rate): {wer_score * 100:.4f}%")
print(f"CER (Character Error Rate): {cer_score * 100:.4f}%")

print('-------------------------------------------------------------')
for i in range(5):
    print(f'Pred : {predictions[i]}')
    print(f'Ref : {references[i]}\n')

# 현재 시각 가져오기
end_datetime = datetime.now()
# 현재 시각 출력
print("\n시작 시각:", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print("끝 시각:", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))