import os
import librosa
import torch
import json
from datetime import datetime

from silero_vad import get_speech_timestamps

# 현재 시각 가져오기
start_datetime = datetime.now()

#############  경로 설정 #############
dir_path = r"D:\Dysarthria Dataset\audio\119"
output_jsonl_path = "vad_segments_119.jsonl"
sample_rate = 16000  # Silero VAD는 16kHz만 지원

############# Silero VAD 모델 로드 #############
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False   # 이미 캐시된 모델이 있다면 다시 다운로드하지 않음
)

############# 추출된 음성 구간 저장할 리스트 #############
jsonl_lines = []

############# wav 파일 처리 #############
for filename in os.listdir(dir_path):
    if not filename.endswith('.wav'):   # 확장자 .wav인 파일만 처리
        continue

    file_path = os.path.join(dir_path, filename)

    try:
        # wav 파일 로드
        # Silero VAD가 지원하는 샘플레이트인 16kHz로 리샘플링
        # mono 변환 : 오디오 정보를 하나의 채널로
        wav, sr = librosa.load(file_path, sr=16000, mono=True)
        print(f"Loading file : {file_path}")

        # numpy -> torch tensor로 변환
        wav_tensor = torch.from_numpy(wav)

        # VAD로 음성 구간 추출
        speech_segments = get_speech_timestamps(
            wav_tensor,                      # 입력 데이터
            model,                           # Silero VAD 모델
            sampling_rate=sample_rate,       # 샘플링 레이트
            threshold=0.7,                   # 민감도
            min_speech_duration_ms = 2000,   # 음성 최소 시간(ms) 2초
            min_silence_duration_ms = 2000,  # 정적 최소 시간(ms) 2초
            speech_pad_ms = 500              # 앞뒤 여유 시간(ms) 0.5초
        )

        i = 1
        # 결과 저장
        for seg in speech_segments:
            start_time = round(seg['start'] / sample_rate, 3)
            end_time = round(seg['end'] / sample_rate, 3)
            duration = round(end_time - start_time, 3)

            jsonl_line = {
                "audio": file_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            }

            jsonl_lines.append(jsonl_line)
            print(f"Finished processing line : {filename} ({i})")
            i += 1

    except Exception as e:
        print(f"Error {file_path} : {e}")

# ====== JSONL로 저장 ======
with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for line in jsonl_lines:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"Saved a total of {len(jsonl_lines)} speech segments to {output_jsonl_path}")

# 현재 시각 가져오기
end_datetime = datetime.now()
# 현재 시각 출력
print("\n시작 시각:", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print("끝 시각:", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))
