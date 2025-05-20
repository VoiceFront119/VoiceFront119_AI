import os
import json
from pydub import AudioSegment
from datetime import datetime

# 현재 시각 가져오기
start_datetime = datetime.now()

# 저장할 디렉토리
output_dir = r"D:\Dysarthria Dataset\trimmed_audio\119"

# start_from = "ID-02-28-N-PCS-02-04-M-82-KK"

# VAD jsonl 불러오기
with open("vad_segments_119.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 오디오 파일별 잘린 개수 관리를 위한 딕셔너리
counter = {}
# start_processing = False

for line in lines:
    data = json.loads(line)
    file_path = data["audio"]

    # 시간 단위 s -> ms
    start = int(data["start_time"] * 1000)
    end = int(data["end_time"] * 1000)

    audio_name = os.path.splitext(os.path.basename(file_path))[0]   # 오디오 파일 이름만 추출

    # # 시작할 파일이 나올 때까지 건너뛰기
    # if not start_processing:
    #     if audio_name == start_from:
    #         start_processing = True
    #     else:
    #         continue  # 아직 시작점이 아니면 다음 줄로 넘어감

    origin_audio = AudioSegment.from_wav(file_path)   # 원본 오디오 파일 불러오기
    print(f'Processing {audio_name}')
    segment = origin_audio[start:end]   # 구간 자르기

    # 각 파일별로 자른 순서 번호 붙이기
    count = counter.get(audio_name, 0) + 1
    counter[audio_name] = count

    output_path = os.path.join(output_dir, f"{audio_name}_{count:03}.wav")
    segment.export(output_path, format="wav")

print("Audio successfully segmented")

# 현재 시각 가져오기
end_datetime = datetime.now()
# 현재 시각 출력
print("\n시작 시각:", start_datetime.strftime("%Y-%m-%d %H:%M:%S"))
print("끝 시각:", end_datetime.strftime("%Y-%m-%d %H:%M:%S"))