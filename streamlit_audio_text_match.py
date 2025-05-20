# conda activate 2025capstone
# streamlit run D:\2025Capstone\streamlit_audio_text_match.py

import streamlit as st
import os
import pandas as pd

# Streamlit 페이지 설정
st.set_page_config(page_title="Audio-Text 매칭 툴", layout="centered")

# 디렉토리 설정
audio_dir = r"D:\Dysarthria Dataset\trimmed_audio\119"   # 분할된 오디오 파일 경로
output_csv = r"D:\Dysarthria Dataset\audio_text_matches_119.csv"

# 오디오 파일 불러오기
audio_files = sorted([
    f for f in os.listdir(audio_dir) if f.endswith(".wav")
])


# 세션 상태 초기화
# 'matches' : 매칭 리스트
if "matches" not in st.session_state:
    st.session_state.matches = []
# 'current_idx' : 현재 작업 중인 오디오 인덱스
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
# 'start_selected' : 시작 오디오 파일 선택 여부
if "start_selected" not in st.session_state:
    st.session_state.start_selected = False
# 'input_text' : 입력 텍스트
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

indexed_audio_files = [f"{idx + 1}. {name}" for idx, name in enumerate(audio_files)]

# 작업 시작할 오디오 파일이 선택되면 매칭 작업 시작
if not st.session_state.start_selected:
    st.title("매칭 시작할 오디오 파일 선택")
    selected_display = st.selectbox("오디오 파일을 선택하세요. : ", indexed_audio_files)

    if st.button("오디오 - 텍스트 매칭 시작"):
        selected_file = selected_display.split(". ", 1)[1]
        st.session_state.current_idx = audio_files.index(selected_file)
        st.session_state.start_selected = True
        st.rerun()  # 화면 새로고침

else:
    # 현재 진행 상황
    total_files = len(audio_files)
    completed_files = st.session_state.current_idx

    st.title(f"오디오 - 텍스트 매칭 중... ({completed_files}개 / {total_files}개)")
    st.progress(completed_files / total_files)   # 진행률 바

    # 현재 작업할 오디오 파일 불러오기
    if st.session_state.current_idx < len(audio_files):
        current_file = audio_files[st.session_state.current_idx]
        st.subheader(f"파일명: {current_file}")

        # 오디오 재생
        audio_file_path = os.path.join(audio_dir, current_file)
        st.audio(audio_file_path, format='audio/wav')

        # 음성에 맞는 텍스트 입력
        col1, col2 = st.columns([4, 1])
        with col1:
            # 텍스트 입력창
            # input_text = st.text_input("이 오디오에 맞는 텍스트를 입력하세요:")
            text = st.text_input("이 오디오에 맞는 텍스트를 입력하세요.", value=st.session_state.input_text)
            st.session_state.input_text = text  # 입력한 텍스트 세션 상태에 저장

        with col2:
            # 텍스트 지우기 버튼
            if st.button("지우기"):
                st.session_state.input_text = ""

        # 작업 저장
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ 다음 오디오 파일"):
                if st.session_state.input_text.strip() == "":
                    st.warning("매칭되는 텍스트를 입력하세요.")  # 텍스트 입력 없으면 경고 메시지
                else:
                    # 매칭 추가
                    st.session_state.matches.append({
                        "audio_file_name": current_file,
                        "text": text
                    })
                    st.session_state.current_idx += 1
                    st.session_state.input_text = ""  # 텍스트 입력창 초기화

        with col2:
            if st.button("💾 중간 저장"):
                # 기존 CSV 파일 읽어오기
                if os.path.exists(output_csv):
                    df_existing = pd.read_csv(output_csv)
                else:
                    df_existing = pd.DataFrame(columns=["audio_file_name", "text"])

                # 새 데이터 프레임
                df_new = pd.DataFrame(st.session_state.matches)
                # 기존 데이터와 새 데이터 결합
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
                st.success(f"중간 저장 완료! ({len(st.session_state.matches)}개 데이터 저장됨)")
                
                # 매칭 리스트 초기화
                st.session_state.matches = []

        # 초기화 버튼
        if st.button("작업 초기화"):
            for key in ["matches", "current_idx", "start_selected", "text_input"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()  # 화면 새로고침

    else:
        st.success("모든 오디오에 대해 텍스트 매칭이 완료되었습니다")

        # 결과 저장
        if os.path.exists(output_csv):
            df_existing = pd.read_csv(output_csv)
        else:
            df_existing = pd.DataFrame(columns=["audio_file_name", "text"])

        # 새 데이터 프레임
        df_new = pd.DataFrame(st.session_state.matches)
        # 기존 데이터와 새 데이터 결합
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
        st.success(f"저장 완료 : {output_csv}")

        # 초기화 버튼
        if st.button("작업 초기화"):
            for key in ["matches", "current_idx", "start_selected", "text_input"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()  # 화면 새로고침