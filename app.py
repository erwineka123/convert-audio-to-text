from html import escape
from io import BytesIO
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st
import av
from faster_whisper import WhisperModel
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


DEFAULT_PROMPT = (
    "Audio ini menggunakan Bahasa Indonesia formal. "
    "Topik seputar politeknik pelayaran. Istilah yang mungkin muncul: "
    "Politeknik Pelayaran Surabaya, Poltekpel Surabaya, Ketua Dewan Pengawas, "
    "taruna, taruni, sambutan, akademik."
)
MODEL_OPTIONS = {
    "Tiny — paling ringan": "tiny",
    "Base — seimbang untuk CPU": "base",
    "Small — lebih akurat, lebih berat": "small",
}
CHUNK_SECONDS = 300
CHUNK_OVERLAP_SECONDS = 2


@st.cache_resource(show_spinner="Memuat model Whisper...")
def load_model(model_name: str) -> WhisperModel:
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Durasi audio tidak valid.")
    return duration


def transcribe_audio_in_chunks(
    model: WhisperModel,
    audio_path: Path,
    chunk_directory: Path,
    language: str | None,
    prompt: str | None,
    status,
) -> tuple[list[dict], str, float, float]:
    duration = get_audio_duration(audio_path)
    step_seconds = CHUNK_SECONDS - CHUNK_OVERLAP_SECONDS
    chunk_path = chunk_directory / "audio_chunk.wav"
    transcription_segments = []
    detected_language = language
    language_probability = 0.0
    chunk_start = 0.0
    chunk_number = 0

    while chunk_start < duration:
        chunk_number += 1
        chunk_duration = min(CHUNK_SECONDS, duration - chunk_start)
        status.update(
            label=(
                f"Memproses bagian {chunk_number} "
                f"(mulai {format_timestamp(chunk_start)} dari "
                f"{format_timestamp(duration)})..."
            )
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(chunk_start),
                "-i",
                str(audio_path),
                "-t",
                str(chunk_duration),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        segments, info = model.transcribe(
            str(chunk_path),
            language=detected_language,
            beam_size=5,
            vad_filter=True,
            initial_prompt=prompt,
        )
        if chunk_number == 1:
            detected_language = info.language
            language_probability = float(info.language_probability)

        for segment in segments:
            if (
                chunk_start > 0
                and segment.end <= CHUNK_OVERLAP_SECONDS
            ):
                continue
            transcription_segments.append(
                {
                    "id": len(transcription_segments) + 1,
                    "start": float(segment.start + chunk_start),
                    "end": float(segment.end + chunk_start),
                    "text": segment.text.strip(),
                }
            )

        chunk_start += step_seconds

    return (
        transcription_segments,
        detected_language or "unknown",
        language_probability,
        duration,
    )


def create_transcript_pdf(result: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"Transkrip - {result['audio_file']}",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TranscriptTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16324F"),
        spaceAfter=12,
    )
    segment_style = ParagraphStyle(
        "TranscriptSegment",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=8,
    )

    content = [
        Paragraph("Transkrip Audio", title_style),
        Paragraph(f"<b>File audio:</b> {escape(result['audio_file'])}", styles["BodyText"]),
        Paragraph(f"<b>Bahasa:</b> {escape(result['language'])}", styles["BodyText"]),
        Paragraph(f"<b>Model:</b> {escape(result['model'])}", styles["BodyText"]),
        Spacer(1, 16),
    ]
    for segment in result["segments"]:
        timestamp = (
            f"[{format_timestamp(segment['start'])} - "
            f"{format_timestamp(segment['end'])}]"
        )
        content.append(
            Paragraph(
                f"<b>{timestamp}</b> {escape(segment['text'])}",
                segment_style,
            )
        )

    document.build(content)
    return buffer.getvalue()


st.set_page_config(page_title="Audio ke Teks", page_icon="🎙️", layout="wide")
st.title("🎙️ Audio ke Teks")
st.write("Transkripsikan audio menggunakan Faster-Whisper.")
st.info(
    "Aplikasi ini berjalan di CPU. Untuk audio panjang, proses bisa memakan "
    "waktu lama. Model `large-v3` memerlukan GPU dan tidak tersedia pada "
    "Streamlit Community Cloud."
)

with st.form("transcription_form"):
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=["mp3", "m4a", "wav", "mpeg", "mpga", "webm", "ogg", "flac"],
    )
    model_label = st.selectbox("Model", list(MODEL_OPTIONS), index=1)
    language_label = st.selectbox(
        "Bahasa audio",
        ["Bahasa Indonesia", "Deteksi otomatis"],
    )
    prompt = st.text_area(
        "Konteks/istilah khusus (opsional)",
        value=DEFAULT_PROMPT,
        help="Konteks dapat membantu pengenalan nama dan istilah khusus.",
    )
    submitted = st.form_submit_button("Mulai transkripsi", type="primary")

if submitted:
    st.session_state.pop("transcription_result", None)
    if uploaded_file is None:
        st.error("Silakan pilih file audio terlebih dahulu.")
    else:
        model_name = MODEL_OPTIONS[model_label]
        language = "id" if language_label == "Bahasa Indonesia" else None
        started_at = time.perf_counter()
        uploaded_bytes = uploaded_file.getbuffer()
        if not uploaded_bytes.nbytes:
            st.error("File audio kosong. Silakan pilih file audio lain.")
        elif not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            st.error(
                "FFmpeg tidak tersedia di server. Pastikan `packages.txt` "
                "berisi `ffmpeg`, lalu deploy ulang aplikasi."
            )
        else:
            status = st.status("Memulai transkripsi...", expanded=True)
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    audio_path = temp_path / (
                        f"input{Path(uploaded_file.name).suffix or '.audio'}"
                    )
                    audio_path.write_bytes(uploaded_bytes)
                    model = load_model(model_name)
                    (
                        segments_data,
                        detected_language,
                        language_probability,
                        duration,
                    ) = transcribe_audio_in_chunks(
                        model,
                        audio_path,
                        temp_path,
                        language,
                        prompt.strip() or None,
                        status,
                    )
            except subprocess.CalledProcessError as error:
                status.update(label="Gagal membaca atau memotong audio.", state="error")
                details = (error.stderr or "").strip()
                st.error(
                    "FFmpeg gagal memproses file audio. "
                    f"{details[:500] or 'Coba unggah file audio lain.'}"
                )
            except (TypeError, ValueError, RuntimeError, av.error.FFmpegError) as error:
                status.update(label="Transkripsi gagal.", state="error")
                st.error(f"{type(error).__name__}: {error}")
            except MemoryError:
                status.update(label="Memori server tidak mencukupi.", state="error")
                st.error(
                    "Server kehabisan memori. Coba model Tiny atau bagi audio "
                    "menjadi file yang lebih pendek."
                )
            else:
                elapsed_seconds = time.perf_counter() - started_at
                status.update(
                    label="Transkripsi selesai.",
                    state="complete",
                    expanded=False,
                )
                st.session_state["transcription_result"] = {
                    "audio_file": uploaded_file.name,
                    "model": model_name,
                    "language": detected_language,
                    "language_probability": language_probability,
                    "duration_seconds": duration,
                    "processing_time_seconds": elapsed_seconds,
                    "segments": segments_data,
                }

result = st.session_state.get("transcription_result")
if result:
    st.subheader("Hasil transkripsi")
    st.caption(
        f"Bahasa terdeteksi: {result['language']} · "
        f"Model: {result['model']} · "
        f"Waktu proses: {format_timestamp(result['processing_time_seconds'])}"
    )

    segments_data = result["segments"]
    transcript = "\n".join(
        f"[{format_timestamp(segment['start'])} - "
        f"{format_timestamp(segment['end'])}] {segment['text']}"
        for segment in segments_data
    )
    st.text_area("Teks transkrip", value=transcript, height=360)

    pdf_result = create_transcript_pdf(result)
    pdf_name = f"transkrip_{Path(result['audio_file']).stem}.pdf"
    st.download_button(
        "Unduh transkrip PDF",
        data=pdf_result,
        file_name=pdf_name,
        mime="application/pdf",
        type="primary",
    )
