from html import escape
from io import BytesIO
import tempfile
import time
from pathlib import Path

import streamlit as st
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
        suffix = Path(uploaded_file.name).suffix or ".audio"
        started_at = time.perf_counter()
        segments_data = []

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / f"upload{suffix}"
            audio_path.write_bytes(uploaded_file.getvalue())
            model = load_model(model_name)
            status = st.status("Memulai transkripsi...", expanded=True)
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=prompt.strip() or None,
            )

            for segment in segments:
                segments_data.append(
                    {
                        "id": len(segments_data) + 1,
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text.strip(),
                    }
                )
                if len(segments_data) % 10 == 0:
                    status.update(
                        label=(
                            f"Memproses {len(segments_data)} segmen "
                            f"(sampai {format_timestamp(segment.end)})..."
                        )
                    )

        elapsed_seconds = time.perf_counter() - started_at
        status.update(label="Transkripsi selesai.", state="complete", expanded=False)
        st.session_state["transcription_result"] = {
            "audio_file": uploaded_file.name,
            "model": model_name,
            "language": info.language,
            "language_probability": float(info.language_probability),
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
