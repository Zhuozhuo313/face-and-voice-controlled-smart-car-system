#VAD+KWS+speaker  实现语音唤醒服务
import sounddevice as sd
import sherpa_onnx
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

g_sample_rate = 16000

def create_recognizer():
    recognizer = sherpa_onnx.KeywordSpotter(
        tokens="./model/KWS/sherpa-onnx-kws-zipformer-wenetspeech/tokens.txt",
        encoder="./model/KWS/sherpa-onnx-kws-zipformer-wenetspeech/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        decoder="./model/KWS/sherpa-onnx-kws-zipformer-wenetspeech/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
        joiner="./model/KWS/sherpa-onnx-kws-zipformer-wenetspeech/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
        keywords_file="./model/KWS/sherpa-onnx-kws-zipformer-wenetspeech/keywords.txt"
    )
    return recognizer


def load_speaker_embedding_model():
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model="./model/Sperker_recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return extractor

def load_speaker_file() -> Dict[str, List[str]]:
    speaker="./model/Sperker_recognition/speaker.txt"
    if not Path(speaker).is_file():
        raise ValueError(f"--speaker-file {speaker} does not exist")

    ans = defaultdict(list)
    with open(speaker,encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, filename = fields
            ans[speaker_name].append(filename)
    return ans

def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_speaker_embedding(
    filenames: List[str],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    assert len(filenames) > 0, "filenames is empty"

    ans = None
    for filename in filenames:
        print(f"processing {filename}")
        samples, sample_rate = load_audio(filename)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        if ans is None:
            ans = embedding
        else:
            ans += embedding

    return ans / len(filenames)



#load---------------------------------------------------------------

extractor = load_speaker_embedding_model()
speaker_file = load_speaker_file()
recognizer=create_recognizer() 
stream_recognizer = recognizer.create_stream()

manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

for name, filename_list in speaker_file.items():
    embedding = compute_speaker_embedding(
        filenames=filename_list,
        extractor=extractor,
    )
    status = manager.add(name, embedding)
    if not status:
        raise RuntimeError(f"Failed to register speaker {name}")


vad_config = sherpa_onnx.VadModelConfig()
vad_config.silero_vad.model = "./model/VAD/silero_vad.onnx"
vad_config.silero_vad.min_silence_duration = 0.25
vad_config.silero_vad.min_speech_duration = 0.25
vad_config.sample_rate = g_sample_rate
if not vad_config.validate():
    raise ValueError("Errors in vad config")

vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

#load completely-----------------------------------------------------------------------------


def vaw():

    window_size = vad_config.silero_vad.window_size
    
    samples_per_read = int(0.1 * g_sample_rate)  # 0.1 second = 100 ms

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")

    buffer = np.array([])
    #buffer = []
    with sd.InputStream(channels=1, dtype="float32", samplerate=g_sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            buffer = np.concatenate([buffer, samples])

            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]
            
            while not vad.empty():
                if len(vad.front.samples) < 0.5 * g_sample_rate:
                    # this segment is too short, skip it
                    vad.pop()
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(
                    sample_rate=g_sample_rate, waveform=vad.front.samples
                )
                stream.input_finished()


                embedding = extractor.compute(stream)
                embedding = np.array(embedding)
                name = manager.search(embedding, threshold=0.3)
                if not name:
                    name = "unknown"

                # Now for  KWS_ASR
                
                stream_recognizer.accept_waveform(
                    sample_rate=g_sample_rate, waveform=vad.front.samples
                )
                tail_paddings = np.zeros(int(0.66 * g_sample_rate), dtype=np.float32)
                stream_recognizer.accept_waveform(g_sample_rate, tail_paddings)
                while recognizer.is_ready(stream_recognizer):
                    recognizer.decode_stream(stream_recognizer)
                result = recognizer.get_result(stream_recognizer)
                
                vad.pop()
                if result:
                    print(f"\r{name}: {result}")
                    return name,result



if __name__ == "__main__":
    try:
        while True:
            vaw()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
















