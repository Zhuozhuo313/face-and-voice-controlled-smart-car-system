import wave
import time
import sherpa_onnx
import soundfile as sf
import sounddevice as sd
import numpy as np

def play_sound(file_path):
    # 打开WAV文件
    with wave.open(file_path, 'rb') as wf:
        # 获取音频数据
        data = wf.readframes(wf.getnframes())
        # 获取采样率
        sample_rate = wf.getframerate()
        # 将音频数据转换为numpy数组
        audio_data = np.frombuffer(data, dtype=np.int16)
        # 播放音频
        sd.play(audio_data, sample_rate)
        # 等待播放完成
        sd.wait()


def create_tts():
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model="./model/TTS/sherpa-onnx-vits-zh-ll/model.onnx",
                lexicon="./model/TTS/sherpa-onnx-vits-zh-ll/lexicon.txt",
                dict_dir="./model/TTS/sherpa-onnx-vits-zh-ll/dict",
                tokens="./model/TTS/sherpa-onnx-vits-zh-ll/tokens.txt",
            ),
        ),
        rule_fsts="./model/TTS/sherpa-onnx-vits-zh-ll/phone.fst",
        max_num_sentences=1,
    )
    if not tts_config.validate():
        raise ValueError(f"Invalid configuration: {tts_config}")

    return  sherpa_onnx.OfflineTts(tts_config)

TTs = create_tts()

def tts(text,sid=1,speed=1.5):
    
    outwave="./model/TTS/sherpa-onnx-vits-zh-ll/output.wav"
    audio = TTs.generate(text, sid, speed)

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        return

    sf.write(
        outwave,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    play_sound(outwave)


if __name__ == "__main__":
    

    s=time.time()
    tts("世界你好")
    tts("你好世界")
    print(time.time()-s)






