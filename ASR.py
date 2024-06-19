import sounddevice as sd
import sherpa_ncnn
import time

#带端点检测的实时语音识别器，导入模型文件
def create_recognizer():
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./model/ASR/zipformer-small-96/tokens.txt",
        encoder_param="./model/ASR/zipformer-small-96/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./model/ASR/zipformer-small-96/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./model/ASR/zipformer-small-96/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./model/ASR/zipformer-small-96/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./model/ASR/zipformer-small-96/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./model/ASR/zipformer-small-96/joiner_jit_trace-pnnx.ncnn.bin",
        decoding_method="modified_beam_search",
        enable_endpoint_detection=True,  #启用端点检测
        rule1_min_trailing_silence=0.5,  #规则1: 最短静音时间，单位s
        rule2_min_trailing_silence=0.5, #规则2：两段话之间最短静音时间
        rule3_min_utterance_length=5, #规则3：最长语音时间
    )
    return recognizer




recognizer = create_recognizer()


def realTime_ASR():
    start=time.time()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)

            is_endpoint = recognizer.is_endpoint

            result = recognizer.text
            if result and (last_result != result):
                last_result = result
                print("\r{}".format(result), end="", flush=True)

            if is_endpoint:
                recognizer.reset()
                if result:
                    print("\r{}".format(result), flush=True)
                    return result
            end=time.time()
            totalTime=end-start
            if totalTime>30 and not result:
                recognizer.reset()
                return "长时间未收到指令"





if __name__ == "__main__":

    while True:
        result=realTime_ASR()
        print(result)





    















