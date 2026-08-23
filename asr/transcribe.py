import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class ASR:
    def __init__(self, model_dir, device="cuda:0", dtype=torch.float16,
                 khasi_language_token="cy", num_beams=1, max_new_tokens=64):
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device
        )
        self.device = device
        self.dtype = dtype
        self.khasi_language_token = khasi_language_token
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def transcribe_khasi(self, audio, sampling_rate=16000):
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        predicted_ids = self.model.generate(
            input_features,
            language=self.khasi_language_token,
            task="transcribe",
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


class HindiASR:
    """Dedicated ONNX conformer + CTC decoder path for Hindi (Devanagari output)."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def transcribe(self, audio, sampling_rate=16000):
        # audio: float32 numpy array, 16kHz mono
        inputs = {self.session.get_inputs()[0].name: audio}
        outputs = self.session.run(None, inputs)
        return self._decode_ctc(outputs[0])

    def _decode_ctc(self, logits):
        raise NotImplementedError("Fill in CTC decode logic matching hi-conformer.onnx output.")


if __name__ == "__main__":
    import soundfile as sf

    asr = ASR(model_dir="./checkpoints")
    audio, sr = sf.read("sample_khasi.wav")
    print(asr.transcribe_khasi(audio, sampling_rate=sr))
