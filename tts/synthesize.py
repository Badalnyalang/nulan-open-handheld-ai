import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer


class TTS:
    """
    MMS-TTS (VITS), two small per-language checkpoints:
      - Hindi: facebook/mms-tts-hin (native)
      - Khasi: facebook/mms-tts-vie (Vietnamese used as proxy, both Austroasiatic/Mon-Khmer)
    """

    def __init__(self, hindi_repo="facebook/mms-tts-hin",
                 khasi_proxy_repo="facebook/mms-tts-vie",
                 device="cuda:0", sample_rate=24000):
        self.device = device
        self.sample_rate = sample_rate

        self.hindi_model = VitsModel.from_pretrained(hindi_repo).to(device)
        self.hindi_tokenizer = AutoTokenizer.from_pretrained(hindi_repo)

        self.khasi_model = VitsModel.from_pretrained(khasi_proxy_repo).to(device)
        self.khasi_tokenizer = AutoTokenizer.from_pretrained(khasi_proxy_repo)

    def synthesize(self, text, language, output_path=None):
        model, tokenizer = (
            (self.hindi_model, self.hindi_tokenizer) if language == "hi"
            else (self.khasi_model, self.khasi_tokenizer)
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            waveform = model(**inputs).waveform
        audio = waveform.squeeze().cpu().numpy()
        if output_path:
            sf.write(output_path, audio, model.config.sampling_rate)
        return audio, model.config.sampling_rate


if __name__ == "__main__":
    tts = TTS()
    tts.synthesize("Ka slap ka long ka jingpynmih ki dieng...", language="kha",
                    output_path="test_khasi.wav")
    tts.synthesize("प्रकाश संश्लेषण वह प्रक्रिया है", language="hi",
                    output_path="test_hindi.wav")
