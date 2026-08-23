import sys
import time
sys.path.append("..")

from asr.transcribe import ASR
from lid.detect_language import LanguageRouter
from retrieval.ne_embed import Retriever
from tts.synthesize import TTS


class SunoSutraPipeline:
    def __init__(self, config):
        print("Loading ASR...")
        self.asr = ASR(
            model_dir=config["asr"]["model_dir"],
            khasi_language_token=config["asr"]["khasi_language_token"],
            device=config["asr"]["device"],
        )

        print("Loading LID...")
        self.lid = LanguageRouter(
            model_path=config["lid"]["model_path"],
            hindi_confidence_threshold=config["lid"]["hindi_confidence_threshold"],
        )

        print("Loading retrieval...")
        self.retriever = Retriever(
            embed_model_dir=config["retrieval"]["embed_model_dir"],
            kb_index_path=config["retrieval"]["kb_index_path"],
            kb_answers_path=config["retrieval"]["kb_answers_path"],
        )

        print("Loading TTS...")
        self.tts = TTS(
            hindi_repo=config["tts"]["hindi_repo"],
            khasi_proxy_repo=config["tts"]["khasi_proxy_repo"],
            device=config["tts"]["device"],
        )

        print("All models loaded.")

    def run(self, audio, sampling_rate=16000, output_path="answer.wav"):
        timings = {}

        t = time.time()
        query_text = self.asr.transcribe_khasi(audio, sampling_rate=sampling_rate)
        timings["asr"] = time.time() - t

        t = time.time()
        language = self.lid.route(query_text)
        timings["lid"] = time.time() - t

        t = time.time()
        match = self.retriever.match(query_text, top_k=1)[0]
        timings["retrieval"] = time.time() - t

        answer_text = match["answers"].get(language, match["answers"].get("kha"))

        t = time.time()
        audio_out, sr = self.tts.synthesize(answer_text, language=language, output_path=output_path)
        timings["tts"] = time.time() - t

        timings["total"] = sum(timings.values())
        return {
            "query_text": query_text,
            "detected_language": language,
            "matched_id": match["id"],
            "match_score": match["score"],
            "answer_text": answer_text,
            "audio_path": output_path,
            "timings": timings,
        }


if __name__ == "__main__":
    import yaml
    import soundfile as sf

    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = SunoSutraPipeline(config)
    audio, sr = sf.read("sample_query.wav")
    result = pipeline.run(audio, sampling_rate=sr)
    print(result)
