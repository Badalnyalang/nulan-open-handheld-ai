"""
Runs the pipeline twice on the same query: run 1 (cold-ish, first load) and
run 2 (warm). This is the script that produced the validated latency numbers
in the submission (Run 1: 4.9s, Run 2: 3.6s, against a 5-8s target budget).
"""
import time
import yaml
import soundfile as sf
from run_pipeline import SunoSutraPipeline


def bench(pipeline, audio, sr, label):
    print(f"\n=== {label} ===")
    result = pipeline.run(audio, sampling_rate=sr, output_path=f"{label}.wav")
    t = result["timings"]
    print(f"[ASR]       {t['asr']:.2f}s")
    print(f"[LID]       {t['lid']:.4f}s")
    print(f"[RETRIEVAL] {t['retrieval']:.2f}s -> matched id={result['matched_id']}, "
          f"score={result['match_score']:.3f}")
    print(f"[TTS]       {t['tts']:.2f}s")
    print(f"=== TOTAL: {t['total']:.2f}s ===")
    return result


if __name__ == "__main__":
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = SunoSutraPipeline(config)
    audio, sr = sf.read("sample_query.wav")

    bench(pipeline, audio, sr, "run1_cold")
    bench(pipeline, audio, sr, "run2_warm")
