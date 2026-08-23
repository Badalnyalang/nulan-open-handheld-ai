import fasttext


class LanguageRouter:
    """
    IndicLID (AI4Bharat, fastText, roman-script model).
    Only used to confidently detect Hindi and route away from the Khasi default,
    IndicLID does not reliably identify Khasi itself, so Khasi is the fallback
    whenever Hindi confidence is below threshold.
    """

    def __init__(self, model_path, hindi_confidence_threshold=0.85):
        self.model = fasttext.load_model(model_path)
        self.hindi_confidence_threshold = hindi_confidence_threshold

    def route(self, text):
        labels, scores = self.model.predict(text)
        label = labels[0].replace("__label__", "")
        score = scores[0]
        if label == "hin_Latn" and score >= self.hindi_confidence_threshold:
            return "hi"
        return "kha"


if __name__ == "__main__":
    router = LanguageRouter(model_path="./models/indiclid-ftr/model_baseline_roman.bin")
    for text in ["Kumno phi long?", "aap kaise hain", "How are you?"]:
        print(text, "->", router.route(text))
