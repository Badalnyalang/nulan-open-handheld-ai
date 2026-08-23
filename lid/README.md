# LID (Language Identification)

Uses IndicLID (AI4Bharat, fastText, roman-script model) purely as a Hindi-vs-default-Khasi
router, not a full language classifier.

IndicLID does not reliably identify Khasi (it is not in its training set), so the routing
logic is simple by design: if Hindi is detected with high confidence, route to the Hindi
path, otherwise assume Khasi (the primary/default language).

CPU-only, ~0.0004s per call, negligible in the latency budget.

Model: AI4Bharat's IndicLID, part of the open BHASHINI-linked NLP ecosystem.
