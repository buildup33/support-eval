# Evaluation methodology

Support Eval's published measurements are small, synthetic product checks. They are intended to
make evaluator limitations visible, not to establish an industry benchmark or compare Support Eval
with other products.

## Datasets

The repository contains two independently authored synthetic datasets:

| Dataset | Conversations | Criteria per conversation | Human judgments |
| --- | ---: | ---: | ---: |
| Original benchmark | 10 | 6 | 60 |
| Frozen holdout | 15 | 6 | 90 |

The corresponding source files are:

- `examples/benchmark-conversations.json`
- `examples/benchmark-rubric.json`
- `examples/benchmark-expected.json`
- `examples/holdout-conversations.json`
- `examples/holdout-rubric.json`
- `examples/holdout-expected.json`

All conversations are synthetic. Expected PASS, FAIL, and UNCERTAIN labels and expected flag states
were written independently from provider output.

## Measurement procedure

1. Freeze the conversations, rubric, and human-authored expected outcomes.
2. Evaluate every conversation with Groq using `openai/gpt-oss-20b` and structured output.
3. Validate criterion IDs, message evidence references, and result fields before scoring.
4. Apply deterministic application scoring and flagging.
5. Compare `results.json` with the expected file using `support-eval compare-benchmark`.

Criterion agreement is the number of exact PASS, FAIL, or UNCERTAIN matches divided by all expected
criterion judgments. Flag agreement is the number of exact conversation-level flag matches divided
by all conversations.

## Frozen measurements

| Dataset | Measurement | Result |
| --- | --- | ---: |
| Original benchmark | Criterion agreement | 51/60 (85.00%) |
| Frozen holdout | Criterion agreement | 77/90 (85.56%) |
| Frozen holdout | Flag agreement | 10/15 (66.67%) |

The same Groq provider and model family were used for these measurements. The original benchmark
was used to diagnose general evaluator behavior before the independent holdout was created and run,
so the two results should not be read as a before/after trend. The frozen holdout is the more useful
unseen check for the current MVP evaluation approach.

## Limitations

- The datasets contain only 25 synthetic conversations in total.
- Human expected judgments can be subjective, especially around uncertainty and closing quality.
- The datasets do not represent the distribution, policies, languages, or risks of every support
  organization.
- Measurements reflect one Groq/model configuration and a particular evaluation contract.
- Model, rubric, prompt, or provider changes can change the results.
- Public benchmark data can influence future development and is not a permanently unseen test set.
- Criterion agreement does not measure factual correctness against external systems.
- Flag agreement is materially lower than criterion agreement and reinforces the need for human
  review.

These measurements are not an industry benchmark, an external product comparison, a claim of
state-of-the-art performance, or evidence that AI judgments are safe for unsupervised consequential
decisions.

## Reproducing the comparison

Generating new provider results makes real API requests. Once a result already exists, comparison
itself is local and offline:

```console
support-eval compare-benchmark benchmark-results/results.json --expected examples/benchmark-expected.json
support-eval compare-benchmark holdout-results/results.json --expected examples/holdout-expected.json
```

Do not alter expected answers after seeing provider output merely to improve agreement.
