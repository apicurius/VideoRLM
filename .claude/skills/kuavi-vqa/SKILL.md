---
name: kuavi-vqa
description: Answer multiple-choice questions about video content using embeddings
argument-hint: <question> -- <choice1> | <choice2> [| choice3] [| choice4]
disable-model-invocation: true
---

Parse `$ARGUMENTS`:
- Text before `--` is the **question**
- Text after `--` is a pipe-separated list of **choices**

Example: `What color is the car? -- red | blue | green | black`

Call `kuavi_discriminative_vqa` with:
- `question`: the parsed question text (trimmed)
- `candidates`: list of parsed choices (each trimmed)

Present the answer with confidence scores for each candidate.
