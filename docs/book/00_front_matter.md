# Beyond Quadratic Attention

## The complete book of the project: what we built, why, how, what we found, and everything that went wrong along the way

*Written for TheOneAboveALL, and for anyone who wants to understand this project from zero.*

*Every number in this book was measured on this machine during the project. The raw data lives in
`experiments/results/` and the generated reports live in `experiments/reports/`. Wherever a number
appears, the run it came from is named so that you can open the file and check it yourself.*

---

## How to read this book

This book assumes **no prior knowledge**. It begins with what a vector is and ends with the details of
how a fused attention kernel moves memory through a CPU cache. If you already know a topic, skip
ahead; every chapter starts with a short "what you will learn" box and ends with a "key points to
remember" box.

There are three kinds of text:

* **Explanations**, which teach an idea from first principles.
* **"In the code" boxes**, which point to the exact file and function that implements the idea, so you
  can open it and read along. File paths are relative to the project folder
  `C:\Users\omjee\OneDrive\Desktop\PROJECTS\Beyond Quadratic Attention`.
* **"What actually happened" boxes**, which tell the true story of the project: the mistakes, wrong
  turns, bugs and how each was found and fixed. These matter. Real research is mostly finding out
  why something does not behave as expected, and this project had many such moments. Chapter 22
  collects all of them in one place.

### A map of the book

| Part | Chapters | What it covers |
|---|---|---|
| I. Foundations | 1–5 | The project in one chapter; the mathematics you need; how computers store and compute numbers; Transformers and attention from scratch; why attention is "quadratic" |
| II. What we built | 6–10 | A tour of the codebase; the reference implementation; tiled exact attention and online softmax; PyTorch's optimized kernels; the approximate methods |
| III. Measuring correctly | 11–13 | The numerical-correctness laboratory; the complexity model; the benchmarking system |
| IV. Experiments and results | 14–16 | Exact methods; approximate methods, tile sizes, threads and dtypes; what a single attention operation can retrieve |
| V. Models and quality | 17–19 | The tiny Transformer and its training; trained-task results; the MQAR investigation |
| VI. Operating the lab | 20–21 | Commands, interface, reports, reproduction; the test suite |
| VII. Reference | 22–26 | The complete bugs-and-fixes logbook; limitations and open questions; glossary; self-test questions; file index |

### If you only have one hour

Read Chapter 1 (the whole project in one chapter), Chapter 22 (the logbook) and Chapter 25
(self-test questions). You will then know what was done, what was found and what went wrong.

### If you want to be able to defend this project in an interview

Read Chapters 4, 5, 8, 11, 13, 14 and 15 carefully, then answer every question in Chapter 25 without
looking.
