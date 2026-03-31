# Framework Diagram Prompt

**Paper**: DRG: An Empirical Study of Dual-Reader Revision Gates for Quality and Compliance

## Image Generation Prompt

Create a clean, professional, vector-art methodology/architecture diagram for a top-tier ML conference paper. Layout left-to-right data flow on a white or very light grey background. Use a sophisticated harmonious palette: muted blue #4477AA for core modules, teal #44AA99 for secondary modules, warm accent #CCBB44 for gate/decision highlights, soft purple #AA3377 for metric callouts, charcoal text #222222. All shapes: rounded-corner boxes (radius ~6–8px), flat design with subtle 1–2px soft shadow. Typeface: clean sans-serif (Inter or Helvetica Neue), weights: regular for labels, medium for module titles. Minimal text only: component names and very short annotations.

Sequence and elements (left-to-right):
- Input Prompt (box, #4477AA) — label: "Input Prompt"
- Arrow → LLM Generation (box, #44AA99) — label: "LLM Generation"
- Arrow → Dual Readers cluster: two side-by-side rounded boxes: left "Quality Reader" (#4477AA) with small annotation under label: "helpfulness, accuracy"; right "Compliance Reader" (#44AA99) with annotation: "policy, refusals". Connect both with arrows from LLM.
- Arrows from each reader → Gate Logic (diamond or rounded box highlighted #CCBB44) — label: "Gate Logic" and small inline choices: "Either-Fail / Disagreement"
- Arrow → Revision Step (box, #AA3377) — label: "Constrained Revision (single step)" and tiny note: "consumes per-reader critiques"
- Arrow → Final Output (box, muted blue #4477AA) — label: "Final Output"

Additional elements:
- Audit Log box (smaller, top-right, border #AA3377) connected with dashed lines from Dual Readers and Gate Logic — label: "Audit log — per-condition seed aggregates & prereg plan"
- Metric callout (soft purple #AA3377 small rounded badge top-right): "Pilot: DRG = 0.75 ±0.125 vs baseline 0.666667 ±0.072169 (+12.5%); p=0.1785"
Styling: thin arrows (2–3px) with clear arrowheads, consistent spacing, aligned baseline, high information density but uncluttered. Output as flat vector illustration suitable for single-column figure.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
