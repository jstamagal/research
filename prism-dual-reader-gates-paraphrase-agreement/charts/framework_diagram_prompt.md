# Framework Diagram Prompt

**Paper**: PRISM: Dual-Reader Gates Fall Short of Paraphrase Agreement

## Image Generation Prompt

Create a clean academic vector-style methodology framework diagram for a top-tier ML conference paper, titled subtly at top: “PRISM: Dual-Reader Gates Fall Short of Paraphrase Agreement”. White or very light gray background (#F7F7F5), flat design, rounded rectangles, thin dark-gray outlines, subtle soft shadows, crisp typography, no photorealism. Use a harmonious muted palette: blue #4477AA, teal #44AA99, warm accent #CCBB44, soft purple #AA3377, light neutral fills.

Layout: left-to-right pipeline with three horizontal tiers: generation pipeline (top), evaluation gates (middle), outcomes/metrics (bottom). Start with a leftmost input box “User Prompt”. Arrow to “Base LLM Generation”. Then split into two experimental branches: upper branch “Single-Objective Instruction” and lower branch “Dual-Objective PRISM”. In the PRISM branch, show two parallel reader modules: “Quality Reader” (blue) and “Compliance Reader” (teal), each receiving the draft response. Add small annotations inside/under boxes only: “helpfulness / clarity / completeness” for Quality, “policy / safety / refusal” for Compliance. Their outputs feed into a central decision box in purple: “Dual-Reader Gate”. Add a small note: “agreement?”.

From the gate, show two arrows: pass to “Final Response”; fail to “Self-Revision Trigger”, then loop back to “Revised Response” and re-enter both readers. For the single-objective branch, show one evaluator box and a simpler pass/revise loop. On the right, place a compact metrics panel with three stacked boxes: “Output Quality”, “Refusal Rate”, “Hedging Behavior”. Add a small side callout box near the dual-reader gate: “Paraphrase Agreement Test”. Emphasize directional flow, balanced spacing, high information density without clutter, minimal text, publication-ready architecture overview.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
