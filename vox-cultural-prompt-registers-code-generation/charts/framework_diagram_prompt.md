# Framework Diagram Prompt

**Paper**: VOX: Cultural Prompt Registers Reshape Multilingual Code Generation

## Image Generation Prompt

Create a clean academic methodology framework diagram for a top-tier ML conference paper, titled “VOX: Cultural Prompt Registers Reshape Multilingual Code Generation.” Use a left-to-right flow on a white or very light gray background (#F7F8FA), vector-art flat design, rounded rectangles, subtle soft shadows, thin consistent outlines, no photorealism. Color palette should be sophisticated and muted: blue #4477AA, teal #44AA99, warm accent #CCBB44, soft purple #AA3377, with neutral grays #D9DEE7 and #4A5568 for text/arrows.

Structure the figure as 5 main stages with concise labels. Stage 1: “Programming Tasks” and “Test Bench” input modules, showing multilingual code-generation benchmarks, multiple languages, problem prompts, unit tests. Stage 2: “Prompt Register Constructor” branching into three culturally embedded system prompt types: “Filial Shame”, “Devotion / Abandonment”, and “Corporate Authority”; include a parallel small module “Neutral Control.” Also show “Language Setting” with “Monolingual” and “Code-switched.” Stage 3: “Prompt Assembly” combining task prompt + register + language condition, feeding into “LLM Inference” with multiple model icons/boxes. Stage 4: “Generated Code” flowing into “Execution & Scoring,” with sub-boxes “Pass@k / Correctness,” “Compilation / Runtime,” and “Behavioral Analysis.” Stage 5: “Comparative Analysis” with outputs “Across Registers,” “Across Languages,” “Across Models,” and “Interaction Effects.”

Use directional arrows with branch-merge logic, compact annotations like “system prompt,” “code output,” “unit tests,” “comparison.” Keep text minimal, balanced spacing, aligned grid, visually dense but uncluttered, publication-ready.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
