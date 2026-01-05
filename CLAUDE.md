
## Code Formatting
- code should adhere to "black" style
- write concise code wherever possible
- chain together python expressions when the intermediate results aren't used, except for improved readability of the code
- use newlines within functions and methods strategically, to group related lines of code together, but don't overdo it
- shorten string and symbol names when it will help code fit onto fewer lines

## Naming Conventions
- use short names (such as for functions and variables), abbreviating words when the name is longer than ~10 characters
- suggested variable names:
  - model for models
  - tokenizer for tokenizers
  - ckpt for checkpoints

## Type Annotations
- use mypy type annotations when possible

## Comments and Documentation
- docstrings should usually just be one short line
- try not to include inline comments more often than once per every three lines of executable code
- when formatting shell commands in markdown files, prefer to indent it with 4 spaces rather than using a code block with "```"
- capitalize header text in markdown files
- for inline comments that are sentence fragments, don't capitalize the first letter and don't put a period at then end
- for prose in markdown files and docstring comments, use full sentences with capital letters and ending with periods.

## Imports
- write imports in a way that's compatible with isort

## Logging
- print/log information sparingly

## Git and Commits
- put unrelated changes in separate commits when possible
- don't put any boilerplate content in commit messages (like "Generated with" or "Co-Authored-By")