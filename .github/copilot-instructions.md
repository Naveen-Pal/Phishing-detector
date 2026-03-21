# Copilot instructions for this repository

## Project purpose and architecture
- Starting file is `phishing_detector.py`.
- Follow the top standard practice to write quality code.
- Suggest what can be done better and do it only after I allow.
- Try to make codebase as small as possible for better understanding.
- Detection is a layered pipeline in `classify_message()`:
  1. Deterministic Rule based prediction score on Text 
  2. Deterministic Rule based prediction score on URL
  3. Based on ML model overall text, URL
  4. Also Use the sender information context to find if it is phishing or not.


## Key files
- `phishing_detector.py`: all core logic, CLI entrypoint, printing, exit semantics.
- `README.md`: expected usage, output examples, dependency/runtime expectations.
- `requirements.txt`: runtime deps (`transformers`, `torch`, `requests`).