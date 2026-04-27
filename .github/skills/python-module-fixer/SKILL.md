---
name: python-module-fixer
description: '**WORKFLOW SKILL** — Fix ModuleNotFoundError in Python applications by diagnosing the missing module and installing it. USE FOR: resolving import errors like the one with gensim in streamlit_app.py. DO NOT USE FOR: non-Python errors or system-level issues.'
---

# Python Module Fixer

## Overview
This skill provides a step-by-step workflow to resolve ModuleNotFoundError exceptions in Python applications by identifying missing dependencies and installing them.

## Workflow Steps

1. **Parse the Error Message**
   - Extract the module name from the ModuleNotFoundError (e.g., 'gensim' from the traceback).

2. **Check Existing Dependencies**
   - Read the project's dependency file (requirements.txt, pyproject.toml, etc.) to see if the module is already listed.
   - If using a virtual environment, ensure it's activated.

3. **Add Missing Dependency**
   - If the module is not in the dependency file, add it with an appropriate version (or latest if unspecified).
   - Update the dependency file accordingly.

4. **Install the Package**
   - Use the appropriate package manager (pip, conda, etc.) to install the missing package.
   - For pip: `pip install <module>`
   - Ensure the installation is in the correct environment.

5. **Verify the Fix**
   - Run the application or script to confirm the import error is resolved.
   - Check for any additional missing dependencies that may surface.

## Tools and Commands
- Use `read_file` to inspect dependency files.
- Use `replace_string_in_file` to update requirements.txt.
- Use `install_python_packages` to install packages.
- Use `run_in_terminal` to execute installation commands or run the app.
- Configure the Python environment first using `configure_python_environment`.

## Example Usage
When encountering a ModuleNotFoundError like:
```
ModuleNotFoundError: No module named 'gensim'
```

Apply this skill to:
1. Identify 'gensim' as the missing module.
2. Check if 'gensim' is in requirements.txt.
3. If not, add `gensim` to requirements.txt.
4. Run `pip install gensim`.
5. Run the Streamlit app to verify.

## Related Agents
- Python Dependency Fixer: Can be invoked as a subagent for automated dependency resolution.