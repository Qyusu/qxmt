# Contributing to QXMT
Thank you for your interest in contributing to QXMT! Contributions are welcome and appreciated. This document outlines the process for contributing to the project.

## How to Contribute
You can contribute to this project in the following ways:

1. **Reporting bugs**: If you find any bugs, let us know.
2. **Proposing new features**: Suggest new features or improvements.
3. **Contributing code**: Fix bugs or add new features through code contributions.
4. **Improving documentation**: Help improve or translate the documentation.

## Reporting Bugs
When reporting a bug, please provide the following details:

- Environment details (operating system, Python version, etc.)
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant error messages or screenshots

Please submit bug reports through the [Issues](https://github.com/Qyusu/qxmt/issues) section.

## Proposing New Features
We welcome ideas for new features or improvements. When proposing a new feature, please explain what problem it solves and how it benefits the project. Feature proposals can be submitted through the [Issues](https://github.com/Qyusu/qxmt/issues) section.

## Contributing Code
To contribute code, follow these steps:

1. Fork the QXMT repository.
2. Create a new branch for your feature or bug fix (e.g., feature/my-new-feature).
3. Make your changes and write clear commit messages.
4. Ensure that all tests pass locally.
5. Submit a pull request (PR).

### Code Style
Please follow these coding style guidelines:

- Use [Black](https://github.com/psf/black) as the code formatter with the option `--line-length=120`.
- Ensure the code is type-hinted and checked with mypy. The mypy configuration can be found in [mypy.ini](./mypy.ini).
- For docstrings, use the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for functions and classes.

### Testing
When contributing code, make sure to add or update relevant tests. Ensure that the test suite passes locally before submitting your PR.

You can run the tests with the following command:
``` bash
poetry run pytest
```

### Creating Pull Requests
Before creating a pull request:

- Ensure that all tests pass and check code style.
- Rebase your branch on the latest main branch, if needed.
- The pull request will be reviewed, and feedback may be provided. If changes are requested, please update your PR accordingly.

## Community Guidelines
We expect contributors to be respectful to others and provide constructive feedback. Let’s maintain a positive and collaborative environment where everyone can contribute comfortably.

## License
By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project. For more details, please refer to the [LICENSE](./LICENSE) file.
