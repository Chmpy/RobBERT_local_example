<p align="center">
  <img src="RobBERT_local_example.png" width="60%" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">RobBERT Local Example</h1>
</p>
<p align="center">
    <img src="https://img.shields.io/github/license/Chmpy/RobBERT_local_example?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Chmpy/RobBERT_local_example?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Chmpy/RobBERT_local_example?style=default&color=0080ff" alt="repo-top-language">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [📍 Overview](#-overview)
- [🗂️ Repository Structure](#️-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [⚙️ Installation](#️-installation)
  - [🤖 Usage](#-usage)
- [🎗 License](#-license)
</details>
<hr>

## 📍 Overview

The RobBERT_Local_Example repository provides a comprehensive exploration of the Dutch-based BERT model, RobBERT. This project is designed to guide users from basic to advanced usage, starting with the high-level pipeline API from Hugging Face and progressively diving into the low-level APIs. The goal is to offer a deep understanding of the model’s execution process and its application to various natural language processing tasks.

---

## 🗂️ Repository Structure

```
└── RobBERT_local_example/
    ├── LICENSE
    ├── convert_tf_models.sh
    ├── convert_tf_models_quantized.sh
    ├── main.py
    ├── requirements.txt
    ├── tf-main.py
    ├── tflite-main.py
    └── utils.py
```

---

## 🚀 Getting Started

**System Requirements:**

* **Python**: `version 3.11.5`

### ⚙️ Installation

<h4>From <code>source</code></h4>

> 1. Clone the RobBERT_local_example repository:
>
> ```console
> $ git clone https://github.com/Chmpy/RobBERT_local_example
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd RobBERT_local_example
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

---

### 🤖 Usage

<h4>From <code>source</code></h4>

> Run RobBERT_local_example using the command below:
> ```console
> $ python main.py
> $ python tf-main.py
> $ python tflite-main.py
> ```

> Run the model conversion scripts before executing tflite-main.py:
> ```console
> $ ./convert_tf_models.sh
> $ ./convert_tf_models_quantized.sh
## 🎗 License

This project is protected under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.
