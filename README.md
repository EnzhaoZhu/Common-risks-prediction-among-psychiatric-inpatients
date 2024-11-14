# Language Model Code for Psychiatric Risk Prediction

## Language Models

This repository provides several language model variants to suit different tasks and configurations:

- **Model 1**: A simple language model that uses only the text of the patient's medical records.
- **Model 1a**: A multitask variant of Model 1, designed to handle multiple classification tasks simultaneously.
- **Model 2**: A multimodal language model that incorporates both textual and structured data (numeric features).
- **Model 2a**: A multitask variant of Model 2, for handling multiple tasks with both textual and structured data.

### Models Overview

| Model      | Description                                                         | Input Type        | Multitask |
|------------|---------------------------------------------------------------------|-------------------|-----------|
| **Model 1**| Text-based language model, uses the first medical record only       | Text              | No        |
| **Model 1a**| Multitask language model with the first medical record only        | Text              | Yes       |
| **Model 2**| Multimodal language model, combines text and structured numeric data | Text + Numeric    | No        |
| **Model 2a**| Multitask multimodal language model, combines text and numeric data | Text + Numeric    | Yes       |

### Model Components

1. **Config Class**: The configuration file that sets model parameters and training configurations, such as batch size, learning rate, and dropout probabilities. The config file uses `argparse` for argument parsing.
   
2. **Model Classes**: The core models of this repository:
   - `Model1` (Text-only)
   - `Model1a` (Text-only, Multitask)
   - `Model2` (Multimodal: Text + Numeric)
   - `Model2a` (Multimodal, Multitask: Text + Numeric)


---

## Privacy and Ethical Considerations

Due to the **Mental Health Law** and privacy protection policies in China, particularly those surrounding sensitive medical data, the **data processing** and **feature engineering** code cannot be shared. These processes are conducted within a private enviornment, and are not accessible via external networks.

For **ethical reasons**, we cannot share any model training code or data processing scripts related to the dataset used in the experiments, as this would potentially expose patient information.

---

## Future Directions

- **Long-text Handling**:  
  Current models are limited by the input length, especially for long medical records. Future research will explore advanced architectures and techniques, such as hierarchical models and memory networks, to handle longer sequences more effectively.

- **Model Efficiency**:  
  We are exploring ways to improve model efficiency. Approaches such as **Mamba architecture**, **knowledge distillation**, and **model pruning** will be examined to make the models lighter and more suitable for deployment in real-world clinical environments.

- **Causal Inference**:  
  Future work will focus on incorporating **causal inference techniques** to enhance model interpretability and make better clinical decisions. This will help identify causal relationships between medical features and outcomes, offering more reliable risk predictions for psychiatric inpatients.

---

## Requirements

To run the code, you will need the following dependencies:

- **Python** 3.8
- **PyTorch** (version 1.8.0 or above)
- **Transformers** (version 4.29.0 or above)
- **CUDA** (optional, for GPU acceleration)

You can install the required dependencies using the following:

```bash
pip install -r requirements.txt


