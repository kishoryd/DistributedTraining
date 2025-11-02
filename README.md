# **Deep Learning Training Examples with PyTorch**

This repository provides a comprehensive guide and practical examples for training deep learning models using PyTorch across various parallelism strategies. Whether you are working on single-GPU training or scaling to multi-GPU setups with Distributed Data Parallel (DDP)  these examples will guide you through the process.

---

## **Contents**

### 01. **Introduction to Deep Learning**
- Foundational concepts of deep learning and PyTorch.
- **HPC Environment Setup:**
  - **Using SLURM for job scheduling**: Submitting and managing training jobs.
  - **Loading necessary modules**: Configuring PyTorch and CUDA on an HPC cluster.

---

### 02. **Single-GPU Training**
- Efficiently training models on a single GPU.

---

### 03. **Multi-GPU Training with Data Parallelism (DP)**
- Scaling models across multiple GPUs using `torch.nn.DataParallel`.
- **Key Considerations:**
  - Understanding inter-GPU communication overhead.
  - Differences between DP and DDP for better performance.

---

### 04. **Distributed Data Parallel (DDP) Training**
- Leveraging `torch.nn.parallel.DistributedDataParallel` for efficient multi-GPU training.
- Setting up process groups and distributed samplers
- **Advantages of DDP Over DP:**
  - Lower communication overhead.
  - Better scalability across multiple nodes.

---


### 05. **Containerized Training with Enroot and NGC Containers**
- Running PyTorch training using **NVIDIA Enroot** and **NGC Containers** on HPC.
- **Topics Covered:**
  - Importing and running **NGC PyTorch containers** with Enroot.
  - Running single and multi-GPU PyTorch workloads inside containers.
  - Using **SLURM** to launch containerized PyTorch jobs on GPU clusters.

---

## **Resources**

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- [Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
- [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [NVIDIA Enroot](https://github.com/NVIDIA/enroot)
- [NGC PyTorch Containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)

---

## **Note**
- If you are already familiar with deep learning with PyTorchand HPC, you can skip [01. Introduction to Deep Learning](./01_introduction/) and go directly to [02. Single-GPU Training](./02_singlegpu_training/).
