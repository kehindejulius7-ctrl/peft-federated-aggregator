# Parameter-Efficient Federated Aggregator for Heterogeneous Edge Nodes

## Overview
This repository contains a production-grade implementation of Parameter-Efficient Federated Learning (PEFT-FL). Designed for decentralized edge environments, the framework integrates low-rank adapter bottlenecking, Dirichlet Non-IID data distribution modeling, and bandwidth transmission benchmarking.

## Key Features
* **PEFT Bottleneck Adapters:** Injects low-rank adapter layers (`PEFTAdapterBlock`) into deep feature extractors, freezing backbone representations to reduce uplink payload size.
* **Dirichlet Non-IID Partitioning:** Simulates real-world client data skew using Dirichlet distribution over class targets ($\alpha = 0.5$).
* **Weighted FedAvg Aggregation:** Central server aggregates updates proportional to edge node sample volumes ($n_k / N$).
* **Dynamic Bandwidth Profiling:** Measures upstream communication savings between full-parameter synchronization and adapter-only payloads.

## Architecture & Pipeline
1. **Backbone:** Deep Convolutional Neural Network (CNN) Feature Extractor.
2. **Adapter Layer:** Latent bottleneck down-projection, non-linear activation, zero-initialized up-projection.
3. **Optimization:** SGD with momentum applied strictly to trainable parameters (adapters + head).

## Quickstart

### Prerequisites
```bash
pip install torch torchvision numpy
