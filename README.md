# Generative Pre-trained Fluid Dynamics

## Abstract
This project explores the use of **Generative Pre-trained Transformers (GPT)** for modeling and simulating **incompressible** fluid dynamics phenomena. By leveraging the power of large language model (LLMs) architectures, this project aims to develop a novel approach to fluid simulation that combines the predictive accuracy of deep learning with the physical consistency of traditional methods.

## From CFD to GPT

Traditional fluid dynamics simulations rely on solving complex partial differential equations (PDEs) such as the Navier-Stokes equations. While fairly accurate, these methods can be computationally expensive and may struggle to capture complex, turbulent flows. This project investigates an alternative approach where a GPT-based model is trained to predict fluid behavior directly from input data.

The core idea is to treat fluid dynamics as a sequence modeling problem, where the state of the fluid at different time steps or spatial locations is represented as a sequence of tokens. The GPT model then learns the underlying physical dynamics and can generate future states or simulate flow patterns.

## Key Features

- **GPT-based Fluid Modeling**: Utilizes transformer architecture for fluid dynamics simulation
- **Sequence-to-Sequence Prediction**: Models fluid behavior as temporal or spatial sequences
- **Physical Consistency**: Aims to maintain physical constraints while learning from data
- **Scalable Architecture**: Leverages parallel processing capabilities of transformers
