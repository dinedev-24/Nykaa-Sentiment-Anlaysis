#!/bin/bash

# Upgrade pip and install necessary dependencies
pip install setuptools wheel pip --upgrade


# Install Rust for tokenizers
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install dependencies from requirements.txt
pip install -r requirements.txt
