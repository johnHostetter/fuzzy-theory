"""
Shared test fixtures for the tests package.
"""

import torch

AVAILABLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
