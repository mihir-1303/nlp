from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def delivery_payload():
    return {
        "Days for shipment (scheduled)": 2,
        "Order Item Quantity": 2,
        "Order Item Product Price": 100.0,
        "Order Item Discount": 10.0,
        "Order Item Discount Rate": 0.10,
        "Latitude": 12.9716,
        "Longitude": 77.5946,
        "Type": "DEBIT",
        "Shipping Mode": "Standard Class",
        "Market": "Pacific Asia",
        "Order Region": "Southeast Asia",
        "Category Name": "Cleats",
        "Department Name": "Footwear",
        "Customer Segment": "Consumer",
    }
