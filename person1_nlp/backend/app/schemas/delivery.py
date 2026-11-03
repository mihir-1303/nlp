from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


Threshold = Annotated[float, Field(ge=0.0, le=1.0)]


class DeliveryPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days_for_shipment_scheduled: float = Field(..., alias="Days for shipment (scheduled)")
    order_item_quantity: float = Field(..., alias="Order Item Quantity")
    order_item_product_price: float = Field(..., alias="Order Item Product Price")
    order_item_discount: float = Field(..., alias="Order Item Discount")
    order_item_discount_rate: float = Field(..., alias="Order Item Discount Rate")
    latitude: float = Field(..., alias="Latitude")
    longitude: float = Field(..., alias="Longitude")
    type: str = Field(..., alias="Type", min_length=1)
    shipping_mode: str = Field(..., alias="Shipping Mode", min_length=1)
    market: str = Field(..., alias="Market", min_length=1)
    order_region: str = Field(..., alias="Order Region", min_length=1)
    category_name: str = Field(..., alias="Category Name", min_length=1)
    department_name: str = Field(..., alias="Department Name", min_length=1)
    customer_segment: str = Field(..., alias="Customer Segment", min_length=1)
    threshold: Threshold | None = None

    @field_validator("type", "shipping_mode", "market", "order_region", "category_name", "department_name", "customer_segment")
    @classmethod
    def categorical_values_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("categorical values must not be blank")
        return value


class DeliveryPredictionResponse(BaseModel):
    estimated_late_delivery_probability: float
    prediction: int
    warning: bool
    threshold_used: float
    target: str = "Late_delivery_risk"
