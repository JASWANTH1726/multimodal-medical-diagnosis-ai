"""API request/response schemas"""

from pydantic import BaseModel
from typing import Optional, Dict, List


class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    image: str  # Base64 encoded image
    text: str   # Clinical notes
    return_explanation: Optional[bool] = False


class AlternativePrediction(BaseModel):
    """Alternative prediction"""
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: str
    confidence: float
    alternatives: Optional[List[AlternativePrediction]] = None
    explanation: Optional[Dict] = None
    processing_time_ms: float
    model_version: str = "0.1.0"
    request_id: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    version: str
    trained_date: str
    supported_diagnoses: List[str]
    performance_metrics: Dict
    inference_time_ms: float
