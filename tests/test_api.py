"""Tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test suite for health check endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_check_fields(self):
        """Test health check response has required fields"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data


class TestModelInfoEndpoint:
    """Test suite for model info endpoint"""
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "supported_diagnoses" in data
        assert len(data["supported_diagnoses"]) > 0
    
    def test_model_info_performance_metrics(self):
        """Test model info includes performance metrics"""
        response = client.get("/model/info")
        data = response.json()
        assert "performance_metrics" in data
        metrics = data["performance_metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


class TestStatsEndpoint:
    """Test suite for statistics endpoint"""
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "predictions_today" in data
        assert "average_confidence" in data
