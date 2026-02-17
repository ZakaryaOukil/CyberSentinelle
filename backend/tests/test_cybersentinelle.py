"""
CyberSentinelle Network Intrusion Detection API Tests
Tests all endpoints for the IDS web application
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://ids-detection.preview.emergentagent.com')


class TestHealthEndpoint:
    """Health check endpoint tests"""
    
    def test_health_check(self):
        """Test the health endpoint returns status healthy"""
        response = requests.get(f"{BASE_URL}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        print(f"Health check passed: {data}")


class TestDatasetEndpoints:
    """Dataset info and EDA endpoint tests"""
    
    def test_dataset_info(self):
        """Test the dataset info endpoint"""
        response = requests.get(f"{BASE_URL}/api/dataset/info")
        assert response.status_code == 200
        data = response.json()
        
        # Verify expected fields
        assert "total_samples" in data
        assert "num_features" in data
        assert "columns" in data
        assert "label_distribution" in data
        assert "attack_categories" in data
        
        print(f"Dataset info: {data['total_samples']} samples, {data['num_features']} features")
    
    def test_dataset_eda(self):
        """Test the EDA endpoint returns visualization data"""
        response = requests.get(f"{BASE_URL}/api/dataset/eda")
        assert response.status_code == 200
        data = response.json()
        
        # Verify chart data structure for frontend
        assert "numeric_stats" in data
        assert "feature_distributions" in data
        assert "correlation_matrix" in data
        assert "top_features" in data
        
        # Verify label distribution for attack chart
        assert "label" in data["feature_distributions"]
        label_dist = data["feature_distributions"]["label"]
        assert "normal" in label_dist
        print(f"EDA data: Label distribution has {len(label_dist)} categories")
        
        # Verify protocol distribution for pie chart
        assert "protocol_type" in data["feature_distributions"]
        protocol_dist = data["feature_distributions"]["protocol_type"]
        assert "tcp" in protocol_dist
        print(f"Protocol distribution: {protocol_dist}")
        
        # Verify top features for bar chart
        assert len(data["top_features"]) > 0
        assert "feature" in data["top_features"][0]
        assert "variance" in data["top_features"][0]
        print(f"Top features: {[f['feature'] for f in data['top_features'][:5]]}")


class TestModelEndpoints:
    """Model training and metrics endpoint tests"""
    
    def test_model_metrics(self):
        """Test model metrics endpoint returns trained model results"""
        response = requests.get(f"{BASE_URL}/api/model/metrics")
        assert response.status_code == 200
        data = response.json()
        
        # Verify results structure
        assert "results" in data
        results = data["results"]
        
        # Verify Random Forest metrics
        assert "random_forest" in results
        rf = results["random_forest"]
        assert rf["accuracy"] > 0.8, "Model accuracy should be above 80%"
        assert "precision" in rf
        assert "recall" in rf
        assert "f1_score" in rf
        assert "roc_data" in rf  # For ROC curve chart
        assert "feature_importance" in rf  # For feature importance chart
        
        print(f"Random Forest - Accuracy: {rf['accuracy']:.4f}, Precision: {rf['precision']:.4f}")
        
        # Verify Decision Tree metrics
        assert "decision_tree" in results
        dt = results["decision_tree"]
        assert dt["accuracy"] > 0.8
        print(f"Decision Tree - Accuracy: {dt['accuracy']:.4f}")
        
        # Verify ROC data for curve chart
        roc = rf["roc_data"]
        assert "fpr" in roc
        assert "tpr" in roc
        assert "auc" in roc
        assert roc["auc"] > 0.8, "AUC should be above 0.8"
        print(f"ROC AUC: {roc['auc']:.4f}")


class TestPredictionEndpoint:
    """Prediction endpoint tests - Normal and Attack scenarios"""
    
    def test_predict_normal_traffic(self):
        """Test prediction with normal traffic features (DEMO NORMAL)"""
        payload = {
            "features": {
                "duration": 0,
                "src_bytes": 200,
                "dst_bytes": 1000,
                "count": 2,
                "srv_count": 2,
                "same_srv_rate": 1,
                "logged_in": 1,
                "serror_rate": 0,
                "srv_serror_rate": 0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF"
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/model/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "prediction" in data
        assert "probability" in data
        assert "attack_category" in data
        assert "risk_level" in data
        
        # Normal traffic should be predicted as Normal
        assert data["prediction"] == "Normal", f"Expected Normal, got {data['prediction']}"
        assert data["probability"]["normal"] > data["probability"]["attack"]
        assert data["risk_level"] == "LOW"
        
        print(f"Normal traffic prediction: {data['prediction']} (confidence: {data['probability']['normal']:.2%})")
    
    def test_predict_attack_traffic(self):
        """Test prediction with attack traffic features (DEMO ATTAQUE)"""
        payload = {
            "features": {
                "duration": 0,
                "src_bytes": 0,
                "dst_bytes": 0,
                "count": 500,
                "srv_count": 500,
                "same_srv_rate": 1,
                "logged_in": 0,
                "serror_rate": 1,
                "srv_serror_rate": 1,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF"
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/model/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        # Attack traffic should be predicted as Attack
        assert data["prediction"] == "Attack", f"Expected Attack, got {data['prediction']}"
        assert data["attack_category"] != "Normal"
        assert data["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
        
        print(f"Attack traffic prediction: {data['prediction']} ({data['attack_category']}, {data['risk_level']})")
        print(f"Confidence: {data['probability']['attack']:.2%}")


class TestClusteringEndpoints:
    """Clustering endpoint tests"""
    
    def test_clustering_results(self):
        """Test clustering results endpoint"""
        response = requests.get(f"{BASE_URL}/api/clustering/results")
        assert response.status_code == 200
        data = response.json()
        
        # Verify clustering data structure
        assert "n_clusters" in data
        assert "silhouette_score" in data
        assert "pca_data" in data  # For 2D scatter plot
        assert "cluster_distribution" in data  # For cluster sizes bar chart
        
        # Verify data quality
        assert data["n_clusters"] >= 2
        assert len(data["pca_data"]) > 0
        
        # Verify PCA data structure for scatter chart
        pca_point = data["pca_data"][0]
        assert "x" in pca_point
        assert "y" in pca_point
        assert "cluster" in pca_point
        
        print(f"Clustering: {data['n_clusters']} clusters, silhouette: {data['silhouette_score']:.4f}")
        print(f"Cluster distribution: {data['cluster_distribution']}")


class TestMonitorEndpoint:
    """Live monitor traffic endpoint tests"""
    
    def test_monitor_traffic(self):
        """Test monitor traffic endpoint"""
        response = requests.get(f"{BASE_URL}/api/monitor/traffic")
        assert response.status_code == 200
        data = response.json()
        
        # Verify monitor data structure
        assert "status" in data
        print(f"Monitor status: {data.get('status', 'unknown')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
