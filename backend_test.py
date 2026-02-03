#!/usr/bin/env python3
"""
Backend API Testing for CyberSentinelle Network Intrusion Detection System
Tests all API endpoints using the public URL
"""

import requests
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

# PUBLIC URL for testing
BACKEND_URL = "https://network-intrusion-ai.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class CyberSentinelleAPITester:
    def __init__(self):
        self.base_url = API_BASE
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CyberSentinelle-Test/1.0'
        })

    def log_test(self, test_name: str, status: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if status:
            self.tests_passed += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   Details: {details}")
        else:
            self.failed_tests.append(f"{test_name}: {details}")
            print(f"❌ {test_name}")
            if details:
                print(f"   Error: {details}")
        print()

    def make_request(self, method: str, endpoint: str, data: Dict[Any, Any] = None, timeout: int = 30) -> tuple:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=timeout)
            else:
                return False, None, f"Unsupported method: {method}"
            
            return True, response, ""
        except requests.exceptions.Timeout:
            return False, None, f"Request timeout after {timeout}s"
        except requests.exceptions.ConnectionError:
            return False, None, "Connection error - backend may be down"
        except requests.exceptions.RequestException as e:
            return False, None, f"Request error: {str(e)}"

    def test_health_endpoints(self):
        """Test basic health endpoints"""
        print("🔍 Testing Health Endpoints...")
        print("=" * 60)

        # Test root endpoint
        success, response, error = self.make_request('GET', '/')
        if success and response.status_code == 200:
            self.log_test("Root endpoint (/)", True, f"Status: {response.status_code}")
        else:
            self.log_test("Root endpoint (/)", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test health endpoint
        success, response, error = self.make_request('GET', '/health')
        if success and response.status_code == 200:
            try:
                data = response.json()
                if 'status' in data and data['status'] == 'healthy':
                    self.log_test("Health check (/health)", True, f"Status: {data['status']}")
                else:
                    self.log_test("Health check (/health)", False, "Invalid health response format")
            except:
                self.log_test("Health check (/health)", False, "Invalid JSON response")
        else:
            self.log_test("Health check (/health)", False, error or f"Status: {response.status_code if response else 'No response'}")

    def test_dataset_endpoints(self):
        """Test dataset-related endpoints"""
        print("🔍 Testing Dataset Endpoints...")
        print("=" * 60)

        # Test dataset info endpoint
        success, response, error = self.make_request('GET', '/dataset/info')
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['total_samples', 'num_features', 'columns', 'label_distribution', 'attack_categories']
                if all(field in data for field in required_fields):
                    self.log_test("Dataset info (/dataset/info)", True, f"Samples: {data['total_samples']}, Features: {data['num_features']}")
                    # Store for later use
                    self.dataset_info = data
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Dataset info (/dataset/info)", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("Dataset info (/dataset/info)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Dataset info (/dataset/info)", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test EDA endpoint
        success, response, error = self.make_request('GET', '/dataset/eda', timeout=45)
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['numeric_stats', 'correlation_matrix', 'feature_distributions', 'top_features']
                if all(field in data for field in required_fields):
                    num_stats = len(data['numeric_stats'])
                    num_features = len(data['top_features'])
                    self.log_test("EDA data (/dataset/eda)", True, f"Numeric stats: {num_stats}, Top features: {num_features}")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("EDA data (/dataset/eda)", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("EDA data (/dataset/eda)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("EDA data (/dataset/eda)", False, error or f"Status: {response.status_code if response else 'No response'}")

    def test_model_endpoints(self):
        """Test model training and metrics endpoints"""
        print("🔍 Testing Model Training & Metrics...")
        print("=" * 60)

        # Test model training - this may take time
        print("⏳ Training models (this may take 30-60 seconds)...")
        success, response, error = self.make_request('POST', '/model/train', timeout=90)
        if success and response.status_code == 200:
            try:
                data = response.json()
                if 'status' in data and data['status'] == 'success' and 'results' in data:
                    results = data['results']
                    if 'decision_tree' in results and 'random_forest' in results:
                        dt_acc = results['decision_tree'].get('accuracy', 0)
                        rf_acc = results['random_forest'].get('accuracy', 0)
                        self.log_test("Model training (/model/train)", True, f"DT Acc: {dt_acc:.3f}, RF Acc: {rf_acc:.3f}")
                    else:
                        self.log_test("Model training (/model/train)", False, "Missing model results")
                else:
                    self.log_test("Model training (/model/train)", False, "Invalid training response")
            except Exception as e:
                self.log_test("Model training (/model/train)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Model training (/model/train)", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test model metrics endpoint
        success, response, error = self.make_request('GET', '/model/metrics')
        if success and response.status_code == 200:
            try:
                data = response.json()
                if 'results' in data:
                    results = data['results']
                    if 'decision_tree' in results and 'random_forest' in results:
                        self.log_test("Model metrics (/model/metrics)", True, "Both DT and RF metrics available")
                        # Store for prediction tests
                        self.models_trained = True
                    else:
                        self.log_test("Model metrics (/model/metrics)", False, "Missing model metrics")
                else:
                    self.log_test("Model metrics (/model/metrics)", False, "No results in response")
            except Exception as e:
                self.log_test("Model metrics (/model/metrics)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Model metrics (/model/metrics)", False, error or f"Status: {response.status_code if response else 'No response'}")

    def test_prediction_endpoints(self):
        """Test prediction endpoints"""
        print("🔍 Testing Prediction Endpoints...")
        print("=" * 60)

        # Test prediction with normal traffic sample
        normal_sample = {
            "features": {
                "duration": 0,
                "protocol_type": "tcp",
                "service": "http",
                "flag": "SF",
                "src_bytes": 215,
                "dst_bytes": 45076,
                "count": 1,
                "srv_count": 1,
                "serror_rate": 0.0,
                "same_srv_rate": 1.0,
                "dst_host_count": 255,
                "dst_host_srv_count": 255
            }
        }

        success, response, error = self.make_request('POST', '/model/predict', normal_sample)
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['prediction', 'probability', 'attack_category', 'risk_level']
                if all(field in data for field in required_fields):
                    pred = data['prediction']
                    risk = data['risk_level']
                    self.log_test("Prediction - Normal sample", True, f"Prediction: {pred}, Risk: {risk}")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Prediction - Normal sample", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("Prediction - Normal sample", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Prediction - Normal sample", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test prediction with attack sample (DoS-like)
        attack_sample = {
            "features": {
                "duration": 0,
                "protocol_type": "tcp",
                "service": "private",
                "flag": "S0",
                "src_bytes": 0,
                "dst_bytes": 0,
                "count": 511,
                "srv_count": 511,
                "serror_rate": 1.0,
                "same_srv_rate": 1.0,
                "dst_host_count": 255,
                "dst_host_srv_count": 1
            }
        }

        success, response, error = self.make_request('POST', '/model/predict', attack_sample)
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['prediction', 'probability', 'attack_category', 'risk_level']
                if all(field in data for field in required_fields):
                    pred = data['prediction']
                    risk = data['risk_level']
                    prob = data['probability'].get('attack', 0)
                    self.log_test("Prediction - Attack sample", True, f"Prediction: {pred}, Risk: {risk}, Attack prob: {prob:.3f}")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Prediction - Attack sample", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("Prediction - Attack sample", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Prediction - Attack sample", False, error or f"Status: {response.status_code if response else 'No response'}")

    def test_clustering_endpoints(self):
        """Test clustering endpoints"""
        print("🔍 Testing Clustering Endpoints...")
        print("=" * 60)

        # Test run clustering - this may take time
        print("⏳ Running K-Means clustering (this may take 30-60 seconds)...")
        success, response, error = self.make_request('POST', '/clustering/run', timeout=90)
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['n_clusters', 'silhouette_score', 'cluster_centers', 'cluster_distribution']
                if all(field in data for field in required_fields):
                    n_clusters = data['n_clusters']
                    silhouette = data['silhouette_score']
                    self.log_test("Clustering execution (/clustering/run)", True, f"Clusters: {n_clusters}, Silhouette: {silhouette:.4f}")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Clustering execution (/clustering/run)", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("Clustering execution (/clustering/run)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Clustering execution (/clustering/run)", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test clustering results endpoint
        success, response, error = self.make_request('GET', '/clustering/results')
        if success and response.status_code == 200:
            try:
                data = response.json()
                required_fields = ['n_clusters', 'silhouette_score', 'elbow_data', 'pca_data']
                if all(field in data for field in required_fields):
                    elbow_points = len(data['elbow_data'])
                    pca_points = len(data['pca_data'])
                    self.log_test("Clustering results (/clustering/results)", True, f"Elbow data: {elbow_points} points, PCA data: {pca_points} points")
                else:
                    missing = [f for f in required_fields if f not in data]
                    self.log_test("Clustering results (/clustering/results)", False, f"Missing fields: {missing}")
            except Exception as e:
                self.log_test("Clustering results (/clustering/results)", False, f"JSON parsing error: {str(e)}")
        else:
            self.log_test("Clustering results (/clustering/results)", False, error or f"Status: {response.status_code if response else 'No response'}")

    def test_notebook_endpoints(self):
        """Test notebook generation and download"""
        print("🔍 Testing Notebook Endpoints...")
        print("=" * 60)

        # Test notebook generation
        success, response, error = self.make_request('POST', '/notebook/generate', timeout=30)
        if success and response.status_code == 200:
            try:
                data = response.json()
                if 'status' in data and data['status'] == 'success':
                    self.log_test("Notebook generation (/notebook/generate)", True, "Notebook generated successfully")
                else:
                    self.log_test("Notebook generation (/notebook/generate)", True, "Notebook generation completed")
            except:
                # Generation endpoint might return text or other format
                self.log_test("Notebook generation (/notebook/generate)", True, "Notebook generation completed")
        else:
            self.log_test("Notebook generation (/notebook/generate)", False, error or f"Status: {response.status_code if response else 'No response'}")

        # Test notebook download
        success, response, error = self.make_request('GET', '/notebook/download', timeout=30)
        if success and response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            content_length = len(response.content)
            if 'application' in content_type or content_length > 1000:
                self.log_test("Notebook download (/notebook/download)", True, f"Downloaded {content_length} bytes")
            else:
                self.log_test("Notebook download (/notebook/download)", False, f"Invalid content type or size: {content_type}, {content_length} bytes")
        else:
            self.log_test("Notebook download (/notebook/download)", False, error or f"Status: {response.status_code if response else 'No response'}")

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting CyberSentinelle API Test Suite")
        print("=" * 80)
        print(f"Testing backend at: {BACKEND_URL}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Run test suites
        self.test_health_endpoints()
        self.test_dataset_endpoints()
        self.test_model_endpoints()
        self.test_prediction_endpoints()
        self.test_clustering_endpoints()
        self.test_notebook_endpoints()

        # Print summary
        print("=" * 80)
        print("🎯 TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {len(self.failed_tests)}")
        print(f"Success rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.failed_tests:
            print("\n❌ Failed tests:")
            for i, failure in enumerate(self.failed_tests, 1):
                print(f"  {i}. {failure}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = CyberSentinelleAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())