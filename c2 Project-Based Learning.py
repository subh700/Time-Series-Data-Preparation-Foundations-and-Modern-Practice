class ProjectCatalog:
    """Catalog of hands-on time series forecasting projects."""
    
    def __init__(self):
        self.projects = self._catalog_projects()
    
    def _catalog_projects(self) -> Dict[str, List[Dict]]:
        """Catalog projects by difficulty level."""
        
        return {
            "beginner": [
                {
                    "title": "Stock Price Prediction",
                    "description": "Predict daily stock prices using historical data",
                    "dataset": "Yahoo Finance API",
                    "duration": "3-5 days",
                    "techniques": ["Moving averages", "ARIMA", "Prophet"],
                    "learning_goals": ["Data loading", "Basic forecasting", "Visualization"],
                    "difficulty": 2
                },
                
                {
                    "title": "Weather Temperature Forecasting", 
                    "description": "Forecast daily temperature using weather data",
                    "dataset": "OpenWeatherMap API",
                    "duration": "3-5 days",
                    "techniques": ["Seasonal decomposition", "Exponential smoothing"],
                    "learning_goals": ["Seasonality handling", "API integration"],
                    "difficulty": 2
                },
                
                {
                    "title": "Website Traffic Prediction",
                    "description": "Predict daily website visitors using Google Analytics data",
                    "dataset": "Google Analytics or simulated data",
                    "duration": "4-6 days", 
                    "techniques": ["Prophet", "Trend analysis", "Holiday effects"],
                    "learning_goals": ["Business forecasting", "Holiday modeling"],
                    "difficulty": 3
                },
                
                {
                    "title": "Energy Consumption Forecasting",
                    "description": "Forecast household energy usage patterns",
                    "dataset": "UCI Household Electric Power Consumption",
                    "duration": "5-7 days",
                    "techniques": ["Multiple seasonality", "SARIMA"],
                    "learning_goals": ["Multiple patterns", "Feature engineering"],
                    "difficulty": 3
                }
            ],
            
            "intermediate": [
                {
                    "title": "Retail Sales Demand Forecasting",
                    "description": "Multi-product demand forecasting for retail chain",
                    "dataset": "Walmart Sales Data (Kaggle)",
                    "duration": "1-2 weeks",
                    "techniques": ["Hierarchical forecasting", "ML ensemble", "External factors"],
                    "learning_goals": ["Multiple series", "External regressors", "Business constraints"],
                    "difficulty": 4
                },
                
                {
                    "title": "Cryptocurrency Price Volatility",
                    "description": "Predict crypto price volatility using multiple indicators",
                    "dataset": "CoinGecko API + Social sentiment",
                    "duration": "2-3 weeks",
                    "techniques": ["GARCH models", "Deep learning", "Sentiment analysis"],
                    "learning_goals": ["Volatility modeling", "Multi-modal data", "High-frequency"],
                    "difficulty": 5
                },
                
                {
                    "title": "Air Quality Prediction System",
                    "description": "Real-time air quality forecasting with alerts",
                    "dataset": "Environmental monitoring stations",
                    "duration": "2-3 weeks", 
                    "techniques": ["LSTM", "Real-time processing", "Anomaly detection"],
                    "learning_goals": ["Real-time systems", "Health applications", "Alert systems"],
                    "difficulty": 4
                },
                
                {
                    "title": "Supply Chain Optimization",
                    "description": "Inventory forecasting for supply chain optimization",
                    "dataset": "Manufacturing/logistics data",
                    "duration": "2-4 weeks",
                    "techniques": ["Multi-echelon forecasting", "Optimization", "Uncertainty quantification"],
                    "learning_goals": ["Business optimization", "Probabilistic forecasting"],
                    "difficulty": 5
                }
            ],
            
            "advanced": [
                {
                    "title": "Algorithmic Trading System",
                    "description": "Automated trading based on price forecasting",
                    "dataset": "Financial market data feeds",
                    "duration": "4-6 weeks",
                    "techniques": ["High-frequency forecasting", "Reinforcement learning", "Risk management"],
                    "learning_goals": ["Production systems", "Risk management", "Real-time decisions"],
                    "difficulty": 8
                },
                
                {
                    "title": "Smart Grid Energy Management",
                    "description": "Optimize energy distribution using demand forecasting",
                    "dataset": "Smart meter data + weather + pricing",
                    "duration": "4-8 weeks",
                    "techniques": ["Physics-informed models", "Multi-scale forecasting", "Control systems"],
                    "learning_goals": ["Complex systems", "Physics integration", "Multi-objective optimization"],
                    "difficulty": 9
                },
                
                {
                    "title": "Pandemic Spread Modeling",
                    "description": "Epidemiological forecasting for public health",
                    "dataset": "Health surveillance data + mobility + demographics",
                    "duration": "6-10 weeks",
                    "techniques": ["Compartmental models", "Agent-based modeling", "Uncertainty quantification"],
                    "learning_goals": ["Scientific modeling", "Policy impact", "Social responsibility"],
                    "difficulty": 9
                }
            ]
        }
    
    def get_project_recommendation(self, skill_level: str, interests: List[str]) -> Dict:
        """Recommend projects based on skill level and interests."""
        
        if skill_level not in self.projects:
            skill_level = "beginner"
        
        projects = self.projects[skill_level]
        
        # Simple keyword matching for recommendations
        scored_projects = []
        for project in projects:
            score = 0
            project_text = f"{project['title']} {project['description']} {' '.join(project['techniques'])}".lower()
            
            for interest in interests:
                if interest.lower() in project_text:
                    score += 1
            
            scored_projects.append((project, score))
        
        # Sort by score and return top recommendation
        scored_projects.sort(key=lambda x: x[1], reverse=True)
        return scored_projects[0][0] if scored_projects else projects[0]
    
    def create_project_tracker(self) -> pd.DataFrame:
        """Create a project progress tracker."""
        
        tracker_data = []
        for level, projects in self.projects.items():
            for project in projects:
                tracker_data.append({
                    'Level': level.title(),
                    'Project': project['title'],
                    'Duration': project['duration'],
                    'Difficulty': '⭐' * project['difficulty'],
                    'Status': '⬜ Not Started',
                    'Completion': '0%'
                })
        
        return pd.DataFrame(tracker_data)

# Demonstrate project catalog
projects = ProjectCatalog()

print("🛠️ HANDS-ON PROJECT CATALOG")
print("=" * 50)

# Show project tracker
tracker_df = projects.create_project_tracker()
print("\n📋 PROJECT PROGRESS TRACKER:")
print(tracker_df.to_string(index=False))

# Show recommendations
print(f"\n🎯 PROJECT RECOMMENDATIONS:")

sample_interests = ["finance", "energy", "healthcare"]
for interest in sample_interests:
    rec = projects.get_project_recommendation("intermediate", [interest])
    print(f"\nFor {interest} interest:")
    print(f"  Recommended: {rec['title']}")
    print(f"  Duration: {rec['duration']}")
    print(f"  Main Technique: {rec['techniques'][0]}")

print(f"\n🏆 LEARNING PATH SUGGESTIONS:")
print("1. Complete 2-3 beginner projects to build foundation")
print("2. Choose 1-2 intermediate projects in your domain of interest")
print("3. Undertake 1 advanced project as a capstone")
print("4. Contribute to open-source projects or create your own")
