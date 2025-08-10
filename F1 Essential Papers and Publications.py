class EssentialReadingList:
    """Curated list of essential papers and resources for time series forecasting."""
    
    def __init__(self):
        self.paper_categories = self._organize_papers()
        self.books = self._catalog_books()
        self.online_resources = self._catalog_online_resources()
    
    def _organize_papers(self) -> Dict[str, List[Dict]]:
        """Organize essential papers by category and importance."""
        
        return {
            "foundational_classics": [
                {
                    "title": "Time Series Analysis: Forecasting and Control",
                    "authors": "Box, G.E.P., Jenkins, G.M., Reinsel, G.C.",
                    "year": 2015,
                    "journal": "Book (5th Edition)",
                    "importance": "★★★★★",
                    "summary": "The definitive reference for classical time series methods, introducing ARIMA methodology",
                    "key_contributions": ["ARIMA methodology", "Box-Jenkins approach", "Seasonal modeling"],
                    "why_essential": "Foundation of modern time series analysis",
                    "difficulty": "Intermediate",
                    "pages": 712
                },
                
                {
                    "title": "Forecasting: principles and practice",
                    "authors": "Hyndman, R.J., Athanasopoulos, G.",
                    "year": 2021,
                    "journal": "Online Book (3rd Edition)",
                    "importance": "★★★★★",
                    "summary": "Comprehensive, practical guide to forecasting with modern approaches",
                    "key_contributions": ["Practical methodology", "R implementations", "Modern techniques"],
                    "why_essential": "Best practical introduction to forecasting",
                    "difficulty": "Beginner to Intermediate",
                    "url": "https://otexts.com/fpp3/"
                },
                
                {
                    "title": "The M4 Competition: 100,000 time series and 61 forecasting methods",
                    "authors": "Makridakis, S., Spiliotis, E., Assimakopoulos, V.",
                    "year": 2020,
                    "journal": "International Journal of Forecasting",
                    "importance": "★★★★☆",
                    "summary": "Largest forecasting competition comparing classical and ML methods",
                    "key_contributions": ["Comprehensive benchmarking", "Method comparison", "Performance insights"],
                    "why_essential": "Empirical evidence on method effectiveness",
                    "difficulty": "Intermediate",
                    "doi": "10.1016/j.ijforecast.2019.04.014"
                }
            ],
            
            "deep_learning_revolution": [
                {
                    "title": "Attention Is All You Need",
                    "authors": "Vaswani, A., et al.",
                    "year": 2017,
                    "journal": "NIPS",
                    "importance": "★★★★★",
                    "summary": "Introduced the Transformer architecture revolutionizing sequence modeling",
                    "key_contributions": ["Transformer architecture", "Self-attention mechanism", "Parallel processing"],
                    "why_essential": "Foundation of modern deep learning for sequences",
                    "difficulty": "Advanced",
                    "arxiv": "1706.03762"
                },
                
                {
                    "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
                    "authors": "Zhou, H., et al.",
                    "year": 2021,
                    "journal": "AAAI",
                    "importance": "★★★★☆",
                    "summary": "Efficient Transformer variant specifically designed for long-term time series forecasting",
                    "key_contributions": ["ProbSparse attention", "Long sequence handling", "Time series Transformers"],
                    "why_essential": "Key advancement in Transformer-based forecasting",
                    "difficulty": "Advanced",
                    "arxiv": "2012.07436"
                },
                
                {
                    "title": "Are Transformers Effective for Time Series Forecasting?",
                    "authors": "Zeng, A., et al.",
                    "year": 2023,
                    "journal": "AAAI",
                    "importance": "★★★★☆",
                    "summary": "Critical analysis showing simple linear models can outperform complex Transformers",
                    "key_contributions": ["Transformer limitations", "Linear model effectiveness", "Benchmark analysis"],
                    "why_essential": "Important reality check on method complexity",
                    "difficulty": "Intermediate",
                    "arxiv": "2205.13504"
                }
            ],
            
            "foundation_models_era": [
                {
                    "title": "A decoder-only foundation model for time-series forecasting",
                    "authors": "Das, A., et al.",
                    "year": 2024,
                    "journal": "Google Research",
                    "importance": "★★★★☆",
                    "summary": "TimesFM - first large-scale foundation model for time series forecasting",
                    "key_contributions": ["Foundation model approach", "Zero-shot forecasting", "Large-scale pretraining"],
                    "why_essential": "Paradigm shift toward foundation models",
                    "difficulty": "Advanced",
                    "url": "https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/"
                },
                
                {
                    "title": "Foundation Models for Time Series Analysis: A Tutorial and Survey",
                    "authors": "Jin, M., et al.",
                    "year": 2024,
                    "journal": "arXiv preprint",
                    "importance": "★★★★☆",
                    "summary": "Comprehensive survey of foundation models for time series",
                    "key_contributions": ["Survey of FM approaches", "Taxonomy", "Future directions"],
                    "why_essential": "Complete overview of emerging paradigm",
                    "difficulty": "Intermediate to Advanced",
                    "arxiv": "2403.14735"
                }
            ],
            
            "specialized_topics": [
                {
                    "title": "Forecasting with Exponential Smoothing: The State Space Approach",
                    "authors": "Hyndman, R.J., et al.",
                    "year": 2008,
                    "journal": "Book",
                    "importance": "★★★☆☆",
                    "summary": "Comprehensive treatment of exponential smoothing in state space framework",
                    "key_contributions": ["State space formulation", "ETS models", "Automatic forecasting"],
                    "why_essential": "Deep dive into exponential smoothing",
                    "difficulty": "Intermediate to Advanced"
                },
                
                {
                    "title": "Hierarchical Forecasting",
                    "authors": "Athanasopoulos, G., et al.",
                    "year": 2017,
                    "journal": "Various papers",
                    "importance": "★★★☆☆",
                    "summary": "Methods for forecasting hierarchical and grouped time series",
                    "key_contributions": ["Reconciliation methods", "Optimal combination", "Coherent forecasts"],
                    "why_essential": "Important for business applications",
                    "difficulty": "Advanced"
                }
            ],
            
            "evaluation_and_methodology": [
                {
                    "title": "Another look at measures of forecast accuracy",
                    "authors": "Hyndman, R.J., Koehler, A.B.",
                    "year": 2006,
                    "journal": "International Journal of Forecasting",
                    "importance": "★★★★☆",
                    "summary": "Comprehensive analysis of forecasting accuracy measures",
                    "key_contributions": ["MASE metric", "Metric comparison", "Evaluation best practices"],
                    "why_essential": "Essential for proper model evaluation",
                    "difficulty": "Intermediate",
                    "doi": "10.1016/j.ijforecast.2006.03.001"
                },
                
                {
                    "title": "A comprehensive survey of time series forecasting",
                    "authors": "Cheng, M., et al.",
                    "year": 2024,
                    "journal": "arXiv preprint",
                    "importance": "★★★★☆",
                    "summary": "Recent comprehensive survey covering classical to modern methods",
                    "key_contributions": ["Method taxonomy", "Performance comparison", "Future directions"],
                    "why_essential": "Up-to-date overview of the field",
                    "difficulty": "Intermediate",
                    "arxiv": "2411.05793"
                }
            ]
        }
    
    def _catalog_books(self) -> List[Dict]:
        """Catalog essential books for different learning stages."""
        
        return [
            {
                "title": "Forecasting: Principles and Practice (3rd Ed)",
                "authors": "Hyndman & Athanasopoulos", 
                "level": "Beginner to Intermediate",
                "focus": "Practical forecasting methods",
                "pros": ["Free online", "R code included", "Comprehensive coverage"],
                "cons": ["R-focused", "Limited deep learning"],
                "recommendation": "Start here for practical forecasting",
                "url": "https://otexts.com/fpp3/"
            },
            
            {
                "title": "Time Series Analysis and Its Applications (4th Ed)",
                "authors": "Shumway & Stoffer",
                "level": "Intermediate to Advanced",
                "focus": "Statistical theory and methods",
                "pros": ["Rigorous treatment", "R code", "Spectral analysis"],
                "cons": ["Mathematical heavy", "Less business focus"],
                "recommendation": "For theoretical understanding",
                "isbn": "978-3319524511"
            },
            
            {
                "title": "The Analysis of Time Series: An Introduction (6th Ed)",
                "authors": "Chatfield",
                "level": "Beginner to Intermediate", 
                "focus": "Accessible introduction",
                "pros": ["Clear explanations", "Good balance", "Updated editions"],
                "cons": ["Less modern methods", "Limited code"],
                "recommendation": "Good starter book",
                "isbn": "978-1584883173"
            },
            
            {
                "title": "Deep Learning for Time Series Forecasting",
                "authors": "Brownlee",
                "level": "Intermediate",
                "focus": "Practical deep learning",
                "pros": ["Python code", "Practical examples", "Step-by-step"],
                "cons": ["Limited theory", "Narrow focus"],
                "recommendation": "For hands-on deep learning",
                "url": "https://machinelearningmastery.com/"
            },
            
            {
                "title": "Hands-On Time Series Analysis with Python",
                "authors": "Peixeiro",
                "level": "Beginner to Intermediate",
                "focus": "Python implementation",
                "pros": ["Python-focused", "Modern libraries", "Practical approach"],
                "cons": ["Less theory", "Limited coverage"],
                "recommendation": "For Python practitioners",
                "isbn": "978-1484259924"
            }
        ]
    
    def _catalog_online_resources(self) -> Dict[str, List[Dict]]:
        """Catalog essential online resources."""
        
        return {
            "courses_and_tutorials": [
                {
                    "name": "Time Series Analysis in Python",
                    "provider": "DataCamp",
                    "type": "Interactive course",
                    "level": "Beginner",
                    "cost": "Subscription",
                    "duration": "4 hours",
                    "focus": "Python libraries and basic methods"
                },
                
                {
                    "name": "TensorFlow Time Series",
                    "provider": "TensorFlow.org",
                    "type": "Official tutorial",
                    "level": "Intermediate",
                    "cost": "Free",
                    "duration": "Self-paced",
                    "focus": "Deep learning with TensorFlow",
                    "url": "https://www.tensorflow.org/tutorials/structured_data/time_series"
                },
                
                {
                    "name": "Time Series Forecasting with Python",
                    "provider": "Machine Learning Mastery",
                    "type": "Blog series",
                    "level": "All levels",
                    "cost": "Free/Paid books",
                    "focus": "Practical implementations"
                }
            ],
            
            "datasets_and_competitions": [
                {
                    "name": "M5 Forecasting Competition",
                    "provider": "Kaggle",
                    "type": "Competition dataset",
                    "description": "Retail sales forecasting with 42,840 time series",
                    "value": "Real-world complexity and benchmarks"
                },
                
                {
                    "name": "Time Series Forecasting Datasets",
                    "provider": "Kaggle",
                    "type": "Dataset collection",
                    "description": "Various time series datasets for practice",
                    "value": "Diverse domains and characteristics"
                },
                
                {
                    "name": "UCI Time Series Data Archive",
                    "provider": "UC Irvine",
                    "type": "Data repository",
                    "description": "Classical time series datasets",
                    "value": "Standard benchmarks"
                }
            ],
            
            "tools_and_libraries": [
                {
                    "name": "sktime Documentation",
                    "type": "Library docs",
                    "description": "Comprehensive time series ML in Python",
                    "value": "Unified API for time series tasks",
                    "url": "https://www.sktime.org/"
                },
                
                {
                    "name": "Darts User Guide",
                    "type": "Library docs", 
                    "description": "Modern forecasting library",
                    "value": "Easy-to-use API for modern methods",
                    "url": "https://unit8co.github.io/darts/"
                },
                
                {
                    "name": "Prophet Documentation",
                    "type": "Library docs",
                    "description": "Facebook's forecasting tool",
                    "value": "Business-friendly forecasting",
                    "url": "https://facebook.github.io/prophet/"
                }
            ]
        }
    
    def create_reading_roadmap(self, level: str, focus_area: str) -> List[Dict]:
        """Create personalized reading roadmap."""
        
        roadmaps = {
            "beginner": {
                "business_forecasting": [
                    {"resource": "Forecasting: Principles and Practice", "weeks": "1-4", "type": "book"},
                    {"resource": "Prophet Documentation", "weeks": "3-4", "type": "online"},
                    {"resource": "Basic time series papers", "weeks": "5-6", "type": "papers"},
                    {"resource": "Kaggle time series course", "weeks": "5-6", "type": "course"}
                ],
                "technical_deep_dive": [
                    {"resource": "The Analysis of Time Series", "weeks": "1-3", "type": "book"},
                    {"resource": "Box-Jenkins methodology", "weeks": "4-5", "type": "papers"},
                    {"resource": "Python implementation practice", "weeks": "4-6", "type": "hands-on"},
                    {"resource": "TensorFlow time series tutorial", "weeks": "6-8", "type": "online"}
                ]
            },
            
            "intermediate": {
                "research_oriented": [
                    {"resource": "Recent survey papers", "weeks": "1-2", "type": "papers"},
                    {"resource": "Transformer papers for time series", "weeks": "3-4", "type": "papers"}, 
                    {"resource": "Foundation model papers", "weeks": "5-6", "type": "papers"},
                    {"resource": "Implement cutting-edge methods", "weeks": "7-12", "type": "hands-on"}
                ],
                "production_focused": [
                    {"resource": "MLOps for time series", "weeks": "1-2", "type": "online"},
                    {"resource": "Evaluation methodology papers", "weeks": "2-3", "type": "papers"},
                    {"resource": "Deployment case studies", "weeks": "4-5", "type": "case-studies"},
                    {"resource": "Build end-to-end system", "weeks": "6-12", "type": "project"}
                ]
            }
        }
        
        return roadmaps.get(level, {}).get(focus_area, [])
    
    def get_paper_recommendations(self, interests: List[str]) -> List[Dict]:
        """Get paper recommendations based on interests."""
        
        all_papers = []
        for category, papers in self.paper_categories.items():
            all_papers.extend(papers)
        
        # Simple keyword matching
        recommended = []
        for paper in all_papers:
            paper_text = f"{paper['title']} {paper['summary']} {' '.join(paper.get('key_contributions', []))}".lower()
            
            relevance_score = sum(1 for interest in interests if interest.lower() in paper_text)
            
            if relevance_score > 0:
                paper_copy = paper.copy()
                paper_copy['relevance_score'] = relevance_score
                recommended.append(paper_copy)
        
        # Sort by importance and relevance
        recommended.sort(key=lambda x: (x['importance'].count('★'), x['relevance_score']), reverse=True)
        
        return recommended[:10]  # Top 10 recommendations

# Demonstrate reading list system
reading_list = EssentialReadingList()

print("📚 ESSENTIAL READING LIST FOR TIME SERIES FORECASTING")
print("=" * 60)

# Show paper categories
print("\n📄 PAPER CATEGORIES:")
for category, papers in reading_list.paper_categories.items():
    print(f"\n• {category.replace('_', ' ').title()}: {len(papers)} papers")
    
    # Show one example paper from each category
    if papers:
        example = papers[0]
        print(f"  Example: '{example['title']}' ({example['year']})")
        print(f"           {example['importance']} | {example['difficulty']}")

# Show book recommendations
print(f"\n📖 RECOMMENDED BOOKS:")
for book in reading_list.books[:3]:  # Show top 3
    print(f"\n• {book['title']}")
    print(f"  Authors: {book['authors']}")
    print(f"  Level: {book['level']}")
    print(f"  Recommendation: {book['recommendation']}")

# Show reading roadmap example
roadmap = reading_list.create_reading_roadmap("beginner", "business_forecasting")
if roadmap:
    print(f"\n🗺️ BEGINNER BUSINESS FORECASTING ROADMAP:")
    for item in roadmap:
        print(f"  Weeks {item['weeks']}: {item['resource']} ({item['type']})")

# Show personalized recommendations
interests = ["deep learning", "transformer", "production"]
recommendations = reading_list.get_paper_recommendations(interests)

print(f"\n🎯 PERSONALIZED RECOMMENDATIONS (for {interests}):")
for i, paper in enumerate(recommendations[:5], 1):
    print(f"\n{i}. {paper['title']} ({paper['year']})")
    print(f"   Importance: {paper['importance']} | Difficulty: {paper['difficulty']}")
    print(f"   Why relevant: {paper['summary'][:100]}...")

# Show online resources summary
online_resources = reading_list.online_resources
print(f"\n🌐 ONLINE RESOURCES SUMMARY:")
for category, resources in online_resources.items():
    print(f"• {category.replace('_', ' ').title()}: {len(resources)} resources")

print(f"\n📅 RECOMMENDED LEARNING TIMELINE:")
print("• Weeks 1-4: Foundations (books + basic papers)")
print("• Weeks 5-8: Practical skills (tutorials + hands-on)")
print("• Weeks 9-12: Advanced topics (recent papers + projects)")
print("• Weeks 13+: Specialization (domain-specific focus)")

print(f"\n💡 READING STRATEGY TIPS:")
print("1. Start with practical books before diving into papers")
print("2. Balance theory with hands-on implementation")
print("3. Join communities (Reddit, Discord, forums) for discussions")
print("4. Keep a reading log to track progress and insights")
print("5. Implement what you read to solidify understanding")
