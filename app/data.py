import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import random


def _random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return v


@dataclass(frozen=True)
class WealthyProfile:
    name: str
    net_worth: float  # in USD
    embedding: np.ndarray


class WealthyProfileDB:
    """
    Database of mock wealthy people profiles for matching against.
    Has people from all wealth levels .
    """

    def __init__(self):
        np.random.seed(42)
        
        print("Loading profile database...")
        self.profiles = self._create_all_profiles()
        self.embeddings = self._generate_embeddings()
        
        stats = self.get_wealth_distribution_stats()
        print(f"   Loaded {stats['total_profiles']} profiles")
        print(f"   Wealth range: ${stats['min_net_worth']:,.0f} to ${stats['max_net_worth']/1e9:.1f}B")
        print(f"   Class breakdown: {stats['class_distribution']}")
    
    def _create_all_profiles(self) -> List[Dict[str, Any]]:
        """Make our fake dataset with people from all wealth levels."""
        
        profiles = []
        
        # ULTRA-WEALTHY (
        ultra_wealthy = [
            {"name": "Elon Musk", "net_worth": 342_000_000_000, "age": 53, "source": "Tesla, SpaceX", "class": "ultra-wealthy"},
            {"name": "Mark Zuckerberg", "net_worth": 216_000_000_000, "age": 40, "source": "Meta Platforms", "class": "ultra-wealthy"},
            {"name": "Jeff Bezos", "net_worth": 215_000_000_000, "age": 61, "source": "Amazon", "class": "ultra-wealthy"},
            {"name": "Larry Ellison", "net_worth": 192_000_000_000, "age": 80, "source": "Oracle", "class": "ultra-wealthy"},
            {"name": "Bernard Arnault", "net_worth": 178_000_000_000, "age": 76, "source": "LVMH", "class": "ultra-wealthy"},
            {"name": "Warren Buffett", "net_worth": 154_000_000_000, "age": 94, "source": "Berkshire Hathaway", "class": "ultra-wealthy"},
            {"name": "Larry Page", "net_worth": 144_000_000_000, "age": 52, "source": "Google", "class": "ultra-wealthy"},
            {"name": "Sergey Brin", "net_worth": 138_000_000_000, "age": 51, "source": "Google", "class": "ultra-wealthy"},
            {"name": "Bill Gates", "net_worth": 108_000_000_000, "age": 69, "source": "Microsoft", "class": "ultra-wealthy"},
            {"name": "Steve Ballmer", "net_worth": 118_000_000_000, "age": 69, "source": "Microsoft", "class": "ultra-wealthy"},
            {"name": "Mukesh Ambani", "net_worth": 92_500_000_000, "age": 67, "source": "Reliance Industries", "class": "ultra-wealthy"},
            {"name": "Carlos Slim", "net_worth": 82_500_000_000, "age": 85, "source": "Telecom", "class": "ultra-wealthy"},
            {"name": "Francoise Bettencourt Meyers", "net_worth": 81_600_000_000, "age": 71, "source": "L'Oreal", "class": "ultra-wealthy"},
            {"name": "Amancio Ortega", "net_worth": 124_000_000_000, "age": 89, "source": "Zara", "class": "ultra-wealthy"},
            {"name": "Zhong Shanshan", "net_worth": 57_700_000_000, "age": 70, "source": "Beverages", "class": "ultra-wealthy"},
            {"name": "Gautam Adani", "net_worth": 56_300_000_000, "age": 62, "source": "Infrastructure", "class": "ultra-wealthy"},
            {"name": "Ma Huateng", "net_worth": 56_200_000_000, "age": 53, "source": "Tencent", "class": "ultra-wealthy"},
            {"name": "Jack Ma", "net_worth": 28_600_000_000, "age": 60, "source": "Alibaba", "class": "ultra-wealthy"},
            {"name": "Michael Bloomberg", "net_worth": 105_000_000_000, "age": 83, "source": "Bloomberg LP", "class": "ultra-wealthy"},
            {"name": "Jensen Huang", "net_worth": 98_700_000_000, "age": 62, "source": "NVIDIA", "class": "ultra-wealthy"},
        ]
        
        # WEALTHY 
        wealthy = [
            {"name": "Tech Entrepreneur Sarah", "net_worth": 850_000_000, "age": 45, "source": "Software Company", "class": "wealthy"},
            {"name": "Investment Banker Robert", "net_worth": 750_000_000, "age": 52, "source": "Private Equity", "class": "wealthy"},
            {"name": "Real Estate Mogul Diana", "net_worth": 650_000_000, "age": 58, "source": "Commercial Real Estate", "class": "wealthy"},
            {"name": "Pharmaceutical CEO Michael", "net_worth": 550_000_000, "age": 61, "source": "Biotech", "class": "wealthy"},
            {"name": "Media Executive Lisa", "net_worth": 450_000_000, "age": 49, "source": "Entertainment", "class": "wealthy"},
            {"name": "Oil Executive James", "net_worth": 350_000_000, "age": 66, "source": "Energy", "class": "wealthy"},
            {"name": "Fashion Designer Emma", "net_worth": 250_000_000, "age": 43, "source": "Luxury Brand", "class": "wealthy"},
            {"name": "Hedge Fund Manager David", "net_worth": 180_000_000, "age": 55, "source": "Hedge Fund", "class": "wealthy"},
            {"name": "Mining Executive Carlos", "net_worth": 120_000_000, "age": 59, "source": "Mining", "class": "wealthy"},
            {"name": "Restaurant Chain Owner Maria", "net_worth": 85_000_000, "age": 47, "source": "Food Service", "class": "wealthy"},
            {"name": "Professional Athlete Kevin", "net_worth": 75_000_000, "age": 35, "source": "Sports", "class": "wealthy"},
            {"name": "Entertainment Producer Rachel", "net_worth": 65_000_000, "age": 41, "source": "Film/TV", "class": "wealthy"},
            {"name": "Tech Investor Alex", "net_worth": 55_000_000, "age": 38, "source": "Venture Capital", "class": "wealthy"},
            {"name": "Manufacturing Owner Chen", "net_worth": 45_000_000, "age": 54, "source": "Manufacturing", "class": "wealthy"},
            {"name": "Software Executive Amanda", "net_worth": 35_000_000, "age": 42, "source": "SaaS", "class": "wealthy"},
            {"name": "Investment Advisor John", "net_worth": 25_000_000, "age": 48, "source": "Financial Services", "class": "wealthy"},
            {"name": "E-commerce Founder Nina", "net_worth": 20_000_000, "age": 36, "source": "Online Retail", "class": "wealthy"},
            {"name": "Consulting Firm Owner Paul", "net_worth": 18_000_000, "age": 51, "source": "Consulting", "class": "wealthy"},
            {"name": "Medical Practice Owner Dr. Susan", "net_worth": 15_000_000, "age": 46, "source": "Healthcare", "class": "wealthy"},
            {"name": "Construction Company Owner Tony", "net_worth": 12_000_000, "age": 53, "source": "Construction", "class": "wealthy"},
        ]
        
        # UPPER MIDDLE CLASS 
        upper_middle = [
            {"name": "Senior Software Engineer Tom", "net_worth": 8_500_000, "age": 44, "source": "Tech Salary + Stock", "class": "upper-middle"},
            {"name": "Surgeon Dr. Patricia", "net_worth": 6_200_000, "age": 51, "source": "Medical Practice", "class": "upper-middle"},
            {"name": "Corporate Lawyer Jennifer", "net_worth": 4_800_000, "age": 47, "source": "Law Firm Partner", "class": "upper-middle"},
            {"name": "Investment Advisor Mark", "net_worth": 3_900_000, "age": 49, "source": "Financial Planning", "class": "upper-middle"},
            {"name": "Engineering Manager Lisa", "net_worth": 3_200_000, "age": 42, "source": "Aerospace", "class": "upper-middle"},
            {"name": "Pharmaceutical Researcher Dr. Kumar", "net_worth": 2_800_000, "age": 45, "source": "Biotech", "class": "upper-middle"},
            {"name": "Marketing Director Sarah", "net_worth": 2_400_000, "age": 39, "source": "Fortune 500", "class": "upper-middle"},
            {"name": "Real Estate Agent Monica", "net_worth": 2_100_000, "age": 43, "source": "Luxury Properties", "class": "upper-middle"},
            {"name": "Dentist Dr. Williams", "net_worth": 1_900_000, "age": 48, "source": "Private Practice", "class": "upper-middle"},
            {"name": "IT Director James", "net_worth": 1_650_000, "age": 46, "source": "Technology", "class": "upper-middle"},
            {"name": "Financial Analyst Rachel", "net_worth": 1_450_000, "age": 37, "source": "Investment Banking", "class": "upper-middle"},
            {"name": "Architect Michael", "net_worth": 1_250_000, "age": 44, "source": "Design Firm", "class": "upper-middle"},
            {"name": "Sales Director Kevin", "net_worth": 1_100_000, "age": 41, "source": "Medical Devices", "class": "upper-middle"},
            {"name": "University Professor Dr. Chen", "net_worth": 950_000, "age": 52, "source": "Academia + Consulting", "class": "upper-middle"},
            {"name": "Operations Manager Diana", "net_worth": 850_000, "age": 38, "source": "Manufacturing", "class": "upper-middle"},
            {"name": "Pharmacist Robert", "net_worth": 750_000, "age": 45, "source": "Hospital Pharmacy", "class": "upper-middle"},
            {"name": "Product Manager Amy", "net_worth": 680_000, "age": 35, "source": "Tech Company", "class": "upper-middle"},
            {"name": "Consulting Manager Brian", "net_worth": 620_000, "age": 40, "source": "Management Consulting", "class": "upper-middle"},
            {"name": "Airline Pilot Captain Martinez", "net_worth": 580_000, "age": 47, "source": "Commercial Aviation", "class": "upper-middle"},
            {"name": "Physical Therapist Owner Janet", "net_worth": 520_000, "age": 43, "source": "Healthcare Practice", "class": "upper-middle"},
        ]
        
        # MIDDLE CLASS 
        middle_class = [
            {"name": "Teacher Maria", "net_worth": 420_000, "age": 38, "source": "Public Education", "class": "middle"},
            {"name": "Police Officer David", "net_worth": 380_000, "age": 42, "source": "Law Enforcement", "class": "middle"},
            {"name": "Nurse Jennifer", "net_worth": 350_000, "age": 35, "source": "Hospital", "class": "middle"},
            {"name": "Accountant Steve", "net_worth": 320_000, "age": 40, "source": "CPA Firm", "class": "middle"},
            {"name": "Firefighter Captain Lisa", "net_worth": 295_000, "age": 44, "source": "Fire Department", "class": "middle"},
            {"name": "Social Worker Amanda", "net_worth": 270_000, "age": 36, "source": "Non-profit", "class": "middle"},
            {"name": "Electrician Mike", "net_worth": 245_000, "age": 39, "source": "Union Trade", "class": "middle"},
            {"name": "Insurance Agent Carol", "net_worth": 225_000, "age": 41, "source": "Insurance Sales", "class": "middle"},
            {"name": "Bank Manager Robert", "net_worth": 210_000, "age": 45, "source": "Banking", "class": "middle"},
            {"name": "HR Specialist Nancy", "net_worth": 195_000, "age": 37, "source": "Human Resources", "class": "middle"},
            {"name": "Plumber Tony", "net_worth": 180_000, "age": 43, "source": "Skilled Trade", "class": "middle"},
            {"name": "Administrative Assistant Karen", "net_worth": 165_000, "age": 46, "source": "Corporate", "class": "middle"},
            {"name": "Mechanic Carlos", "net_worth": 150_000, "age": 38, "source": "Auto Repair", "class": "middle"},
            {"name": "Postal Worker James", "net_worth": 140_000, "age": 48, "source": "USPS", "class": "middle"},
            {"name": "Library Manager Susan", "net_worth": 130_000, "age": 44, "source": "Public Library", "class": "middle"},
            {"name": "Customer Service Rep Linda", "net_worth": 120_000, "age": 35, "source": "Call Center", "class": "middle"},
            {"name": "Truck Driver Paul", "net_worth": 115_000, "age": 41, "source": "Transportation", "class": "middle"},
            {"name": "Dental Hygienist Rachel", "net_worth": 110_000, "age": 33, "source": "Dental Office", "class": "middle"},
            {"name": "Security Guard Marcus", "net_worth": 105_000, "age": 39, "source": "Corporate Security", "class": "middle"},
            {"name": "Office Manager Patricia", "net_worth": 102_000, "age": 42, "source": "Small Business", "class": "middle"},
        ]
        
        # LOWER MIDDLE CLASS 
        lower_middle = [
            {"name": "Retail Manager Jessica", "net_worth": 85_000, "age": 34, "source": "Retail Chain", "class": "lower-middle"},
            {"name": "Factory Supervisor Miguel", "net_worth": 78_000, "age": 37, "source": "Manufacturing", "class": "lower-middle"},
            {"name": "Medical Assistant Rosa", "net_worth": 72_000, "age": 29, "source": "Healthcare", "class": "lower-middle"},
            {"name": "Warehouse Manager Kevin", "net_worth": 68_000, "age": 41, "source": "Logistics", "class": "lower-middle"},
            {"name": "Daycare Owner Maria", "net_worth": 64_000, "age": 38, "source": "Childcare", "class": "lower-middle"},
            {"name": "Delivery Driver Tom", "net_worth": 58_000, "age": 32, "source": "Package Delivery", "class": "lower-middle"},
            {"name": "Restaurant Manager Lisa", "net_worth": 54_000, "age": 35, "source": "Food Service", "class": "lower-middle"},
            {"name": "Construction Worker Juan", "net_worth": 51_000, "age": 39, "source": "Construction", "class": "lower-middle"},
            {"name": "Sales Associate Jennifer", "net_worth": 47_000, "age": 28, "source": "Retail", "class": "lower-middle"},
            {"name": "Bus Driver William", "net_worth": 44_000, "age": 45, "source": "Public Transit", "class": "lower-middle"},
            {"name": "Hairstylist Sandra", "net_worth": 41_000, "age": 31, "source": "Beauty Salon", "class": "lower-middle"},
            {"name": "Maintenance Worker Robert", "net_worth": 38_000, "age": 43, "source": "Building Maintenance", "class": "lower-middle"},
            {"name": "Cashier Supervisor Angela", "net_worth": 35_000, "age": 36, "source": "Grocery Store", "class": "lower-middle"},
            {"name": "Home Health Aide Carmen", "net_worth": 32_000, "age": 42, "source": "Healthcare", "class": "lower-middle"},
            {"name": "Food Prep Worker Jose", "net_worth": 29_000, "age": 27, "source": "Restaurant", "class": "lower-middle"},
            {"name": "Janitorial Supervisor Mark", "net_worth": 27_000, "age": 48, "source": "Cleaning Services", "class": "lower-middle"},
            {"name": "Assembly Line Worker Diana", "net_worth": 26_000, "age": 33, "source": "Manufacturing", "class": "lower-middle"},
            {"name": "Pet Groomer Sarah", "net_worth": 25_500, "age": 26, "source": "Pet Services", "class": "lower-middle"},
            {"name": "Landscaper Carlos", "net_worth": 25_200, "age": 35, "source": "Landscaping", "class": "lower-middle"},
            {"name": "Childcare Worker Monica", "net_worth": 25_000, "age": 24, "source": "Daycare", "class": "lower-middle"},
        ]
        
        # LOWER INCOME/POVERTY CLASS
        lower_income = [
            {"name": "Fast Food Worker Ashley", "net_worth": 22_000, "age": 23, "source": "Fast Food", "class": "lower-income"},
            {"name": "Retail Cashier Michael", "net_worth": 19_500, "age": 26, "source": "Retail", "class": "lower-income"},
            {"name": "Cleaning Staff Maria", "net_worth": 17_800, "age": 34, "source": "Janitorial", "class": "lower-income"},
            {"name": "Part-time Server Lisa", "net_worth": 16_200, "age": 21, "source": "Restaurant", "class": "lower-income"},
            {"name": "Grocery Bagger James", "net_worth": 15_000, "age": 19, "source": "Grocery Store", "class": "lower-income"},
            {"name": "Security Guard Patricia", "net_worth": 14_500, "age": 29, "source": "Security", "class": "lower-income"},
            {"name": "Home Care Aide Rosa", "net_worth": 13_800, "age": 31, "source": "Home Care", "class": "lower-income"},
            {"name": "Dishwasher Carlos", "net_worth": 12_500, "age": 25, "source": "Restaurant", "class": "lower-income"},
            {"name": "Farm Worker Miguel", "net_worth": 11_200, "age": 28, "source": "Agriculture", "class": "lower-income"},
            {"name": "Parking Attendant Kevin", "net_worth": 10_800, "age": 33, "source": "Parking Services", "class": "lower-income"},
            {"name": "Hotel Housekeeper Ana", "net_worth": 9_500, "age": 27, "source": "Hospitality", "class": "lower-income"},
            {"name": "Laundromat Worker David", "net_worth": 8_900, "age": 22, "source": "Laundry Services", "class": "lower-income"},
            {"name": "Street Vendor Juan", "net_worth": 7_800, "age": 35, "source": "Street Vending", "class": "lower-income"},
            {"name": "Car Wash Worker Tony", "net_worth": 6_500, "age": 24, "source": "Car Wash", "class": "lower-income"},
            {"name": "Unemployed Single Mom Jennifer", "net_worth": 5_200, "age": 30, "source": "Unemployment/Assistance", "class": "lower-income"},
            {"name": "Disabled Veteran Robert", "net_worth": 4_800, "age": 38, "source": "Disability Benefits", "class": "lower-income"},
            {"name": "Student Worker Amanda", "net_worth": 3_500, "age": 20, "source": "Part-time + Student Loans", "class": "lower-income"},
            {"name": "Elderly Retiree Dorothy", "net_worth": 2_800, "age": 72, "source": "Social Security", "class": "lower-income"},
            {"name": "Homeless Shelter Worker Marcus", "net_worth": 1_500, "age": 26, "source": "Non-profit", "class": "lower-income"},
            {"name": "Recent Immigrant Sofia", "net_worth": 800, "age": 32, "source": "Odd Jobs", "class": "lower-income"},
        ]
        
        # Combine all profiles
        profiles.extend(ultra_wealthy)
        profiles.extend(wealthy)
        profiles.extend(upper_middle)
        profiles.extend(middle_class)
        profiles.extend(lower_middle)
        profiles.extend(lower_income)
        
        return profiles
    
    def _generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings that actually correlate with wealth levels.
        
        Creates distinct patterns for different wealth classes so that
        rich people are more similar to other rich people than to poor people.
        """
        embeddings = []
        
        print("Generating wealth-correlated embeddings...")
        print("   Creating distinct patterns for each wealth class")
        
        # Define VERY distinct base patterns for different wealth classes
        # These will be in the first 20 dimensions for strong separation
        wealth_base_patterns = {
            "ultra-wealthy": np.array([0.9, 0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.8, 0.7, 0.9,
                                     0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.9, 0.8]),
            "wealthy": np.array([0.7, 0.6, 0.7, 0.5, 0.6, 0.7, 0.4, 0.6, 0.5, 0.7,
                               0.6, 0.7, 0.5, 0.6, 0.4, 0.7, 0.6, 0.5, 0.7, 0.6]),
            "upper-middle": np.array([0.5, 0.4, 0.5, 0.3, 0.4, 0.5, 0.2, 0.4, 0.3, 0.5,
                                    0.4, 0.5, 0.3, 0.4, 0.2, 0.5, 0.4, 0.3, 0.5, 0.4]),
            "middle": np.array([0.3, 0.2, 0.3, 0.1, 0.2, 0.3, 0.0, 0.2, 0.1, 0.3,
                              0.2, 0.3, 0.1, 0.2, 0.0, 0.3, 0.2, 0.1, 0.3, 0.2]),
            "lower-middle": np.array([0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.2, 0.0, -0.1, 0.1,
                                    0.0, 0.1, -0.1, 0.0, -0.2, 0.1, 0.0, -0.1, 0.1, 0.0]),
            "lower-income": np.array([-0.1, -0.2, -0.1, -0.3, -0.2, -0.1, -0.4, -0.2, -0.3, -0.1,
                                    -0.2, -0.1, -0.3, -0.2, -0.4, -0.1, -0.2, -0.3, -0.1, -0.2])
        }
        
        # Industry patterns (next 10 dimensions)
        industry_patterns = {
            "tech": np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]),
            "finance": np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
            "traditional": np.array([0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]),
            "service": np.array([-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1])
        }
        
        for i, profile in enumerate(self.profiles):
            # Start with wealth class pattern (first 20 dimensions)
            wealth_class = profile["class"]
            base_pattern = wealth_base_patterns.get(wealth_class, wealth_base_patterns["middle"])
            
            # Determine industry type from source and name
            source = profile["source"].lower()
            name = profile["name"].lower()
            
            if any(word in source + name for word in ["tech", "software", "ai", "tesla", "spacex", "nvidia", "microsoft", "google", "amazon", "meta"]):
                industry_type = "tech"
            elif any(word in source + name for word in ["bank", "finance", "investment", "hedge", "capital", "berkshire"]):
                industry_type = "finance"
            elif any(word in source + name for word in ["retail", "food", "service", "cleaning", "delivery", "uber", "walmart"]):
                industry_type = "service"
            else:
                industry_type = "traditional"
            
            industry_pattern = industry_patterns[industry_type]
            
            embedding = np.zeros(2048) 
            embedding[:20] = base_pattern
            embedding[20:30] = industry_pattern
            
            age_factor = profile["age"] / 100.0  # 0 to ~0.8
            age_pattern = np.linspace(age_factor, age_factor * 0.5, 10)
            embedding[30:40] = age_pattern
            
            name_hash = hash(profile["name"]) % 10000
            np.random.seed(name_hash)
            
            personal_noise = np.random.normal(0, 0.02, 2048 - 40)  # Reduced noise
            embedding[40:] = personal_noise
            
            embedding += np.random.normal(0, 0.005, 2048)  # Very small noise
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        
        # Check similarity within and across classes
        mean_similarity = np.mean([
            np.dot(embeddings_array[i], embeddings_array[j])
            for i in range(len(embeddings_array))
            for j in range(i+1, len(embeddings_array))
        ])
        
        print(f"Generated {len(embeddings)} wealth-correlated embeddings")
        print(f"   Average similarity: {mean_similarity:.3f}")
        
        self._validate_embedding_quality(embeddings_array)
        
        return embeddings_array
    
    def _validate_embedding_quality(self, embeddings: np.ndarray):
        class_groups = {}
        for i, profile in enumerate(self.profiles):
            wealth_class = profile["class"]
            if wealth_class not in class_groups:
                class_groups[wealth_class] = []
            class_groups[wealth_class].append(i)
        
        within_class_sims = []
        between_class_sims = []
        
        for class_name, indices in class_groups.items():
            # Within class similarity
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    sim = np.dot(embeddings[indices[i]], embeddings[indices[j]])
                    within_class_sims.append(sim)
            
            # Between class similarity (just sample a few)
            other_indices = [idx for other_class, other_indices in class_groups.items() 
                           if other_class != class_name for idx in other_indices[:3]]
            for idx1 in indices[:3]:
                for idx2 in other_indices[:3]:
                    sim = np.dot(embeddings[idx1], embeddings[idx2])
                    between_class_sims.append(sim)
        
        avg_within = np.mean(within_class_sims) if within_class_sims else 0
        avg_between = np.mean(between_class_sims) if between_class_sims else 0
        
        print(f"   Within-class similarity: {avg_within:.3f}")
        print(f"   Between-class similarity: {avg_between:.3f}")
        
        if avg_within > avg_between:
            print("   Good: Same wealth class people are more similar")
        else:
            print("   Warning: Embeddings might not be wealth-correlated enough")
    
    def get_top_k_similar(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Find the most similar people to the uploaded photo."""
        from .utils import cosine_similarity
        
        similarities = []
        for i, profile_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, profile_embedding)
            similarities.append((self.profiles[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_profile_count_by_class(self) -> Dict[str, int]:
        """Count how many people we have in each wealth category."""
        class_counts = {}
        for profile in self.profiles:
            class_type = profile["class"]
            class_counts[class_type] = class_counts.get(class_type, 0) + 1
        return class_counts
    
    def get_wealth_distribution_stats(self) -> Dict[str, Any]:
        """Get some basic stats about our fake dataset."""
        net_worths = [p["net_worth"] for p in self.profiles]
        
        return {
            "total_profiles": len(self.profiles),
            "min_net_worth": min(net_worths),
            "max_net_worth": max(net_worths),
            "median_net_worth": np.median(net_worths),
            "mean_net_worth": np.mean(net_worths),
            "class_distribution": self.get_profile_count_by_class()
        } 