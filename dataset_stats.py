#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.data import WealthyProfileDB
import numpy as np

def format_currency(amount):
    """Format currency with appropriate units (K, M, B)."""
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.0f}K"
    else:
        return f"${amount:,.0f}"

def main():
    print("=" * 80)
    print("WEALTH POTENTIAL ESTIMATOR - DATASET STATISTICS")
    print("=" * 80)
    
    # Initialize the database
    print("Loading comprehensive dataset...")
    db = WealthyProfileDB()
    
    # Get statistics
    stats = db.get_wealth_distribution_stats()
    class_counts = stats["class_distribution"]
    
    print(f"\nDATASET OVERVIEW")
    print(f"{'─' * 40}")
    print(f"Total Profiles: {stats['total_profiles']:,}")
    print(f"Min Net Worth: {format_currency(stats['min_net_worth'])}")
    print(f"Max Net Worth: {format_currency(stats['max_net_worth'])}")
    print(f"Median Net Worth: {format_currency(stats['median_net_worth'])}")
    print(f"Mean Net Worth: {format_currency(stats['mean_net_worth'])}")
    
    print(f"\n💰 ECONOMIC CLASS DISTRIBUTION")
    print(f"{'─' * 50}")
    
    # Define class order and descriptions
    class_info = {
        "ultra-wealthy": ("Ultra-Wealthy (Forbes Billionaires)", "$1B+", "Top 1%"),
        "wealthy": ("Wealthy (Multi-millionaires)", "$10M-$999M", "Top 5%"),
        "upper-middle": ("Upper Middle Class", "$500K-$10M", "Top 20%"),
        "middle": ("Middle Class", "$100K-$500K", "Middle 20%"),
        "lower-middle": ("Lower Middle Class", "$25K-$100K", "Next 20%"),
        "lower-income": ("Lower Income/Poverty", "$0-$25K", "Bottom 20%")
    }
    
    total_profiles = stats['total_profiles']
    
    for class_key, (name, range_str, percentile) in class_info.items():
        count = class_counts.get(class_key, 0)
        percentage = (count / total_profiles) * 100
        print(f"{name:<35} {count:>3} profiles ({percentage:>5.1f}%) | {range_str:<12} | {percentile}")
    
    print(f"\n SAMPLE PROFILES BY CLASS")
    print(f"{'─' * 70}")
    
    # Show sample profiles from each class
    for class_key, (class_name, _, _) in class_info.items():
        print(f"\n{class_name}:")
        class_profiles = [p for p in db.profiles if p["class"] == class_key]
        
        # Show first 3 profiles from each class
        for i, profile in enumerate(class_profiles[:3]):
            net_worth_str = format_currency(profile["net_worth"])
            print(f"  • {profile['name']:<25} {net_worth_str:>8} | Age {profile['age']:>2} | {profile['source']}")
        
        if len(class_profiles) > 3:
            print(f"  ... and {len(class_profiles) - 3} more profiles")
    

    
    print(f"\n🔬 TECHNICAL FEATURES")
    print(f"{'─' * 40}")
    print("• Deterministic 2048-dimensional embeddings")
    print("• Wealth-class specific feature patterns")
    print("• Age-based appearance adjustments")
    print("• Industry/profession-specific signals")
    print("• Cosine similarity matching")
    print("• Weighted averaging for net worth estimation")
    
    print(f"\n📈 EXPECTED IMPROVEMENTS")
    print(f"{'─' * 45}")
    print("• More accurate predictions across all economic levels")
    print("• Better representation of middle and lower classes")
    print("• Reduced bias toward ultra-wealthy predictions")
    print("• More diverse and realistic matching profiles")
    print("• Enhanced model training with balanced data")
    
    print("\n" + "=" * 80)
    print("Dataset ready for production use!")
    print("=" * 80)

if __name__ == "__main__":
    main() 