#!/usr/bin/env python3
"""
Example usage of the perplexity calculation script.
This creates a sample data file and runs the perplexity calculation.
"""

import json
import os

# Create sample data that matches the WebShop format
sample_data = [
    {
        "prompt": """You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: find me a blue cotton shirt under $30.
You are now at step 1 and your current observation is: You are on the homepage of the WebShop. You can search for products, browse categories, or view featured items. The search bar is prominently displayed at the top of the page.
Your admissible actions of the current situation are: 
[
search[query]: Search for products matching the query
click[element]: Click on a specific element
back: Go back to the previous page
].

Now it's your turn to take one action for the current step."""
    },
    {
        "prompt": """You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: buy a wireless mouse with good battery life.
Prior to this step, you have already taken 2 step(s). Below are the most recent 2 observations and the corresponding actions you took: Step 1: Homepage displayed. Action: search[wireless mouse]. Step 2: Search results showing various wireless mice.
You are now at step 3 and your current observation is: Search results page showing 15 wireless mice. The first result is a Logitech M705 Marathon Mouse with 3-year battery life for $39.99. The second result is a VicTsing Wireless Mouse with 15-month battery life for $12.99. The third result is a Jelly Comb 2.4G Slim Wireless Mouse with 18-month battery life for $9.99.
Your admissible actions of the current situation are: 
[
click[product_1]: Click on Logitech M705 Marathon Mouse
click[product_2]: Click on VicTsing Wireless Mouse  
click[product_3]: Click on Jelly Comb Wireless Mouse
next_page: View more results
sort[criteria]: Sort results by price/rating/relevance
filter[criteria]: Filter by brand/price/features
].

Now it's your turn to take one action for the current step."""
    },
    {
        "prompt": """You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: find a laptop backpack with multiple compartments.
You are now at step 1 and your current observation is: WebShop homepage is displayed with various categories including Electronics, Clothing, Home & Kitchen, Sports & Outdoors, and Bags & Luggage. There's a search bar at the top and featured deals showing discounts on select items.
Your admissible actions of the current situation are: 
[
search[query]: Search for specific products
click[Bags & Luggage]: Browse the Bags & Luggage category
click[featured_deal_1]: View featured laptop bag deal
].

Now it's your turn to take one action for the current step."""
    }
]

# Save sample data
sample_file = "sample_webshop_data.json"
with open(sample_file, 'w') as f:
    json.dump(sample_data, f, indent=2)

print(f"Created sample data file: {sample_file}")
print("\nTo run the perplexity calculation, use:")
print(f"python calculate_observation_perplexity.py --data_file {sample_file} --output_file perplexity_results.csv")
print("\nFor GPU acceleration (if available):")
print(f"python calculate_observation_perplexity.py --data_file {sample_file} --device cuda --output_file perplexity_results.csv")
print("\nTo use a different model (e.g., GPT-2 medium):")
print(f"python calculate_observation_perplexity.py --data_file {sample_file} --model_name gpt2-medium --output_file perplexity_results.csv")
