"""
Reference data for common crop symptoms.

NOTE:
This module is intended only for UI/reference suggestions and should
not be used directly in any core prediction logic.
"""

CROP_SYMPTOMS = {
    "Rice": ["Brown leaf spots"],
    "Maize": ["Seed rot", "seedling blights"],
    "Cotton": ["Wilted leaves", "boll rot"],
    "Soybean": ["Yellow patches on leaves"],
    "Groundnut": ["Seed rot", "seedling blight", "collar rot"],
    "Sorghum": ["Grain mold", "charcoal rot", "downy mildew"],
    "Wheat": ["Yellowing leaves"],
    "Barley": ["Net Blotch", "Barley Stripe", "Bacterial Blight"],
    "Chickpea": ["Ascochyta blight", "Fusarium wilt", "Botrytis gray mold"],
    "Mustard": ["Alternaria blight", "white rust", "Sclerotinia stem rot"],
    "Peas": ["Fungal, bacterial and viral infections"],
    "Linseed": ["Rust", "Fusarium Wilt", "Powdery Mildew"],
    "Watermelon": ["Fusarium wilt", "Gummy stem blight", "Anthracnose"],
    "Muskmelon": ["Fusarium wilt", "downy mildew", "gummy stem blight"],
    "Cucumber": ["Downy mildew", "powdery mildew", "angular leaf spot"],
    "Bitter Gourd": ["Powdery mildew", "downy mildew", "mosaic virus"],
    "Pumpkin": ["Powdery mildew", "downy mildew"],
    "Ridge Gourd": ["Powdery mildew", "downy mildew"],
}

# Reference data for common crop diseases (separate from symptoms)
# NOTE: This is for UI/reference only and should not be used in prediction logic.
CROP_DISEASES = {
    "Rice": ["Brown leaf spots"],
    "Maize": ["Seed rot and seedling blights caused by fungi like Pythium"],
    "Cotton": ["Wilted leaves", "boll rot"],
    "Soybean": ["Yellow patches on leaves"],
    "Groundnut": ["Seed rot", "seedling blight", "collar rot"],
    "Sorghum": ["Fungal infections like grain mold", "charcoal rot", "downy mildew"],
    "Wheat": ["Yellowing leaves"],
    "Barley": ["Net Blotch", "Barley Stripe", "Bacterial Blight"],
    "Chickpea": ["Ascochyta blight", "Fusarium wilt", "Botrytis gray mold"],
    "Mustard": ["Alternaria blight", "white rust", "Sclerotinia stem rot"],
    "Peas": ["A mix of fungal, bacterial, and viral infections"],
    "Linseed": ["Rust", "Fusarium Wilt", "Powdery Mildew"],
    "Watermelon": ["Fusarium wilt", "Gummy stem blight", "Anthracnose"],
    "Muskmelon": ["Fusarium wilt", "downy mildew", "gummy stem blight"],
    "Cucumber": ["Downy mildew", "powdery mildew", "angular leaf spot"],
    "Bitter Gourd": ["Powdery mildew", "downy mildew", "mosaic virus"],
    "Pumpkin": ["Powdery mildew", "downy mildew"],
    "Ridge Gourd": ["Fungal infections like powdery mildew", "downy mildew"],
}
