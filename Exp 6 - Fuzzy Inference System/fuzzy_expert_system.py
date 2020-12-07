import numpy as np

HIGH_MILEAGE_MEMBERSHIP_CONSTANT = "high_mileage_membership"
LOW_MILEAGE_MEMBERSHIP_CONSTANT = "low_mileage_membership"
HEAVY_DAMAGE_MEMBERSHIP_CONSTANT = "heavy_damage_membership"
LIGHT_DAMAGE_MEMBERSHIP_CONSTANT = "light_damage_membership"
GOOD_SALE_MEMBERSHIP_CONSTANT = "good_sale_price_membership"
BAD_SALE_MEMBERSHIP_CONSTANT = "bad_sale_price_membership"

def get_mileage_membership(mileage: int):
    high_mileage_membership = get_high_mileage_membership(mileage)
    low_mileage_membership = 1 - high_mileage_membership
    return {HIGH_MILEAGE_MEMBERSHIP_CONSTANT: high_mileage_membership, LOW_MILEAGE_MEMBERSHIP_CONSTANT: low_mileage_membership}
    
def get_high_mileage_membership(mileage: int):
    if mileage >= 0 and mileage <= 50000:
        return 0
    elif mileage >= 100000 and mileage <= 140000:
        return 1
    else:
        return (mileage-50000)/(100000-50000)
    
def get_low_mileage_membership(mileage: int):
    if mileage >= 100000 and mileage <= 140000:
        return 0
    elif mileage >= 0 and mileage <= 50000:
        return 1
    else:
        return (mileage-50000)/(100000-50000)

def get_damage_level_membership(damage: int):
    heavy_damage_membership = get_heavy_damage_membership(damage)
    light_damage_membership = 1 - heavy_damage_membership
    return {HEAVY_DAMAGE_MEMBERSHIP_CONSTANT: heavy_damage_membership, LIGHT_DAMAGE_MEMBERSHIP_CONSTANT: light_damage_membership}

def get_heavy_damage_membership(damage):
    if damage >= 0 and damage <= 3:
        return 0
    elif damage >= 7 and damage <= 10:
        return 1
    else:
        return (damage-3)/(7-3)
    
def get_light_damage_membership(damage):
    if damage >= 7 and damage <= 10:
        return 0
    elif damage >=0 and damage <= 3:
        return 1
    else:
        return (damage-3)/(7-3)

def fuzzify(fuzzy_set:dict):
    sale_price = {}
    sale_price[BAD_SALE_MEMBERSHIP_CONSTANT] = min(fuzzy_set[HEAVY_DAMAGE_MEMBERSHIP_CONSTANT], fuzzy_set[HIGH_MILEAGE_MEMBERSHIP_CONSTANT])
    sale_price[GOOD_SALE_MEMBERSHIP_CONSTANT] = min(fuzzy_set[LIGHT_DAMAGE_MEMBERSHIP_CONSTANT], fuzzy_set[LOW_MILEAGE_MEMBERSHIP_CONSTANT])
    return sale_price

def sale(x):
	return (x * 10000) + 5000

# Mean Max Membership
def mean_max_defuzz(price_membership):
	return (sale(price_membership) + 20000) / 2.0

# Weighted Average Method
def weighted_avg_defuzz(price_membership1, price_membership2):
	a = sale(price_membership1) * price_membership1
	b = sale(price_membership2) * price_membership2
	return (a + b) / (price_membership1 + price_membership2)

# First of Maxima Method
def first_maxima_defuzz(price_membership):
	return sale(price_membership)

if __name__ == "__main__":
    print("Enter the mileage displayed on odometer: ")
    mileage = int(input())
    print("Enter the damage level: ")
    damage = int(input())
    mileage_membership = get_mileage_membership(mileage)
    print("Mileage Membership: ", str(mileage_membership))
    damage_membership = get_damage_level_membership(damage)
    print("Damage Membership: ", str(damage_membership))
    fuzzy_set = fuzzify({HIGH_MILEAGE_MEMBERSHIP_CONSTANT: mileage_membership[HIGH_MILEAGE_MEMBERSHIP_CONSTANT], LOW_MILEAGE_MEMBERSHIP_CONSTANT: mileage_membership[LOW_MILEAGE_MEMBERSHIP_CONSTANT], HEAVY_DAMAGE_MEMBERSHIP_CONSTANT: damage_membership[HEAVY_DAMAGE_MEMBERSHIP_CONSTANT], LIGHT_DAMAGE_MEMBERSHIP_CONSTANT: damage_membership[LIGHT_DAMAGE_MEMBERSHIP_CONSTANT]})
    print("After fuzzification: ", str(fuzzy_set))
    mean_max_price = mean_max_defuzz(max(fuzzy_set[GOOD_SALE_MEMBERSHIP_CONSTANT], fuzzy_set[BAD_SALE_MEMBERSHIP_CONSTANT]))
    weighted_avg_price = weighted_avg_defuzz(fuzzy_set[GOOD_SALE_MEMBERSHIP_CONSTANT], fuzzy_set[BAD_SALE_MEMBERSHIP_CONSTANT])
    first_maxima_price = first_maxima_defuzz(max(fuzzy_set[GOOD_SALE_MEMBERSHIP_CONSTANT], fuzzy_set[BAD_SALE_MEMBERSHIP_CONSTANT]))
    print("Mean Max Membership 	 : ", round(mean_max_price, 2))
    print("Weighted Average Method : ", round(weighted_avg_price, 2))
    print("First of Maxima Method  : ", round(first_maxima_price, 2))
