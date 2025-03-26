import math

# TODO:
# [ ] Update velocity to be corresponding to a wireless connection
# [ ] Add some noise to estimation

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius
    lat1, lon1 = map(math.radians, [lat1, lon1])
    lat2, lon2 = map(math.radians, [lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def estimate_latency(lat1, lon1, lat2, lon2):
    distance = haversine(lat1, lon1, lat2, lon2)
    
    # Speed ​​of light in optical fiber in km/s
    speed_of_light_in_fiber = 200000  # km/s
    
    # Theoretical minimum latency
    latency = (distance / speed_of_light_in_fiber) * 1000  # in ms
    
    return latency