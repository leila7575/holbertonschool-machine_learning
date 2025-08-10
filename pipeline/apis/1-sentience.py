#!/usr/bin/env python3
"""sentientPlanets returns list of home planets of sentient species"""


import requests


def sentientPlanets():
    """returns list of home planets of sentient species"""
    url = 'https://swapi-api.hbtn.io/api/species/'
    sentient_planet = []
    while url:
        r = requests.get(url)
        data = r.json()
        for species in data["results"]:
            classification = species['classification']
            designation = species['designation']
            if 'sentient' in classification or 'sentient' in designation:
                planet_url = species['homeworld']
                if planet_url:
                    planet_r = requests.get(planet_url)
                    planet_data = planet_r.json()
                    sentient_planet.append(planet_data['name'])
        url = data.get("next")

    return sentient_planet
